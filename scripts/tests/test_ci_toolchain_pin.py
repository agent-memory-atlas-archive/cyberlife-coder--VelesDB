"""Every job that builds Rust installs the pinned toolchain before it builds.

``rust-toolchain.toml`` pins a channel and the components that go with it.
rustup honours that file on every ``cargo`` call made inside the repository,
whatever toolchain an earlier step installed. A job that installs ``stable``
-- or the pinned channel without the file's components -- therefore makes
rustup install the missing toolchain, or its missing components, in the
middle of whichever step first reaches cargo.

That implicit install is where ``Python Integrations`` and ``Python SDK
Tests`` died, intermittently, inside ``pip install ./crates/velesdb-python``
(runs 34509954412, 34597461423, 34656653039, 34760724019)::

    info: syncing channel updates for 1.90-x86_64-unknown-linux-gnu
    info: downloading 6 components
    info: rolling back changes
    error: failed to install component: 'clippy-preview-x86_64-unknown-linux-gnu',
    detected conflict: 'bin/cargo-clippy'

The jobs that install the channel explicitly, components included, never
failed that way. This suite pins that shape for every workflow:

* a workflow that uses ``RUST_VERSION`` declares it, equal to the file's
  channel -- a called workflow does not inherit its caller's ``env``;
* each ``dtolnay/rust-toolchain`` step installs ``${{ env.RUST_VERSION }}``
  with every component the file names, or ``nightly``;
* no step reaches cargo -- directly, or through a build that spawns it --
  before that install, unless the call names its toolchain (``cargo +nightly``)
  or an earlier step set a rustup directory override, both of which outrank
  the file;
* no ``actions/cache`` step saves ``~/.rustup`` or ``~/.cargo/bin``, where a
  half-installed toolchain would come back as exactly this conflict.

Stdlib only: the required job that runs it has a bare interpreter.
"""

from __future__ import annotations

import math
import re
import tomllib
import unittest
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
TOOLCHAIN_FILE = REPO_ROOT / "rust-toolchain.toml"

PINNED = "${{ env.RUST_VERSION }}"
TOOLCHAIN_ACTION = "dtolnay/rust-toolchain@"
CACHE_ACTION = "actions/cache@"

# A cargo call that resolves its toolchain through rust-toolchain.toml: not
# `~/.cargo/...`, not `cargo-foo`, not `cargo +nightly`.
PLAIN_CARGO = r"(?<![\w./+-])cargo(?![\w-])(?!\s+\+)"
# ...or a build that spawns one.
BUILD_RE = re.compile(
    PLAIN_CARGO
    + r"|\bmaturin\s+(?:build|develop)\b"
    + r"|\bwasm-pack\b"
    + r"|\bnapi\s+build\b"
    + r"|\bpip\s+install\b[^\n]*\./crates/"
)
OVERRIDE_RE = re.compile(r"\brustup\s+override\s+set\b")
TOOLCHAIN_PATHS_RE = re.compile(r"\.rustup\b|\.cargo/bin\b")

KEY_RE = re.compile(r"^( *)([A-Za-z0-9_-]+):(?:\s+(.*?))?\s*$")
COMMENT_RE = re.compile(r"^\s*#")
BLOCK_MARKERS = ("", "|", "|-", ">", ">-")


def _unquote(value: str) -> str:
    value = re.sub(r"\s+#.*$", "", value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return value


def _depth(line: str) -> float:
    """Indentation of `line`; a blank line nests under whatever precedes it."""
    return len(line) - len(line.lstrip(" ")) if line.strip() else math.inf


def _mapping(lines: list[str], indent: int) -> dict[str, str | list[str]]:
    """Keys at exactly `indent`: an inline value, or the lines nested under it."""
    out: dict[str, str | list[str]] = {}
    key = None
    for line in lines:
        depth = _depth(line)
        if depth < indent:
            break
        match = KEY_RE.match(line) if depth == indent else None
        if match:
            key = match.group(2)
            raw = match.group(3) or ""
            out[key] = [] if raw in BLOCK_MARKERS else _unquote(raw)
        elif isinstance(out.get(key), list):
            out[key].append(line)
    return out


def _child(keys: dict[str, str | list[str]], name: str, indent: int) -> dict[str, str | list[str]]:
    value = keys.get(name)
    return _mapping(value, indent) if isinstance(value, list) else {}


def _text(value: str | list[str] | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else "\n".join(value)


@dataclass
class Step:
    uses: str = ""
    run: str = ""
    inputs: dict[str, str] = field(default_factory=dict)


@dataclass
class Workflow:
    env: dict[str, str]
    jobs: dict[str, list[Step]]


def _steps(lines: str | list[str] | None) -> list[Step]:
    items: list[list[str]] = []
    for line in lines if isinstance(lines, list) else []:
        if re.match(r"^ {6}- ", line):
            items.append([" " * 8 + line[8:]])
        elif items:
            items[-1].append(line)
    steps = []
    for item in items:
        keys = _mapping(item, 8)
        inputs = {name: _text(value).strip() for name, value in _child(keys, "with", 10).items()}
        steps.append(Step(uses=_text(keys.get("uses")), run=_text(keys.get("run")), inputs=inputs))
    return steps


def parse(text: str) -> Workflow:
    top = _mapping([line for line in text.splitlines() if not COMMENT_RE.match(line)], 0)
    return Workflow(
        env={name: _text(value) for name, value in _child(top, "env", 2).items()},
        jobs={name: _steps(_child({"job": body}, "job", 4).get("steps")) for name, body in _child(top, "jobs", 2).items()},
    )


def toolchain_file() -> tuple[str, frozenset[str]]:
    table = tomllib.loads(TOOLCHAIN_FILE.read_text(encoding="utf-8"))["toolchain"]
    return str(table["channel"]), frozenset(table.get("components", ()))


def _components(step: Step) -> set[str]:
    return {part.strip() for part in step.inputs.get("components", "").split(",") if part.strip()}


def _env_findings(workflow: Workflow, text: str, channel: str) -> list[str]:
    found = []
    declared = workflow.env.get("RUST_VERSION")
    if declared is not None and declared != channel:
        found.append(f"RUST_VERSION is {declared!r}, rust-toolchain.toml pins {channel!r}")
    if PINNED in text and declared is None:
        found.append("uses RUST_VERSION without declaring it in the workflow env")
    return found


def _toolchain_finding(where: str, step: Step, components: frozenset[str]) -> tuple[bool, str | None]:
    """(installs the pinned toolchain, what is wrong with the step)."""
    toolchain = step.inputs.get("toolchain", "")
    if toolchain == PINNED:
        missing = sorted(components - _components(step))
        return True, f"{where} installs {PINNED} without {missing}" if missing else None
    if toolchain == "nightly":
        return False, None
    return False, f"{where} installs {toolchain or 'the action ref'!r}, not {PINNED}"


def _cache_finding(where: str, step: Step) -> str | None:
    saved = step.inputs.get("path", "")
    if step.uses.startswith(CACHE_ACTION) and TOOLCHAIN_PATHS_RE.search(saved):
        return f"{where} caches a toolchain path: {saved!r}"
    return None


def _job_findings(job: str, steps: list[Step], components: frozenset[str]) -> list[str]:
    found: list[str | None] = []
    ready = False  # the pinned toolchain is installed, or a rustup override outranks the file
    for index, step in enumerate(steps):
        where = f"{job}: step {index}"
        if step.uses.startswith(TOOLCHAIN_ACTION):
            pinned, finding = _toolchain_finding(where, step, components)
            ready |= pinned
            found.append(finding)
        found.append(_cache_finding(where, step))
        ready |= bool(OVERRIDE_RE.search(step.run))
        if BUILD_RE.search(step.run) and not ready:
            found.append(f"{where} reaches cargo before the pinned toolchain is installed")
    return [finding for finding in found if finding]


def findings(text: str, channel: str, components: frozenset[str]) -> list[str]:
    workflow = parse(text)
    found = _env_findings(workflow, text, channel)
    for job, steps in workflow.jobs.items():
        found += _job_findings(job, steps, components)
    return found


def real_findings() -> dict[str, list[str]]:
    channel, components = toolchain_file()
    report = {}
    for workflow in sorted(WORKFLOW_DIR.glob("*.yml")):
        found = findings(workflow.read_text(encoding="utf-8"), channel, components)
        if found:
            report[workflow.name] = found
    return report


GOOD = """\
env:
  RUST_VERSION: "1.90"
jobs:
  py:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      # a comment between two steps
      - name: Install the pinned Rust toolchain
        uses: dtolnay/rust-toolchain@stable # Intentionally unpinned - uses toolchain input
        if: always()
        with:
          toolchain: ${{ env.RUST_VERSION }}
          components: rustfmt, clippy
      - name: Cache Cargo registry
        uses: actions/cache@v6
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
          key: k
      - name: Build
        run: |
          # PEP 517 build
          pip install ./crates/velesdb-python --force-reinstall
  deep:
    runs-on: ubuntu-latest
    steps:
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: nightly
      - run: cargo +nightly install cargo-fuzz
      - run: |
          rustup override set nightly
          cargo miri test
"""

COMPONENTS = frozenset({"rustfmt", "clippy"})


def _variant(old: str, new: str) -> list[str]:
    if old not in GOOD:
        raise ValueError(f"not in the reference workflow: {old!r}")
    return findings(GOOD.replace(old, new), "1.90", COMPONENTS)


class FindingsTests(unittest.TestCase):
    """The checker itself: silent on the right shape, loud on each wrong one."""

    def test_the_right_shape_is_silent(self) -> None:
        self.assertEqual([], findings(GOOD, "1.90", COMPONENTS))

    def test_the_parser_sees_every_step(self) -> None:
        workflow = parse(GOOD)
        self.assertEqual(["py", "deep"], list(workflow.jobs))
        self.assertEqual(4, len(workflow.jobs["py"]))
        self.assertEqual("rustfmt, clippy", workflow.jobs["py"][1].inputs["components"])
        self.assertIn("pip install ./crates/velesdb-python", workflow.jobs["py"][3].run)

    def test_the_failing_shape_is_refused(self) -> None:
        # What Python Integrations shipped: `stable`, no inputs, then pip.
        found = _variant(
            "        with:\n          toolchain: ${{ env.RUST_VERSION }}\n          components: rustfmt, clippy\n",
            "",
        )
        self.assertTrue(any("'the action ref', not" in f for f in found), found)
        self.assertTrue(any("py: step 3 reaches cargo" in f for f in found), found)

    def test_stable_is_refused(self) -> None:
        found = _variant("toolchain: ${{ env.RUST_VERSION }}", "toolchain: stable")
        self.assertTrue(any("'stable', not" in f for f in found), found)

    def test_missing_components_are_refused(self) -> None:
        found = _variant("components: rustfmt, clippy", "components: llvm-tools-preview")
        self.assertEqual([f"py: step 1 installs {PINNED} without ['clippy', 'rustfmt']"], found)

    def test_a_build_before_the_install_is_refused(self) -> None:
        text = GOOD.replace(
            "      - uses: actions/checkout@v7\n",
            "      - uses: actions/checkout@v7\n      - run: cargo machete\n",
        )
        found = findings(text, "1.90", COMPONENTS)
        self.assertTrue(any("py: step 1 reaches cargo" in f for f in found), found)

    def test_a_plain_cargo_in_a_nightly_job_is_refused(self) -> None:
        found = _variant("cargo +nightly install cargo-fuzz", "cargo install cargo-fuzz")
        self.assertEqual(["deep: step 1 reaches cargo before the pinned toolchain is installed"], found)

    def test_a_cache_holding_a_toolchain_is_refused(self) -> None:
        for path in ("~/.rustup", "~/.cargo/bin"):
            with self.subTest(path=path):
                found = _variant("~/.cargo/git", path)
                self.assertTrue(any("caches a toolchain path" in f for f in found), found)

    def test_a_drifted_pin_is_refused(self) -> None:
        self.assertEqual(
            ["RUST_VERSION is '1.89', rust-toolchain.toml pins '1.90'"],
            _variant('RUST_VERSION: "1.90"', 'RUST_VERSION: "1.89"'),
        )

    def test_an_undeclared_pin_is_refused(self) -> None:
        found = _variant('env:\n  RUST_VERSION: "1.90"\n', "")
        self.assertIn("uses RUST_VERSION without declaring it in the workflow env", found)


class RealWorkflowTests(unittest.TestCase):
    def test_the_toolchain_file_names_a_channel_and_components(self) -> None:
        channel, components = toolchain_file()
        self.assertRegex(channel, r"^\d+\.\d+(\.\d+)?$")
        self.assertTrue(components, "rust-toolchain.toml names no components: nothing to pin")

    def test_every_workflow_parses_into_jobs_with_steps(self) -> None:
        workflows = sorted(WORKFLOW_DIR.glob("*.yml"))
        self.assertTrue(workflows)
        for path in workflows:
            with self.subTest(workflow=path.name):
                jobs = parse(path.read_text(encoding="utf-8")).jobs
                self.assertTrue(jobs, "no job parsed")
                self.assertTrue(any(jobs.values()), "no step parsed")

    def test_the_python_build_jobs_install_the_pinned_toolchain_before_pip(self) -> None:
        _, components = toolchain_file()
        jobs = parse((WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")).jobs
        for name in ("python-integrations", "python-sdk-tests"):
            with self.subTest(job=name):
                steps = jobs[name]
                build = next(i for i, s in enumerate(steps) if "pip install ./crates/" in s.run)
                install = next(
                    i for i, s in enumerate(steps) if s.uses.startswith(TOOLCHAIN_ACTION)
                )
                self.assertLess(install, build)
                self.assertEqual(PINNED, steps[install].inputs.get("toolchain"))
                self.assertLessEqual(components, _components(steps[install]))

    def test_no_job_reaches_cargo_without_the_pinned_toolchain(self) -> None:
        report = real_findings()
        self.assertEqual(
            {},
            report,
            "\n".join(f"{wf}: {f}" for wf, found in report.items() for f in found),
        )


if __name__ == "__main__":
    unittest.main()
