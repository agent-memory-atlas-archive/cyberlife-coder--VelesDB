"""Every job that builds Rust installs rust-toolchain.toml's toolchain, from that file, first.

rustup resolves ``rust-toolchain.toml`` on every ``cargo`` or ``rustc`` call made
inside the repository, whatever toolchain a workflow step installed. A version
written in a workflow is therefore dead, and worse than dead: when it is not
the file's, or lacks the file's components, the first cargo call makes rustup
install the file's toolchain in the middle of a build step. That implicit
install is where ``Python Integrations`` and ``Python SDK Tests`` died,
intermittently, inside ``pip install ./crates/velesdb-python`` (runs
34509954412, 34597461423, 34656653039, 34760724019)::

    info: syncing channel updates for 1.90-x86_64-unknown-linux-gnu
    info: downloading 6 components
    info: rolling back changes
    error: failed to install component: 'clippy-preview-x86_64-unknown-linux-gnu',
    detected conflict: 'bin/cargo-clippy'

So the file is the only pin, and this suite holds every workflow to it:

* no toolchain version in a workflow -- no ``RUST_VERSION``, no ``toolchain:``,
  ``RUSTUP_TOOLCHAIN``, ``rustup ... <toolchain>`` or ``cargo +<toolchain>``
  other than ``nightly``, and the file's channel nowhere outside a comment;
* before a job's first cargo call -- a build step, or ``Swatinem/rust-cache``,
  which runs ``cargo metadata`` -- the job runs ``rustup toolchain install``
  with no toolchain name, which installs exactly what the file names,
  components included; unless every call in the job is pinned to ``nightly``
  (``RUSTUP_TOOLCHAIN`` or a ``rustup override``), which outranks the file;
* each place that pins ``nightly`` carries a comment saying why;
* no ``actions/cache`` step saves ``~/.rustup`` or ``~/.cargo/bin``.

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

ALLOWED = "nightly"
TOOLCHAIN_ACTION = "dtolnay/rust-toolchain@"
CACHE_ACTION = "actions/cache@"
RUST_CACHE_ACTION = "Swatinem/rust-cache@"

# A cargo call that resolves its toolchain through rust-toolchain.toml: not
# `~/.cargo/...`, not `cargo-foo`, not `cargo +nightly`...
PLAIN_CARGO = r"(?<![\w./+-])cargo(?![\w-])(?!\s+\+)"
# ...or a build that spawns one.
BUILD_RE = re.compile(
    PLAIN_CARGO
    + r"|\bmaturin\s+(?:build|develop)\b"
    + r"|\bwasm-pack\b"
    + r"|\bnapi\s+build\b"
    + r"|\bpip\s+install\b[^\n]*\./crates/"
)
RUSTUP_RE = re.compile(r"\brustup\s+(toolchain\s+install|install|update|default|override\s+set|run)\b([^\n;&|]*)")
PLUS_RE = re.compile(r"(?<![\w./-])(?:cargo|rustc|rustdoc)\s+\+(\S+)")
KEYED_RE = re.compile(r"^\s*(?:toolchain|RUSTUP_TOOLCHAIN):\s*(\S.*?)\s*$")
OVERRIDE_RE = re.compile(r"\brustup\s+override\s+set\b")
TOOLCHAIN_PATHS_RE = re.compile(r"\.rustup\b|\.cargo/bin\b")
VALUE_FLAGS = frozenset({"--profile", "-c", "--component", "-t", "--target"})

# Lines of the step or key that holds a pin, walked over to reach its comment.
STRUCTURAL_RE = re.compile(r"^\s*(?:- )?(?:name|uses|if|with|env|run)\s*:")

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
class Job:
    env: dict[str, str]
    steps: list[Step]


@dataclass
class Workflow:
    env: dict[str, str]
    jobs: dict[str, Job]


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


def _job(body: str | list[str]) -> Job:
    keys = _child({"job": body}, "job", 4)
    env = {name: _text(value) for name, value in _child(keys, "env", 6).items()}
    return Job(env=env, steps=_steps(keys.get("steps")))


def parse(text: str) -> Workflow:
    top = _mapping([line for line in text.splitlines() if not COMMENT_RE.match(line)], 0)
    return Workflow(
        env={name: _text(value) for name, value in _child(top, "env", 2).items()},
        jobs={name: _job(body) for name, body in _child(top, "jobs", 2).items()},
    )


def toolchain_file() -> tuple[str, frozenset[str]]:
    table = tomllib.loads(TOOLCHAIN_FILE.read_text(encoding="utf-8"))["toolchain"]
    return str(table["channel"]), frozenset(table.get("components", ()))


def _named(args: str) -> str | None:
    """The toolchain a rustup command line names, or None when it names none."""
    skip = False
    for token in args.split():
        if skip:
            skip = False
        elif token in VALUE_FLAGS:
            skip = True
        elif not token.startswith("-"):
            return token
    return None


def _pins(line: str) -> list[tuple[str, bool]]:
    """(toolchain, is a pin site that must say why) for each toolchain the line names."""
    pins = [(match.group(1), False) for match in PLUS_RE.finditer(line)]
    pins += [(name, True) for match in RUSTUP_RE.finditer(line) if (name := _named(match.group(2)))]
    keyed = KEYED_RE.match(line)
    if keyed:
        pins.append((_unquote(keyed.group(1)), True))
    return pins


def _says_why(raw: list[str], index: int) -> bool:
    """A comment sits right above the pin, or above the step or key holding it."""
    indent = _depth(raw[index])
    for line in reversed(raw[:index]):
        if COMMENT_RE.match(line):
            return True
        if not (STRUCTURAL_RE.match(line) or _depth(line) >= indent) or not line.strip():
            return False
    return False


def _unexplained_nightly(raw: list[str], index: int, pins: list[tuple[str, bool]]) -> bool:
    return any(site for name, site in pins if name == ALLOWED) and not _says_why(raw, index)


def _line_findings(raw: list[str], index: int, channel_re: re.Pattern[str]) -> list[str]:
    line, where = raw[index], f"line {index + 1}"
    pins = _pins(line)
    found = [f"{where} names toolchain {name!r}" for name, _ in pins if name != ALLOWED]
    if _unexplained_nightly(raw, index, pins):
        found.append(f"{where} pins nightly without a comment saying why")
    if "RUST_VERSION" in line or channel_re.search(line):
        found.append(f"{where} copies the toolchain version: {line.strip()!r}")
    return found


def _literal_findings(text: str, channel: str) -> list[str]:
    raw = text.splitlines()
    channel_re = re.compile(r"(?<![\w.])" + re.escape(channel) + r"(?![\w.])")
    found = []
    for index, line in enumerate(raw):
        if not COMMENT_RE.match(line):
            found += _line_findings(raw, index, channel_re)
    return found


def _installs_from_the_file(run: str) -> bool:
    return any(m.group(1).startswith("toolchain") and _named(m.group(2)) is None for m in RUSTUP_RE.finditer(run))


def _reaches_cargo(step: Step) -> bool:
    return step.uses.startswith(RUST_CACHE_ACTION) or bool(BUILD_RE.search(step.run))


def _step_finding(where: str, step: Step) -> str | None:
    if step.uses.startswith(TOOLCHAIN_ACTION) and "toolchain" not in step.inputs:
        return f"{where} installs the action's ref, not rust-toolchain.toml's toolchain"
    if step.uses.startswith(CACHE_ACTION) and TOOLCHAIN_PATHS_RE.search(step.inputs.get("path", "")):
        return f"{where} caches a toolchain path: {step.inputs['path']!r}"
    return None


def _job_findings(name: str, job: Job, pinned_everywhere: bool) -> list[str]:
    found: list[str | None] = []
    ready = pinned_everywhere or "RUSTUP_TOOLCHAIN" in job.env
    for index, step in enumerate(job.steps):
        where = f"{name}: step {index}"
        found.append(_step_finding(where, step))
        ready |= _installs_from_the_file(step.run)
        ready |= bool(OVERRIDE_RE.search(step.run))
        if _reaches_cargo(step) and not ready:
            found.append(f"{where} reaches cargo before rust-toolchain.toml's toolchain is installed")
    return [finding for finding in found if finding]


def findings(text: str, channel: str) -> list[str]:
    workflow = parse(text)
    found = _literal_findings(text, channel)
    for name, job in workflow.jobs.items():
        found += _job_findings(name, job, "RUSTUP_TOOLCHAIN" in workflow.env)
    return found


def real_findings() -> dict[str, list[str]]:
    channel, _ = toolchain_file()
    report = {}
    for workflow in sorted(WORKFLOW_DIR.glob("*.yml")):
        found = findings(workflow.read_text(encoding="utf-8"), channel)
        if found:
            report[workflow.name] = found
    return report


INSTALL = "        run: rustup toolchain install --no-self-update --profile minimal\n"
GOOD = """\
jobs:
  py:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      # a comment between two steps
      - name: Install the toolchain rust-toolchain.toml pins
        if: always()
""" + INSTALL + """\
      - name: Cache Cargo registry
        uses: actions/cache@v6
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
          key: k
      - uses: Swatinem/rust-cache@v2
      - name: Build
        run: |
          # PEP 517 build
          pip install ./crates/velesdb-python --force-reinstall
  fuzz:
    runs-on: ubuntu-latest
    env:
      # Nightly on purpose: cargo-fuzz instruments with -Zsanitizer.
      RUSTUP_TOOLCHAIN: nightly
    steps:
      # Installed explicitly, so rustup never installs it mid-step.
      - uses: dtolnay/rust-toolchain@abc # nightly
        with:
          toolchain: nightly
      - uses: Swatinem/rust-cache@v2
      - run: cargo install cargo-fuzz
      - run: cargo +nightly fuzz run target
  miri:
    runs-on: ubuntu-latest
    steps:
      # Miri ships with nightly only; the override keeps every call on it.
      - run: |
          rustup toolchain install nightly --component miri
          rustup override set nightly
          cargo miri test
"""


def _variant(old: str, new: str) -> list[str]:
    if old not in GOOD:
        raise ValueError(f"not in the reference workflow: {old!r}")
    return findings(GOOD.replace(old, new), "1.90")


def _has(found: list[str], fragment: str) -> bool:
    return any(fragment in finding for finding in found)


class FindingsTests(unittest.TestCase):
    """The checker itself: silent on the right shape, loud on each wrong one."""

    def test_the_right_shape_is_silent(self) -> None:
        self.assertEqual([], findings(GOOD, "1.90"))

    def test_the_parser_sees_every_step_and_the_job_env(self) -> None:
        workflow = parse(GOOD)
        self.assertEqual(["py", "fuzz", "miri"], list(workflow.jobs))
        self.assertEqual(5, len(workflow.jobs["py"].steps))
        self.assertEqual("nightly", workflow.jobs["fuzz"].env["RUSTUP_TOOLCHAIN"])
        self.assertIn("pip install ./crates/velesdb-python", workflow.jobs["py"].steps[4].run)

    def test_the_failing_shape_is_refused(self) -> None:
        # What Python Integrations shipped: `stable` from the action's ref, then pip.
        found = _variant(
            "      - name: Install the toolchain rust-toolchain.toml pins\n        if: always()\n" + INSTALL,
            "      - uses: dtolnay/rust-toolchain@stable\n",
        )
        self.assertTrue(_has(found, "py: step 1 installs the action's ref"), found)
        self.assertTrue(_has(found, "py: step 3 reaches cargo"), found)
        self.assertTrue(_has(found, "py: step 4 reaches cargo"), found)

    def test_a_version_literal_is_refused(self) -> None:
        found = _variant(INSTALL, '        uses: dtolnay/rust-toolchain@stable\n        with:\n          toolchain: "1.90"\n')
        self.assertTrue(_has(found, "names toolchain '1.90'"), found)
        self.assertTrue(_has(found, "copies the toolchain version"), found)

    def test_a_rust_version_copy_is_refused(self) -> None:
        found = findings('env:\n  RUST_VERSION: "1.90"\n' + GOOD, "1.90")
        self.assertTrue(_has(found, "copies the toolchain version"), found)

    def test_a_named_rustup_install_is_refused(self) -> None:
        found = _variant("install --no-self-update", "install 1.86 --no-self-update")
        self.assertTrue(_has(found, "names toolchain '1.86'"), found)
        self.assertTrue(_has(found, "py: step 3 reaches cargo"), found)

    def test_a_cargo_plus_version_is_refused(self) -> None:
        found = _variant("cargo +nightly fuzz", "cargo +1.86 fuzz")
        self.assertEqual(1, len(found), found)
        self.assertTrue(_has(found, "names toolchain '1.86'"), found)

    def test_a_build_before_the_install_is_refused(self) -> None:
        text = GOOD.replace(
            "      - uses: actions/checkout@v7\n",
            "      - uses: actions/checkout@v7\n      - run: cargo machete\n",
        )
        self.assertTrue(_has(findings(text, "1.90"), "py: step 1 reaches cargo"))

    def test_rust_cache_before_the_install_is_refused(self) -> None:
        text = GOOD.replace(
            "      - uses: actions/checkout@v7\n",
            "      - uses: actions/checkout@v7\n      - uses: Swatinem/rust-cache@v2\n",
        )
        self.assertTrue(_has(findings(text, "1.90"), "py: step 1 reaches cargo"))

    def test_a_nightly_job_that_leaves_the_file_in_charge_is_refused(self) -> None:
        found = _variant(
            "    env:\n      # Nightly on purpose: cargo-fuzz instruments with -Zsanitizer.\n      RUSTUP_TOOLCHAIN: nightly\n",
            "",
        )
        self.assertEqual(
            [
                "fuzz: step 1 reaches cargo before rust-toolchain.toml's toolchain is installed",
                "fuzz: step 2 reaches cargo before rust-toolchain.toml's toolchain is installed",
            ],
            found,
        )

    def test_a_nightly_pin_without_a_reason_is_refused(self) -> None:
        for comment in (
            "      # Nightly on purpose: cargo-fuzz instruments with -Zsanitizer.\n",
            "      # Miri ships with nightly only; the override keeps every call on it.\n",
        ):
            with self.subTest(comment=comment):
                self.assertTrue(_has(_variant(comment, ""), "pins nightly without a comment"))

    def test_a_cache_holding_a_toolchain_is_refused(self) -> None:
        for path in ("~/.rustup", "~/.cargo/bin"):
            with self.subTest(path=path):
                self.assertTrue(_has(_variant("~/.cargo/git", path), "caches a toolchain path"))


class RealWorkflowTests(unittest.TestCase):
    def test_the_toolchain_file_names_a_channel_and_components(self) -> None:
        channel, components = toolchain_file()
        self.assertRegex(channel, r"^\d+\.\d+(\.\d+)?$")
        self.assertTrue(components, "rust-toolchain.toml names no components")

    def test_every_workflow_parses_into_jobs_with_steps(self) -> None:
        workflows = sorted(WORKFLOW_DIR.glob("*.yml"))
        self.assertTrue(workflows)
        for path in workflows:
            with self.subTest(workflow=path.name):
                jobs = parse(path.read_text(encoding="utf-8")).jobs
                self.assertTrue(jobs, "no job parsed")
                self.assertTrue(any(job.steps for job in jobs.values()), "no step parsed")

    def test_the_python_build_jobs_install_from_the_file_before_pip(self) -> None:
        jobs = parse((WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")).jobs
        for name in ("python-integrations", "python-sdk-tests"):
            with self.subTest(job=name):
                steps = jobs[name].steps
                build = next(i for i, s in enumerate(steps) if "pip install ./crates/" in s.run)
                install = next((i for i, s in enumerate(steps) if _installs_from_the_file(s.run)), None)
                self.assertIsNotNone(install, "no `rustup toolchain install` from the file")
                self.assertLess(install, build)

    def test_no_workflow_pins_a_toolchain_or_reaches_cargo_before_installing_it(self) -> None:
        report = real_findings()
        self.assertEqual(
            {},
            report,
            "\n".join(f"{wf}: {f}" for wf, found in report.items() for f in found),
        )


if __name__ == "__main__":
    unittest.main()
