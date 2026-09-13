"""Every build of this repository installs rust-toolchain.toml's toolchain, from that file, first.

rustup resolves ``rust-toolchain.toml`` on every ``cargo`` or ``rustc`` call made
inside the repository, whatever toolchain a step installed. A version written in
a workflow is therefore dead, and worse than dead: when it is not the file's, or
lacks the file's components, the first cargo call makes rustup install the
file's toolchain in the middle of a build step. That implicit install is where
``Python Integrations`` and ``Python SDK Tests`` died, intermittently, inside
``pip install ./crates/velesdb-python`` (runs 34509954412, 34597461423,
34656653039, 34760724019)::

    info: syncing channel updates for 1.90-x86_64-unknown-linux-gnu
    info: downloading 6 components
    info: rolling back changes
    error: failed to install component: 'clippy-preview-x86_64-unknown-linux-gnu',
    detected conflict: 'bin/cargo-clippy'

So the file is the only pin. Over every workflow (``*.yml``, ``*.yaml``), every
composite action and every Dockerfile, this suite holds that:

* nothing names a toolchain but ``nightly``: no ``RUST_VERSION``, no
  ``*toolchain:`` input, no ``RUSTUP_TOOLCHAIN`` assignment (key, inline,
  ``export`` or ``$GITHUB_ENV``), no ``rustup toolchain install|add``,
  ``install``, ``update``, ``default``, ``override set|add`` or ``run`` naming
  one, no ``cargo +<toolchain>`` (by path too), and the file's channel nowhere
  outside a comment;
* every call that reaches cargo -- a build, ``cargo +X``, or
  ``Swatinem/rust-cache``, which runs ``cargo metadata`` -- finds its toolchain
  installed by an earlier step of the same job: ``X`` for ``cargo +X``; else the
  ``RUSTUP_TOOLCHAIN`` or rustup override in force; else the file of the
  checkout the call runs in, installed by ``rustup toolchain install`` with no
  toolchain name (a checkout with ``path:`` carries its own copy);
* each ``nightly`` pin carries a ``# nightly: <why>`` comment;
* no ``actions/cache`` step saves ``~/.rustup`` or ``~/.cargo/bin``;
* no tracked file claims loom runs on nightly: it needs only ``--cfg loom``.

Stdlib only: the required job that runs it has a bare interpreter.
"""

from __future__ import annotations

import math
import posixpath
import re
import subprocess
import tempfile
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
CHECKOUT_ACTION = "actions/checkout@"

# A call that resolves its toolchain through rust-toolchain.toml or RUSTUP_TOOLCHAIN:
# cargo, rustc or rustdoc, bare or by path, but not `~/.cargo/...`, `cargo-foo`
# or `cargo +<toolchain>`...
PLAIN_CALL = r"(?<![\w.+-])(?:cargo|rustc|rustdoc)(?:\.exe)?(?![\w.-])(?!\s+\+)"
# ...or a build that spawns one.
BUILD_RE = re.compile(
    PLAIN_CALL
    + r"|\bmaturin\s+(?:build|develop)\b"
    + r"|\bwasm-pack\b"
    + r"|\bnapi\s+build\b"
    + r"|\bpip\s+install\b[^\n]*\./crates/"
)
PLUS_RE = re.compile(r"(?<![\w.-])(?:cargo|rustc|rustdoc)(?:\.exe)?\s+\+([^\s\"'`;&|)]+)")
RUSTUP_RE = re.compile(
    r"\brustup(?:\.exe)?\s+(toolchain\s+(?:install|add)|install|update|default|override\s+(?:set|add)|run)\b([^\n;&|]*)"
)
KEYED_RE = re.compile(r"^\s*(?:-\s+)?(?:[\w-]*toolchain|RUSTUP_TOOLCHAIN)\s*:\s*(\S.*?)\s*$")
ASSIGN_RE = re.compile(r"\bRUSTUP_TOOLCHAIN\s*=\s*[\"']?([^\s\"'`;&|)]+)")
DOCKER_ENV_RE = re.compile(r"^\s*ENV\s+RUSTUP_TOOLCHAIN\s+([^\s=]+)")
EXPORT_RE = re.compile(r"^\s*export\s+RUSTUP_TOOLCHAIN=")
CD_RE = re.compile(r"^\s*cd\s+([^\s;&|]+)")
INSTALL_VERBS = frozenset({"toolchain install", "toolchain add", "install"})
OVERRIDE_VERBS = frozenset({"override set", "override add"})
# RUST_VERSION as a key or a reference, never inside a longer name such as
# CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS (a cargo resolver setting).
RUST_VERSION_RE = re.compile(r"(?<![\w.$])RUST_VERSION\s*[:=]|\benv\.RUST_VERSION\b|\$\{?RUST_VERSION\b")
TOOLCHAIN_PATHS_RE = re.compile(r"\.rustup\b|\.cargo/bin\b")
VALUE_FLAGS = frozenset({"--profile", "-c", "--component", "-t", "--target"})
NIGHTLY_REASON_RE = re.compile(r"^\s*#\s*nightly:\s*\S")
# Lines of the step or key that holds a pin, walked over to reach its comment.
STRUCTURAL_RE = re.compile(r"^\s*(?:- )?(?:name|uses|if|with|env|run|shell|working-directory)\s*:")
DOCKERFILE_NAME_RE = re.compile(r"(?:^|/)(?:Dockerfile(?:\.[\w.-]+)?|[\w.-]+\.Dockerfile)$")
DOC_SUFFIXES = frozenset({".md", ".rs", ".toml", ".yml", ".yaml", ".txt", ".py", ".sh", ".ps1"})
LOOM_CLAIM_EXEMPT = {
    "CHANGELOG.md": "history is not rewritten",
    "scripts/tests/test_ci_toolchain_pin.py": "the checker spells the words it looks for",
}

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


def _indent(lines: list[str]) -> float:
    return next((_depth(line) for line in lines if line.strip()), 0)


def _value(raw: str) -> str | list[str]:
    return [] if raw in BLOCK_MARKERS else _unquote(raw)


def _mapping(lines: list[str]) -> dict[str, str | list[str]]:
    """Keys at the block's own indentation: an inline value, or the lines nested under it."""
    indent = _indent(lines)
    out: dict[str, str | list[str]] = {}
    key = None
    for line in lines:
        depth = _depth(line)
        if depth < indent:
            break
        match = KEY_RE.match(line) if depth == indent else None
        if match:
            key = match.group(2)
            out[key] = _value(match.group(3) or "")
        elif isinstance(out.get(key), list):
            out[key].append(line)
    return out


def _child(keys: dict[str, str | list[str]], name: str) -> dict[str, str | list[str]]:
    value = keys.get(name)
    return _mapping(value) if isinstance(value, list) else {}


def _text(value: str | list[str] | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else "\n".join(value)


def _flat(keys: dict[str, str | list[str]], name: str) -> dict[str, str]:
    return {key: _text(value).strip() for key, value in _child(keys, name).items()}


@dataclass
class Step:
    uses: str = ""
    run: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    directory: str = ""


@dataclass
class Job:
    env: dict[str, str]
    steps: list[Step]
    directory: str = ""


@dataclass
class Workflow:
    env: dict[str, str]
    jobs: dict[str, Job]
    directory: str = ""


def _items(lines: list[str]) -> list[list[str]]:
    """The `- ` items of a block sequence, whatever its indentation."""
    depth = next((_depth(line) for line in lines if line.lstrip().startswith("- ")), None)
    items: list[list[str]] = []
    for line in lines:
        if _depth(line) == depth and line.lstrip().startswith("- "):
            items.append([line.replace("- ", "  ", 1)])
        elif items:
            items[-1].append(line)
    return items


def _step(item: list[str]) -> Step:
    keys = _mapping(item)
    return Step(
        uses=_text(keys.get("uses")),
        run=_text(keys.get("run")),
        inputs=_flat(keys, "with"),
        env=_flat(keys, "env"),
        directory=_text(keys.get("working-directory")),
    )


def _working_directory(keys: dict[str, str | list[str]]) -> str:
    return _text(_child(_child(keys, "defaults"), "run").get("working-directory"))


def _job(body: str | list[str]) -> Job:
    keys = _mapping(body) if isinstance(body, list) else {}
    steps = keys.get("steps")
    return Job(
        env=_flat(keys, "env"),
        steps=[_step(item) for item in _items(steps if isinstance(steps, list) else [])],
        directory=_working_directory(keys),
    )


def parse(text: str) -> Workflow:
    top = _mapping([line for line in text.splitlines() if not COMMENT_RE.match(line)])
    jobs = {name: _job(body) for name, body in _child(top, "jobs").items()}
    if not jobs and isinstance(top.get("runs"), list):
        jobs = {"runs": _job(top["runs"])}  # a composite action: one job, `runs.steps`
    return Workflow(env=_flat(top, "env"), jobs=jobs, directory=_working_directory(top))


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


def _verb(match: re.Match[str]) -> str:
    return " ".join(match.group(1).split())


def _installs(verb: str, named: str | None) -> bool:
    """`rustup toolchain install|add`, `rustup install`, or `rustup update <toolchain>`."""
    return verb in INSTALL_VERBS or (verb == "update" and named is not None)


def _pins(line: str) -> list[tuple[str, bool]]:
    """(toolchain, is a pin site that must say why) for each toolchain the line names."""
    pins = [(match.group(1), False) for match in PLUS_RE.finditer(line)]
    pins += [(name, True) for match in RUSTUP_RE.finditer(line) if (name := _named(match.group(2)))]
    pins += [(_unquote(match.group(1)), True) for match in ASSIGN_RE.finditer(line)]
    keyed = KEYED_RE.match(line) or DOCKER_ENV_RE.match(line)
    if keyed:
        pins.append((_unquote(keyed.group(1)), True))
    return pins


def _gives_a_reason(raw: list[str], position: int) -> bool:
    """The comment block ending at `position` holds a `# nightly: <why>` line."""
    while position >= 0 and COMMENT_RE.match(raw[position]):
        if NIGHTLY_REASON_RE.match(raw[position]):
            return True
        position -= 1
    return False


def _says_why(raw: list[str], index: int) -> bool:
    """A `# nightly: <why>` comment sits right above the pin, or above the step or key holding it."""
    indent = _depth(raw[index])
    for position in range(index - 1, -1, -1):
        line = raw[position]
        if COMMENT_RE.match(line):
            return _gives_a_reason(raw, position)
        if not line.strip() or not (STRUCTURAL_RE.match(line) or _depth(line) >= indent):
            return False
    return False


def _unexplained_nightly(raw: list[str], index: int, pins: list[tuple[str, bool]]) -> bool:
    return any(site for name, site in pins if name == ALLOWED) and not _says_why(raw, index)


def _line_findings(raw: list[str], index: int, channel_re: re.Pattern[str]) -> list[str]:
    line, where = raw[index], f"line {index + 1}"
    pins = _pins(line)
    found = [f"{where} names toolchain {name!r}" for name, _ in pins if name != ALLOWED]
    if _unexplained_nightly(raw, index, pins):
        found.append(f"{where} pins nightly without a comment `# nightly: <why>`")
    if RUST_VERSION_RE.search(line) or channel_re.search(line):
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
    return any(_verb(m) in INSTALL_VERBS and _named(m.group(2)) is None for m in RUSTUP_RE.finditer(run))


def _norm(directory: str) -> str:
    return posixpath.normpath(directory.strip() or ".")


def _within(directory: str, copy: str) -> bool:
    return copy == "." or directory == copy or directory.startswith(copy + "/")


def _cd(line: str, directory: str) -> str:
    match = CD_RE.match(line)
    if match is None or match.group(1).startswith(("/", "$", "~")):
        return directory
    return _norm(posixpath.join(directory, match.group(1)))


def _assignments(line: str) -> tuple[str | None, str | None, str | None]:
    """RUSTUP_TOOLCHAIN the line sets: (via $GITHUB_ENV, via export, for this command only)."""
    match = ASSIGN_RE.search(line)
    if match is None:
        return None, None, None
    if "GITHUB_ENV" in line:
        return match.group(1), None, None
    if EXPORT_RE.match(line):
        return None, match.group(1), None
    return None, None, match.group(1)


class _Walk:
    """One job, step by step: which toolchains it installed, and what each call needs."""

    def __init__(self, name: str, job: Job, workflow: Workflow) -> None:
        self.name, self.job, self.workflow = name, job, workflow
        self.copies = {"."}  # checkouts, each with its own rust-toolchain.toml
        self.installed: set[str] = set()
        self.override: str | None = None  # `rustup override set <toolchain>`
        self.exported: str | None = None  # RUSTUP_TOOLCHAIN written to $GITHUB_ENV
        self.found: list[str] = []

    def copy_of(self, directory: str) -> str:
        return max((copy for copy in self.copies if _within(directory, copy)), key=len)

    def toolchain(self, step: Step, directory: str, line_env: str | None = None) -> str:
        chosen = (
            line_env
            or step.env.get("RUSTUP_TOOLCHAIN")
            or self.exported
            or self.job.env.get("RUSTUP_TOOLCHAIN")
            or self.workflow.env.get("RUSTUP_TOOLCHAIN")
        )
        return chosen or self.override or f"file:{self.copy_of(directory)}"

    def require(self, where: str, needed: str) -> None:
        if needed in self.installed:
            return
        copy = needed.removeprefix("file:")
        if copy == needed:
            self.found.append(f"{where} reaches cargo on {needed!r} before installing it")
        else:
            prefix = "" if copy == "." else f"{copy}/"
            self.found.append(f"{where} reaches cargo before {prefix}rust-toolchain.toml's toolchain is installed")

    def visit(self, index: int, step: Step) -> None:
        where = f"{self.name}: step {index}"
        self._visit_uses(where, step)
        directory = _norm(step.directory or self.job.directory or self.workflow.directory)
        exported: str | None = None
        for line in step.run.splitlines():
            if not COMMENT_RE.match(line):
                directory, exported = self._visit_line(where, step, line, directory, exported)

    def _visit_uses(self, where: str, step: Step) -> None:
        if step.uses.startswith(CHECKOUT_ACTION):
            self.copies.add(_norm(step.inputs.get("path", "")))
        elif step.uses.startswith(TOOLCHAIN_ACTION) and "toolchain" in step.inputs:
            self.installed.add(step.inputs["toolchain"])
        elif step.uses.startswith(TOOLCHAIN_ACTION):
            self.found.append(f"{where} installs the action's ref, not rust-toolchain.toml's toolchain")
        elif step.uses.startswith(RUST_CACHE_ACTION):
            self.require(where, self.toolchain(step, "."))
        elif step.uses.startswith(CACHE_ACTION) and TOOLCHAIN_PATHS_RE.search(step.inputs.get("path", "")):
            self.found.append(f"{where} caches a toolchain path: {step.inputs['path']!r}")

    def _visit_line(self, where: str, step: Step, line: str, directory: str, exported: str | None):
        directory = _cd(line, directory)
        to_github_env, exported_here, inline = _assignments(line)
        self.exported = to_github_env or self.exported
        exported = exported_here or exported
        line_env = self._rustup(line, directory) or inline or exported
        for match in PLUS_RE.finditer(line):
            self.require(where, match.group(1))
        if BUILD_RE.search(line):
            self.require(where, self.toolchain(step, directory, line_env))
        return directory, exported

    def _rustup(self, line: str, directory: str) -> str | None:
        """Record what the line installs or overrides; return the toolchain `rustup run` names."""
        ran = None
        for match in RUSTUP_RE.finditer(line):
            verb, named = _verb(match), _named(match.group(2))
            if _installs(verb, named):
                self.installed.add(named or f"file:{self.copy_of(directory)}")
            elif verb in OVERRIDE_VERBS and named:
                self.override = named
            elif verb == "run":
                ran = named
        return ran


def _job_findings(name: str, job: Job, workflow: Workflow) -> list[str]:
    walk = _Walk(name, job, workflow)
    for index, step in enumerate(job.steps):
        walk.visit(index, step)
    return walk.found


def findings(text: str, channel: str) -> list[str]:
    workflow = parse(text)
    found = _literal_findings(text, channel)
    for name, job in workflow.jobs.items():
        found += _job_findings(name, job, workflow)
    return found


@dataclass
class _Stage:
    """One Dockerfile stage: what it copied and installed so far."""

    copied: bool = False  # rust-toolchain.toml is in the stage
    repository: bool = False  # the repository's sources are in the stage
    default: str | None = None  # `rustup default <toolchain>`
    env: str | None = None  # ENV RUSTUP_TOOLCHAIN
    installed: set[str] = field(default_factory=set)


def _instructions(text: str) -> list[tuple[int, str, str]]:
    """(0-based line, INSTRUCTION, arguments), `\\` continuations joined."""
    out: list[tuple[int, str, str]] = []
    start, body = None, ""
    for number, line in enumerate(text.splitlines()):
        if start is None and (not line.strip() or COMMENT_RE.match(line)):
            continue
        start = number if start is None else start
        body += " " + line.strip().removesuffix("\\")
        if not line.rstrip().endswith("\\"):
            word, _, args = body.strip().partition(" ")
            out.append((start, word.upper(), args))
            start, body = None, ""
    return out


def _docker_copy(stage: _Stage, args: str) -> None:
    sources = [part for part in args.split() if not part.startswith("--")][:-1]
    names = {posixpath.basename(posixpath.normpath(source)) for source in sources}
    stage.copied |= bool(names & {"rust-toolchain.toml", "."})
    stage.repository |= bool(names & {".", "Cargo.toml", "Cargo.lock", "crates"})


def _docker_env(args: str) -> str | None:
    match = ASSIGN_RE.search(args) or re.search(r"\bRUSTUP_TOOLCHAIN\s+([^\s=]+)", args)
    return _unquote(match.group(1)) if match else None


def _docker_rustup(stage: _Stage, args: str) -> None:
    for match in RUSTUP_RE.finditer(args):
        verb, named = _verb(match), _named(match.group(2))
        if verb in INSTALL_VERBS and (named or stage.copied):
            stage.installed.add(named or "file")
        elif verb == "default" and named:
            stage.installed.add(named)
            stage.default = named


def _docker_plus(stage: _Stage, where: str, args: str) -> list[str]:
    return [f"{where} reaches cargo on {m.group(1)!r} before installing it"
            for m in PLUS_RE.finditer(args) if m.group(1) not in stage.installed]


def _docker_build(stage: _Stage, where: str, args: str) -> list[str]:
    if not (stage.repository and BUILD_RE.search(args)):
        return []
    needed = stage.env or ("file" if stage.copied else stage.default)
    if needed in stage.installed:
        return []
    if needed is None:
        return [f"{where} builds the repository with the base image's toolchain, not rust-toolchain.toml's"]
    if needed == "file":
        return [f"{where} reaches cargo before rust-toolchain.toml's toolchain is installed"]
    return [f"{where} reaches cargo on {needed!r} before installing it"]


def dockerfile_findings(text: str, channel: str) -> list[str]:
    found = _literal_findings(text, channel)
    stage = _Stage()
    for number, word, args in _instructions(text):
        if word == "FROM":
            stage = _Stage()
        elif word in ("COPY", "ADD") and "--from" not in args:
            _docker_copy(stage, args)
        elif word == "ENV":
            stage.env = _docker_env(args) or stage.env
        elif word == "RUN":
            _docker_rustup(stage, args)
            found += _docker_plus(stage, f"line {number + 1}", args) + _docker_build(stage, f"line {number + 1}", args)
    return found


def workflow_files(root: Path = REPO_ROOT) -> list[Path]:
    workflows, actions = root / ".github" / "workflows", root / ".github" / "actions"
    found = [*workflows.glob("*.yml"), *workflows.glob("*.yaml"), *actions.glob("**/action.yml"), *actions.glob("**/action.yaml")]
    return sorted(set(found))


def _tracked(root: Path) -> list[str]:
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    return [name for name in listed.decode("utf-8").split("\0") if name]


def dockerfiles(root: Path = REPO_ROOT) -> list[Path]:
    return [root / name for name in _tracked(root) if DOCKERFILE_NAME_RE.search(name)]


def loom_nightly_claims(root: Path = REPO_ROOT) -> list[str]:
    claims = []
    for name in _tracked(root):
        if Path(name).suffix not in DOC_SUFFIXES or name in LOOM_CLAIM_EXEMPT:
            continue
        try:
            text = (root / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        claims += [f"{name}:{n}: {line.strip()}" for n, line in enumerate(text.splitlines(), 1)
                   if "loom" in line.lower() and "nightly" in line.lower()]
    return claims


def real_findings() -> dict[str, list[str]]:
    channel, _ = toolchain_file()
    report = {}
    for path in workflow_files():
        found = findings(path.read_text(encoding="utf-8"), channel)
        if found:
            report[path.relative_to(REPO_ROOT).as_posix()] = found
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
      # nightly: cargo-fuzz instruments with -Zsanitizer.
      RUSTUP_TOOLCHAIN: nightly
    steps:
      # nightly: installed here, so rustup never installs it mid-step.
      - uses: dtolnay/rust-toolchain@abc # nightly
        with:
          toolchain: nightly
      - uses: Swatinem/rust-cache@v2
      - run: cargo install cargo-fuzz
      - run: cargo +nightly fuzz run target
  miri:
    runs-on: ubuntu-latest
    steps:
      # nightly: Miri ships with nightly only; the override keeps every call on it.
      - run: |
          rustup toolchain install nightly --component miri
          rustup override set nightly
          cargo miri test
"""
FUZZ_ENV = "    env:\n      # nightly: cargo-fuzz instruments with -Zsanitizer.\n      RUSTUP_TOOLCHAIN: nightly\n"
FUZZ_INSTALL = (
    "      # nightly: installed here, so rustup never installs it mid-step.\n"
    "      - uses: dtolnay/rust-toolchain@abc # nightly\n        with:\n          toolchain: nightly\n"
)
FUZZ_CACHE = "      - uses: Swatinem/rust-cache@v2\n      - run: cargo install cargo-fuzz\n"
AFTER_INSTALL = "      - name: Cache Cargo registry\n"

# The review's job whose steps sit at 4 spaces, where a 6-space parser sees none.
EARLY = """\
jobs:
  early:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v7
    - run: cargo build
"""
COMPOSITE = """\
name: setup
runs:
  using: composite
  steps:
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo build
      shell: bash
"""
BASE = """\
jobs:
  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/checkout@v7
        with:
          ref: develop
          path: base
      - name: Install the toolchain rust-toolchain.toml pins
""" + INSTALL + """\
      - name: Baseline
        working-directory: base
        run: cargo bench
"""
BASE_INSTALL = (
    "      - name: Install the toolchain base/rust-toolchain.toml pins\n        working-directory: base\n" + INSTALL
)
DOCKERFILE_WITHOUT_THE_FILE = """\
FROM rust:1.98-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --bin velesdb-server
FROM debian:bookworm-slim
COPY --from=builder /app/target/release/velesdb-server /usr/local/bin/
"""
DOCKERFILE_FROM_THE_FILE = """\
FROM rust:1.98-bookworm AS builder
WORKDIR /app
COPY rust-toolchain.toml ./
RUN rustup toolchain install --no-self-update --profile minimal
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --bin velesdb-server
FROM debian:bookworm-slim
COPY --from=builder /app/target/release/velesdb-server /usr/local/bin/
"""
DOCKERFILE_NIGHTLY = """\
FROM rust:slim-bookworm AS builder
WORKDIR /app
# nightly: this image exists to benchmark a nightly build.
RUN rustup default nightly && \\
    apt-get update
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --bin velesdb-server
"""
# Six ways of choosing a toolchain the round-1 guard let through, each as a step.
PIN_SITES = (
    ("      - run: rustup override add stable\n", "stable"),
    ("      - run: rustup toolchain add 1.86\n", "1.86"),
    ("      - run: RUSTUP_TOOLCHAIN=1.86 cargo build\n", "1.86"),
    ("      - run: echo RUSTUP_TOOLCHAIN=stable >> $GITHUB_ENV\n", "stable"),
    ("      - run: ~/.cargo/bin/cargo +1.86 build\n", "1.86"),
    ("      - uses: PyO3/maturin-action@v1\n        with:\n          rust-toolchain: stable\n", "stable"),
)


def _variant(old: str, new: str, text: str = GOOD) -> list[str]:
    if old not in text:
        raise ValueError(f"not in the reference workflow: {old!r}")
    return findings(text.replace(old, new), "1.90")


def _add_step(snippet: str) -> list[str]:
    return _variant(AFTER_INSTALL, snippet + AFTER_INSTALL)


def _has(found: list[str], fragment: str) -> bool:
    return any(fragment in finding for finding in found)


def _items_under_steps(lines: list[str]) -> int:
    """The `- ` items of the first `steps:` block in `lines`, at whatever indentation."""
    count, item_depth, inside = 0, None, False
    for line in lines:
        depth = len(line) - len(line.lstrip(" "))
        if not inside:
            inside = re.match(r"^\s*steps:\s*$", line) is not None
            continue
        if item_depth is None:
            item_depth = depth
        if depth < item_depth or (depth == item_depth and not line.lstrip().startswith("- ")):
            break
        count += depth == item_depth
    return count


def _job_heads(lines: list[str]) -> list[int]:
    """Where each job's key sits, two spaces in, after `jobs:`."""
    start = lines.index("jobs:")
    return [i for i, line in enumerate(lines) if i > start and re.match(r"^  [A-Za-z0-9_-]+:\s*$", line)]


def _written_steps(text: str) -> dict[str, int]:
    """Each job's `- ` items under `steps:`, counted without the guard's parser."""
    lines = [line for line in text.splitlines() if line.strip() and not COMMENT_RE.match(line)]
    if "jobs:" not in lines:
        return {"runs": _items_under_steps(lines)}
    heads = _job_heads(lines)
    ends = heads[1:] + [len(lines)]
    return {lines[h].strip()[:-1]: _items_under_steps(lines[h:e]) for h, e in zip(heads, ends)}


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

    # Independent check (a): a cargo resolver setting is not a RUST_VERSION copy.
    def test_only_a_rust_version_key_or_reference_is_a_copy(self) -> None:
        resolver = "env:\n  CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS: fallback\n"
        self.assertEqual([], findings(resolver + GOOD, "1.90"))
        for copy in ('  RUST_VERSION: "1.89"', "  X: ${{ env.RUST_VERSION }}", "  X: $RUST_VERSION", "  X: ${RUST_VERSION}"):
            with self.subTest(copy=copy):
                self.assertTrue(_has(findings("env:\n" + copy + "\n" + GOOD, "1.90"), "copies the toolchain version"))

    def test_a_named_rustup_install_is_refused(self) -> None:
        found = _variant("install --no-self-update", "install 1.86 --no-self-update")
        self.assertTrue(_has(found, "names toolchain '1.86'"), found)
        self.assertTrue(_has(found, "py: step 3 reaches cargo"), found)

    def test_a_cargo_plus_version_is_refused(self) -> None:
        found = _variant("cargo +nightly fuzz", "cargo +1.86 fuzz")
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
        self.assertEqual(
            [
                "fuzz: step 1 reaches cargo before rust-toolchain.toml's toolchain is installed",
                "fuzz: step 2 reaches cargo before rust-toolchain.toml's toolchain is installed",
            ],
            _variant(FUZZ_ENV, ""),
        )

    def test_a_nightly_pin_without_a_reason_is_refused(self) -> None:
        for comment in (
            "      # nightly: cargo-fuzz instruments with -Zsanitizer.\n",
            "      # nightly: Miri ships with nightly only; the override keeps every call on it.\n",
        ):
            with self.subTest(comment=comment):
                self.assertTrue(_has(_variant(comment, ""), "pins nightly without a comment"))

    def test_a_cache_holding_a_toolchain_is_refused(self) -> None:
        for path in ("~/.rustup", "~/.cargo/bin"):
            with self.subTest(path=path):
                self.assertTrue(_has(_variant("~/.cargo/git", path), "caches a toolchain path"))

    # Round 2, finding 1: every way of choosing a toolchain is a pin site.
    def test_every_way_of_naming_a_toolchain_is_refused(self) -> None:
        for snippet, name in PIN_SITES:
            with self.subTest(snippet=snippet):
                found = _add_step(snippet)
                self.assertTrue(_has(found, f"names toolchain {name!r}"), found)

    # Round 2, finding 2: a nightly exception holds only if nightly is installed first and said why.
    def test_the_nightly_exceptions_hold_to_what_they_claim(self) -> None:
        with self.subTest(case="rust-cache above the nightly install"):
            found = _variant(FUZZ_INSTALL + FUZZ_CACHE, "      - uses: Swatinem/rust-cache@v2\n" + FUZZ_INSTALL
                             + "      - run: cargo install cargo-fuzz\n")
            self.assertTrue(_has(found, "fuzz: step 0 reaches cargo on 'nightly' before installing it"), found)
        with self.subTest(case="cargo +nightly in a job that never installs nightly"):
            found = _add_step("      - run: cargo +nightly build\n")
            self.assertTrue(_has(found, "py: step 2 reaches cargo on 'nightly' before installing it"), found)
        with self.subTest(case="# TODO instead of the reason"):
            found = _variant("# nightly: cargo-fuzz instruments with -Zsanitizer.", "# TODO")
            self.assertTrue(_has(found, "pins nightly without a comment"), found)

    # Round 2, finding 4: steps at any indentation, and composite actions, are read.
    def test_steps_at_any_indentation_are_seen(self) -> None:
        self.assertEqual({"early": 2}, _written_steps(EARLY))
        self.assertEqual({"early": 2}, {name: len(job.steps) for name, job in parse(EARLY).jobs.items()})
        self.assertTrue(_has(findings(EARLY, "1.90"), "early: step 1 reaches cargo"))

    def test_a_composite_action_is_checked_like_a_job(self) -> None:
        found = findings(COMPOSITE, "1.90")
        self.assertTrue(_has(found, "installs the action's ref"), found)
        self.assertTrue(_has(found, "reaches cargo"), found)

    def test_workflow_files_reads_yaml_and_composite_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wanted = [".github/actions/setup/action.yml", ".github/actions/x/y/action.yaml",
                      ".github/workflows/a.yml", ".github/workflows/b.yaml"]
            for name in wanted + [".github/actions/setup/README.md", ".github/workflows/c.txt"]:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text("x\n", encoding="utf-8")
            self.assertEqual(wanted, sorted(p.relative_to(root).as_posix() for p in workflow_files(root)))

    # Round 2, finding 5: a Dockerfile that builds the repository installs from the file too.
    def test_a_dockerfile_building_the_repository_without_the_file_is_refused(self) -> None:
        found = dockerfile_findings(DOCKERFILE_WITHOUT_THE_FILE, "1.90")
        self.assertTrue(_has(found, "rust-toolchain.toml"), found)

    def test_a_dockerfile_that_installs_from_the_file_is_silent(self) -> None:
        self.assertEqual([], dockerfile_findings(DOCKERFILE_FROM_THE_FILE, "1.90"))

    def test_a_nightly_dockerfile_must_say_why_and_name_nothing_else(self) -> None:
        self.assertEqual([], dockerfile_findings(DOCKERFILE_NIGHTLY, "1.90"))
        no_reason = DOCKERFILE_NIGHTLY.replace("# nightly: this image exists to benchmark a nightly build.\n", "")
        self.assertTrue(_has(dockerfile_findings(no_reason, "1.90"), "pins nightly without a comment"))
        pinned = DOCKERFILE_NIGHTLY.replace("rustup default nightly", "rustup default 1.86")
        self.assertTrue(_has(dockerfile_findings(pinned, "1.90"), "names toolchain '1.86'"))

    # Round 2, finding 6: a build in another checkout needs that checkout's toolchain.
    def test_a_build_in_another_checkout_needs_that_checkouts_toolchain(self) -> None:
        wanted = "regression: step 3 reaches cargo before base/rust-toolchain.toml's toolchain is installed"
        self.assertTrue(_has(findings(BASE, "1.90"), wanted), findings(BASE, "1.90"))
        cd_base = _variant("        working-directory: base\n        run: cargo bench\n",
                           "        run: cd base && cargo bench\n", BASE)
        self.assertTrue(_has(cd_base, wanted), cd_base)
        self.assertEqual([], _variant("      - name: Baseline\n", BASE_INSTALL + "      - name: Baseline\n", BASE))


class RealWorkflowTests(unittest.TestCase):
    def test_the_toolchain_file_names_a_channel_and_components(self) -> None:
        channel, components = toolchain_file()
        self.assertRegex(channel, r"^\d+\.\d+(\.\d+)?$")
        self.assertTrue(components, "rust-toolchain.toml names no components")

    def test_every_workflow_parses_into_jobs_with_steps(self) -> None:
        workflows = workflow_files()
        self.assertTrue(workflows)
        for path in workflows:
            with self.subTest(workflow=path.name):
                jobs = parse(path.read_text(encoding="utf-8")).jobs
                self.assertTrue(jobs, "no job parsed")
                self.assertTrue(any(job.steps for job in jobs.values()), "no step parsed")

    def test_every_job_parses_as_many_steps_as_it_writes(self) -> None:
        for path in workflow_files():
            with self.subTest(workflow=path.name):
                text = path.read_text(encoding="utf-8")
                parsed = {name: len(job.steps) for name, job in parse(text).jobs.items()}
                self.assertEqual(_written_steps(text), parsed)

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

    def test_every_dockerfile_that_builds_rust_installs_from_the_file(self) -> None:
        channel, _ = toolchain_file()
        paths = dockerfiles()
        self.assertIn("Dockerfile", [path.relative_to(REPO_ROOT).as_posix() for path in paths])
        for path in paths:
            with self.subTest(dockerfile=path.relative_to(REPO_ROOT).as_posix()):
                self.assertEqual([], dockerfile_findings(path.read_text(encoding="utf-8"), channel))

    # Round 2, finding 3: loom no longer runs on nightly, and no line may say it does.
    def test_no_tracked_file_says_loom_runs_on_nightly(self) -> None:
        claims = loom_nightly_claims()
        self.assertEqual([], claims, "\n".join(claims))


if __name__ == "__main__":
    unittest.main()
