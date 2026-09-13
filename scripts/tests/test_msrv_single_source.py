"""The MSRV has one source, and every copy of it agrees with that source.

`rust-version` in the workspace `Cargo.toml` is the MSRV a crates.io consumer
is promised. Two kinds of file restate it: `rust-toolchain.toml`, so a local
build uses that compiler, and the `RUST_VERSION` a workflow installs, so CI
does. Each says in a comment that it matches, and nothing checked it: a bump
that missed one copy would leave CI proving a toolchain nobody declared.

The workflows are read with a regex rather than PyYAML, so this suite needs
nothing beyond the standard library.
"""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

#: A `RUST_VERSION:` key, quoted or not. Anchored at the start of the line, so
#: a commented-out declaration or a `${{ env.RUST_VERSION }}` use is not read.
RUST_VERSION_RE = re.compile(r"""^[ \t]*RUST_VERSION:[ \t]*["']?([^"'\s#]+)""", re.MULTILINE)


def declared_msrv() -> str:
    manifest = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    return manifest["workspace"]["package"]["rust-version"]


def toolchain_channel() -> str:
    pinned = tomllib.loads((REPO_ROOT / "rust-toolchain.toml").read_text(encoding="utf-8"))
    return pinned["toolchain"]["channel"]


def rust_versions(workflow_text: str) -> "list[str]":
    return RUST_VERSION_RE.findall(workflow_text)


def workflow_rust_versions() -> "dict[str, list[str]]":
    """Each workflow that declares `RUST_VERSION`, to the values it declares."""
    found = {}
    for workflow in sorted(WORKFLOW_DIR.glob("*.yml")):
        values = rust_versions(workflow.read_text(encoding="utf-8"))
        if values:
            found[workflow.name] = values
    return found


class ReaderTests(unittest.TestCase):
    """The regex, on synthetic text, before it judges the real workflows."""

    def test_reads_a_quoted_and_an_unquoted_declaration(self) -> None:
        text = 'env:\n  RUST_VERSION: "1.90"\njobs:\n  x:\n    env:\n      RUST_VERSION: 1.91\n'
        self.assertEqual(rust_versions(text), ["1.90", "1.91"])

    def test_ignores_a_comment_and_a_use(self) -> None:
        text = "  # RUST_VERSION: '1.80'\n      toolchain: ${{ env.RUST_VERSION }}\n"
        self.assertEqual(rust_versions(text), [])


class MsrvAgreementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.msrv = declared_msrv()

    def test_ci_installs_the_declared_msrv(self) -> None:
        # The named requirement, and the positive control for the sweep below:
        # a reader that found no declaration anywhere would pass it vacuously.
        values = workflow_rust_versions().get("ci.yml", [])
        self.assertTrue(values, "ci.yml declares no RUST_VERSION this reader can see")
        self.assertEqual(
            set(values),
            {self.msrv},
            f"ci.yml installs Rust {values}, Cargo.toml declares rust-version = "
            f"{self.msrv!r}: change both in the same commit",
        )

    def test_every_workflow_installs_the_declared_msrv(self) -> None:
        for name, values in workflow_rust_versions().items():
            with self.subTest(workflow=name):
                self.assertEqual(
                    set(values),
                    {self.msrv},
                    f"{name} installs Rust {values}, Cargo.toml declares "
                    f"rust-version = {self.msrv!r}",
                )

    def test_the_toolchain_file_pins_the_declared_msrv(self) -> None:
        self.assertEqual(
            toolchain_channel(),
            self.msrv,
            "rust-toolchain.toml pins another compiler than Cargo.toml's rust-version",
        )


if __name__ == "__main__":
    unittest.main()
