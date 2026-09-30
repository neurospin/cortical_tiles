#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guards against the legacy package name resurfacing in tracked file paths.

The package was renamed to ``cortical_tiles``. Its former snake_case name
must not appear in any tracked file's *path* (e.g. a stray
``docs/deep_folding.png``) — a stale filename is a trap for the next
reader to link to or reference by the old name.

This does NOT check tracked file *content*: several notebooks legitimately
reference ``/neurospin/dico/data/deep_folding/...``, a real, permanent
Neurospin data-storage directory unrelated to the package rename (confirmed
to exist on disk; a hypothetical ``.../cortical_tiles/...`` sibling does
not). Renaming those literals would break the notebooks. See TASK-041's
same finding in champollion_pipeline's ledger.

The check reads git's own view of the repository (``git ls-files``), so
untracked build artefacts such as a stale ``*.egg-info`` directory are
ignored.

The legacy name is assembled at runtime so that this file does not itself
contain it.

Covers REQ-UPSTREAM-03 (tracked in champollion_pipeline/elm/REQUIREMENTS.md).
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The legacy package identifier, as an exact snake_case literal. Prose
#: spellings with a space or different case are out of scope.
LEGACY_NAME = "deep" + "_folding"


def _git(*args):
    """Run git at the repository root, returning (returncode, stdout)."""
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout


@pytest.fixture(scope="module")
def git_checkout():
    """Skip (never fail) when the sources are not a git checkout."""
    returncode, toplevel = _git("rev-parse", "--show-toplevel")
    if returncode != 0:
        pytest.skip(f"{REPO_ROOT} is not a git checkout")
    if Path(toplevel.strip()).resolve() != REPO_ROOT:
        pytest.skip(f"{REPO_ROOT} is not the root of its git checkout")
    return REPO_ROOT


def test_no_tracked_path_contains_legacy_name(git_checkout):
    """No tracked file path contains the legacy package name."""
    returncode, listing = _git("ls-files")
    assert returncode == 0, "git ls-files failed"

    offenders = sorted(path for path in listing.splitlines() if LEGACY_NAME in path)
    assert not offenders, f"{len(offenders)} tracked path(s) contain '{LEGACY_NAME}': {offenders}"
