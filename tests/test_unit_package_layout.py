#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package-layout guards for the TASK-131 dead-code deletion.

User decision 2026-10-06 (champollion_pipeline TASK-131): "delete everything
but what's inside brainvisa/utils". Nine dead modules leave the package; the
seven dead-for-coverage modules under ``brainvisa/utils/`` stay.

- REQ-CTDEADCODE-1 — the ``cortical_tiles`` package contains none of the nine
  deleted modules.
- REQ-CTDEADCODE-2 — importing a remaining ``cortical_tiles`` module does not
  raise an ``ImportError`` for one of those nine modules (guards against a
  surviving importer of a deleted module).

Requirements tracked in champollion_pipeline/.alm/REQUIREMENTS.md.

The import walk runs in a child interpreter: several modules execute
script-style code at import time (reading ``/neurospin`` paths, printing,
configuring logging), which must not leak into the pytest process. Modules
that fail to import for any other reason (missing data, script-style
sibling imports) are out of scope here and do not fail this guard.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "cortical_tiles"

#: REQ-CTDEADCODE-1: module files deleted from the package (TASK-131).
DELETED_MODULES = (
    "preprocessing/transforms.py",
    "preprocessing/pynet_transforms.py",
    "preprocessing/create_sets.py",
    "preprocessing/datasets.py",
    "brainvisa/benchmark_pipeline.py",
    "brainvisa/put_together_datasets.py",
    "brainvisa/dataset_to_sparse.py",
    "brainvisa/generate_sparse_dataset.py",
    "utils/split_train_test.py",
)

#: Dotted names of the deleted modules, e.g. ``cortical_tiles.utils.split_train_test``.
DELETED_DOTTED = tuple(
    "cortical_tiles." + module[: -len(".py")].replace("/", ".") for module in DELETED_MODULES
)
#: Bare stems, to also catch script-style ``import datasets`` importers.
DELETED_STEMS = tuple(Path(module).stem for module in DELETED_MODULES)


@pytest.mark.parametrize("module", DELETED_MODULES)
def test_deleted_dead_module_is_absent(module):
    """REQ-CTDEADCODE-1: the module file is gone from the package."""
    path = PACKAGE_DIR / module
    assert not path.exists(), (
        f"{path.relative_to(REPO_ROOT)} must be deleted from the cortical_tiles package "
        f"(REQ-CTDEADCODE-1, TASK-131 user decision 2026-10-06)"
    )


_IMPORT_WALK = textwrap.dedent(
    """
    import importlib, json, pkgutil, sys
    import cortical_tiles

    results = []
    def _on_error(name):
        exc = sys.exc_info()[1]
        results.append({"module": name, "error": type(exc).__name__,
                        "missing": getattr(exc, "name", None), "message": str(exc)[:300]})

    for info in pkgutil.walk_packages(cortical_tiles.__path__, "cortical_tiles.", onerror=_on_error):
        try:
            importlib.import_module(info.name)
        except BaseException as exc:  # noqa: BLE001 - script-style modules may sys.exit at import
            results.append({"module": info.name, "error": type(exc).__name__,
                            "missing": getattr(exc, "name", None), "message": str(exc)[:300]})
    sys.stdout.write("\\n@@RESULT@@" + json.dumps({"package": cortical_tiles.__file__, "failures": results}) + "\\n")
    """
)


@pytest.fixture(scope="module")
def import_walk():
    """Import every module of the package in a child interpreter; return its failures."""
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_WALK],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    marker = "@@RESULT@@"
    assert marker in completed.stdout, (
        f"import walk did not complete (rc={completed.returncode}); stderr tail:\n{completed.stderr[-2000:]}"
    )
    walk = json.loads(completed.stdout.rsplit(marker, 1)[1].strip())
    assert Path(walk["package"]).resolve().parent == PACKAGE_DIR.resolve(), (
        f"import walk loaded cortical_tiles from {walk['package']}, not this checkout ({PACKAGE_DIR})"
    )
    return walk


def _names_deleted_module(missing):
    if not missing:
        return False
    return missing in DELETED_DOTTED or missing.split(".")[-1] in DELETED_STEMS


def test_no_remaining_module_imports_a_deleted_module(import_walk):
    """REQ-CTDEADCODE-2: no remaining module fails on an import of a deleted module."""
    offenders = [
        failure
        for failure in import_walk["failures"]
        if failure["module"] not in DELETED_DOTTED
        and failure["error"] in ("ImportError", "ModuleNotFoundError")
        and _names_deleted_module(failure["missing"])
    ]
    assert not offenders, (
        "remaining cortical_tiles modules still import a module deleted by REQ-CTDEADCODE-1 "
        f"(REQ-CTDEADCODE-2): {json.dumps(offenders, indent=2)}"
    )
