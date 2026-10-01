#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guards eight upstream test modules against working-directory coupling.

These modules historically built their paths from ``os.getcwd()`` or from
bare relative ``data/...`` literals, so they only worked when pytest was
launched from the repository root, and they wrote their outputs into the
tracked ``data/`` tree. Each must instead resolve inputs from its own
``__file__`` location and write outputs beneath pytest's ``tmp_path``.

The check is a static AST scan: it needs no BrainVISA, no data, and runs
in milliseconds, so it guards the property even where the guarded modules
themselves are skipped.

Covers REQ-UPSTREAM-04 (tracked in champollion_pipeline/elm/REQUIREMENTS.md).
"""

import ast
import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

GUARDED_MODULES = [
    "test_compute_bounding_box",
    "test_compute_mask",
    "test_resample_files",
    "test_generate_crops",
    "test_generate_skeletons",
    "test_generate_foldlabels",
    "test_generate_distmaps",
    "test_generate_ICBM2009c_transforms",
]

#: A relative path literal rooted at the repository's ``data`` directory,
#: e.g. ``"data/test"``, ``"./data/source"`` or plain ``"data"``.
RELATIVE_DATA_LITERAL = re.compile(r"^(\./)?data(/|$)")

#: Calls that return the process working directory.
CWD_CALLS = {("os", "getcwd"), ("os", "getcwdb"), ("Path", "cwd")}

#: Fixtures that provide a per-test temporary output directory.
TMP_FIXTURES = {"tmp_path", "tmp_path_factory"}

#: Callables whose non-first positional arguments are joined onto an
#: anchor, so a relative literal there does not resolve against the cwd.
JOIN_CALLS = {"join", "joinpath", "Path", "PurePath"}


def _parse(module_name):
    source_path = TESTS_DIR / f"{module_name}.py"
    return ast.parse(source_path.read_text(), filename=str(source_path))


def _parents(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _call_name(call):
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_anchored(node, parent):
    """True when a literal is joined onto some preceding anchor path."""
    if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div):
        return parent.right is node
    if isinstance(parent, ast.Call) and _call_name(parent) in JOIN_CALLS:
        return node in parent.args[1:]
    return False


@pytest.mark.parametrize("module_name", GUARDED_MODULES)
def test_module_does_not_read_process_cwd(module_name):
    """No call to os.getcwd(), os.getcwdb() or Path.cwd()."""
    offenders = []
    for node in ast.walk(_parse(module_name)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and (node.func.value.id, node.func.attr) in CWD_CALLS
        ):
            offenders.append(f"line {node.lineno}: {node.func.value.id}.{node.func.attr}()")
    assert not offenders, f"{module_name}.py reads the process working directory: {offenders}"


@pytest.mark.parametrize("module_name", GUARDED_MODULES)
def test_module_has_no_cwd_relative_data_path(module_name):
    """No bare relative 'data/...' literal that resolves against the cwd."""
    tree = _parse(module_name)
    parents = _parents(tree)
    offenders = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and RELATIVE_DATA_LITERAL.match(node.value)
            and not _is_anchored(node, parents.get(node))
        ):
            offenders.append(f"line {node.lineno}: {node.value!r}")
    assert not offenders, f"{module_name}.py uses cwd-relative data paths: {offenders}"


@pytest.mark.parametrize("module_name", GUARDED_MODULES)
def test_module_writes_outputs_under_tmp_path(module_name):
    """At least one test function requests tmp_path/tmp_path_factory."""
    requesting = [
        node.name
        for node in ast.walk(_parse(module_name))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and TMP_FIXTURES & {arg.arg for arg in node.args.args}
    ]
    assert requesting, (
        f"{module_name}.py has no test requesting {sorted(TMP_FIXTURES)}; "
        "its outputs are not sent to a pytest temporary directory"
    )
