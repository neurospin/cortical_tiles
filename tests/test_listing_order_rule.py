"""Listing-order rule of the cortical_tiles package (TASK-READORDER-BDRABCZUK-4550903D1CB2).

User rule: files must always be read in the same order, for reproducibility. A lint-style test:
it parses every module of the cortical_tiles package and fails on a listing call (glob, iglob,
rglob, listdir, scandir, walk, iterdir) that is not inside a sorted() call and has no entry in
tests/listing_order_exemptions.py. Each exemption entry carries the reason its order never reaches
a result.

A call counts as sorted when it lies in a positional argument of a sorted() call, at any depth
of that argument expression (so sorted(p for p in d.iterdir() if p.is_dir()) is sorted). A module
Python cannot parse (cortical_tiles/visualization/visualize_volume_as_bucket.py on 2026-11-10)
cannot show sorted(): each of its lines naming a listing call outside a comment is reported.

* REQ-READORDER-BDRABCZUK-5A6652ECC6B8 - the test fails on an unsorted listing call not exempted.
* REQ-READORDER-BDRABCZUK-04E14C889178 - the test fails on an exemption entry without a reason.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.listing_order_exemptions import EXEMPTIONS

LISTING_NAMES = frozenset({"glob", "iglob", "rglob", "listdir", "scandir", "walk", "iterdir"})
REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "cortical_tiles"
MODULE_SCOPE = "<module>"
UNPARSABLE_SCOPE = "<unparsable>"
LISTING_CALL_TEXT = re.compile(r"\b(?:" + "|".join(sorted(LISTING_NAMES)) + r")\s*\(")
SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class ListingCall:
    """One listing call outside sorted(), with the key the exemption list uses"""

    path: str
    scope: str
    call: str
    line: int

    @property
    def key(self):
        return (self.path, self.scope, self.call)


def _callee_name(call):
    """Returns the called name of call: f for f(...), attr for x.attr(...), else None"""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_sorted_call(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sorted"


def _listing_lines_of_unparsable(source, rel_path):
    """Returns one call per line of source naming a listing call, for a module Python cannot parse

    Such a module cannot be checked for sorted(), so each line that names a listing call is
    reported, its stripped text standing for the call.
    """
    return [
        ListingCall(rel_path, UNPARSABLE_SCOPE, line.strip(), number)
        for number, line in enumerate(source.splitlines(), start=1)
        if LISTING_CALL_TEXT.search(line.split("#", 1)[0])
    ]


def find_unsorted_listing_calls(source, rel_path):
    """Returns the listing calls of source that lie outside every sorted() argument"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _listing_lines_of_unparsable(source, rel_path)
    found = []

    def visit(node, scope, in_sorted):
        if isinstance(node, ast.Call) and not in_sorted and _callee_name(node) in LISTING_NAMES:
            found.append(ListingCall(rel_path, ".".join(scope) or MODULE_SCOPE, ast.unparse(node), node.lineno))
        if isinstance(node, SCOPE_NODES):
            scope = (*scope, node.name)
        if _is_sorted_call(node):
            visit(node.func, scope, in_sorted)
            for arg in node.args:
                visit(arg, scope, True)
            for keyword in node.keywords:
                visit(keyword, scope, in_sorted)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, scope, in_sorted)

    visit(tree, (), False)
    return found


def unexempted(calls, exemptions):
    """Returns the calls whose key has no entry in exemptions"""
    return [call for call in calls if call.key not in exemptions]


def empty_reason_keys(exemptions):
    """Returns the exemption keys whose reason is empty after whitespace stripping"""
    return [key for key, reason in exemptions.items() if not str(reason).strip()]


def _scan_package():
    """Returns the modules scanned and the unsorted listing calls of the cortical_tiles package"""
    modules = sorted(PACKAGE_DIR.rglob("*.py"))
    calls = []
    for module in modules:
        rel_path = module.relative_to(REPO_ROOT).as_posix()
        calls.extend(find_unsorted_listing_calls(module.read_text(encoding="utf-8"), rel_path))
    return modules, calls


# --- REQ-READORDER-BDRABCZUK-5A6652ECC6B8: unsorted listing call not exempted -------------------

UNSORTED_CASES = [
    ("import glob\nfiles = glob.glob('d/*')\n", MODULE_SCOPE, "glob.glob('d/*')"),
    ("from glob import glob\nfiles = glob('d/*')\n", MODULE_SCOPE, "glob('d/*')"),
    ("import glob\nfiles = list(glob.iglob('d/*'))\n", MODULE_SCOPE, "glob.iglob('d/*')"),
    (
        "from pathlib import Path\ndef f():\n    return list(Path('d').rglob('*.nii.gz'))\n",
        "f",
        "Path('d').rglob('*.nii.gz')",
    ),
    ("import os\nclass C:\n    def m(self):\n        return os.listdir('d')\n", "C.m", "os.listdir('d')"),
    ("import os\nentries = [e.name for e in os.scandir('d')]\n", MODULE_SCOPE, "os.scandir('d')"),
    ("import os\nfor root, dirs, files in os.walk('d'):\n    pass\n", MODULE_SCOPE, "os.walk('d')"),
    ("from pathlib import Path\nnames = [p.name for p in Path('d').iterdir()]\n", MODULE_SCOPE, "Path('d').iterdir()"),
    ("import os\nfiles = min(os.listdir('d'))\n", MODULE_SCOPE, "os.listdir('d')"),
    ("import os\nfiles = sorted(['a'], key=lambda p: len(os.listdir(p)))\n", MODULE_SCOPE, "os.listdir(p)"),
    (
        "import os\nif True:\nfiles = sorted(os.listdir('d'))\n# glob.glob('d/*')\n",
        UNPARSABLE_SCOPE,
        "files = sorted(os.listdir('d'))",
    ),
]


@pytest.mark.parametrize(("source", "scope", "call"), UNSORTED_CASES)
def test_unsorted_listing_call_is_reported(source, scope, call):
    calls = find_unsorted_listing_calls(source, "cortical_tiles/m.py")
    assert [c.key for c in unexempted(calls, {})] == [("cortical_tiles/m.py", scope, call)]


SORTED_CASES = [
    "import glob\nfiles = sorted(glob.glob('d/*'))\n",
    "import os\nfiles = sorted(os.listdir('d'))\n",
    "from pathlib import Path\ndirs = sorted(p for p in Path('d').iterdir() if p.is_dir())\n",
    "import os\ndef f():\n    return [x for x in sorted(os.listdir('d'))]\n",
    "import glob\nfiles = sorted(glob.glob('a/*') + glob.glob('b/*'))\n",
    "import glob\nname = glob.escape('d')\n",
]


@pytest.mark.parametrize("source", SORTED_CASES)
def test_sorted_listing_call_is_not_reported(source):
    assert find_unsorted_listing_calls(source, "cortical_tiles/m.py") == []


def test_exempt_listing_call_is_not_reported():
    source = "import os\ndef f(d):\n    if os.listdir(d) == []:\n        return os.listdir(d + '/x')\n"
    calls = find_unsorted_listing_calls(source, "cortical_tiles/m.py")
    exemptions = {("cortical_tiles/m.py", "f", "os.listdir(d)"): "emptiness test, order unused"}
    assert [c.key for c in unexempted(calls, exemptions)] == [("cortical_tiles/m.py", "f", "os.listdir(d + '/x')")]


def test_package_has_no_unexempted_unsorted_listing_call():
    modules, calls = _scan_package()
    assert modules, f"no module found under {PACKAGE_DIR}"
    missing = unexempted(calls, EXEMPTIONS)
    report = "\n".join(f"    {c.key!r}: '',  # line {c.line}" for c in missing)
    assert not missing, (
        f"{len(missing)} listing calls of cortical_tiles are neither inside sorted() nor in "
        f"tests/listing_order_exemptions.py:\n{report}"
    )


# --- REQ-READORDER-BDRABCZUK-04E14C889178: exemption entry without a reason ---------------------

KEY = ("cortical_tiles/m.py", "f", "os.listdir(d)")


@pytest.mark.parametrize(("reason", "reported"), [("", True), ("   \n\t", True), ("emptiness test", False)])
def test_empty_exemption_reason_is_reported(reason, reported):
    assert empty_reason_keys({KEY: reason}) == ([KEY] if reported else [])


def test_exemption_reasons_are_not_empty():
    assert empty_reason_keys(EXEMPTIONS) == []
