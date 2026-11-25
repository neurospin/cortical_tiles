"""compute_mask.py and compute_bounding_box.py against their reference, and the shared subject_cli module.

Refactor Task TASK-ARGDISC-BDRABCZUK-FD9002DE9CD3 merges the duplicated command line and per-subject loop of the two
scripts into cortical_tiles/brainvisa/subject_cli.py without changing behaviour. The reference is cortical_tiles
commit 13c0b13: its two scripts are read with ``git show`` and imported from a temporary folder, so every
differential test compares the live code, new(x), with reference(x) on the same input.

* REQ-ARGDISC-BDRABCZUK-FCDA39485EC4 - compute_mask keeps the 13c0b13 option names, help texts and defaults.
* REQ-ARGDISC-BDRABCZUK-7EB26AD0D374 - compute_bounding_box keeps the 13c0b13 option names, help texts and defaults.
* REQ-ARGDISC-BDRABCZUK-6510E2C60F39 - compute_mask writes the 13c0b13 files.
* REQ-ARGDISC-BDRABCZUK-5D734B6E014A - compute_mask ends with the 13c0b13 exit status and error message.
* REQ-ARGDISC-BDRABCZUK-9F64E3BD111B - compute_bounding_box writes the 13c0b13 files.
* REQ-ARGDISC-BDRABCZUK-9BCCDE3CD0A9 - compute_bounding_box ends with the 13c0b13 exit status and error message.
* REQ-ARGDISC-BDRABCZUK-1E7E121C882C - subject_cli yields the graph path of each subject.
* REQ-ARGDISC-BDRABCZUK-490DF8147D66 - subject_cli raises the 13c0b13 error for a subject without graph.
* REQ-ARGDISC-BDRABCZUK-4C4F38B8E36B - both scripts build their parser through one subject_cli function.
* REQ-ARGDISC-BDRABCZUK-2811536886C4 - both scripts find the graph of a subject through subject_cli.

The pins above the subject_cli section pass on the unchanged code; the subject_cli section fails until green.
Defect DEF-ARGDISC-BDRABCZUK-5A5388AFB61F (compute_bounding_box fails with the default -n all) is not fixed by
this Task: the differential tests keep that ValueError as it is.
"""

import ast
import contextlib
import hashlib
import importlib.util
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from soma import aims

from cortical_tiles.brainvisa import compute_bounding_box as cbb
from cortical_tiles.brainvisa import compute_mask as cm
from cortical_tiles.brainvisa import subject_cli
from cortical_tiles.brainvisa.utils.subjects import get_all_subjects_as_dictionary

REFERENCE = "13c0b13"
REPO = Path(__file__).resolve().parents[1]
BRAINVISA = REPO / "cortical_tiles" / "brainvisa"
SCRIPTS = ("compute_mask", "compute_bounding_box")
LIVE = {"compute_mask": cm, "compute_bounding_box": cbb}

SIDE = "R"
PATH_TO_GRAPH = "t1mri/default/folds"
GRAPH_PATTERN = f"%(subject)s/{PATH_TO_GRAPH}/%(side)s%(subject)s*.arg"
# 1 mm graph voxels of the sulcus per subject; (10, 10, 10) is shared, so the mask counts 2 there.
SUBJECT_VOXELS = {"subA": [(10, 10, 10), (11, 10, 10)], "subB": [(10, 10, 10), (20, 20, 20)]}
JSON_CLOCK_KEYS = ("timestamp", "date")


# --- reference and fixtures -----------------------------------------------------------------------------------------


def _load_reference(name, folder):
    """Imports <name>.py of commit REFERENCE from folder, under its own file name (same prog and logger name)."""
    source = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{REFERENCE}:cortical_tiles/brainvisa/{name}.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = folder / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"reference_{REFERENCE}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    folder = tmp_path_factory.mktemp(f"reference_{REFERENCE}")
    return {name: _load_reference(name, folder) for name in SCRIPTS}


def _graph(voxels_by_sulcus):
    """In-memory CorticalFoldArg graph, 1 mm voxels, one aims_ss bucket per named sulcus."""
    graph = aims.Graph("CorticalFoldArg")
    graph["voxel_size"] = [1.0, 1.0, 1.0, 1.0]
    graph["boundingbox_min"] = [0, 0, 0]
    graph["boundingbox_max"] = [30, 30, 30]
    for name, voxels in voxels_by_sulcus.items():
        bucket = aims.BucketMap_VOID()
        bucket.setSizeXYZT(1.0, 1.0, 1.0, 1.0)
        for voxel in voxels:
            bucket[0][aims.Point3d(*voxel)] = 0
        vertex = graph.addVertex("fold")
        vertex["name"] = name
        vertex["aims_ss"] = aims.rc_ptr_BucketMap_VOID(bucket)
    return graph


@pytest.fixture
def database(tmp_path, monkeypatch):
    """<src>/<subject>/<PATH_TO_GRAPH>/R<subject>_auto.arg placeholders; aims.read returns their in-memory graphs."""
    src = tmp_path / "src"
    graphs = {}
    for subject, voxels in SUBJECT_VOXELS.items():
        folder = src / subject / PATH_TO_GRAPH
        folder.mkdir(parents=True)
        path = folder / f"{SIDE}{subject}_auto.arg"
        path.write_text("")
        graphs[str(path)] = _graph({"S.C._right": voxels, "F.C.M._right": [(5, 5, 5)]})
    real_read = aims.read

    def read(filename, *args, **kwargs):
        return graphs[filename] if filename in graphs else real_read(filename, *args, **kwargs)

    monkeypatch.setattr(aims, "read", read)
    return src


@contextlib.contextmanager
def _restored_logging():
    """setup_log swaps the root logger's file handler and level: put them back afterwards."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        yield
    finally:
        for handler in root.handlers[:]:
            if handler not in handlers:
                root.removeHandler(handler)
                handler.close()
        for handler in handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(level)


# --- outcome and file snapshots -------------------------------------------------------------------------------------


def _outcome(module, argv, capsys):
    """(how main ended, exit status or error type, error message, argparse error line) for main(argv)."""
    capsys.readouterr()
    try:
        with _restored_logging():
            module.main(argv)
        ended = ("exit", 0, None)
    except SystemExit as exc:
        ended = ("exit", exc.code, None)
    except Exception as exc:  # noqa: BLE001 - the differential compares any error
        ended = ("raise", type(exc).__name__, str(exc))
    errors = [line for line in capsys.readouterr().err.splitlines() if ": error: " in line]
    return (*ended, errors)


def _snapshot(folder):
    """{relative path: content} of every file below folder, the folder path replaced by <OUT>."""
    snapshot = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        name = str(path.relative_to(folder))
        if name.endswith(".nii.gz"):
            volume = np.asarray(aims.read(str(path)))
            snapshot[name] = (volume.shape, str(volume.dtype), hashlib.sha256(volume.tobytes()).hexdigest())
        elif name.endswith(".json"):
            data = json.loads(path.read_text())
            for key in JSON_CLOCK_KEYS:
                data.pop(key, None)
            snapshot[name] = json.dumps(data, sort_keys=True).replace(str(folder), "<OUT>")
        else:
            snapshot[name] = path.read_text().replace(str(folder), "<OUT>")
    return snapshot


def _argv(src, out, extra):
    return ["-s", str(src), "-p", PATH_TO_GRAPH, "-u", "S.C.", "-i", SIDE, "-o", str(out), *extra]


def _assert_writes_like_reference(name, reference, src, tmp_path, extra, capsys):
    ref_out, live_out = tmp_path / "reference_out", tmp_path / "live_out"
    ref_end = _outcome(reference[name], _argv(src, ref_out, extra), capsys)
    live_end = _outcome(LIVE[name], _argv(src, live_out, extra), capsys)
    assert live_end == ref_end
    ref_files = _snapshot(ref_out)
    assert ref_files, "the reference wrote nothing: the comparison would be vacuous"
    assert _snapshot(live_out) == ref_files


def _assert_ends_like_reference(name, reference, src, tmp_path, extra, capsys):
    ref_end = _outcome(reference[name], _argv(src, tmp_path / "reference_out", extra), capsys)
    live_end = _outcome(LIVE[name], _argv(src, tmp_path / "live_out", extra), capsys)
    assert live_end == ref_end


WRITE_CASES = {
    "all_subjects_verbose": ["-n", "all", "-x", "2", "-v"],
    "one_subject": ["-n", "1", "-x", "2"],
    "new_sulcus_debug": ["-n", "2", "-w", "S.X.", "-x", "2", "-vv"],
    "default_n_and_voxel_size": [],
}
END_CASES = {
    "help": ["--help"],
    "unknown_option": ["--bogus"],
    "negative_n": ["-n", "-1"],
    "text_n": ["-n", "x"],
    "zero_n": ["-n", "0"],
    "subject_without_graph": ["-n", "3", "-x", "2", "--with-subject-without-graph"],  # -n all: see default_n
    "default_n": ["-x", "2"],
}


def _end_case(src, case):
    extra = list(END_CASES[case])
    if "--with-subject-without-graph" in extra:
        extra.remove("--with-subject-without-graph")
        (src / "subAA").mkdir()  # sorted between subA and subB: subA is processed first, then the error
    return extra


# --- REQ-ARGDISC-BDRABCZUK-FCDA39485EC4 / 7EB26AD0D374: options, help texts, defaults -------------------------------

COMMON_HELP = [
    "-h, --help show this help message and exit",
    (
        "-s SRC_DIR [SRC_DIR ...], --src_dir SRC_DIR [SRC_DIR ...] Source directory where the MRI data lies. "
        "If there are several directories, add all directories one after the other. Example: -s DIR_1 DIR_2. "
        "Default is : /neurospin/dico/data/bv_databases/human/manually_labeled/pclean/all"
    ),
    (
        "-w NEW_SULCUS, --new_sulcus NEW_SULCUS Sulcus name around which we determine the bounding box. "
        "Default is : None (same name as 'sulcus')"
    ),
    "-i SIDE, --side SIDE Hemisphere side. Default is : R",
    (
        "-p PATH_TO_GRAPH, --path_to_graph PATH_TO_GRAPH Relative path to manually labelled graph. "
        "Default is t1mri/t1/default_analysis/folds/3.3/base2018_manual"
    ),
    (
        "-n NB_SUBJECTS, --nb_subjects NB_SUBJECTS Number of subjects to take into account, or 'all'. "
        "0 subject is allowed, for debug purpose. Default is : all"
    ),
    (
        "-v, --verbose Verbose mode: If no option is provided then logging.INFO is selected. "
        "If one option -v (or -vv) or more is provided then logging.DEBUG is selected."
    ),
]
SCRIPT_HELP = {
    "compute_mask": [
        (
            "-o OUTPUT_DIR, --output_dir OUTPUT_DIR Output directory where to store the output mask files. "
            "Default is : test/mask/1mm"
        ),
        (
            "-u SULCUS, --sulcus SULCUS Sulcus name around which we determine the bounding box/mask. "
            "Default is : F.C.M.ant."
        ),
        "-x OUT_VOXEL_SIZE, --out_voxel_size OUT_VOXEL_SIZE Voxel size of mask. Default is : 1.0",
    ],
    "compute_bounding_box": [
        (
            "-o OUTPUT_DIR, --output_dir OUTPUT_DIR Output directory where to store the output bbox json files. "
            "Default is : test/bbox/1mm"
        ),
        "-u SULCUS, --sulcus SULCUS Sulcus name around which we determine the bounding box. Default is : F.C.M.ant.",
        "-x OUT_VOXEL_SIZE, --out_voxel_size OUT_VOXEL_SIZE Voxel size of bounding box. Default is : None",
    ],
}
SCRIPT_DEFAULTS = {
    "src_dir": "/neurospin/dico/data/bv_databases/human/manually_labeled/pclean/all",
    "path_to_graph": "t1mri/t1/default_analysis/folds/3.3/base2018_manual",
    "sulcus": "F.C.M.ant.",
    "new_sulcus": None,
    "side": "R",
    "out_voxel_size": 1.0,
    "nb_subjects": -1,
}
DESCRIPTION = {
    "compute_mask": "Computes mask around the named sulcus",
    "compute_bounding_box": "Computes bounding box around the named sulcus",
}
OUTPUT_KEY = {"compute_mask": "mask_dir", "compute_bounding_box": "bbox_dir"}


def _assert_keeps_reference_option_help(name, capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "1000")  # argparse breaks long words at the terminal width
    capsys.readouterr()
    LIVE[name].main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert text.startswith(f"usage: {name}.py ")
    assert f" {DESCRIPTION[name]} options: " in text
    for entry in [*COMMON_HELP, *SCRIPT_HELP[name]]:
        # an entry is whole: the next option or the end of the text follows it
        assert f"{entry} -" in text or text.endswith(entry), entry


def _assert_keeps_reference_defaults(name, tmp_path):
    out = str(tmp_path / "out")
    with _restored_logging():
        params = LIVE[name].parse_args(["-o", out])
    expected = {**SCRIPT_DEFAULTS, OUTPUT_KEY[name]: out}
    assert {key: params[key] for key in expected} == expected


def test_compute_mask_keeps_reference_option_help(capsys, monkeypatch):
    """REQ-ARGDISC-BDRABCZUK-FCDA39485EC4: prog, description and every option entry of 13c0b13."""
    _assert_keeps_reference_option_help("compute_mask", capsys, monkeypatch)


def test_compute_mask_keeps_reference_defaults(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-FCDA39485EC4: default values of 13c0b13 (src_dir default stays a string)."""
    _assert_keeps_reference_defaults("compute_mask", tmp_path)


def test_compute_bounding_box_keeps_reference_option_help(capsys, monkeypatch):
    """REQ-ARGDISC-BDRABCZUK-7EB26AD0D374: prog, description and every option entry of 13c0b13."""
    _assert_keeps_reference_option_help("compute_bounding_box", capsys, monkeypatch)


def test_compute_bounding_box_keeps_reference_defaults(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-7EB26AD0D374: default values of 13c0b13."""
    _assert_keeps_reference_defaults("compute_bounding_box", tmp_path)


# --- REQ-ARGDISC-BDRABCZUK-6510E2C60F39 / 9F64E3BD111B: files written ----------------------------------------------


@pytest.mark.parametrize("case", list(WRITE_CASES))
def test_compute_mask_writes_reference_files(case, reference, database, tmp_path, capsys):
    """REQ-ARGDISC-BDRABCZUK-6510E2C60F39: mask, samples, log and command-line files equal the reference's."""
    _assert_writes_like_reference("compute_mask", reference, database, tmp_path, WRITE_CASES[case], capsys)


@pytest.mark.parametrize("case", list(WRITE_CASES))
def test_compute_bounding_box_writes_reference_files(case, reference, database, tmp_path, capsys):
    """REQ-ARGDISC-BDRABCZUK-9F64E3BD111B: JSON (clock entries excepted), log and command-line files equal."""
    _assert_writes_like_reference("compute_bounding_box", reference, database, tmp_path, WRITE_CASES[case], capsys)


# --- REQ-ARGDISC-BDRABCZUK-5D734B6E014A / 9BCCDE3CD0A9: exit status and error message -------------------------------


@pytest.mark.parametrize("case", list(END_CASES))
def test_compute_mask_ends_like_reference(case, reference, database, tmp_path, capsys):
    """REQ-ARGDISC-BDRABCZUK-5D734B6E014A: same exit status, error type, message and argparse error line."""
    _assert_ends_like_reference("compute_mask", reference, database, tmp_path, _end_case(database, case), capsys)


@pytest.mark.parametrize("case", list(END_CASES))
def test_compute_bounding_box_ends_like_reference(case, reference, database, tmp_path, capsys):
    """REQ-ARGDISC-BDRABCZUK-9BCCDE3CD0A9: same ending; case default_n keeps DEF-ARGDISC-BDRABCZUK-5A5388AFB61F."""
    extra = _end_case(database, case)
    _assert_ends_like_reference("compute_bounding_box", reference, database, tmp_path, extra, capsys)


def test_compute_bounding_box_default_n_still_fails_as_in_reference(reference, database, tmp_path, capsys):
    """Guard of the default_n case above: the reference itself raises the Defect's ValueError, so the pin is real."""
    ended = _outcome(reference["compute_bounding_box"], _argv(database, tmp_path / "out", ["-x", "2"]), capsys)
    assert ended[:2] == ("raise", "ValueError")


# --- subject_cli: REQ-ARGDISC-BDRABCZUK-1E7E121C882C / 490DF8147D66 ------------------------------------------------


class _Messages(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _reference_loop(module, subjects):
    """(graph paths read, log messages, error) of the per-subject loop of a reference script."""
    reader = MagicMock(return_value=([0] * 3, [0] * 3))
    messages = _Messages()
    module.log.addHandler(messages)
    old_level = module.log.level
    module.log.setLevel(logging.INFO)
    error = None
    try:
        if hasattr(module, "increment_mask"):
            with patch.object(module, "increment_one_mask", reader), patch.object(module, "aims", MagicMock()):
                module.increment_mask(subjects, None, "S.C._right", (1.0, 1.0, 1.0), "unused")
        else:
            with patch.object(module, "get_one_bounding_box", reader):
                module.get_bounding_boxes(subjects, "S.C._right")
    except RuntimeError as exc:
        error = str(exc)
    finally:
        module.log.removeHandler(messages)
        module.log.setLevel(old_level)
    read = [call.args[0] for call in reader.call_args_list]
    return read, messages.messages, error


def _live_loop(subjects):
    """(graph paths, log messages, error) of subject_cli.iter_subject_graphs; checks the yielded subjects."""
    log = logging.getLogger("test_unit_subject_cli.live_loop")
    log.setLevel(logging.INFO)
    messages = _Messages()
    log.addHandler(messages)
    read, yielded, error = [], [], None
    try:
        for subject, path in subject_cli.iter_subject_graphs(subjects, log):
            yielded.append(subject)
            read.append(path)
    except RuntimeError as exc:
        error = str(exc)
    finally:
        log.removeHandler(messages)
    assert yielded == subjects[: len(yielded)]
    return read, messages.messages, error


def _make_subjects(src, layout):
    """Creates <src>/<subject>/<PATH_TO_GRAPH>/<file> for {subject: [file names]}; lists subjects as the scripts do."""
    for subject, files in layout.items():
        folder = src / subject / PATH_TO_GRAPH
        folder.mkdir(parents=True)
        for file_name in files:
            (folder / file_name).write_text("")
    return get_all_subjects_as_dictionary([str(src)], [GRAPH_PATTERN], SIDE)


SUBJECT_NAMES = st.lists(st.sampled_from(["s01", "s02", "s10", "s2", "sA"]), min_size=1, max_size=4, unique=True)
SUFFIXES = st.lists(st.sampled_from(["", "_auto", "_manual", "_v2", "B"]), min_size=1, max_size=3, unique=True)


@settings(max_examples=25, deadline=None)
@given(layout=SUBJECT_NAMES.flatmap(lambda names: st.fixed_dictionaries({n: SUFFIXES for n in names})))
def test_iter_subject_graphs_matches_reference_loops(reference, layout):
    """REQ-ARGDISC-BDRABCZUK-1E7E121C882C: same graph per subject, order and log messages as both reference loops."""
    with tempfile.TemporaryDirectory() as folder:
        files = {s: [f"{SIDE}{s}{suffix}.arg" for suffix in suffixes] + [f"L{s}.arg"] for s, suffixes in layout.items()}
        subjects = _make_subjects(Path(folder) / "src", files)
        live = _live_loop(subjects)
        for name in SCRIPTS:
            expected = _reference_loop(reference[name], subjects)
            assert expected[0], "the reference read no graph: the comparison would be vacuous"
            assert live == expected, name


@pytest.mark.parametrize("name", SCRIPTS)
def test_iter_subject_graphs_raises_reference_error(name, reference, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-490DF8147D66: same RuntimeError, after the subjects listed before the failing one."""
    subjects = _make_subjects(tmp_path / "src", {"s1": [f"{SIDE}s1.arg"], "s2": [], "s3": [f"{SIDE}s3.arg"]})
    expected = _reference_loop(reference[name], subjects)
    assert expected[2] and expected[2].startswith("No graph file!")
    assert len(expected[0]) == 1
    assert _live_loop(subjects) == expected


# --- subject_cli: REQ-ARGDISC-BDRABCZUK-4C4F38B8E36B / 2811536886C4 (structure) -------------------------------------


def _tree(name):
    return ast.parse((BRAINVISA / f"{name}.py").read_text())


def _called_names(node):
    """Last name of every called function below node: glob.glob -> glob, argparse.ArgumentParser -> ArgumentParser."""
    names = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            func = call.func
            names.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
    return names


def _subject_cli_parser_builders():
    """Top-level subject_cli functions that build an ArgumentParser, directly or through another such function."""
    functions = {f.name: _called_names(f) for f in _tree("subject_cli").body if isinstance(f, ast.FunctionDef)}
    builders = {name for name, called in functions.items() if "ArgumentParser" in called}
    grown = True
    while grown:
        more = {name for name, called in functions.items() if called & builders} - builders
        builders |= more
        grown = bool(more)
    return builders


def _imports_subject_cli(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("subject_cli"):
            return True
        if isinstance(node, ast.ImportFrom) and any(alias.name == "subject_cli" for alias in node.names):
            return True
    return False


def test_scripts_build_their_parser_in_subject_cli():
    """REQ-ARGDISC-BDRABCZUK-4C4F38B8E36B: no ArgumentParser in either script; one common subject_cli builder."""
    builders = _subject_cli_parser_builders()
    used = []
    for name in SCRIPTS:
        tree = _tree(name)
        assert _imports_subject_cli(tree), name
        called = _called_names(tree)
        assert "ArgumentParser" not in called, name
        used.append(called & builders)
    assert used[0] & used[1], f"no common subject_cli parser builder: {used}"


def test_scripts_find_graphs_through_subject_cli():
    """REQ-ARGDISC-BDRABCZUK-2811536886C4: no glob call in either script; both loop with iter_subject_graphs."""
    assert "glob" in _called_names(_tree("subject_cli"))
    for name in SCRIPTS:
        tree = _tree(name)
        assert _imports_subject_cli(tree), name
        called = _called_names(tree)
        assert not called & {"glob", "iglob"}, name
        assert "iter_subject_graphs" in called, name
