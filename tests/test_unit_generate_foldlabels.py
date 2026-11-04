"""Unit tests for cortical_tiles.brainvisa.generate_foldlabels (REQ-CTILESTEST-127..137).

champollion_pipeline stage 2 converts every subject's Morphologist graph into a raw foldlabel volume
with ``generate_foldlabels`` -> ``GraphConvert2FoldLabel`` (junction 'thin'). The tracked subject
graphs take ~5 s each to convert, and aims cannot write a synthetic graph with buckets, so the
graph files here are empty placeholders ``<src>/<subject>/<path_to_graph>/R<subject>.arg`` and the
``aims`` reference of cortical_tiles.brainvisa.utils.foldlabel is replaced by a reader that serves a
tiny in-memory ``CorticalFoldArg`` graph (writing still uses the real ``aims.write``). Outputs go to
pytest tmp dirs whose names never contain the file prefix 'foldlabel_'.

Characterization tests: behaviour recorded on cortical_tiles dev 4141bd8. Tests flagged "DEFECT"
pin current behaviour that looks wrong; it is recorded, not fixed (tests-only rule of TASK-114).
"""

import csv
import inspect
import logging
import os
import types

import numpy as np
import pytest
from cortical_tiles.brainvisa import generate_foldlabels as gf
from cortical_tiles.brainvisa.utils import foldlabel as fl
from cortical_tiles.brainvisa.utils.skip_list import SkipEntry, load_skip_entries, load_skipped, record_skip
from pathos import multiprocessing as pathos_mp
from soma import aims

SIDE = "R"
PATH_TO_GRAPH = "t1mri/folds"
BAD_SUBJECT = "s3bad"


def _bucket(voxels):
    bucket = aims.BucketMap_VOID()
    bucket.setSizeXYZT(1.0, 1.0, 1.0, 1.0)
    for voxel in voxels:
        bucket[0][aims.Point3d(*voxel)] = 0
    return aims.rc_ptr_BucketMap_VOID(bucket)


def _graph(ss=((1, 1, 1), (2, 2, 2))):
    g = aims.Graph("CorticalFoldArg")
    g["voxel_size"] = [2.0, 2.0, 2.0, 1.0]
    g["boundingbox_min"] = [0, 0, 0]
    g["boundingbox_max"] = [4, 5, 6]
    v1 = g.addVertex("fold")
    v1["aims_ss"] = _bucket(ss)
    v1["aims_bottom"] = _bucket([(4, 4, 4)])  # thin 7000+n vs wide 2000+n
    g.addVertex("fold")["aims_ss"] = _bucket([(3, 3, 3)])
    return g


@pytest.fixture
def good_graph():
    """The single graph instance served for every good subject. aims vertex iteration order (hence the n
    of 1000+n) differs between two instances built identically, so the expected volume is computed from
    this same instance (forked parallel workers inherit it unchanged)."""
    return _graph()


@pytest.fixture
def expected(good_graph):
    def build(junction="thin"):
        return np.asarray(fl.generate_foldlabel_from_graph(good_graph, junction)).copy()

    return build


@pytest.fixture
def graph_reader(monkeypatch, good_graph):
    """Serves good_graph for every .arg path; a path containing BAD_SUBJECT gets an out-of-box voxel."""
    reads = []

    def read(path):
        reads.append(path)
        assert path.endswith(".arg")
        return _graph(ss=[(40, 40, 40)]) if BAD_SUBJECT in path else good_graph

    monkeypatch.setattr(fl, "aims", types.SimpleNamespace(read=read, write=aims.write))
    return reads


def _add_subject(src, subject, graph_names=None):
    folder = src / subject / PATH_TO_GRAPH
    folder.mkdir(parents=True)
    for name in graph_names or [f"{SIDE}{subject}.arg"]:
        (folder / name).write_text("")


@pytest.fixture
def src(tmp_path):
    src = tmp_path / "graphs"
    src.mkdir()
    for subject in ("s1", "s2"):
        _add_subject(src, subject)
    return src


def _skip_dir(out):
    """Skip-list directory of the converter writing into ``out``: a sibling of ``out``, inside the test's tmp dir."""
    return out.parent / "skip"


def _converter(src, out, bids=False, parallel=False, junction="thin", side=SIDE):
    return gf.GraphConvert2FoldLabel(
        src_dir=str(src),
        foldlabel_dir=str(out),
        side=side,
        junction=junction,
        parallel=parallel,
        path_to_graph=PATH_TO_GRAPH,
        bids=bids,
        qc_path="",
        skip_dir=str(_skip_dir(out)),
    )


def _read(path):
    return np.asarray(aims.read(str(path))).copy()


def _niftis(folder):
    return sorted(name for name in os.listdir(folder) if name.endswith(".nii.gz"))


def _clear_pathos_pools():
    state = pathos_mp._ProcessPool__STATE
    for pool in list(state.values()):
        pool.terminate()
        pool.join()
    state.clear()


# --- REQ-CTILESTEST-127 / 128: get_foldlabel_filename ----------------------------------------------


def test_foldlabel_filename_without_bids(tmp_path):
    """REQ-CTILESTEST-127: non-BIDS name is <foldlabel_dir>/<side>/<side>foldlabel_<subject>.nii.gz."""
    conv = _converter(tmp_path / "graphs", tmp_path / "labels")
    name = conv.get_foldlabel_filename("s1", "/x/ses-1/acq-2/run-3/R.arg")
    assert name == f"{tmp_path / 'labels'}/{SIDE}/{SIDE}foldlabel_s1.nii.gz"
    assert (tmp_path / "labels" / SIDE).is_dir()


@pytest.mark.parametrize(
    "graph_file, suffix",
    [
        ("/d/sub-1/ses-A/anat/run-3_acq-B/R.arg", "_ses-A_acq-B_run-3"),
        ("/d/sub-1/ses-A/R.arg", "_ses-A"),
        ("/d/sub-1/acq-hr/R.arg", "_acq-hr"),
        ("/d/sub-1/R.arg", ""),
    ],
)
def test_foldlabel_filename_with_bids(tmp_path, graph_file, suffix):
    """REQ-CTILESTEST-128: BIDS name appends the _ses-, _acq-, _run- entities found in the graph path, in that order."""
    conv = _converter(tmp_path / "graphs", tmp_path / "labels", bids=True)
    expected = f"{tmp_path / 'labels'}/{SIDE}/{SIDE}foldlabel_s1{suffix}.nii.gz"
    assert conv.get_foldlabel_filename("s1", graph_file) == expected


# --- REQ-CTILESTEST-129..133: generate_one_foldlabel -----------------------------------------------


def test_one_foldlabel_without_graph_reports_missing_file(src, tmp_path, graph_reader):
    """REQ-CTILESTEST-129: no <side>*.arg -> (subject, "No graph file: <pattern> doesn't exist"), nothing written."""
    (src / "s9").mkdir()
    conv = _converter(src, tmp_path / "labels")
    pattern = f"{src}/s9/{PATH_TO_GRAPH}/{SIDE}*.arg"
    assert conv.generate_one_foldlabel("s9") == ("s9", f"No graph file: {pattern} doesn't exist")
    assert _niftis(tmp_path / "labels" / SIDE) == []
    assert graph_reader == []


def test_one_foldlabel_reports_generation_error(src, tmp_path, graph_reader):
    """REQ-CTILESTEST-130: a conversion error -> (subject, its message), no exception, nothing written."""
    _add_subject(src, BAD_SUBJECT)
    conv = _converter(src, tmp_path / "labels")
    graph_file = f"{src}/{BAD_SUBJECT}/{PATH_TO_GRAPH}/{SIDE}{BAD_SUBJECT}.arg"
    assert conv.generate_one_foldlabel(BAD_SUBJECT) == (BAD_SUBJECT, f"Error generating foldlabel from {graph_file}")
    assert _niftis(tmp_path / "labels" / SIDE) == []


@pytest.mark.parametrize("junction", ["thin", "wide"])
def test_one_foldlabel_writes_volume(src, tmp_path, graph_reader, expected, junction):
    """REQ-CTILESTEST-131: success -> (subject, None) and the foldlabel of the subject's graph is written."""
    conv = _converter(src, tmp_path / "labels", junction=junction)
    assert conv.generate_one_foldlabel("s1") == ("s1", None)
    np.testing.assert_array_equal(_read(tmp_path / "labels" / SIDE / f"{SIDE}foldlabel_s1.nii.gz"), expected(junction))


def test_one_foldlabel_keeps_existing_file(src, tmp_path, graph_reader):
    """REQ-CTILESTEST-132: an existing foldlabel file is not regenerated and the graph is not read."""
    conv = _converter(src, tmp_path / "labels")
    existing = tmp_path / "labels" / SIDE / f"{SIDE}foldlabel_s1.nii.gz"
    sentinel = np.full((2, 2, 2, 1), 9, dtype=np.int16)
    aims.write(aims.Volume(sentinel), str(existing))

    assert conv.generate_one_foldlabel("s1") == ("s1", None)
    np.testing.assert_array_equal(_read(existing), sentinel)
    assert graph_reader == []


def test_one_foldlabel_bids_writes_one_file_per_graph(src, tmp_path, graph_reader):
    """REQ-CTILESTEST-133: BIDS mode converts every matching graph of the subject, one file per session."""
    session_root = tmp_path / "bids"
    session_root.mkdir()
    for session in ("ses-1", "ses-2"):
        _add_subject(session_root / session, "s1")
    conv = gf.GraphConvert2FoldLabel(
        src_dir=str(session_root),
        foldlabel_dir=str(tmp_path / "labels"),
        side=SIDE,
        junction="thin",
        parallel=False,
        path_to_graph=PATH_TO_GRAPH,
        bids=True,
        qc_path="",
        skip_dir=str(tmp_path / "skip"),
    )
    conv.src_dir = f"{session_root}/*"  # glob spans both sessions

    assert conv.generate_one_foldlabel("s1") == ("s1", None)
    assert _niftis(tmp_path / "labels" / SIDE) == [
        f"{SIDE}foldlabel_s1_ses-1.nii.gz",
        f"{SIDE}foldlabel_s1_ses-2.nii.gz",
    ]


# --- REQ-CTILESTEST-134..136: compute --------------------------------------------------------------


def test_compute_logs_summary(src, tmp_path, graph_reader, expected, capsys, caplog):
    """REQ-CTFAIL-BDRABCZUK-759C66EFFD0F (supersedes REQ-CTILESTEST-134): one INFO record of the
    generate_foldlabels logger holds 'Succeeded: k/n', 'Skipped (n-k):' and one '<subject>: <reason>' line
    per failure; nothing goes to stdout."""
    _add_subject(src, BAD_SUBJECT)
    (src / "s0").mkdir()  # no graph
    caplog.set_level(logging.INFO)
    _converter(src, tmp_path / "labels").compute(nb_subjects=-1)

    summaries = [
        record
        for record in caplog.records
        if record.name == gf.log.name and record.levelno == logging.INFO and "Succeeded: " in record.getMessage()
    ]
    assert len(summaries) == 1
    message = summaries[0].getMessage()
    assert "Succeeded: 2/4" in message
    assert "Skipped (2):" in message
    assert f"    s0: No graph file: {src}/s0/{PATH_TO_GRAPH}/{SIDE}*.arg doesn't exist" in message
    assert f"    {BAD_SUBJECT}: Error generating foldlabel from" in message
    assert capsys.readouterr().out == ""
    for subject in ("s1", "s2"):
        np.testing.assert_array_equal(
            _read(tmp_path / "labels" / SIDE / f"{SIDE}foldlabel_{subject}.nii.gz"), expected()
        )


def test_compute_does_not_list_converted_subjects_as_not_processed(src, tmp_path, graph_reader):
    """REQ-CTFAIL-BDRABCZUK-59946FB937BD (supersedes REQ-CTDEFECTS-18): converted subjects are absent from
    the side's not_processed_files_<side>.csv."""
    (src / "s1.minf").write_text("")
    _add_subject(src, BAD_SUBJECT)
    _converter(src, tmp_path / "labels").compute(nb_subjects=-1)

    assert _niftis(tmp_path / "labels" / SIDE) == [f"{SIDE}foldlabel_s1.nii.gz", f"{SIDE}foldlabel_s2.nii.gz"]
    with open(tmp_path / "labels" / f"not_processed_files_{SIDE}.csv") as handle:
        assert sorted(row[0] for row in csv.reader(handle)) == [BAD_SUBJECT]


def test_compute_not_processed_csv_excludes_qc_rejected_subjects(src, tmp_path, graph_reader):
    """REQ-CTFAIL-BDRABCZUK-59946FB937BD (supersedes REQ-CTDEFECTS-18): a subject rejected by the QC file is
    absent from the side's not_processed_files_<side>.csv."""
    _add_subject(src, BAD_SUBJECT)
    _add_subject(src, "s9rejected")
    qc_path = tmp_path / "qc.tsv"
    qc_path.write_text(f"participant_id\tqc\ns1\t1\ns2\t1\n{BAD_SUBJECT}\t1\ns9rejected\t0\n")
    conv = gf.GraphConvert2FoldLabel(
        src_dir=str(src),
        foldlabel_dir=str(tmp_path / "labels"),
        side=SIDE,
        junction="thin",
        parallel=False,
        path_to_graph=PATH_TO_GRAPH,
        bids=False,
        qc_path=str(qc_path),
        skip_dir=str(tmp_path / "skip"),
    )
    conv.compute(nb_subjects=-1)

    assert _niftis(tmp_path / "labels" / SIDE) == [f"{SIDE}foldlabel_s1.nii.gz", f"{SIDE}foldlabel_s2.nii.gz"]
    with open(tmp_path / "labels" / f"not_processed_files_{SIDE}.csv") as handle:
        assert sorted(row[0] for row in csv.reader(handle)) == [BAD_SUBJECT]


def test_compute_parallel_matches_serial(src, tmp_path, graph_reader, expected):
    """REQ-CTILESTEST-136: parallel=True writes the same foldlabel files as the serial mode."""
    _clear_pathos_pools()
    try:
        _converter(src, tmp_path / "labels", parallel=True).compute(nb_subjects=-1)
    finally:
        _clear_pathos_pools()
    for subject in ("s1", "s2"):
        np.testing.assert_array_equal(
            _read(tmp_path / "labels" / SIDE / f"{SIDE}foldlabel_{subject}.nii.gz"), expected()
        )


# --- REQ-CTFAIL-BDRABCZUK-E22179FF84DB / 20790A85DD4E / A64ED2E15FF0 / 59946FB937BD / 2311D75B2EFD /
# 8F6ACD36E459 / 0AEBFF72FBD7 ----------------------------------------------------------------------
# The failure rule itself (ERROR record, traceback, skip-list format) belongs to skip_list.py
# (REQ-CTFAIL-BDRABCZUK-826AB670054E and siblings); these tests pin only what generate_foldlabels adds.


@pytest.fixture
def raising_generator(monkeypatch):
    """generate_foldlabel_from_graph_file raises RuntimeError('boom') from KeyError for BAD_SUBJECT's graph,
    writes nothing for the others; the raised exceptions are collected."""
    raised = []

    def generate(graph_file, foldlabel_file, junction):
        if BAD_SUBJECT in graph_file:
            try:
                raise KeyError("aims_ss")
            except KeyError as cause:
                exc = RuntimeError("boom")
                exc.__cause__ = cause
                raised.append(exc)
                raise exc

    monkeypatch.setattr(gf, "generate_foldlabel_from_graph_file", generate)
    return raised


def test_failing_subject_is_recorded_in_side_skip_list(src, tmp_path, raising_generator):
    """REQ-CTFAIL-BDRABCZUK-E22179FF84DB: a subject whose generation raises is in the side's skip list, with
    step generate_foldlabels and the type and message of the raised exception (not only str(exc))."""
    _add_subject(src, BAD_SUBJECT)
    out = tmp_path / "labels"
    _converter(src, out).compute(nb_subjects=-1)

    assert load_skip_entries(_skip_dir(out), SIDE) == [
        SkipEntry(subject=BAD_SUBJECT, step="generate_foldlabels", exc_type="RuntimeError", message="boom")
    ]
    assert load_skip_entries(_skip_dir(out), "L") == []


def test_failing_subject_error_record_carries_the_raised_exception(src, tmp_path, raising_generator, caplog):
    """REQ-CTFAIL-BDRABCZUK-E22179FF84DB: record_skip receives the raised exception object itself, so the
    ERROR record carries its traceback and __cause__ chain."""
    _add_subject(src, BAD_SUBJECT)
    caplog.set_level(logging.ERROR)
    _converter(src, tmp_path / "labels").compute(nb_subjects=-1)

    assert len(raising_generator) == 1
    carried = [record for record in caplog.records if record.exc_info and record.exc_info[1] is raising_generator[0]]
    assert len(carried) == 1
    assert isinstance(carried[0].exc_info[1].__cause__, KeyError)


def test_parallel_failing_subject_is_recorded_in_side_skip_list(src, tmp_path, graph_reader):
    """REQ-CTFAIL-BDRABCZUK-E22179FF84DB: in parallel mode a failing subject is recorded in the side's skip
    list too, and only that subject."""
    _add_subject(src, BAD_SUBJECT)
    out = tmp_path / "labels"
    _clear_pathos_pools()
    try:
        _converter(src, out, parallel=True).compute(nb_subjects=-1)
    finally:
        _clear_pathos_pools()

    entries = load_skip_entries(_skip_dir(out), SIDE)
    assert [(entry.subject, entry.step) for entry in entries] == [(BAD_SUBJECT, "generate_foldlabels")]


def test_subject_without_graph_is_recorded_as_file_not_found(src, tmp_path, graph_reader):
    """REQ-CTFAIL-BDRABCZUK-20790A85DD4E: a subject with no graph file of the side is in the side's skip list
    with exception type FileNotFoundError."""
    (src / "s0").mkdir()  # no graph
    out = tmp_path / "labels"
    _converter(src, out).compute(nb_subjects=-1)

    entries = load_skip_entries(_skip_dir(out), SIDE)
    assert [(entry.subject, entry.step, entry.exc_type) for entry in entries] == [
        ("s0", "generate_foldlabels", "FileNotFoundError")
    ]


def test_skip_listed_subject_is_not_processed(src, tmp_path, graph_reader):
    """REQ-CTFAIL-BDRABCZUK-A64ED2E15FF0: a subject already in the side's skip list (here from an earlier step)
    gets no fold-label file and its graph is not read; the other subjects are converted."""
    out = tmp_path / "labels"
    record_skip(_skip_dir(out), SIDE, "generate_skeletons", "s2", RuntimeError("no skeleton"))
    _converter(src, out).compute(nb_subjects=-1)

    assert _niftis(out / SIDE) == [f"{SIDE}foldlabel_s1.nii.gz"]
    assert [path for path in graph_reader if "/s2/" in path] == []


def test_left_and_right_not_processed_lists_are_kept_apart(tmp_path, graph_reader):
    """REQ-CTFAIL-BDRABCZUK-59946FB937BD: an L run then an R run into the same folder leave one
    not_processed_files_<side>.csv per side, each listing that side's subjects without a fold label."""
    src = tmp_path / "graphs"
    src.mkdir()
    for subject in ("s1", "s2", BAD_SUBJECT):
        _add_subject(src, subject, graph_names=[f"L{subject}.arg", f"R{subject}.arg"])
    _add_subject(src, "s5ronly", graph_names=["Rs5ronly.arg"])
    out = tmp_path / "labels"
    _converter(src, out, side="L").compute(nb_subjects=-1)
    _converter(src, out, side="R").compute(nb_subjects=-1)

    lists = {}
    for side in ("L", "R"):
        with open(out / f"not_processed_files_{side}.csv") as handle:
            lists[side] = sorted(row[0] for row in csv.reader(handle))
    assert lists == {"L": [BAD_SUBJECT, "s5ronly"], "R": [BAD_SUBJECT]}


def _generate_foldlabels(src, foldlabel_dir, **extra):
    gf.generate_foldlabels(
        src_dir=str(src),
        foldlabel_dir=str(foldlabel_dir),
        path_to_graph=PATH_TO_GRAPH,
        side=SIDE,
        junction="thin",
        bids=False,
        parallel=False,
        nb_subjects=-1,
        qc_path="",
        **extra,
    )


def test_generate_foldlabels_records_failures_in_skip_dir(src, tmp_path, graph_reader):
    """REQ-CTFAIL-BDRABCZUK-2311D75B2EFD: the failure is recorded in the skip list of the skip_dir argument (the
    run's shared skip list), not under foldlabel_dir or the directories above it."""
    _add_subject(src, BAD_SUBJECT)
    output_dir = tmp_path / "run"
    shared = tmp_path / "shared"
    _generate_foldlabels(src, output_dir / "foldlabels" / "raw", skip_dir=str(shared))

    assert load_skipped(shared, SIDE) == {BAD_SUBJECT}
    assert sorted(path.name for path in output_dir.rglob("skip_list_*.jsonl")) == []


def test_generate_foldlabels_reads_skip_list_of_skip_dir(src, tmp_path, graph_reader):
    """REQ-CTFAIL-BDRABCZUK-2311D75B2EFD: a subject in the skip list of the skip_dir argument is not converted."""
    shared = tmp_path / "shared"
    record_skip(shared, SIDE, "generate_skeletons", "s2", RuntimeError("no skeleton"))
    foldlabel_dir = tmp_path / "run" / "foldlabels" / "raw"
    _generate_foldlabels(src, foldlabel_dir, skip_dir=str(shared))

    assert _niftis(foldlabel_dir / SIDE) == [f"{SIDE}foldlabel_s1.nii.gz"]


def test_generate_foldlabels_without_skip_dir_uses_foldlabel_dir(src, tmp_path, graph_reader):
    """REQ-CTFAIL-BDRABCZUK-8F6ACD36E459: called without skip_dir, generate_foldlabels records the failure in the
    skip list of its foldlabel_dir argument."""
    _add_subject(src, BAD_SUBJECT)
    foldlabel_dir = tmp_path / "labels"
    _generate_foldlabels(src, foldlabel_dir)

    assert load_skipped(foldlabel_dir, SIDE) == {BAD_SUBJECT}


def test_main_passes_skip_dir_option(tmp_path, monkeypatch):
    """REQ-CTFAIL-BDRABCZUK-0AEBFF72FBD7: --skip_dir DIR on the command line reaches generate_foldlabels as
    skip_dir=DIR."""
    received = []
    monkeypatch.setattr(gf, "generate_foldlabels", lambda **kwargs: received.append(kwargs))
    shared = str(tmp_path / "shared")

    gf.main(["-s", str(tmp_path), "-o", str(tmp_path / "labels"), "--skip_dir", shared])

    assert [kwargs["skip_dir"] for kwargs in received] == [shared]


# --- REQ-CTILESTEST-137: parse_args ---------------------------------------------------------------


@pytest.mark.parametrize("extra", [[], ["-j", "wide", "-n", "3", "-b"]])
def test_parse_args_returns_generate_foldlabels_keyword_arguments(tmp_path, extra):
    """REQ-CTDEFECTS-19 (inverts REQ-CTILESTEST-137), REQ-CTFAIL-BDRABCZUK-0AEBFF72FBD7: parse_args keys ==
    generate_foldlabels parameter names (skip_dir included)."""
    params = gf.parse_args(["-s", str(tmp_path), "-o", str(tmp_path / "labels"), *extra])
    assert set(params) == set(inspect.signature(gf.generate_foldlabels).parameters)


def test_main_calls_generate_foldlabels_with_parsed_arguments(tmp_path, monkeypatch):
    """REQ-CTDEFECTS-20: main(argv) calls generate_foldlabels once with keyword arguments == parse_args(argv)."""
    received = []
    monkeypatch.setattr(gf, "generate_foldlabels", lambda **kwargs: received.append(kwargs))
    argv = ["-s", str(tmp_path), "-o", str(tmp_path / "labels"), "-j", "wide", "-n", "3", "-b", "-q", "qc.tsv"]

    gf.main(argv)

    assert received == [gf.parse_args(argv)]
