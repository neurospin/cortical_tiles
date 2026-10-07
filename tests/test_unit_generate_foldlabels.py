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
import os
import types

import numpy as np
import pytest
from cortical_tiles.brainvisa import generate_foldlabels as gf
from cortical_tiles.brainvisa.utils import foldlabel as fl
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


def _converter(src, out, bids=False, parallel=False, junction="thin"):
    return gf.GraphConvert2FoldLabel(
        src_dir=str(src),
        foldlabel_dir=str(out),
        side=SIDE,
        junction=junction,
        parallel=parallel,
        path_to_graph=PATH_TO_GRAPH,
        bids=bids,
        qc_path="",
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
    )
    conv.src_dir = f"{session_root}/*"  # glob spans both sessions

    assert conv.generate_one_foldlabel("s1") == ("s1", None)
    assert _niftis(tmp_path / "labels" / SIDE) == [
        f"{SIDE}foldlabel_s1_ses-1.nii.gz",
        f"{SIDE}foldlabel_s1_ses-2.nii.gz",
    ]


# --- REQ-CTILESTEST-134..136: compute --------------------------------------------------------------


def test_compute_prints_summary(src, tmp_path, graph_reader, expected, capsys):
    """REQ-CTILESTEST-134: stdout reports 'Succeeded: k/n' and one '<subject>: <reason>' line per failure."""
    _add_subject(src, BAD_SUBJECT)
    (src / "s0").mkdir()  # no graph
    _converter(src, tmp_path / "labels").compute(nb_subjects=-1)

    out = capsys.readouterr().out
    assert "Succeeded: 2/4" in out
    assert "Skipped (2):" in out
    assert f"    s0: No graph file: {src}/s0/{PATH_TO_GRAPH}/{SIDE}*.arg doesn't exist" in out
    assert f"    {BAD_SUBJECT}: Error generating foldlabel from" in out
    for subject in ("s1", "s2"):
        np.testing.assert_array_equal(
            _read(tmp_path / "labels" / SIDE / f"{SIDE}foldlabel_{subject}.nii.gz"), expected()
        )


def test_compute_does_not_list_converted_subjects_as_not_processed(src, tmp_path, graph_reader):
    """REQ-CTDEFECTS-18 (inverts REQ-CTILESTEST-135): converted subjects are absent from not_processed_files.csv."""
    (src / "s1.minf").write_text("")
    _add_subject(src, BAD_SUBJECT)
    _converter(src, tmp_path / "labels").compute(nb_subjects=-1)

    assert _niftis(tmp_path / "labels" / SIDE) == [f"{SIDE}foldlabel_s1.nii.gz", f"{SIDE}foldlabel_s2.nii.gz"]
    with open(tmp_path / "labels" / "not_processed_files.csv") as handle:
        assert sorted(row[0] for row in csv.reader(handle)) == [BAD_SUBJECT]


def test_compute_not_processed_csv_excludes_qc_rejected_subjects(src, tmp_path, graph_reader):
    """REQ-CTDEFECTS-18: a subject rejected by the QC file is absent from not_processed_files.csv."""
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
    )
    conv.compute(nb_subjects=-1)

    assert _niftis(tmp_path / "labels" / SIDE) == [f"{SIDE}foldlabel_s1.nii.gz", f"{SIDE}foldlabel_s2.nii.gz"]
    with open(tmp_path / "labels" / "not_processed_files.csv") as handle:
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


# --- REQ-CTILESTEST-137: parse_args ---------------------------------------------------------------


@pytest.mark.parametrize("extra", [[], ["-j", "wide", "-n", "3", "-b"]])
def test_parse_args_returns_generate_foldlabels_keyword_arguments(tmp_path, extra):
    """REQ-CTDEFECTS-19 (inverts REQ-CTILESTEST-137): parse_args keys == generate_foldlabels parameter names."""
    params = gf.parse_args(["-s", str(tmp_path), "-o", str(tmp_path / "labels"), *extra])
    assert set(params) == set(inspect.signature(gf.generate_foldlabels).parameters)


def test_main_calls_generate_foldlabels_with_parsed_arguments(tmp_path, monkeypatch):
    """REQ-CTDEFECTS-20: main(argv) calls generate_foldlabels once with keyword arguments == parse_args(argv)."""
    received = []
    monkeypatch.setattr(gf, "generate_foldlabels", lambda **kwargs: received.append(kwargs))
    argv = ["-s", str(tmp_path), "-o", str(tmp_path / "labels"), "-j", "wide", "-n", "3", "-b", "-q", "qc.tsv"]

    gf.main(argv)

    assert received == [gf.parse_args(argv)]
