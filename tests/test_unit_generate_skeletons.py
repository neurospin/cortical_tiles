"""Unit tests for cortical_tiles.brainvisa.generate_skeletons (REQ-CTILESTEST-154..157).

champollion_pipeline stage 2 starts by converting every subject's Morphologist graph into a
raw skeleton (``generate_skeletons``). The graph -> volume conversion itself is covered by
test_generate_skeletons.py and test_unit_utils_skeleton_foldlabel.py; these tests pin the
orchestration around it: BIDS file naming, subjects without a graph, the skipped-subjects
QC csv, and the CLI. A recorder replaces ``generate_skeleton_from_graph_file`` (it writes
an empty marker file) so no graph is read; all paths are in a pytest tmp dir. The parallel
path runs in a fresh pathos pool so the forked workers see the recorder.
"""

import csv
import inspect
import os

import pytest
from cortical_tiles.brainvisa import generate_skeletons as gs
from pathos import multiprocessing as pathos_mp

PATH_TO_GRAPH = "t1mri/default_acquisition/default_analysis/folds/3.1"


def _clear_pathos_pools():
    """p_map reuses a cached pathos pool whose forked workers predate monkeypatching."""
    state = pathos_mp._ProcessPool__STATE
    for pool in list(state.values()):
        pool.terminate()
        pool.join()
    state.clear()


@pytest.fixture
def fresh_process_pool():
    _clear_pathos_pools()
    yield
    _clear_pathos_pools()


def _converter(tmp_path, bids=False, parallel=False, side="L"):
    return gs.GraphConvert2Skeleton(
        src_dir=str(tmp_path / "morpho"),
        skeleton_dir=str(tmp_path / "skeletons"),
        side=side,
        junction="thin",
        parallel=parallel,
        path_to_graph=PATH_TO_GRAPH,
        bids=bids,
        qc_path="",
        njobs=1,
    )


@pytest.fixture
def fake_conversion(monkeypatch):
    calls = []

    def _fake(graph_file, skeleton_file, junction):
        calls.append((graph_file, skeleton_file, junction))
        open(skeleton_file, "w").close()

    monkeypatch.setattr(gs, "generate_skeleton_from_graph_file", _fake)
    return calls


def _make_subject(tmp_path, subject, with_graph, side="L"):
    folds = tmp_path / "morpho" / subject / PATH_TO_GRAPH
    folds.mkdir(parents=True)
    if with_graph:
        (folds / f"{side}{subject}.arg").write_text("")


# --- REQ-CTILESTEST-154: BIDS skeleton filename --------------------------------------------------


@pytest.mark.parametrize(
    "graph_file, suffix",
    [
        ("/d/sub-1/ses-A/anat/t1mri/acq-B/run-3/folds/Lsub-1.arg", "_ses-A_acq-B_run-3"),
        ("/d/sub-1/run-3_acq-B_ses-A/Lsub-1.arg", "_ses-A_acq-B_run-3"),  # fixed order ses, acq, run
        ("/d/sub-1/ses-A/Lsub-1.arg", "_ses-A"),
        ("/d/sub-1/default/Lsub-1.arg", ""),
    ],
)
def test_bids_skeleton_filename_appends_session_acquisition_run(tmp_path, graph_file, suffix):
    """REQ-CTILESTEST-154: bids -> <dir>/<side>skeleton_generated_<subject>[_ses-*][_acq-*][_run-*].nii.gz."""
    conv = _converter(tmp_path, bids=True)
    expected = f"{tmp_path}/skeletons/L/Lskeleton_generated_sub-1{suffix}.nii.gz"
    assert conv.get_skeleton_filename("sub-1", graph_file) == expected


def test_non_bids_skeleton_filename_ignores_bids_entities(tmp_path):
    """REQ-CTILESTEST-154: bids False -> no entity suffix even if the graph path has one."""
    conv = _converter(tmp_path, bids=False)
    name = conv.get_skeleton_filename("sub-1", "/d/sub-1/ses-A/Lsub-1.arg")
    assert name == f"{tmp_path}/skeletons/L/Lskeleton_generated_sub-1.nii.gz"


# --- REQ-CTILESTEST-155: subject without graph ---------------------------------------------------


def test_subject_without_graph_is_returned_and_not_converted(tmp_path, fake_conversion):
    """REQ-CTILESTEST-155: no graph matches -> returns (subject, glob pattern), no conversion, no file."""
    _make_subject(tmp_path, "s01", with_graph=False)
    conv = _converter(tmp_path)
    result = conv.generate_one_skeleton("s01")
    assert result == ("s01", f"{tmp_path}/morpho/s01/{PATH_TO_GRAPH}/L*.arg")
    assert fake_conversion == []
    assert os.listdir(tmp_path / "skeletons" / "L") == []


# --- REQ-CTILESTEST-156: skipped_subjects.csv ----------------------------------------------------


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.parametrize("bids", [False, True], ids=["plain", "bids"])
def test_compute_writes_skipped_subjects_csv(tmp_path, fake_conversion, fresh_process_pool, parallel, bids):
    """REQ-CTILESTEST-156: compute writes skipped_subjects.csv: header + one row per subject without graph."""
    _make_subject(tmp_path, "s01", with_graph=True)
    _make_subject(tmp_path, "s02", with_graph=False)
    _make_subject(tmp_path, "s03", with_graph=False)
    conv = _converter(tmp_path, bids=bids, parallel=parallel)

    conv.compute(nb_subjects=-1)

    with open(tmp_path / "skeletons" / "L" / "skipped_subjects.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["subject", "expected_graph_path"]
    assert sorted(rows[1:]) == [[s, f"{tmp_path}/morpho/{s}/{PATH_TO_GRAPH}/L*.arg"] for s in ("s02", "s03")]
    assert os.path.exists(tmp_path / "skeletons" / "L" / "Lskeleton_generated_s01.nii.gz")


def test_compute_without_skipped_subject_writes_no_csv(tmp_path, fake_conversion):
    """REQ-CTILESTEST-156 (guard): no skipped subject -> no skipped_subjects.csv."""
    _make_subject(tmp_path, "s01", with_graph=True)
    _converter(tmp_path).compute(nb_subjects=-1)
    assert not os.path.exists(tmp_path / "skeletons" / "L" / "skipped_subjects.csv")


# --- REQ-CTILESTEST-157: main / parse_args -------------------------------------------------------


def test_main_calls_generate_skeletons_with_its_own_parameter_names(tmp_path, monkeypatch):
    """REQ-CTILESTEST-157: main(argv) calls generate_skeletons once, keyword names == its parameters."""
    received = []
    monkeypatch.setattr(gs, "generate_skeletons", lambda **kwargs: received.append(kwargs))
    monkeypatch.chdir(tmp_path)

    gs.main(["-s", "morpho", "-o", "skel", "-i", "R", "-p", "PG", "-q", "qc.csv", "-j", "wide", "-b", "-a", "-n", "2"])

    assert len(received) == 1
    params = received[0]
    expected_names = set(inspect.signature(gs.GraphConvert2Skeleton.__init__).parameters) - {"self", "njobs"}
    assert set(params) == expected_names | {"nb_subjects"}
    assert params == dict(
        src_dir=str(tmp_path / "morpho"),
        skeleton_dir=str(tmp_path / "skel"),
        side="R",
        path_to_graph="PG",
        qc_path="qc.csv",
        junction="wide",
        bids=True,
        parallel=True,
        nb_subjects=2,
    )
