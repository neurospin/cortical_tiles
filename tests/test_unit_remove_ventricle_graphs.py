"""Graph discovery in cortical_tiles.brainvisa.remove_ventricle (TASK-ARGDISC-BDRABCZUK-D7BACC319268).

Without a path_to_graph, the whole-brain ventricle removal reads each subject's labelled graph through the
cortical_tiles graph resolver, in the classic branch and in the BIDS branch, and never falls back to an unlabelled
graph (an unlabelled graph holds no ventricle label). A given path_to_graph keeps today's fixed path.

The subject trees hold empty .arg files with synthetic names (tests/graph_trees.py); every .arg read is replaced by a
recorder that returns a tiny in-memory graph, so no real data is used and no .arg file is opened.
"""

import os

import numpy as np
import pytest
from soma import aims

from cortical_tiles.brainvisa import remove_ventricle as rv
from tests.graph_trees import (
    CLASSIC_FOLDS,
    CLI_LABEL,
    CLI_SUBJECT,
    LABEL,
    OTHER_LABEL,
    SUBJECT,
    UNLABELLED,
    classic_subject,
    cli_subject,
    option_help,
    touch,
)

SRC_FILENAME = "skeleton_generated_"
OUTPUT_FILENAME = "skeleton_generated_without_ventricle_"
VENTRICLE_VOXEL = (1, 1, 1)
BIDS_SUBJECT = "sub-01"


def _ventricle_graph():
    """A graph whose only vertex is a ventricle with one simple-surface voxel, VENTRICLE_VOXEL."""
    bucket = aims.BucketMap_VOID()
    bucket.setSizeXYZT(1.0, 1.0, 1.0, 1.0)
    bucket[0][aims.Point3d(*VENTRICLE_VOXEL)] = 0
    graph = aims.Graph("CorticalFoldArg")
    graph["voxel_size"] = [1.0, 1.0, 1.0, 1.0]
    vertex = graph.addVertex("fold")
    vertex["label"] = "ventricle_left"
    vertex["aims_ss"] = aims.rc_ptr_BucketMap_VOID(bucket)
    return graph


def _write_source(tmp_path, side, subject):
    """Writes the 6x6x6 source volume (value 30) of one subject in <tmp>/src/<side>; returns its path."""
    folder = tmp_path / "src" / side
    folder.mkdir(parents=True, exist_ok=True)
    volume = aims.Volume([6, 6, 6], dtype="S16")
    volume.header()["voxel_size"] = [1.0, 1.0, 1.0, 1.0]
    volume.fill(30)
    path = folder / f"{side}{SRC_FILENAME}{subject}.nii.gz"
    aims.write(volume, str(path))
    return path


def _outputs(tmp_path, side):
    """The output volumes written in <tmp>/out/<side>, .minf files left out, sorted."""
    return sorted(name for name in os.listdir(tmp_path / "out" / side) if not name.endswith(".minf"))


@pytest.fixture
def graph_reads(monkeypatch):
    """Replaces aims.read of a .arg file by a recorder returning _ventricle_graph(); returns the recorded paths."""
    recorded = []
    read = rv.aims.read

    def _read(path, *arguments, **options):
        if str(path).endswith(".arg"):
            recorded.append(str(path))
            return _ventricle_graph()
        return read(path, *arguments, **options)

    monkeypatch.setattr(rv.aims, "read", _read)
    return recorded


def _remover(tmp_path, morpho_dir, labelling_session=LABEL, side="F", bids=False, path_to_graph=None):
    return rv.RemoveVentricleFromVolume(
        src_dir=str(tmp_path / "src"),
        output_dir=str(tmp_path / "out"),
        morpho_dir=str(morpho_dir),
        path_to_graph=path_to_graph,
        labelling_session=labelling_session,
        src_filename=SRC_FILENAME,
        output_filename=OUTPUT_FILENAME,
        side=side,
        bids=bids,
        parallel=False,
    )


# --- REQ-ARGDISC-BDRABCZUK-C1D84B71D106: classic branch, graph discovery ----------------------------------------------


def test_discovery_reads_labelled_graphs_of_classic_layout(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-C1D84B71D106: no path_to_graph: per side, the labelling session graph (classic)."""
    morpho = tmp_path / "morpho"
    left = classic_subject(morpho, side="L")
    right = classic_subject(morpho, side="R")
    assert _remover(tmp_path, morpho).get_labelled_graph(SUBJECT) == [left[LABEL], right[LABEL]]


def test_discovery_reads_labelled_graphs_of_morphologist_cli_layout(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-C1D84B71D106: no path_to_graph: per side, the labelling session graph (cli layout)."""
    morpho = tmp_path / "morpho"
    left = cli_subject(morpho, side="L")
    right = cli_subject(morpho, side="R")
    remover = _remover(tmp_path, morpho, labelling_session=CLI_LABEL)
    assert remover.get_labelled_graph(CLI_SUBJECT) == [left[CLI_LABEL], right[CLI_LABEL]]


def test_discovery_reads_graph_of_the_labelling_session_among_several(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-C1D84B71D106: two labelling sets: the one named by labelling_session, not the other."""
    morpho = tmp_path / "morpho"
    left = classic_subject(morpho, labels=(LABEL, OTHER_LABEL), side="L")
    remover = _remover(tmp_path, morpho, labelling_session=OTHER_LABEL, side="L")
    assert remover.get_labelled_graph(SUBJECT) == [left[OTHER_LABEL]]


def test_remove_ventricle_without_path_to_graph_writes_volume_from_discovered_graph(tmp_path, graph_reads):
    """REQ-ARGDISC-BDRABCZUK-C1D84B71D106: remove_ventricle() without path_to_graph reads the found graph, writes.

    The morphologist-cli tree lies off today's default path_to_graph (t1mri/BL/default_analysis/folds/3.1).
    """
    morpho = tmp_path / "morpho"
    graph = cli_subject(morpho, side="L")[CLI_LABEL]
    _write_source(tmp_path, "L", CLI_SUBJECT)

    rv.remove_ventricle(
        src_dir=str(tmp_path / "src"),
        output_dir=str(tmp_path / "out"),
        morpho_dir=str(morpho),
        labelling_session=CLI_LABEL,
        side="L",
    )

    assert graph_reads == [graph]
    output = f"L{OUTPUT_FILENAME}{CLI_SUBJECT}.nii.gz"
    assert _outputs(tmp_path, "L") == [output]
    result = np.asarray(aims.read(str(tmp_path / "out" / "L" / output)))[..., 0]
    assert result[VENTRICLE_VOXEL] == 0 and result[0, 0, 0] == 30


# --- REQ-ARGDISC-BDRABCZUK-D2F2EEEB42F8: BIDS branch, graph discovery -------------------------------------------------


def _bids_classic_graph(morpho, session, side):
    """Classic BIDS tree <morpho>/sub-01/t1mri/<session>/...: unlabelled and labelled graph; returns the labelled."""
    folds = f"{BIDS_SUBJECT}/t1mri/{session}/default_analysis/folds/3.1"
    touch(morpho, f"{folds}/{side}{BIDS_SUBJECT}.arg")
    return touch(morpho, f"{folds}/{LABEL}/{side}{BIDS_SUBJECT}_{LABEL}.arg")


def test_bids_discovery_reads_labelled_graphs_of_the_session_key_in_classic_layout(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-D2F2EEEB42F8: two session folders: the graphs of the subject's session key."""
    morpho = tmp_path / "morpho"
    graphs = {
        (session, side): _bids_classic_graph(morpho, session, side) for session in ("ses-1", "ses-2") for side in "LR"
    }
    remover = _remover(tmp_path, morpho, bids=True)
    assert remover.get_labelled_graph(f"{BIDS_SUBJECT}_ses-2") == [graphs["ses-2", "L"], graphs["ses-2", "R"]]


def test_bids_discovery_reads_labelled_graphs_of_the_session_key_in_morphologist_cli_layout(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-D2F2EEEB42F8: morphologist-cli ses-<n>/run-0/ana-0/...: the graphs of the session key."""
    morpho = tmp_path / "morpho"
    graphs = {
        (session, side): cli_subject(morpho, folder=BIDS_SUBJECT, folds=f"{session}/run-0/ana-0/folds/3.1", side=side)
        for session in ("ses-1", "ses-2")
        for side in "LR"
    }
    remover = _remover(tmp_path, morpho, labelling_session=CLI_LABEL, bids=True)
    expected = [graphs["ses-2", "L"][CLI_LABEL], graphs["ses-2", "R"][CLI_LABEL]]
    assert remover.get_labelled_graph(f"{BIDS_SUBJECT}_ses-2_run-0") == expected


# --- REQ-ARGDISC-BDRABCZUK-2B5E5A6BA347 and -8D59CA724571: labelled graph only ----------------------------------------


def _subject_without_labelled_graph(morpho, case):
    """Lays out SUBJECT so that at least one side has no graph of LABEL; returns its unlabelled graphs."""
    if case == "unlabelled_only":
        labels = {"L": (), "R": ()}
    elif case == "other_labelling_set_only":
        labels = {"L": (OTHER_LABEL,), "R": (OTHER_LABEL,)}
    else:  # right_side_unlabelled
        labels = {"L": (LABEL,), "R": ()}
    return [classic_subject(morpho, labels=labels[side], side=side)[UNLABELLED] for side in "LR"]


CASES = ["unlabelled_only", "other_labelling_set_only", "right_side_unlabelled"]


@pytest.mark.parametrize("case", CASES)
def test_discovery_without_labelled_graph_logs_missing_graph_error(tmp_path, caplog, graph_reads, case):
    """REQ-ARGDISC-BDRABCZUK-2B5E5A6BA347: a side without labelling session graph: one error, subject and missing graph.

    The unlabelled graph the resolver would fall back to is never read.
    """
    morpho = tmp_path / "morpho"
    unlabelled = _subject_without_labelled_graph(morpho, case)
    _write_source(tmp_path, "F", SUBJECT)

    with caplog.at_level("ERROR"):
        _remover(tmp_path, morpho).remove_ventricle_from_one_subject(SUBJECT)

    errors = [record.getMessage() for record in caplog.records if record.levelname == "ERROR"]
    assert len(errors) == 1
    assert errors[0].startswith(f"{SUBJECT}: FileNotFoundError") and "Labelled graph not found" in errors[0]
    assert not set(unlabelled) & set(graph_reads)


@pytest.mark.parametrize("case", CASES)
def test_discovery_without_labelled_graph_writes_no_volume(tmp_path, graph_reads, case):
    """REQ-ARGDISC-BDRABCZUK-8D59CA724571: no output volume for the subject without labelled graph; one for subj02."""
    morpho = tmp_path / "morpho"
    _subject_without_labelled_graph(morpho, case)
    for side in "LR":
        classic_subject(morpho, subject="subj02", side=side)
    for subject in (SUBJECT, "subj02"):
        _write_source(tmp_path, "F", subject)

    _remover(tmp_path, morpho).compute(number_subjects=-1)

    assert _outputs(tmp_path, "F") == [f"F{OUTPUT_FILENAME}subj02.nii.gz"]


# --- REQ-ARGDISC-BDRABCZUK-CC76CFF3255F: a given path_to_graph keeps today's path -------------------------------------


def test_given_path_to_graph_keeps_todays_labelled_graph_path(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-CC76CFF3255F: path_to_graph given: today's fixed path."""
    morpho = tmp_path / "morpho"
    classic_subject(morpho, side="L")
    remover = _remover(tmp_path, morpho, path_to_graph=CLASSIC_FOLDS)
    expected = [str(morpho / SUBJECT / CLASSIC_FOLDS / LABEL / f"{side}{SUBJECT}_{LABEL}.arg") for side in "LR"]
    assert remover.get_labelled_graph(SUBJECT) == expected


# --- REQ-ARGDISC-BDRABCZUK-50CBF194489A and -336515F15D07: command line -----------------------------------------------


def test_command_line_without_p_passes_no_path_to_graph(tmp_path, monkeypatch):
    """REQ-ARGDISC-BDRABCZUK-50CBF194489A: without -p, main calls remove_ventricle with path_to_graph None."""
    received = []
    monkeypatch.setattr(rv, "remove_ventricle", lambda **kwargs: received.append(kwargs))
    rv.main(["-o", str(tmp_path / "out"), "-i", "F"])
    assert len(received) == 1 and received[0]["path_to_graph"] is None


def test_p_help_names_graph_discovery_as_default(capsys, monkeypatch):
    """REQ-ARGDISC-BDRABCZUK-336515F15D07: the -p help names graph discovery below each subject folder as default."""
    assert "Default: graph discovery below each subject folder" in option_help(rv, "-p", capsys, monkeypatch)
