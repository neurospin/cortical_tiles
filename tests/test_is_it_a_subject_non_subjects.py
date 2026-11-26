"""Non-subject entries of the Morphologist output root (DEF-INPUTKIND-BDRABCZUK-B02AC57FDC8E).

Stage 1 writes a ``qc/`` folder and an ``input_qc.tsv`` file beside the subject folders of the
Morphologist output root. The cortical_tiles steps list ``glob(src_dir/*)`` and keep the entries
``is_it_a_subject`` accepts, so both used to be counted as subjects, logged as skipped with
"No graph file", and stage 2 reported a partial run. ``is_it_a_subject`` mirrors the pipeline's
``is_non_subject`` (layout-free subject-folder rule, TASK-ARGDISC-BDRABCZUK-8958D29D2B79): a plain file, a
dot-folder and a qc folder holding no .arg file outside dot folders are no subjects; every other directory is
one. No t1mri folder name plays a part.
"""

import os

import pytest
from cortical_tiles.brainvisa import generate_skeletons as gs
from cortical_tiles.brainvisa.utils.skip_list import load_skip_entries
from cortical_tiles.brainvisa.utils.subjects import is_it_a_subject

PATH_TO_GRAPH = "t1mri/default_acquisition/default_analysis/folds/3.1"


# --- REQ-INPUTKIND-BDRABCZUK-79DC0919E3F4: a path that is not a directory is no subject -------------


@pytest.mark.parametrize("name", ["input_qc.tsv", "sub-01", "notes.txt", "sub-01.minf"])
def test_plain_file_is_not_a_subject(tmp_path, name):
    """REQ-INPUTKIND-BDRABCZUK-79DC0919E3F4: a plain file of the Morphologist root is no subject."""
    path = tmp_path / name
    path.write_text("")

    assert is_it_a_subject(str(path)) is False


# --- REQ-ARGDISC-BDRABCZUK-20C979C71097: a qc directory holding no .arg file outside dot folders is no subject


@pytest.mark.parametrize(
    "dirs, files",
    [
        ([], []),
        ([], ["report.html", "t1mri"]),
        (["ses-01"], []),
        (["anat/t1mri"], []),
        (["ses-01/anat/t1mri"], []),
    ],
    ids=["empty", "files-only", "session-without-t1mri", "t1mri-too-deep", "session-t1mri-too-deep"],
)
def test_qc_directory_is_not_a_subject(tmp_path, dirs, files):
    """REQ-ARGDISC-BDRABCZUK-20C979C71097: a qc folder holding no .arg file is no subject."""
    path = tmp_path / "qc"
    path.mkdir()
    for rel in dirs:
        (path / rel).mkdir(parents=True)
    for rel in files:
        (path / rel).write_text("")

    assert is_it_a_subject(str(path)) is False


@pytest.mark.parametrize(
    "t1mri",
    ["t1mri", "ses-01/t1mri", "ses-M00/t1mri", "t1mri/default_acquisition/default_analysis/folds/3.1"],
)
def test_qc_directory_holding_t1mri_without_graph_is_not_a_subject(tmp_path, t1mri):
    """REQ-ARGDISC-BDRABCZUK-20C979C71097: a t1mri folder no longer makes qc a subject; only a graph does."""
    path = tmp_path / "qc"
    (path / t1mri).mkdir(parents=True)
    (path / t1mri / "qc_report.txt").write_text("")

    assert is_it_a_subject(str(path)) is False


@pytest.mark.parametrize(
    "graph",
    [
        "t1mri/.snapshots/Lqc.arg",
        "t1mri/default_acquisition/.trash/Rqc.arg",
        "t1mri/default_acquisition/default_analysis/folds/3.1/.Lqc.arg",
    ],
    ids=["graph-in-dot-folder", "graph-in-nested-dot-folder", "dot-named-graph"],
)
def test_qc_directory_with_graphs_only_in_dot_entries_is_not_a_subject(tmp_path, graph):
    """REQ-ARGDISC-BDRABCZUK-20C979C71097: a .arg file inside a dot folder, or named with a leading dot, does not count."""
    path = tmp_path / "qc"
    (path / graph).parent.mkdir(parents=True)
    (path / graph).write_text("")

    assert is_it_a_subject(str(path)) is False


# --- REQ-ARGDISC-BDRABCZUK-48D6860C368B: a qc directory holding a .arg file outside dot folders is a subject


@pytest.mark.parametrize(
    "graph",
    [
        "Lqc.arg",
        "folds/3.1/Lqc.arg",
        "anat/ses-01/Rqc_ses-01.arg",
        "ses-1/run-0/qc_ses-1_run-0_hemi-L.arg",
    ],
    ids=["graph-at-top", "graph-in-folds", "graph-in-session", "graph-morphologist-cli-like"],
)
def test_qc_directory_holding_a_graph_is_a_subject(tmp_path, graph):
    """REQ-ARGDISC-BDRABCZUK-48D6860C368B: a qc folder holding a .arg file at any depth is a subject, no t1mri needed."""
    path = tmp_path / "qc"
    (path / graph).parent.mkdir(parents=True, exist_ok=True)
    (path / graph).write_text("")

    assert is_it_a_subject(str(path)) is True


# --- REQ-INPUTKIND-BDRABCZUK-2B25FC141E82: a dot-prefixed directory is no subject -------------------


@pytest.mark.parametrize("name", [".snapshots", ".git", ".sub-01"])
def test_dot_directory_is_not_a_subject(tmp_path, name):
    """REQ-INPUTKIND-BDRABCZUK-2B25FC141E82: a folder whose name starts with a dot is no subject, t1mri or not."""
    path = tmp_path / name
    (path / "t1mri").mkdir(parents=True)

    assert is_it_a_subject(str(path)) is False


# --- REQ-INPUTKIND-BDRABCZUK-0E39BDF5EF35: any other directory is a subject -------------------------


@pytest.mark.parametrize("name", ["sub-01", "s01", "012", "qc_subject", "QC", "sub.01"])
def test_subject_directory_is_a_subject(tmp_path, name):
    """REQ-INPUTKIND-BDRABCZUK-0E39BDF5EF35: a directory not named qc and not dot-prefixed is a subject."""
    path = tmp_path / name
    path.mkdir()

    assert is_it_a_subject(str(path)) is True


# --- regression of the defect: generate_skeletons on a Morphologist root of a --bids run -------------


@pytest.fixture
def fake_conversion(monkeypatch):
    """Replaces the graph -> skeleton conversion by an empty marker file."""

    def _fake(graph_file, skeleton_file, junction):
        open(skeleton_file, "w").close()

    monkeypatch.setattr(gs, "generate_skeleton_from_graph_file", _fake)


def test_generate_skeletons_records_no_skip_for_root_qc_entries(tmp_path, fake_conversion):
    """REQ-INPUTKIND-BDRABCZUK-79DC0919E3F4, REQ-ARGDISC-BDRABCZUK-20C979C71097: no skip for qc/, input_qc.tsv."""
    morpho = tmp_path / "morpho"
    folds = morpho / "s01" / PATH_TO_GRAPH
    folds.mkdir(parents=True)
    (folds / "Ls01.arg").write_text("")
    (morpho / "qc").mkdir()
    (morpho / "input_qc.tsv").write_text("participant_id\tqc\ns01\t1\n")
    skip_dir = tmp_path / "skiplists"

    gs.generate_skeletons(
        src_dir=str(morpho),
        skeleton_dir=str(tmp_path / "skeletons"),
        path_to_graph=PATH_TO_GRAPH,
        side="L",
        junction="thin",
        bids=False,
        parallel=False,
        nb_subjects=-1,
        qc_path="",
        njobs=1,
        skip_dir=str(skip_dir),
    )

    assert [e.subject for e in load_skip_entries(skip_dir, "L")] == []
    assert os.listdir(tmp_path / "skeletons" / "L") != []


def test_generate_skeletons_records_no_skip_for_root_dot_folder(tmp_path, fake_conversion):
    """REQ-INPUTKIND-BDRABCZUK-2B25FC141E82: no skip for a dot-folder beside the subject folders."""
    morpho = tmp_path / "morpho"
    folds = morpho / "s01" / PATH_TO_GRAPH
    folds.mkdir(parents=True)
    (folds / "Ls01.arg").write_text("")
    (morpho / ".snapshots").mkdir()
    skip_dir = tmp_path / "skiplists"

    gs.generate_skeletons(
        src_dir=str(morpho),
        skeleton_dir=str(tmp_path / "skeletons"),
        path_to_graph=PATH_TO_GRAPH,
        side="L",
        junction="thin",
        bids=False,
        parallel=False,
        nb_subjects=-1,
        qc_path="",
        njobs=1,
        skip_dir=str(skip_dir),
    )

    assert [e.subject for e in load_skip_entries(skip_dir, "L")] == []
    assert os.listdir(tmp_path / "skeletons" / "L") != []
