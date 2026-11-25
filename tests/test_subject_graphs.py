"""Tests for the per-subject graph resolver cortical_tiles.brainvisa.subject_graphs.resolve_subject_graphs
(TASK-ARGDISC-BDRABCZUK-55F1E122DB45, epic champollion_pipeline:TASK-ARGDISC-BDRABCZUK-D91B3C7ACD31).

The trees are made of empty files with synthetic names only. Layouts: classic BrainVISA
(t1mri/<acq>/default_analysis/folds/3.1/<session>), morphologist-cli (ses-1/run-0/ana-0/folds/3.1/sul-0_auto)
and a SENIOR-like multi-acquisition one (3TV2, 3TV10).

* REQ-ARGDISC-BDRABCZUK-95E2B7F8CFE9 - the .arg files of a subject are found without a fixed path_to_graph.
* REQ-ARGDISC-BDRABCZUK-EF5C30A027EE - the .arg files the caller lists are returned per side.
* REQ-ARGDISC-BDRABCZUK-957FFCD9F8C5 - a path_to_graph keeps today's glob <subject>/<path_to_graph>/<side>*.arg.
* REQ-ARGDISC-BDRABCZUK-CDAFC60D5AC7 - graph discovery returns the labelled .arg file at any depth.
* REQ-ARGDISC-BDRABCZUK-8EA395AFE4E2 - the side comes from the hemi- entity, else from the first letter.
* REQ-ARGDISC-BDRABCZUK-000BFB310C79 - a .arg file in a folder ending with _auto or _manual is labelled.
* REQ-ARGDISC-BDRABCZUK-F25183B74458 - without a labelled .arg file, the unlabelled one is returned.
* REQ-ARGDISC-BDRABCZUK-2F54654004F5 - a WARNING names the subject folder per side resolved to an unlabelled file.
* REQ-ARGDISC-BDRABCZUK-D2C5FD392292 - two or more labelling sets without a label raise a ValueError listing them.
* REQ-ARGDISC-BDRABCZUK-F6E003EFB3A6 - a label selects its labelling set.
* REQ-ARGDISC-BDRABCZUK-8A31C044EAF4 - several candidates: the first path in natural sort order is kept.
* REQ-ARGDISC-BDRABCZUK-A6C34B4EF807 - each candidate left out is logged at INFO level.
* REQ-ARGDISC-BDRABCZUK-748B93FC1B51 - a side for which no .arg file qualifies gets an empty list.
* REQ-ARGDISC-BDRABCZUK-79B17D42F1AD - a label with no .arg file of its set on a side falls back to the unlabelled one.
* REQ-ARGDISC-BDRABCZUK-1A08DF844E0D - a graphs map listing no .arg file for a side gives that side an empty list.
* REQ-ARGDISC-BDRABCZUK-A9CA002C7448 - a graphs map beside a path_to_graph raises a ValueError.
"""

import logging
import os

import pytest

from cortical_tiles.brainvisa.subject_graphs import GraphSpec, resolve_subject_graphs

SUBJECT = "subj01"
CLASSIC_FOLDS = "t1mri/BL/default_analysis/folds/3.1"
CLI_SUBJECT = "sub-01"
CLI_FOLDS = "ses-1/run-0/ana-0/folds/3.1"


def _touch(root, relative):
    """Creates the empty file root/relative and its folders; returns its path as a string."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return str(path)


def _as_str(result):
    """Returns the resolver result with every path as a string."""
    return {side: [os.fspath(path) for path in paths] for side, paths in result.items()}


def _unlabelled(subject_dir, subject=SUBJECT, folds=CLASSIC_FOLDS):
    """Creates the classic unlabelled graphs of subject; returns {side: path}."""
    return {side: _touch(subject_dir, f"{folds}/{side}{subject}.arg") for side in ("L", "R")}


def _labelled(subject_dir, session, subject=SUBJECT, folds=CLASSIC_FOLDS, sides=("L", "R")):
    """Creates the classic labelled graphs of subject in folder session; returns {side: path}."""
    return {side: _touch(subject_dir, f"{folds}/{session}/{side}{subject}_{session}.arg") for side in sides}


@pytest.fixture
def subject_dir(tmp_path):
    """The classic subject folder graphs/subj01 (empty)."""
    path = tmp_path / "graphs" / SUBJECT
    path.mkdir(parents=True)
    return path


@pytest.fixture
def cli_subject_dir(tmp_path):
    """A morphologist-cli subject folder: unlabelled hemi-L/R graphs and the sul-0_auto labelled set."""
    path = tmp_path / "morphologist" / CLI_SUBJECT
    stem = f"{CLI_SUBJECT}_ses-1_run-0"
    for side in ("L", "R"):
        _touch(path, f"{CLI_FOLDS}/{stem}_hemi-{side}.arg")
        _touch(path, f"{CLI_FOLDS}/sul-0_auto/{stem}_sul-0_hemi-{side}_auto.arg")
    return path


@pytest.fixture
def senior_subject_dir(tmp_path):
    """A SENIOR-like subject folder: acquisitions 3TV10 and 3TV2, one labelled set each, plus unlabelled graphs."""
    path = tmp_path / "derivatives" / "subjects_f" / SUBJECT
    for acquisition in ("3TV10", "3TV2"):
        folds = f"t1mri/{acquisition}/default_analysis/folds/3.1"
        _unlabelled(path, folds=folds)
        _labelled(path, "default_session_auto", folds=folds)
    return path


def _senior_graph(subject_dir, acquisition, side):
    """Returns the path of the labelled graph of side in acquisition of the SENIOR-like tree."""
    return str(
        subject_dir
        / f"t1mri/{acquisition}/default_analysis/folds/3.1/default_session_auto"
        / f"{side}{SUBJECT}_default_session_auto.arg"
    )


# --- REQ-ARGDISC-BDRABCZUK-EF5C30A027EE: caller-listed graph files ------------------------------------------


def test_listed_graph_files_are_returned_per_side(subject_dir):
    """The listed files are returned, not the labelled ones discovery would pick."""
    unlabelled = _unlabelled(subject_dir)
    _labelled(subject_dir, "default_session_auto")
    spec = GraphSpec(graphs={SUBJECT: {"L": [unlabelled["L"]], "R": [unlabelled["R"]]}})

    result = _as_str(resolve_subject_graphs(str(subject_dir), spec))

    assert result == {"L": [unlabelled["L"]], "R": [unlabelled["R"]]}


# --- REQ-ARGDISC-BDRABCZUK-957FFCD9F8C5: legacy path_to_graph glob ------------------------------------------


def test_path_to_graph_override_returns_legacy_glob_matches(subject_dir):
    """<subject>/<path_to_graph>/<side>*.arg: the labelled graphs one folder below are not matched."""
    unlabelled = _unlabelled(subject_dir)
    _labelled(subject_dir, "default_session_auto")

    result = _as_str(resolve_subject_graphs(str(subject_dir), GraphSpec(path_to_graph=CLASSIC_FOLDS)))

    assert result == {"L": [unlabelled["L"]], "R": [unlabelled["R"]]}


def test_path_to_graph_wildcard_returns_every_match_sorted(subject_dir):
    """A wildcard in path_to_graph keeps every match, sorted, as today's glob does."""
    follow_up = _labelled(subject_dir, "deepcnn_auto", folds="t1mri/FU/default_analysis/folds/3.1")
    baseline = _labelled(subject_dir, "deepcnn_auto", folds="t1mri/BL/default_analysis/folds/3.1")
    spec = GraphSpec(path_to_graph="t1mri/*/default_analysis/folds/3.1/deepcnn_auto")

    result = _as_str(resolve_subject_graphs(str(subject_dir), spec))

    assert result == {"L": [baseline["L"], follow_up["L"]], "R": [baseline["R"], follow_up["R"]]}


# --- REQ-ARGDISC-BDRABCZUK-95E2B7F8CFE9, REQ-ARGDISC-BDRABCZUK-CDAFC60D5AC7: discovery, labelled first ------


def test_discovery_returns_labelled_graph_in_classic_layout(subject_dir):
    """Classic BrainVISA layout: the labelled graph of the default session, without any path_to_graph."""
    _unlabelled(subject_dir)
    labelled = _labelled(subject_dir, "default_session_auto")

    result = _as_str(resolve_subject_graphs(str(subject_dir), GraphSpec()))

    assert result == {"L": [labelled["L"]], "R": [labelled["R"]]}


def test_discovery_returns_labelled_graph_in_morphologist_cli_layout(cli_subject_dir):
    """morphologist-cli layout (no t1mri folder): the sul-0_auto labelled graphs."""
    labelled = cli_subject_dir / CLI_FOLDS / "sul-0_auto"
    stem = f"{CLI_SUBJECT}_ses-1_run-0_sul-0"

    result = _as_str(resolve_subject_graphs(str(cli_subject_dir), GraphSpec()))

    assert result == {
        "L": [str(labelled / f"{stem}_hemi-L_auto.arg")],
        "R": [str(labelled / f"{stem}_hemi-R_auto.arg")],
    }


# --- REQ-ARGDISC-BDRABCZUK-8EA395AFE4E2: side of a graph file ------------------------------------------------


def test_discovery_reads_side_from_hemi_entity(tmp_path):
    """A name starting with sub- takes its side from the hemi- entity (unlabelled graphs only here)."""
    subject_dir = tmp_path / "morphologist" / CLI_SUBJECT
    stem = f"{CLI_SUBJECT}_ses-1_run-0"
    left = _touch(subject_dir, f"{CLI_FOLDS}/{stem}_hemi-L.arg")
    right = _touch(subject_dir, f"{CLI_FOLDS}/{stem}_hemi-R.arg")

    result = _as_str(resolve_subject_graphs(str(subject_dir), GraphSpec()))

    assert result == {"L": [left], "R": [right]}


def test_discovery_reads_side_from_first_letter(tmp_path):
    """Subject R01: the classic names LR01_... and RR01_... both hold an R; the first letter decides."""
    subject = "R01"
    subject_dir = tmp_path / "graphs" / subject
    labelled = _labelled(subject_dir, "default_session_auto", subject=subject)

    result = _as_str(resolve_subject_graphs(str(subject_dir), GraphSpec()))

    assert result == {"L": [labelled["L"]], "R": [labelled["R"]]}


# --- REQ-ARGDISC-BDRABCZUK-000BFB310C79: labelled by folder suffix -------------------------------------------


def test_discovery_treats_manual_folder_graph_as_labelled(subject_dir):
    """A graph in a folder ending with _manual is labelled: it wins over the unlabelled graph."""
    _unlabelled(subject_dir)
    manual = _labelled(subject_dir, "base2018_manual")

    result = _as_str(resolve_subject_graphs(str(subject_dir), GraphSpec()))

    assert result == {"L": [manual["L"]], "R": [manual["R"]]}


# --- REQ-ARGDISC-BDRABCZUK-F25183B74458, REQ-ARGDISC-BDRABCZUK-2F54654004F5: unlabelled fallback ------------


def test_discovery_falls_back_to_unlabelled_graph(subject_dir):
    """No labelled graph at all: the unlabelled graphs are returned."""
    unlabelled = _unlabelled(subject_dir)

    result = _as_str(resolve_subject_graphs(str(subject_dir), GraphSpec()))

    assert result == {"L": [unlabelled["L"]], "R": [unlabelled["R"]]}


def test_discovery_warns_when_unlabelled_graph_is_used(subject_dir, caplog):
    """Left labelled, right unlabelled only: one WARNING naming the subject folder, for the right side."""
    _unlabelled(subject_dir)
    _labelled(subject_dir, "default_session_auto", sides=("L",))

    with caplog.at_level(logging.INFO):
        resolve_subject_graphs(str(subject_dir), GraphSpec())

    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING and SUBJECT in record.getMessage()
    ]
    assert len(warnings) == 1


# --- REQ-ARGDISC-BDRABCZUK-D2C5FD392292, REQ-ARGDISC-BDRABCZUK-F6E003EFB3A6: labelling sets ------------------


def test_discovery_refuses_several_labelling_sets_without_label(subject_dir):
    """Two labelling sets and no label: a ValueError naming both sets."""
    _labelled(subject_dir, "default_session_auto")
    _labelled(subject_dir, "deepcnn_auto")

    with pytest.raises(ValueError) as error:
        resolve_subject_graphs(str(subject_dir), GraphSpec())

    assert "default_session_auto" in str(error.value)
    assert "deepcnn_auto" in str(error.value)


def test_discovery_returns_graph_of_named_labelling_set(subject_dir):
    """Two labelling sets and the label deepcnn_auto: the deepcnn_auto graphs."""
    _labelled(subject_dir, "default_session_auto")
    deepcnn = _labelled(subject_dir, "deepcnn_auto")

    result = _as_str(resolve_subject_graphs(str(subject_dir), GraphSpec(label="deepcnn_auto")))

    assert result == {"L": [deepcnn["L"]], "R": [deepcnn["R"]]}


# --- REQ-ARGDISC-BDRABCZUK-8A31C044EAF4, REQ-ARGDISC-BDRABCZUK-A6C34B4EF807: several acquisitions -----------


def test_discovery_keeps_first_acquisition_in_natural_order(senior_subject_dir):
    """3TV2 comes before 3TV10 in natural sort order (plain sorting puts 3TV10 first)."""
    result = _as_str(resolve_subject_graphs(str(senior_subject_dir), GraphSpec()))

    assert result == {
        "L": [_senior_graph(senior_subject_dir, "3TV2", "L")],
        "R": [_senior_graph(senior_subject_dir, "3TV2", "R")],
    }


def test_discovery_logs_left_out_candidates(senior_subject_dir, caplog):
    """The 3TV10 graph left out by the natural-sort choice is named in an INFO record, per side."""
    with caplog.at_level(logging.INFO):
        resolve_subject_graphs(str(senior_subject_dir), GraphSpec())

    info = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    for side in ("L", "R"):
        assert any(_senior_graph(senior_subject_dir, "3TV10", side) in message for message in info)


# --- REQ-ARGDISC-BDRABCZUK-748B93FC1B51: side without qualifying graph ---------------------------------------


def test_side_without_graph_gets_empty_list(subject_dir):
    """Only left graphs exist: the right side gets an empty list."""
    labelled = _labelled(subject_dir, "default_session_auto", sides=("L",))

    result = _as_str(resolve_subject_graphs(str(subject_dir), GraphSpec()))

    assert result == {"L": [labelled["L"]], "R": []}


def test_discovery_gives_empty_list_when_label_and_unlabelled_graph_are_missing(subject_dir):
    """Label deepcnn_auto exists on the left only and no unlabelled graph exists: the right side gets an empty list,
    not the default_session_auto graph."""
    deepcnn = _labelled(subject_dir, "deepcnn_auto", sides=("L",))
    _labelled(subject_dir, "default_session_auto")

    result = _as_str(resolve_subject_graphs(str(subject_dir), GraphSpec(label="deepcnn_auto")))

    assert result == {"L": [deepcnn["L"]], "R": []}


# --- REQ-ARGDISC-BDRABCZUK-79B17D42F1AD (with REQ-ARGDISC-BDRABCZUK-2F54654004F5): label missing on a side ---


def test_discovery_falls_back_to_unlabelled_graph_when_label_is_missing(subject_dir):
    """Label deepcnn_auto exists on the left only: the right side gets the unlabelled graph, not default_session_auto."""
    unlabelled = _unlabelled(subject_dir)
    _labelled(subject_dir, "default_session_auto")
    deepcnn = _labelled(subject_dir, "deepcnn_auto", sides=("L",))

    result = _as_str(resolve_subject_graphs(str(subject_dir), GraphSpec(label="deepcnn_auto")))

    assert result == {"L": [deepcnn["L"]], "R": [unlabelled["R"]]}


def test_discovery_warns_once_when_label_is_missing_on_one_side(subject_dir, caplog):
    """The right side falls back to its unlabelled graph: one WARNING naming the subject folder."""
    _unlabelled(subject_dir)
    _labelled(subject_dir, "default_session_auto")
    _labelled(subject_dir, "deepcnn_auto", sides=("L",))

    with caplog.at_level(logging.INFO):
        resolve_subject_graphs(str(subject_dir), GraphSpec(label="deepcnn_auto"))

    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING and SUBJECT in record.getMessage()
    ]
    assert len(warnings) == 1


# --- REQ-ARGDISC-BDRABCZUK-1A08DF844E0D: graphs map leaving a subject or a side out --------------------------


def test_graphs_map_without_subject_returns_empty_lists(subject_dir):
    """The map lists another subject only: no graph for this one, although discovery would find labelled graphs."""
    _unlabelled(subject_dir)
    _labelled(subject_dir, "default_session_auto")
    spec = GraphSpec(graphs={"subj02": {"L": ["/elsewhere/Lsubj02.arg"], "R": ["/elsewhere/Rsubj02.arg"]}})

    result = _as_str(resolve_subject_graphs(str(subject_dir), spec))

    assert result == {"L": [], "R": []}


def test_graphs_map_without_side_returns_empty_list_for_that_side(subject_dir):
    """The map lists the left graph of this subject only: the right side gets an empty list."""
    unlabelled = _unlabelled(subject_dir)
    _labelled(subject_dir, "default_session_auto")
    spec = GraphSpec(graphs={SUBJECT: {"L": [unlabelled["L"]]}})

    result = _as_str(resolve_subject_graphs(str(subject_dir), spec))

    assert result == {"L": [unlabelled["L"]], "R": []}


# --- REQ-ARGDISC-BDRABCZUK-A9CA002C7448: graphs map and path_to_graph are exclusive --------------------------


def test_graphs_map_with_path_to_graph_raises_value_error(subject_dir):
    """A graphs map and a path_to_graph together: a ValueError."""
    unlabelled = _unlabelled(subject_dir)
    spec = GraphSpec(graphs={SUBJECT: {"L": [unlabelled["L"]], "R": [unlabelled["R"]]}}, path_to_graph=CLASSIC_FOLDS)

    with pytest.raises(ValueError):
        resolve_subject_graphs(str(subject_dir), spec)


def test_label_beside_graphs_map_is_ignored(subject_dir):
    """A label beside a graphs map: the listed files, no error, no labelling set chosen."""
    unlabelled = _unlabelled(subject_dir)
    _labelled(subject_dir, "deepcnn_auto")
    spec = GraphSpec(graphs={SUBJECT: {"L": [unlabelled["L"]], "R": [unlabelled["R"]]}}, label="deepcnn_auto")

    result = _as_str(resolve_subject_graphs(str(subject_dir), spec))

    assert result == {"L": [unlabelled["L"]], "R": [unlabelled["R"]]}


def test_label_beside_path_to_graph_is_ignored(subject_dir):
    """A label beside a path_to_graph: today's glob matches, no error, no labelling set chosen."""
    unlabelled = _unlabelled(subject_dir)
    _labelled(subject_dir, "deepcnn_auto")
    spec = GraphSpec(path_to_graph=CLASSIC_FOLDS, label="deepcnn_auto")

    result = _as_str(resolve_subject_graphs(str(subject_dir), spec))

    assert result == {"L": [unlabelled["L"]], "R": [unlabelled["R"]]}
