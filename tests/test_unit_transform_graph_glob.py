"""Tests for the graph lookup of cortical_tiles.brainvisa.generate_ICBM2009c_transforms.generate_one_transform
(TASK-CTFAIL-BDRABCZUK-361479576D07, DEF-CTFAIL-BDRABCZUK-428A3CD4576B).

A subject id that is a prefix of another (sub-1, sub-10) gets the transform of its own graph only, in
BIDS and in non-BIDS mode. The subject folder of a hit is matched with ``covered_by``: the folder is the
subject, or the subject followed by ses/acq/run parts. Output names are unchanged.

The graph files are empty placeholders; the module's ``aims`` is a fake that writes the path of the
graph it was given into the transform file, so a test can tell whose graph a transform comes from.
"""

import types
from pathlib import Path

import pytest

from cortical_tiles.brainvisa import generate_ICBM2009c_transforms as gt

SIDE = "R"
PATH_TO_GRAPH = "t1mri/folds"
SHORT = "sub-1"
LONG = "sub-10"


def _graph(src, folder, name=None):
    """Creates an empty graph file in subject folder ``folder``; returns its path."""
    directory = src / folder / PATH_TO_GRAPH
    directory.mkdir(parents=True, exist_ok=True)
    graph = directory / f"{SIDE}{name or folder}.arg"
    graph.write_text("")
    return graph


@pytest.fixture
def fake_aims(monkeypatch):
    """Fake aims: write stores the graph path in the transform file. glob lists the folder of the longer id first."""
    state = types.SimpleNamespace(reads=[])

    def read(path):
        state.reads.append(path)
        return path

    def write(transform, path):
        Path(path).write_text(str(transform))

    graph_manip = types.SimpleNamespace(getICBM2009cTemplateTransform=lambda graph: graph)
    monkeypatch.setattr(gt, "aims", types.SimpleNamespace(read=read, write=write, GraphManip=graph_manip))
    real_glob = gt.glob.glob
    monkeypatch.setattr(gt.glob, "glob", lambda pattern: sorted(real_glob(pattern), key=lambda path: LONG not in path))
    return state


def _step(tmp_path, bids):
    src = tmp_path / "graphs"
    src.mkdir(exist_ok=True)
    step = gt.GraphGenerateTransform(
        src_dir=str(src),
        transform_dir=str(tmp_path / "transforms"),
        side=SIDE,
        parallel=False,
        path_to_graph=PATH_TO_GRAPH,
        bids=bids,
        qc_path="",
        skip_dir=str(tmp_path / "skip"),
    )
    return step, src


def _transform(tmp_path, key):
    return tmp_path / "transforms" / SIDE / f"{SIDE}transform_to_ICBM2009c_{key}.trm"


def test_prefix_subject_gets_its_own_transform_non_bids_mode(tmp_path, fake_aims):
    """Non-BIDS: with sub-1 and sub-10 present, each subject's transform comes from its own graph."""
    step, src = _step(tmp_path, bids=False)
    short_graph, long_graph = _graph(src, SHORT), _graph(src, LONG)

    assert step.generate_one_transform(SHORT) == (SHORT, None)
    assert step.generate_one_transform(LONG) == (LONG, None)

    assert _transform(tmp_path, SHORT).read_text() == str(short_graph)
    assert _transform(tmp_path, LONG).read_text() == str(long_graph)


def test_prefix_subject_gets_its_own_transform_bids_mode(tmp_path, fake_aims):
    """BIDS: sub-1 and sub-10 with the same session each get a transform of their own graph."""
    step, src = _step(tmp_path, bids=True)
    short_graph = _graph(src, f"{SHORT}_ses-01", SHORT)
    long_graph = _graph(src, f"{LONG}_ses-01", LONG)

    assert step.generate_one_transform(SHORT) == (SHORT, None)
    assert step.generate_one_transform(LONG) == (LONG, None)

    assert _transform(tmp_path, f"{SHORT}_ses-01").read_text() == str(short_graph)
    assert _transform(tmp_path, f"{LONG}_ses-01").read_text() == str(long_graph)


@pytest.mark.parametrize("bids", [False, True])
def test_subject_without_graph_never_reads_the_graph_of_a_longer_id(tmp_path, fake_aims, bids):
    """sub-1 has no graph while sub-10 has one: sub-1 reports a missing graph and reads nothing."""
    step, src = _step(tmp_path, bids=bids)
    _graph(src, LONG)

    subject, error = step.generate_one_transform(SHORT)

    assert subject == SHORT
    assert "No graph file" in error
    assert fake_aims.reads == []
    assert not _transform(tmp_path, SHORT).exists()


def test_bids_sessions_of_a_subject_are_all_processed_with_unchanged_names(tmp_path, fake_aims):
    """BIDS: every ses/acq/run folder of the subject keeps its transform name; the longer id is left out."""
    step, src = _step(tmp_path, bids=True)
    folders = (f"{SHORT}_ses-01", f"{SHORT}_ses-02_acq-a_run-1")
    graphs = {folder: _graph(src, folder, SHORT) for folder in folders}
    _graph(src, f"{LONG}_ses-01", LONG)

    assert step.generate_one_transform(SHORT) == (SHORT, None)

    assert _transform(tmp_path, f"{SHORT}_ses-01").read_text() == str(graphs[folders[0]])
    assert _transform(tmp_path, f"{SHORT}_ses-02_acq-a_run-1").read_text() == str(graphs[folders[1]])
    assert not _transform(tmp_path, f"{LONG}_ses-01").exists()
    assert len(fake_aims.reads) == 2


def test_non_bids_exact_folder_still_found(tmp_path, fake_aims):
    """Non-BIDS: the exact subject folder is still found and the output name is the subject id."""
    step, src = _step(tmp_path, bids=False)
    graph = _graph(src, SHORT)

    assert step.generate_one_transform(SHORT) == (SHORT, None)

    assert _transform(tmp_path, SHORT).read_text() == str(graph)
