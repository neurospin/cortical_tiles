"""Graph inputs of generate_sulcal_regions (TASK-ARGDISC-BDRABCZUK-817BCB001B96).

generate_sulcal_regions finds the graphs of a subject through the cortical_tiles graph resolver:
``--path_to_graph`` is an optional override, ``--graphs`` names a graphs map JSON file and
``--graph_label`` names a labelling set; both reach each region run in its configuration (keys
``graphs`` and ``graph_label``). ``run_with_params`` is replaced, on the imported module, by a
recorder (tests/graph_trees.py). Synthetic tmp_path trees only: placeholder .arg files, never opened.
The subject folders of a graphs_dir are listed in sorted order (invariant REQ-INV-BDRABCZUK-90B2C4393F3A).
"""

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cortical_tiles.brainvisa.utils.skip_list import record_skip
from tests.graph_trees import import_generate_sulcal_regions, record_region_runs, touch

SUBJECT = "sub-01"
SKELETONS = "generate_skeletons"


@pytest.fixture
def gsr(monkeypatch):
    return import_generate_sulcal_regions(monkeypatch)


@pytest.fixture
def calls(gsr, monkeypatch):
    """Copies of the configuration of each region run (run_with_params replaced by a recorder)."""
    return record_region_runs(gsr, monkeypatch)


def _dataset(tmp_path, path_to_graph=None):
    """A dataset folder holding pipeline_loop_2mm.json, with one subject folder under graphs_dir."""
    graphs_dir = tmp_path / "graphs"
    (graphs_dir / SUBJECT).mkdir(parents=True)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    config = {
        "graphs_dir": str(graphs_dir),
        "output_dir": str(tmp_path / "out"),
        "path_to_graph": path_to_graph,
        "path_to_skeleton_with_hull": "hull",
        "skel_qc_path": "",
    }
    (dataset / "pipeline_loop_2mm.json").write_text(json.dumps(config))
    return dataset


def _argv(dataset, *extra):
    return ["-d", str(dataset), "--path_sk_with_hull", "H", "-i", "R", "-r", "S.Or.", "-y", "skeleton", *extra]


def _graphs_map(tmp_path):
    graphs = {SUBJECT: {"L": [str(tmp_path / "a" / "Lsub-01.arg")], "R": [str(tmp_path / "a" / "Rsub-01.arg")]}}
    path = tmp_path / "graphs.json"
    path.write_text(json.dumps(graphs))
    return path, graphs


def test_parse_args_without_path_to_graph_gives_none(gsr):
    """REQ-ARGDISC-BDRABCZUK-8D51E1F01BA7: no --path_to_graph is accepted; path_to_graph is None."""
    params = gsr.parse_args(["-d", "D", "--path_sk_with_hull", "H"])

    assert params["path_to_graph"] is None


def test_graphs_option_passes_the_map_to_each_region_run(gsr, calls, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-CF5B9F427C3C: --graphs <file> -> each region run gets the map read from it."""
    path, graphs = _graphs_map(tmp_path)

    gsr.main(_argv(_dataset(tmp_path), "--graphs", str(path)))

    assert calls
    assert all(cfg["graphs"] == graphs for cfg in calls)


def test_graph_label_option_passes_the_label_to_each_region_run(gsr, calls, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-CBE28068CD49: --graph_label <name> -> each region run gets that graph label."""
    gsr.main(_argv(_dataset(tmp_path), "--graph_label", "deepcnn_auto"))

    assert calls
    assert all(cfg["graph_label"] == "deepcnn_auto" for cfg in calls)


def test_graphs_map_beside_path_to_graph_raises_before_any_region(gsr, calls, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-E465DC9D5831: a graphs map and a path_to_graph -> ValueError, no region run."""
    path, _ = _graphs_map(tmp_path)

    with pytest.raises(ValueError):
        gsr.main(_argv(_dataset(tmp_path, path_to_graph="t1mri/default/folds"), "--graphs", str(path)))

    assert calls == []


def _bids_config(tmp_path, **extra):
    config = {
        "graphs_dir": str(tmp_path / "graphs"),
        "output_dir": str(tmp_path / "out"),
        "path_to_graph": None,
        "skel_qc_path": "",
        "bids": True,
    }
    config.update(extra)
    (tmp_path / "graphs" / SUBJECT).mkdir(parents=True, exist_ok=True)
    return config


def test_bids_subject_with_discovered_graph_in_skip_list_is_dropped(gsr, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-DC68CA61DB73: discovery finds the only graph of the side; its key is skipped."""
    config = _bids_config(tmp_path)
    folds = f"graphs/{SUBJECT}/ses-1/morphologist/folds/3.1"
    touch(tmp_path, f"{folds}/default_session_auto/sub-01_ses-1_hemi-L_default_session_auto.arg")
    touch(tmp_path, f"{folds}/sub-01_ses-1_hemi-L.arg")
    record_skip(config["output_dir"], "L", SKELETONS, "sub-01_ses-1", RuntimeError("failed"))

    assert gsr.valid_subjects(config, "L") == set()


def test_bids_subject_with_listed_graph_in_skip_list_is_dropped(gsr, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-DC68CA61DB73: the graphs map lists the only graph of the side; its key is skipped."""
    listed = touch(tmp_path, "elsewhere/ses-2/Lsub-01_ses-2.arg")
    config = _bids_config(tmp_path, graphs={SUBJECT: {"L": [listed], "R": []}})
    record_skip(config["output_dir"], "L", SKELETONS, "sub-01_ses-2", RuntimeError("failed"))

    assert gsr.valid_subjects(config, "L") == set()


FOLDER_NAMES = st.lists(
    st.text(alphabet="abcABC019-_.", min_size=1, max_size=8).filter(lambda name: name not in (".", "..")),
    max_size=8,
    unique=True,
)


@settings(max_examples=40, deadline=None)
@given(names=FOLDER_NAMES)
def test_graph_subject_folders_are_sorted_property(names):
    """REQ-INV-BDRABCZUK-90B2C4393F3A: for any content of graphs_dir, the subject folders come in sorted order."""
    with pytest.MonkeyPatch.context() as patch, tempfile.TemporaryDirectory() as graphs_dir:
        module = import_generate_sulcal_regions(patch)  # no function-scoped fixture under @given
        for name in names:
            (Path(graphs_dir) / name).mkdir()

        result = module._graph_subject_folders(graphs_dir)

    assert result == sorted(result)
