"""Graph files read by compute_mask (TASK-ARGDISC-BDRABCZUK-817BCB001B96).

Without a path_to_graph, compute_mask reads for each subject folder the .arg file of the side that
the cortical_tiles graph resolver returns (REQ-ARGDISC-BDRABCZUK-F888CF4C76DE); with one, it keeps
today's glob <subject folder>/<path_to_graph>/<side><subject>*.arg (REQ-ARGDISC-BDRABCZUK-B58AEA5AE7B1).
The command line passes no path_to_graph without -p (REQ-ARGDISC-BDRABCZUK-01BAF5867D6C) and its -p help
names graph discovery as the default (REQ-ARGDISC-BDRABCZUK-EACFCA042A13); user answer Q10, 2026-11-25.
``increment_one_mask`` is replaced, on the imported module, by a recorder of the graph file it gets,
so the .arg files are empty placeholders in tmp_path trees, never opened (tests/graph_trees.py).
"""

import pytest

from cortical_tiles.brainvisa import compute_mask as cm
from tests.graph_trees import (
    LABEL,
    OTHER_LABEL,
    classic_session_graph,
    classic_subject,
    cli_session_graph,
    option_help,
    record_graph_files,
    subject_named_glob_graph,
)

SIDE = "L"
SULCUS = "S.C."
OUT_VS = 2.0


@pytest.fixture
def read_graphs(monkeypatch):
    """Records the graph file of each increment_one_mask call; the mask (second argument) is returned unchanged."""
    return record_graph_files(monkeypatch, cm, "increment_one_mask", lambda _graph, mask, *_rest: mask)


def _mask_of(src, tmp_path, **kwargs):
    cm.compute_mask(
        src_dir=str(src), mask_dir=str(tmp_path / "masks"), sulcus=SULCUS, side=SIDE, out_voxel_size=OUT_VS, **kwargs
    )


def test_mask_reads_discovered_graph_in_classic_layout(read_graphs, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-F888CF4C76DE: classic t1mri layout, no path_to_graph: the labelled graph."""
    graph = classic_session_graph(tmp_path / "src")
    _mask_of(tmp_path / "src", tmp_path)
    assert read_graphs == [graph]


def test_mask_reads_discovered_graph_in_morphologist_cli_layout(read_graphs, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-F888CF4C76DE: morphologist-cli layout, no path_to_graph: the labelled graph."""
    graph = cli_session_graph(tmp_path / "src")
    _mask_of(tmp_path / "src", tmp_path)
    assert read_graphs == [graph]


def test_mask_command_line_defaults_to_no_path_to_graph(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-01BAF5867D6C (and F888CF4C76DE): the command line without -p gives no path_to_graph."""
    assert cm.parse_args(["-s", str(tmp_path / "src"), "-o", str(tmp_path / "masks")])["path_to_graph"] is None


def test_mask_path_to_graph_help_names_graph_discovery(capsys, monkeypatch):
    """REQ-ARGDISC-BDRABCZUK-EACFCA042A13: the -p help names graph discovery below each subject folder as default."""
    assert "Default: graph discovery below each subject folder" in option_help(cm, "-p", capsys, monkeypatch)


def test_mask_with_path_to_graph_keeps_the_subject_named_glob(read_graphs, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-B58AEA5AE7B1: path_to_graph given: <side><subject>*.arg, not any <side>*.arg."""
    graph = subject_named_glob_graph(tmp_path / "src")
    _mask_of(tmp_path / "src", tmp_path, path_to_graph="graphs")
    assert read_graphs == [graph]


def test_mask_graph_label_option_selects_labelling_set(read_graphs, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-2BDF3B397515: of two labelling sets, --graph_label names the one added to the mask."""
    graphs = classic_subject(tmp_path / "src", labels=(LABEL, OTHER_LABEL))
    argv = ["-s", str(tmp_path / "src"), "-o", str(tmp_path / "masks"), "-u", SULCUS, "-i", SIDE]

    cm.main([*argv, "--graph_label", OTHER_LABEL])

    assert read_graphs == [graphs[OTHER_LABEL]]
