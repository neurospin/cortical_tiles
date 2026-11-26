"""Graph files read by compute_bounding_box (TASK-ARGDISC-BDRABCZUK-817BCB001B96).

Without a path_to_graph, compute_bounding_box takes the sulcus box of each subject folder from the
.arg file of the side that the cortical_tiles graph resolver returns (REQ-ARGDISC-BDRABCZUK-254DEE5D4735);
with one, it keeps today's glob <subject folder>/<path_to_graph>/<side><subject>*.arg
(REQ-ARGDISC-BDRABCZUK-30431DC046C8). The command line passes no path_to_graph without -p
(REQ-ARGDISC-BDRABCZUK-43383CED0624) and its -p help names graph discovery as the default
(REQ-ARGDISC-BDRABCZUK-A87BCC3E2819); user answer Q10, 2026-11-25. ``get_one_bounding_box`` is replaced,
on the imported module, by a recorder of the graph file it gets, so the .arg files are empty placeholders
in tmp_path trees, never opened (tests/graph_trees.py).
"""

import numpy as np
import pytest

from cortical_tiles.brainvisa import compute_bounding_box as cbb
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
SULCUS = "S.T.s.ter.asc.ant."
UNIT_BOX = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))


@pytest.fixture
def read_graphs(monkeypatch):
    """Records the graph file of each get_one_bounding_box call; returns a fixed unit box."""
    return record_graph_files(monkeypatch, cbb, "get_one_bounding_box", lambda *_arguments: UNIT_BOX)


def _box_of(src, tmp_path, path_to_graph=None):
    box_dir = str(tmp_path / "bbox")
    cbb.compute_bounding_box(
        src_dir=[str(src)],
        bbox_dir=box_dir,
        sulcus=SULCUS,
        side=SIDE,
        out_voxel_size=1,
        number_subjects="all",
        path_to_graph=path_to_graph,
    )


def test_box_reads_discovered_graph_in_classic_layout(read_graphs, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-254DEE5D4735: classic t1mri layout, no path_to_graph: the labelled graph."""
    expected = [classic_session_graph(tmp_path / "src")]
    _box_of(tmp_path / "src", tmp_path)
    assert read_graphs == expected


def test_box_reads_discovered_graph_in_morphologist_cli_layout(read_graphs, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-254DEE5D4735: morphologist-cli layout, no path_to_graph: the labelled graph."""
    expected = [cli_session_graph(tmp_path / "src")]
    _box_of(tmp_path / "src", tmp_path)
    assert read_graphs == expected


def test_box_command_line_defaults_to_no_path_to_graph(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-43383CED0624 (and 254DEE5D4735): the command line without -p gives no path_to_graph."""
    params = cbb.parse_args(["-s", str(tmp_path / "src"), "-o", str(tmp_path / "bbox")])

    assert params["path_to_graph"] is None


def test_box_path_to_graph_help_names_graph_discovery(capsys, monkeypatch):
    """REQ-ARGDISC-BDRABCZUK-A87BCC3E2819: the -p help names graph discovery below each subject folder as default."""
    entry = option_help(cbb, "-p", capsys, monkeypatch)

    assert "Default: graph discovery below each subject folder" in entry


def test_box_with_path_to_graph_keeps_the_subject_named_glob(read_graphs, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-30431DC046C8: path_to_graph given: <side><subject>*.arg, not any <side>*.arg."""
    expected = [subject_named_glob_graph(tmp_path / "src")]
    _box_of(tmp_path / "src", tmp_path, path_to_graph="graphs")
    assert read_graphs == expected


def test_box_graph_label_option_selects_labelling_set(read_graphs, tmp_path):
    """REQ-ARGDISC-BDRABCZUK-AB62417152FB: of two labelling sets, --graph_label names the one giving the sulcus box."""
    graphs = classic_subject(tmp_path / "src", labels=(LABEL, OTHER_LABEL))

    # "-n 1": the default "-n all" fails today in compute (number_subjects converted twice), a separate defect.
    cbb.main(
        [
            *("-s", str(tmp_path / "src"), "-o", str(tmp_path / "bbox"), "-u", SULCUS, "-i", SIDE),
            *("-n", "1", "--graph_label", OTHER_LABEL),
        ]
    )

    assert read_graphs == [graphs[OTHER_LABEL]]
