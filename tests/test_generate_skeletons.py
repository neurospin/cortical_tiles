"""Tests for cortical_tiles.brainvisa.generate_skeletons.

Graphs read through the cortical_tiles graph resolver (TASK-ARGDISC-BDRABCZUK-7E631CF8D649): trees of
empty .arg files and the checks from tests/graph_trees.py (ConvertingStep), the skeleton writer replaced by a
recorder.

* REQ-ARGDISC-BDRABCZUK-4F5038C42697 - without path_to_graph, the .arg file the resolver returns is converted.
* REQ-ARGDISC-BDRABCZUK-3646CFB645DB - a path_to_graph keeps today's glob <subject>/<path_to_graph>/<side>*.arg.
* REQ-ARGDISC-BDRABCZUK-2F0A81A6D939 - a graph label selects the .arg file of that labelling set.
* REQ-ARGDISC-BDRABCZUK-201BAEC9D7D2 - no graph label, a subject folder holding two labelling sets: the command line
  exits with status 2 listing the sets, before any graph conversion (pre-scan of every subject folder; subject
  folder subj01, with one labelling set, sorts first; user answer 2026-11-25 16:21).
* REQ-ARGDISC-BDRABCZUK-D689C33110D6 - a graphs map gives the .arg files, per subject folder name and side.
"""

from cortical_tiles.brainvisa import generate_skeletons
from tests.graph_trees import ConvertingStep

STEP = ConvertingStep(generate_skeletons, "generate_skeletons", "generate_skeleton_from_graph_file", "skeleton_dir")


def test_generate_skeletons_help():
    args = "--help"
    argv = args.split(" ")
    generate_skeletons.main(argv)


def test_generate_skeletons_n_0(tmp_path):
    """Tests the function when number of subjects is 0"""
    generate_skeletons.generate_skeletons(skeleton_dir=str(tmp_path), nb_subjects=0)


def test_skeletons_discovery_converts_labelled_graph_of_classic_layout(request):
    STEP.converts_labelled_graph_of_classic_layout(request)


def test_skeletons_discovery_converts_labelled_graph_of_morphologist_cli_layout(request):
    STEP.converts_labelled_graph_of_morphologist_cli_layout(request)


def test_skeletons_command_line_without_path_to_graph_runs_discovery(request):
    STEP.command_line_without_path_to_graph_runs_discovery(request)


def test_skeletons_path_to_graph_keeps_todays_glob(request):
    STEP.path_to_graph_keeps_todays_glob(request)


def test_skeletons_graph_label_selects_labelling_set(request):
    STEP.graph_label_selects_labelling_set(request)


def test_skeletons_command_line_exits_2_before_any_conversion(request):
    STEP.command_line_exits_2_before_any_conversion(request)


def test_skeletons_graphs_map_converts_listed_graph(request):
    STEP.graphs_map_converts_listed_graph(request)
