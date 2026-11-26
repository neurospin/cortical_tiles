"""Command line and per-subject loop shared by compute_mask.py and compute_bounding_box.py.

Both scripts loop over the subjects of manually labelled databases, look up each subject's graph file and offer the
same command line; only a few texts differ, given by each script as a ScriptCli.
"""

import argparse
import glob
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from os.path import join

from cortical_tiles.brainvisa.subject_graphs import (
    GraphSpec,
    check_spec_modes,
    prescan_labelling_sets,
    resolve_subject_graphs,
)
from cortical_tiles.brainvisa.utils.constants import (
    _SIDE_DEFAULT,
    _SULCUS_DEFAULT,
    _SUPERVISED_SRC_DIR_DEFAULT,
    _VOXEL_SIZE_DEFAULT,
)
from cortical_tiles.brainvisa.utils.logs import setup_log
from cortical_tiles.brainvisa.utils.subjects import get_number_subjects
from cortical_tiles.brainvisa.utils.sulcus import complete_sulci_name

_SRC_DIR_HELP = (
    "Source directory where the MRI data lies. If there are several directories, add all directories one after the "
    "other. Example: -s DIR_1 DIR_2. Default is : " + _SUPERVISED_SRC_DIR_DEFAULT
)
_NEW_SULCUS_HELP = "Sulcus name around which we determine the bounding box. Default is : None (same name as 'sulcus')"
_NB_SUBJECTS_HELP = (
    "Number of subjects to take into account, or 'all'. 0 subject is allowed, for debug purpose. Default is : all"
)
_SIDE_HELP = "Hemisphere side. Default is : " + _SIDE_DEFAULT
_VERBOSE_HELP = (
    "Verbose mode: If no option is provided then logging.INFO is selected. If one option -v (or -vv) or more is "
    "provided then logging.DEBUG is selected."
)
_PATH_TO_GRAPH_HELP = (
    "Relative path of the labelled graph folder below each subject folder (legacy glob "
    "<subject>/<path_to_graph>/<side><subject>*.arg). Default: graph discovery below each subject folder."
)
_STEP_PATH_TO_GRAPH_HELP = (
    "Relative path of the graph folder below each subject folder: the .arg files are taken from the glob "
    "<subject folder>/<path_to_graph>/<side>*.arg. Default: graph discovery below each subject folder."
)
_GRAPH_LABEL_HELP = (
    "Labelling set to read during graph discovery: the name of the folder holding the labelled graphs (for example "
    "default_session_auto, sul-0_auto). Needed when a subject holds several labelling sets."
)


@dataclass(frozen=True)
class ScriptCli:
    """Texts that differ between the scripts sharing this command line.

    Attributes:
        prog: program name (basename of the script file)
        description: parser description
        output_key: key of the -o value in the parameter dictionary
        output_default: default of -o
        output_help: help text of -o
        sulcus_help: help text of -u
        voxel_size_help: help text of -x
    """

    prog: str
    description: str
    output_key: str
    output_default: str
    output_help: str
    sulcus_help: str
    voxel_size_help: str


def build_parser(cli: ScriptCli) -> argparse.ArgumentParser:
    """Builds the command-line parser of a subject_cli script, options in their historical order."""
    parser = argparse.ArgumentParser(prog=cli.prog, description=cli.description)
    parser.add_argument("-s", "--src_dir", type=str, default=_SUPERVISED_SRC_DIR_DEFAULT, nargs="+", help=_SRC_DIR_HELP)
    parser.add_argument("-o", "--output_dir", type=str, default=cli.output_default, help=cli.output_help)
    parser.add_argument("-u", "--sulcus", type=str, default=_SULCUS_DEFAULT, help=cli.sulcus_help)
    parser.add_argument("-w", "--new_sulcus", type=str, default=None, help=_NEW_SULCUS_HELP)
    parser.add_argument("-i", "--side", type=str, default=_SIDE_DEFAULT, help=_SIDE_HELP)
    parser.add_argument("-p", "--path_to_graph", type=str, default=None, help=_PATH_TO_GRAPH_HELP)
    parser.add_argument("-n", "--nb_subjects", type=str, default="all", help=_NB_SUBJECTS_HELP)
    parser.add_argument("-v", "--verbose", action="count", default=0, help=_VERBOSE_HELP)
    parser.add_argument("-x", "--out_voxel_size", type=float, default=_VOXEL_SIZE_DEFAULT, help=cli.voxel_size_help)
    # SUPPRESS keeps graph_label out of the namespace when absent: the logged command line stays the historical one
    add_graph_label_option(parser, default=argparse.SUPPRESS)
    return parser


def add_graph_label_option(parser: argparse.ArgumentParser, default: object = None) -> None:
    """Adds the one definition of the --graph_label option (alias --graph-label) to parser."""
    parser.add_argument(
        "--graph_label",
        "--graph-label",
        dest="graph_label",
        type=str,
        default=default,
        help=_GRAPH_LABEL_HELP,
    )


def add_graph_options(parser: argparse.ArgumentParser) -> None:  # noqa: V103 - graph steps
    """Adds -p/--path_to_graph (default: graph discovery) and --graph_label to a graph step's parser."""
    parser.add_argument("-p", "--path_to_graph", type=str, default=None, help=_STEP_PATH_TO_GRAPH_HELP)
    add_graph_label_option(parser)


def graph_params(args: argparse.Namespace) -> dict:  # noqa: V103 - graph steps
    """Returns the graph-input keywords of a graph step's API function, valued from the command line."""
    result = {"graph_label": args.graph_label, "graphs": None}  # no graphs map on the command line
    assert "graph_label" in result, 'SubjectCli.graph_params post: "graph_label" in result'
    assert "graphs" in result, 'SubjectCli.graph_params post: "graphs" in result'
    assert len(result) == 2, "SubjectCli.graph_params post: len(result) == 2"
    return result


class GraphStepInputs:  # noqa: V102 - base class of the graph steps
    """Graph inputs of a graph step: graph discovery, a labelling set, a graphs map or the legacy path_to_graph glob.

    The host class sets src_dir, side and bids; these methods read no other host attribute.
    """

    def set_graph_inputs(self, path_to_graph, graph_label=None, graphs=None) -> None:  # noqa: V105 - graph steps
        """Sets the graph-input attributes and the graph specification (no I/O).

        Raises:
            ValueError: a graphs map is given beside a path_to_graph.
        """
        check_spec_modes(graphs, path_to_graph)
        self.path_to_graph = path_to_graph
        self.graph_label = graph_label
        self.graphs = graphs
        self.graph_spec = GraphSpec(graphs=graphs, path_to_graph=path_to_graph, label=graph_label)

    def prescan_listed(self, subjects: list) -> None:  # noqa: V105 - graph steps
        """Raises SeveralLabellingSetsError naming every listed subject folder with several labelling sets."""
        prescan_labelling_sets([join(self.src_dir, subject) for subject in subjects], self.graph_spec)

    def graph_files(self, folders: list, shown: str) -> tuple:  # noqa: V105 - graph steps
        """Returns (the .arg files of the side in folders, the expected-path text of the no-graph message)."""
        files = [g for folder in folders for g in resolve_subject_graphs(folder, self.graph_spec)[self.side]]
        if self.path_to_graph is not None:
            expected = f"{shown}/{self.path_to_graph}/{self.side}*.arg"
        else:
            mode = "graphs map" if self.graphs is not None else "graph discovery"
            label = f", label {self.graph_label}" if self.graph_label is not None else ""
            expected = f"{shown} ({mode}, side {self.side}{label})"
        result = (files, expected)
        assert len(result) == 2, "SubjectCli.graph_files post: len(result) == 2"
        return result

    def counts_listed_graphs(self) -> bool:  # noqa: V105 - graph steps
        """Tells whether the end-of-step count compares the output with the listed graphs (BIDS, path_to_graph)."""
        return self.bids and self.path_to_graph is not None


def parse_subject_args(argv: list, cli: ScriptCli) -> dict:  # noqa: V103 - called by the scripts
    """Parses argv, sets up the log files and returns the parameter dictionary.

    Raises:
        ValueError: -n is neither 'all' nor a non-negative integer (raised after the log files are set up).
    """
    args = build_parser(cli).parse_args(argv)

    # Sets logger level, files log handler and prints/logs command line
    new_sulcus = args.new_sulcus if args.new_sulcus else args.sulcus
    setup_log(args, log_dir=f"{args.output_dir}", prog_name=cli.prog, suffix=complete_sulci_name(new_sulcus, args.side))

    result = {
        "src_dir": args.src_dir,  # src_dir is a list
        "path_to_graph": args.path_to_graph,
        "graph_label": getattr(args, "graph_label", None),
        cli.output_key: args.output_dir,  # only one directory
        "sulcus": args.sulcus,
        "new_sulcus": args.new_sulcus,
        "side": args.side,
        "out_voxel_size": args.out_voxel_size,
        # Checks if nb_subjects is either the string "all" or a positive integer
        "nb_subjects": get_number_subjects(args.nb_subjects),
    }
    assert "src_dir" in result, 'SubjectCli.parse_subject_args post: "src_dir" in result'
    assert "side" in result, 'SubjectCli.parse_subject_args post: "side" in result'
    assert "nb_subjects" in result, 'SubjectCli.parse_subject_args post: "nb_subjects" in result'
    assert "graph_label" in result, 'SubjectCli.parse_subject_args post: "graph_label" in result'
    return result


def graph_inputs(
    path_to_graph: str | Sequence[str] | None, graph_label: str | None, nb_src_dir: int
) -> tuple[GraphSpec | None, list[str]]:
    """Returns (graph specification, graph file templates) of a subject_cli run.

    No path_to_graph selects graph discovery: a GraphSpec carrying graph_label and one subject-folder template per
    source directory. A path_to_graph gives no GraphSpec and the legacy glob templates (graph_label ignored).
    """
    if path_to_graph is None:
        result = (GraphSpec(label=graph_label), ["%(subject)s"] * nb_src_dir)
    else:
        # we use the '*' glob to take into account different naming conventions
        paths = [path_to_graph] if isinstance(path_to_graph, str) else path_to_graph
        result = (None, ["%(subject)s/" + path + "/%(side)s%(subject)s*.arg" for path in paths])
    assert len(result) == 2, "SubjectCli.graph_inputs post: len(result) == 2"
    return result


def prescan_subjects(subjects: Iterable[Mapping[str, str]], graph_spec: GraphSpec | None) -> None:  # noqa: V103
    """Raises SeveralLabellingSetsError before any graph read when a listed subject holds several labelling sets.

    A no-op without a graph specification (legacy glob). Every listed subject is walked, in list order.
    """
    if graph_spec is None:
        return
    prescan_labelling_sets([join(sub["dir"], sub["subject"]) for sub in subjects], graph_spec)


def iter_subject_graphs(  # noqa: V103 - script loops
    subjects: list, log: logging.Logger, graph_spec: GraphSpec | None = None
) -> Iterator[tuple[dict, str]]:
    """Yields (subject, graph path) per subject, in list order.

    Each subject is logged on the caller's logger and looked up only when the caller asks for it. Without a graph
    specification the subject's graph_file pattern is globbed (legacy); with one, the graph resolver chooses the file.

    Raises:
        RuntimeError: a subject folder contains no graph file matching its pattern, or no graph of its side.
        SeveralLabellingSetsError: graph discovery without a label meets several labelling sets.
    """
    for sub in subjects:
        log.info(sub)
        if graph_spec is None:
            # It substitutes 'subject' in graph_file name
            graph_file = sub["graph_file"] % sub
            # It looks for a graph file .arg
            list_graph_file = sorted(glob.glob(join(sub["dir"], graph_file)))
            if len(list_graph_file) == 0:
                raise RuntimeError(f"No graph file! {sub['dir']} doesn't contain {graph_file}")
            yield sub, list_graph_file[0] % sub
            continue
        found = resolve_subject_graphs(join(sub["dir"], sub["subject"]), graph_spec)[sub["side"]]
        if not found:
            label = f", label {graph_spec.label}" if graph_spec.label is not None else ""
            raise RuntimeError(
                f"No graph file! {sub['dir']}/{sub['subject']} holds no graph for side {sub['side']} "
                f"(graph discovery{label})"
            )
        yield sub, found[0]


class SulcusOverSubjects:  # noqa: V102 - base class of the scripts
    """Source folders, graph patterns, side and sulcus names of a sulcus studied over subjects.

    Attributes:
        src_dir: list of strings naming full path source directories
        graph_spec: graph specification of graph discovery, or None for the legacy path_to_graph glob
        graph_file: labelled graph patterns relative to the subject directory, in the order of src_dir
        side: hemisphere side (L for left, or R for right hemisphere)
        sulcus: complete sulcus name
        new_sulcus: complete new sulcus name (the sulcus name when none is given)
    """

    def __init__(self, src_dir, path_to_graph, sulcus, new_sulcus, side, graph_label=None):
        # Transforms input source dir to a list of strings
        self.src_dir = [src_dir] if isinstance(src_dir, str) else src_dir
        self.graph_spec, self.graph_file = graph_inputs(path_to_graph, graph_label, len(self.src_dir))

        self.side = side
        self.sulcus = complete_sulci_name(sulcus, side)
        self.new_sulcus = complete_sulci_name(new_sulcus if new_sulcus else sulcus, side)
