"""Command line and per-subject loop shared by compute_mask.py and compute_bounding_box.py.

Both scripts loop over the subjects of manually labelled databases, look up each subject's graph file and offer the
same command line; only a few texts differ, given by each script as a ScriptCli.
"""

import argparse
import glob
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from os.path import join

from cortical_tiles.brainvisa.utils.constants import (
    _PATH_TO_GRAPH_SUPERVISED_DEFAULT,
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
    parser.add_argument(
        "-p",
        "--path_to_graph",
        type=str,
        default=_PATH_TO_GRAPH_SUPERVISED_DEFAULT,
        help="Relative path to manually labelled graph. Default is " + _PATH_TO_GRAPH_SUPERVISED_DEFAULT,
    )
    parser.add_argument("-n", "--nb_subjects", type=str, default="all", help=_NB_SUBJECTS_HELP)
    parser.add_argument("-v", "--verbose", action="count", default=0, help=_VERBOSE_HELP)
    parser.add_argument("-x", "--out_voxel_size", type=float, default=_VOXEL_SIZE_DEFAULT, help=cli.voxel_size_help)
    return parser


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
    return result


def iter_subject_graphs(subjects: list, log: logging.Logger) -> Iterator[tuple[dict, str]]:  # noqa: V103 - script loops
    """Yields (subject, graph path) per subject, in list order.

    Each subject is logged on the caller's logger and looked up only when the caller asks for it.

    Raises:
        RuntimeError: a subject folder contains no graph file matching its pattern.
    """
    for sub in subjects:
        log.info(sub)
        # It substitutes 'subject' in graph_file name
        graph_file = sub["graph_file"] % sub
        # It looks for a graph file .arg
        list_graph_file = sorted(glob.glob(join(sub["dir"], graph_file)))
        if len(list_graph_file) == 0:
            raise RuntimeError(f"No graph file! {sub['dir']} doesn't contain {graph_file}")
        yield sub, list_graph_file[0] % sub


class SulcusOverSubjects:  # noqa: V102 - base class of the scripts
    """Source folders, graph patterns, side and sulcus names of a sulcus studied over subjects.

    Attributes:
        src_dir: list of strings naming full path source directories
        graph_file: labelled graph patterns relative to the subject directory, in the order of src_dir
        side: hemisphere side (L for left, or R for right hemisphere)
        sulcus: complete sulcus name
        new_sulcus: complete new sulcus name (the sulcus name when none is given)
    """

    def __init__(self, src_dir, path_to_graph, sulcus, new_sulcus, side):
        # Transforms input source dir to a list of strings
        self.src_dir = [src_dir] if isinstance(src_dir, str) else src_dir

        # we use the '*' glob to take into account different naming conventions
        paths = [path_to_graph] if isinstance(path_to_graph, str) else path_to_graph
        self.graph_file = ["%(subject)s/" + path + "/%(side)s%(subject)s*.arg" for path in paths]

        self.side = side
        self.sulcus = complete_sulci_name(sulcus, side)
        self.new_sulcus = complete_sulci_name(new_sulcus if new_sulcus else sulcus, side)
