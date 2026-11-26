"""Shared command line and base class of the graph steps.

The graph steps are generate_foldlabels, generate_skeletons and generate_ICBM2009c_transforms: each converts the graph
files of every subject of a source folder into one output file per graph.
"""

import argparse
import glob
from dataclasses import dataclass
from os.path import abspath, basename, exists

from cortical_tiles.brainvisa import subject_cli
from cortical_tiles.brainvisa.utils import skip_list
from cortical_tiles.brainvisa.utils.constants import (
    _JUNCTION_DEFAULT,
    _QC_PATH_DEFAULT,
    _SIDE_DEFAULT,
    _SRC_DIR_DEFAULT,
)
from cortical_tiles.brainvisa.utils.folder import create_folder
from cortical_tiles.brainvisa.utils.logs import setup_log
from cortical_tiles.brainvisa.utils.skip_list import session_key
from cortical_tiles.brainvisa.utils.subjects import get_number_subjects, is_it_a_subject

SRC_DIR_HELP = "Source directory where the graph data lies. Default is : " + _SRC_DIR_DEFAULT
SIDE_HELP = "Hemisphere side. Default is : " + _SIDE_DEFAULT
BIDS_HELP = "if the database uses the BIDS format"


@dataclass(frozen=True)  # noqa: V102 - used by the graph step modules
class StepCli:
    prog: str
    description: str
    output_key: str
    output_default: str
    output_help: str
    src_help: str = SRC_DIR_HELP
    side_help: str = SIDE_HELP
    bids_help: str = BIDS_HELP
    junction: bool = True
    skip_dir: bool = False


_QC = (
    ("-q", "--qc_path"),
    {"type": str, "default": _QC_PATH_DEFAULT, "help": "Path to quality check .csv. Default is " + _QC_PATH_DEFAULT},
)
_JUNCTION = (
    ("-j", "--junction"),
    {
        "type": str,
        "default": _JUNCTION_DEFAULT,
        "help": f"junction rendering (either 'wide' or 'thin') Default is {_JUNCTION_DEFAULT}",
    },
)
_PARALLEL = (
    ("-a", "--parallel"),
    {"default": False, "action": "store_true", "help": "if set (-a), launches computation in parallel"},
)
_NB_SUBJECTS = (
    ("-n", "--nb_subjects"),
    {
        "type": str,
        "default": "all",
        "help": "Number of subjects to take into account, or 'all'. "
        "0 subject is allowed, for debug purpose."
        "Default is : all",
    },
)
_SKIP_DIR = (
    ("--skip_dir",),
    {
        "type": str,
        "default": None,
        "help": "Directory of the per-side skip lists skip_list_<side>.jsonl. Default: the output directory.",
    },
)
_VERBOSE = (
    ("-v", "--verbose"),
    {
        "action": "count",
        "default": 0,
        "help": "Verbose mode: "
        "If no option is provided then logging.INFO is selected. "
        "If one option -v (or -vv) or more is provided "
        "then logging.DEBUG is selected.",
    },
)


def _options_after_graph(cli):
    bids = (("-b", "--bids"), {"default": False, "action": "store_true", "help": cli.bids_help})
    return [
        _QC,
        *([_JUNCTION] if cli.junction else []),
        bids,
        _PARALLEL,
        _NB_SUBJECTS,
        *([_SKIP_DIR] if cli.skip_dir else []),
        _VERBOSE,
    ]


def build_parser(cli):
    parser = argparse.ArgumentParser(prog=cli.prog, description=cli.description)
    parser.add_argument("-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT, help=cli.src_help)
    parser.add_argument("-o", "--output_dir", type=str, default=cli.output_default, help=cli.output_help)
    parser.add_argument("-i", "--side", type=str, default=_SIDE_DEFAULT, help=cli.side_help)
    subject_cli.add_graph_options(parser)
    for flags, options in _options_after_graph(cli):
        parser.add_argument(*flags, **options)
    return parser


def parse_step_args(argv, cli):  # noqa: V103 - used by the graph step modules
    args = build_parser(cli).parse_args(argv)
    setup_log(args, log_dir=f"{args.output_dir}", prog_name=cli.prog, suffix="right" if args.side == "R" else "left")
    params = {key: value for key, value in vars(args).items() if key not in ("output_dir", "verbose")}
    params["src_dir"] = abspath(args.src_dir)
    params[cli.output_key] = abspath(args.output_dir)
    params.update(subject_cli.graph_params(args))
    params["nb_subjects"] = get_number_subjects(args.nb_subjects)
    return params


class GraphStep(subject_cli.GraphStepInputs):  # noqa: V102 - used by the graph step modules
    PARALLEL_MESSAGE = "PARALLEL MODE: subjects are computed in parallel."
    SERIAL_MESSAGE = "SERIAL MODE: subjects are scanned serially, without parallelism"

    def __init__(
        self, src_dir, output_dir, side, parallel, path_to_graph, bids, qc_path, skip_dir, graph_label, graphs
    ):
        self.src_dir = src_dir
        self.skip_dir = output_dir if skip_dir is None else skip_dir
        self.side = side
        self.qc_path = qc_path
        self.parallel = parallel
        self.set_graph_inputs(path_to_graph, graph_label, graphs)
        self.bids = bids
        self.side_dir = f"{output_dir}/{self.side}"
        self.skipped_keys = frozenset()
        create_folder(abspath(self.side_dir))

    @classmethod  # noqa: V105 - used by the graph step modules
    def run(cls, arguments):
        arguments = dict(arguments)
        nb_subjects = arguments.pop("nb_subjects")
        cls(**arguments).compute(nb_subjects=nb_subjects)

    def listed_subjects(self):  # noqa: V105 - used by the graph step modules
        filenames = sorted(glob.glob(f"{self.src_dir}/*"))
        return [basename(filename) for filename in filenames if is_it_a_subject(filename)]

    def record_missing(self, subject, message):  # noqa: V105 - used by the graph step modules
        exc = FileNotFoundError(message)
        skip_list.record_skip(self.skip_dir, self.side, self.STEP, subject, exc)
        return str(exc)

    def convert_graphs(self, subject, list_graph_file):  # noqa: V105 - used by the graph step modules
        first_failure = None
        for graph_file in list_graph_file:
            key = session_key(subject, graph_file, self.bids)
            if self.bids and key in self.skipped_keys:
                continue
            try:
                output_file = self.get_output_filename(subject, graph_file)
                if not exists(output_file):
                    self.convert(graph_file, output_file)
            except Exception as exc:  # noqa: BLE001 - any failure of one graph is recorded and the loop goes on
                skip_list.record_skip(self.skip_dir, self.side, self.STEP, key, exc)
                first_failure = str(exc) if first_failure is None else first_failure
            if not self.bids:
                break
        return first_failure

    def run_subjects(self, list_subjects):  # noqa: V105 - used by the graph step modules
        one_subject = getattr(self, self.ONE_SUBJECT)
        if self.parallel:
            self.LOG.info(self.PARALLEL_MESSAGE)
            results = self.map_parallel(one_subject, list_subjects)
        else:
            self.LOG.info(self.SERIAL_MESSAGE)
            results = [one_subject(subject) for subject in list_subjects]
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results

    def log_summary(self, title, results, list_subjects):  # noqa: V105 - used by the graph step modules
        failures = {sub: reason for sub, reason in results if reason is not None}
        n_ok = len(list_subjects) - len(failures)
        lines = [f"--- {title} summary (side={self.side}) ---", f"  Succeeded: {n_ok}/{len(list_subjects)}"]
        if failures:
            lines.append(f"  Skipped ({len(failures)}):")
            for sub, reason in sorted(failures.items()):
                lines.append(f"    {sub}: {reason}")
        self.LOG.info("\n".join(lines))
