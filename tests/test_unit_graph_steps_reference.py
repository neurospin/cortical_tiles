"""Differential tests of the three graph steps against their reference (TASK-ARGDISC-BDRABCZUK-EFCEA137B405).

generate_foldlabels, generate_skeletons and generate_ICBM2009c_transforms move onto one base class and one command
line; their behaviour must stay the one of cortical_tiles commit REFERENCE. Each test loads the step module and
subject_cli.py as they are at REFERENCE (``git show``), runs the reference module and the current module on two
identical synthetic trees (tests/graph_trees.py) and compares what each run did: the return value, exit status or
raised exception, the calls into the per-subject converter, the files written below the tree, the INFO-and-higher log
records and the standard output and error texts. The converter (aims for the transforms) is a recorder, so no .arg
file is opened. Before the comparison the temporary root becomes <ROOT> and traceback lines (and the other indented
lines, whose text the log records already carry) leave the written files. These tests pass at REFERENCE: they are the
guard of the refactor (new(x) == reference(x), CONTRACT 7.9a).

Not run: parallel mode (-a), because the reference module, loaded from text, does not pickle into worker processes.
"""

import inspect
import logging
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from functools import partial

import pytest

import cortical_tiles.brainvisa
from cortical_tiles.brainvisa import generate_foldlabels, generate_ICBM2009c_transforms, generate_skeletons
from cortical_tiles.brainvisa.utils.skip_list import record_skip
from tests.graph_trees import (
    BRAINVISA_DIR,
    CLASSIC_FOLDS,
    LABEL,
    OTHER_LABEL,
    REPO_DIR,
    SIDE,
    SUBJECT,
    classic_subject,
    cli_subject,
    touch,
)

REFERENCE = "6e01c5bdf400281ccc69bae676554b4b31a517a4"
ROOT_MARK = "<ROOT>"
BAD_SUBJECT = "subjbad"  # the recorder raises for the graph of this subject
OTHER_SUBJECT = "subj02"
_LOADED = {}  # reference modules by stem, loaded once per process


# --- the reference -----------------------------------------------------------------------------------------------


def _git_show(stem):
    """The text of cortical_tiles/brainvisa/<stem>.py at REFERENCE."""
    done = subprocess.run(
        ["git", "-C", REPO_DIR, "show", f"{REFERENCE}:cortical_tiles/brainvisa/{stem}.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        pytest.fail(f"cortical_tiles commit {REFERENCE} is not readable: {done.stderr.strip()}")
    return done.stdout


def _execute(stem):
    """Module <stem> executed from its REFERENCE text; __file__ is the current file, so prog and logger names match."""
    module = types.ModuleType(f"tests._reference_{stem}")
    module.__file__ = os.path.join(BRAINVISA_DIR, f"{stem}.py")  # noqa: V101 - read by the executed module
    sys.modules[module.__name__] = module  # dataclasses look their module up
    code = compile(_git_show(stem), f"{REFERENCE[:7]}:{stem}.py", "exec")
    exec(code, module.__dict__)  # noqa: S102 - the reference is this repository's own commit
    return module


def reference(stem):
    """The step module <stem> at REFERENCE, importing subject_cli at REFERENCE too."""
    if "subject_cli" not in _LOADED:
        _LOADED["subject_cli"] = _execute("subject_cli")
    if stem not in _LOADED:
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(sys.modules, "cortical_tiles.brainvisa.subject_cli", _LOADED["subject_cli"])
            patch.setattr(cortical_tiles.brainvisa, "subject_cli", _LOADED["subject_cli"], raising=False)
            _LOADED[stem] = _execute(stem)
    return _LOADED[stem]


# --- the steps and their recorders -------------------------------------------------------------------------------


def _converter(calls):
    """Recorder of writer(graph_file, output_file, junction): writes an empty output, raises for BAD_SUBJECT."""

    def convert(graph_file, output_file, *rest):
        calls.append(("convert", graph_file, output_file, *rest))
        if BAD_SUBJECT in graph_file:
            raise RuntimeError(f"cannot convert {os.path.basename(graph_file)}")
        open(output_file, "w").close()

    return convert


def _fake_aims(calls):
    """Recorder of the aims calls of the transforms step: read raises for BAD_SUBJECT, write an empty file."""

    def read(path):
        calls.append(("read", path))
        if BAD_SUBJECT in path:
            raise RuntimeError(f"cannot read {os.path.basename(path)}")
        return path

    def write(transform, path):
        calls.append(("write", transform, path))
        open(path, "w").close()

    manip = types.SimpleNamespace(getICBM2009cTemplateTransform=lambda graph: f"transform of {graph}")
    return types.SimpleNamespace(read=read, write=write, GraphManip=manip)


@dataclass(frozen=True)
class Step:
    """A graph step: its current module, API function, output keyword, skip-list step and converter name."""

    module: types.ModuleType
    function: str  # name of the API function, also the module stem and the skip-list step
    output_keyword: str
    writer: str | None  # name of the converter in the module; None: the module's aims

    def patch_converter(self, module, patch, calls):
        """Replaces the per-subject converter of module by a recorder appending to calls."""
        if self.writer is None:
            patch.setattr(module, "aims", _fake_aims(calls))
        else:
            patch.setattr(module, self.writer, _converter(calls))


FOLDLABELS = Step(generate_foldlabels, "generate_foldlabels", "foldlabel_dir", "generate_foldlabel_from_graph_file")
SKELETONS = Step(generate_skeletons, "generate_skeletons", "skeleton_dir", "generate_skeleton_from_graph_file")
TRANSFORMS = Step(generate_ICBM2009c_transforms, "generate_ICBM2009c_transforms", "transform_dir", None)


# --- running and comparing ---------------------------------------------------------------------------------------


def _normalise(value, root):
    """value with the text of root replaced by ROOT_MARK in every string it holds."""
    if isinstance(value, str):
        return value.replace(str(root), ROOT_MARK)
    if isinstance(value, (list, tuple)):
        return type(value)(_normalise(item, root) for item in value)
    if isinstance(value, dict):
        return {key: _normalise(item, root) for key, item in value.items()}
    return value


def _written_files(root):
    """{path below root: lines} of every file below root, traceback and indented lines left out."""
    files = {}
    for folder, _folders, names in sorted(os.walk(root)):
        for name in sorted(names):
            path = os.path.join(folder, name)
            with open(path, errors="replace") as handle:
                lines = [line for line in handle.read().splitlines() if not line.startswith(("Traceback", " "))]
            files[os.path.relpath(path, root)] = _normalise(lines, root)
    return files


def _outcome(call, root, request):
    """What call() did: how it ended, INFO-and-higher log records, standard output and error, written files."""
    caplog = request.getfixturevalue("caplog")
    capsys = request.getfixturevalue("capsys")
    caplog.clear()
    capsys.readouterr()
    with caplog.at_level(logging.DEBUG):
        try:
            ended = ("returned", call())
        except SystemExit as stop:
            ended = ("SystemExit", stop.code)
        except Exception as error:  # noqa: BLE001 - the type and text of any raised exception are compared
            ended = (type(error).__name__, str(error))
    captured = capsys.readouterr()
    records = [(rec.name, rec.levelname, rec.getMessage()) for rec in caplog.records if rec.levelno >= logging.INFO]
    result = {"ended": ended, "records": records, "stdout": captured.out, "stderr": captured.err}
    return {**_normalise(result, root), "files": _written_files(root)}


def assert_same_as_reference(request, step, build, call):
    """Runs call(module, root) for the REFERENCE and the current module, each on its own tree made by build(root,
    step), and asserts both runs did the same."""
    old = reference(step.function)
    assert old is not step.module and old.__name__.startswith("tests._reference_")
    tmp_path = request.getfixturevalue("tmp_path")
    outcomes = {}
    for name, module in (("reference", old), ("current", step.module)):
        root = tmp_path / name
        root.mkdir()
        build(root, step)
        calls = []
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("COLUMNS", "120")  # argparse wraps the help text at the terminal width
            step.patch_converter(module, patch, calls)
            outcomes[name] = _outcome(partial(call, module, root), root, request)
        outcomes[name]["converter_calls"] = _normalise(calls, root)
    assert outcomes["current"] == outcomes["reference"]


# --- synthetic trees (src = <root>/graphs, output = <root>/out) --------------------------------------------------


def _nothing(root, step):
    """No subject at all."""
    (root / "graphs").mkdir()


def _one(root, step):
    """One classic subject with one labelling set."""
    classic_subject(root / "graphs")


def _one_right(root, step):
    """One classic subject with its right hemisphere graphs."""
    classic_subject(root / "graphs", side="R")


def _one_two_sets(root, step):
    """One classic subject with two labelling sets."""
    classic_subject(root / "graphs", labels=(LABEL, OTHER_LABEL))


def _several_sets(root, step):
    """subj01 with one labelling set, subj02 with two."""
    classic_subject(root / "graphs")
    classic_subject(root / "graphs", subject=OTHER_SUBJECT, labels=(LABEL, OTHER_LABEL))


def _two(root, step):
    """Two classic subjects, one labelling set each."""
    classic_subject(root / "graphs")
    classic_subject(root / "graphs", subject=OTHER_SUBJECT)


def _failures(root, step):
    """subj01 converts, subjbad's converter raises, subj02 holds no graph."""
    classic_subject(root / "graphs")
    classic_subject(root / "graphs", subject=BAD_SUBJECT)
    touch(root / "graphs", f"{OTHER_SUBJECT}/t1mri/notes.txt")


def _cli(root, step):
    """One morphologist-cli subject folder."""
    cli_subject(root / "graphs")


def _skipped_before(root, step):
    """Two subjects, subj02 in the side's skip list of the output folder before the run."""
    _two(root, step)
    record_skip(str(root / "out"), SIDE, step.function, OTHER_SUBJECT, RuntimeError("earlier failure"))


def _skipped_elsewhere(root, step):
    """Two subjects, subj02 in the side's skip list of <root>/skips before the run."""
    _two(root, step)
    record_skip(str(root / "skips"), SIDE, step.function, OTHER_SUBJECT, RuntimeError("earlier failure"))


def _with_qc(root, step):
    """Two subjects and <root>/qc.tsv keeping subj01 only."""
    _two(root, step)
    (root / "qc.tsv").write_text(f"participant_id\tqc\n{SUBJECT}\t1\n{OTHER_SUBJECT}\t0\n")


# --- scenarios ---------------------------------------------------------------------------------------------------

# (id, tree, extra command-line arguments; {root} is the tree root)
COMMAND_LINES = [
    ("discovery", _one, []),
    ("no_subject", _nothing, []),
    ("path_to_graph", _one, ["-p", CLASSIC_FOLDS]),
    ("graph_label", _one_two_sets, ["--graph_label", OTHER_LABEL]),
    ("graph_label_dash", _one_two_sets, ["--graph-label", LABEL]),
    ("several_labelling_sets", _several_sets, []),
    ("failures", _failures, []),
    ("bids", _cli, ["-b"]),
    ("bids_path_to_graph", _one, ["-b", "-p", CLASSIC_FOLDS]),
    ("nb_subjects", _two, ["-n", "1"]),
    ("nb_subjects_zero", _two, ["-n", "0"]),
    ("bad_nb_subjects", _one, ["-n", "abc"]),
    ("unknown_option", _one, ["--bogus"]),
    ("qc_file", _with_qc, ["-q", "{root}/qc.tsv"]),
    ("verbose", _one, ["-v"]),
    ("right_side", _one_right, ["-i", "R"]),
    ("skip_list", _skipped_before, []),
    ("skip_dir", _skipped_elsewhere, ["--skip_dir", "{root}/skips"]),
    ("junction", _one, ["-j", "wide"]),
]

# (id, extra parse_args arguments; {root} is the tree root)
ARGUMENT_LISTS = [
    ("defaults", []),
    ("path_to_graph", ["-p", CLASSIC_FOLDS]),
    ("graph_label", ["--graph_label", LABEL]),
    ("flags", ["-b", "-a", "-n", "3", "-v", "-v"]),
    ("qc_path", ["-q", "{root}/qc.tsv"]),
    ("junction", ["-j", "wide"]),
    ("skip_dir", ["--skip_dir", "{root}/skips"]),
    ("right_side", ["-i", "R"]),
    ("bad_nb_subjects", ["-n", "-2"]),
    ("unknown_option", ["--bogus"]),
]

# (id, tree, extra keyword arguments of the API function given the tree root)
API_RUNS = [
    ("defaults", _one, lambda root: {}),
    ("path_to_graph", _one, lambda root: {"path_to_graph": CLASSIC_FOLDS}),
    ("graph_label", _one_two_sets, lambda root: {"graph_label": OTHER_LABEL}),
    (
        "graphs_map",
        _one,
        lambda root: {"graphs": {SUBJECT: {SIDE: [touch(root, f"elsewhere/{SIDE}{SUBJECT}_listed.arg")]}}},
    ),
    ("graphs_map_and_path", _one, lambda root: {"graphs": {}, "path_to_graph": CLASSIC_FOLDS}),
    ("several_labelling_sets", _several_sets, lambda root: {}),
    ("failures", _failures, lambda root: {}),
    ("bids", _cli, lambda root: {"bids": True}),
    ("nb_subjects", _two, lambda root: {"nb_subjects": 1}),
    ("qc_file", _with_qc, lambda root: {"qc_path": str(root / "qc.tsv")}),
    ("skip_list", _skipped_before, lambda root: {}),
    ("skip_dir", _skipped_elsewhere, lambda root: {"skip_dir": str(root / "skips")}),
    ("junction", _one, lambda root: {"junction": "wide"}),
    ("njobs", _one, lambda root: {"njobs": 2}),
]


def _argv(root, extra):
    """-s <root>/graphs -o <root>/out -i SIDE, then extra with {root} filled in."""
    return ["-s", str(root / "graphs"), "-o", str(root / "out"), "-i", SIDE] + [
        item.format(root=root) for item in extra
    ]


def _check_command_line(request, step, scenario):
    """main(argv) of the step does what main does at REFERENCE."""
    _id, build, extra = scenario
    assert_same_as_reference(request, step, build, lambda module, root: module.main(_argv(root, extra)))


def _check_parse_args(request, step, arguments):
    """parse_args(argv) of the step returns (or raises) what it does at REFERENCE."""
    _id, extra = arguments
    assert_same_as_reference(request, step, _nothing, lambda module, root: module.parse_args(_argv(root, extra)))


def _check_help(request, step):
    """main(["--help"]) prints the help text it prints at REFERENCE and returns."""
    assert_same_as_reference(request, step, _nothing, lambda module, root: module.main(["--help"]))


def _check_api_run(request, step, run):
    """The API function writes, logs and raises what it does at REFERENCE for one keyword argument set."""
    _id, build, keywords = run

    def call(module, root):
        arguments = {
            "src_dir": str(root / "graphs"),
            step.output_keyword: str(root / "out"),
            "side": SIDE,
            "qc_path": "",
            **keywords(root),
        }
        return getattr(module, step.function)(**arguments)

    assert_same_as_reference(request, step, build, call)


def _check_api_signature(step):
    """The API function has the parameter names, kinds and defaults it has at REFERENCE."""
    old = inspect.signature(getattr(reference(step.function), step.function))
    assert inspect.signature(getattr(step.module, step.function)) == old


def _ids(items):
    return [item[0] for item in items]


# --- the three steps: test_<step>_<what>_matches_reference, the same five tests for each ---------------------------


def _define_tests(name, step):
    """Put the five differential tests of one step into this module under their test_<name>_... names."""

    def help_matches(request):
        _check_help(request, step)

    def parse_args_matches(request, arguments):
        _check_parse_args(request, step, arguments)

    def command_line_matches(request, scenario):
        _check_command_line(request, step, scenario)

    def api_signature_matches():
        _check_api_signature(step)

    def api_run_matches(request, run):
        _check_api_run(request, step, run)

    tests = {
        "help": help_matches,
        "parse_args": pytest.mark.parametrize("arguments", ARGUMENT_LISTS, ids=_ids(ARGUMENT_LISTS))(
            parse_args_matches
        ),
        "command_line": pytest.mark.parametrize("scenario", COMMAND_LINES, ids=_ids(COMMAND_LINES))(
            command_line_matches
        ),
        "api_signature": api_signature_matches,
        "api_run": pytest.mark.parametrize("run", API_RUNS, ids=_ids(API_RUNS))(api_run_matches),
    }
    for what, function in tests.items():
        function.__name__ = f"test_{name}_{what}_matches_reference"
        globals()[function.__name__] = function


for _name, _step in (("foldlabels", FOLDLABELS), ("skeletons", SKELETONS), ("transforms", TRANSFORMS)):
    _define_tests(_name, _step)
