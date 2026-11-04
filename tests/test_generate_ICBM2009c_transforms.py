"""Tests for cortical_tiles.brainvisa.generate_ICBM2009c_transforms (TASK-CTFAIL-BDRABCZUK-973A1019B1AF).

A subject whose ICBM2009c transform generation fails is reported through ``record_skip`` (the real
exception object, so its type and traceback reach the ERROR record and the skip list), subjects
already in the side's skip list are left out, and the end-of-step summary goes through the module
logger instead of standard output.

The skip-list directory of the step is its explicit ``skip_dir`` argument (``run_with_params`` passes
the one shared skip-list directory of the run, the cortical_tiles output root, to every step). The
tests put ``skip_dir`` apart from the parent of ``transform_dir`` so the step cannot guess it.
Called without ``skip_dir`` (a standalone run), the step falls back to its own ``transform_dir``.

The graph files are empty placeholders ``<src>/<subject>/<path_to_graph>/<side><subject>.arg`` and the
module's ``aims`` reference is replaced by a fake whose ``read`` raises for BAD_SUBJECT, so no real
graph is needed. Each test names the behaviour it verifies.
"""

import inspect
import json
import logging
import os
import types
from pathlib import Path

import pytest
from cortical_tiles.brainvisa import generate_ICBM2009c_transforms
from cortical_tiles.brainvisa.generate_ICBM2009c_transforms import (
    generate_ICBM2009c_transforms as run_transforms,
)
from cortical_tiles.brainvisa.utils import skip_list
from cortical_tiles.brainvisa.utils.skip_list import load_skip_entries, record_skip

gt = generate_ICBM2009c_transforms

STEP = "generate_ICBM2009c_transforms"
SIDE = "R"
OPPOSITE_SIDE = "L"
PATH_TO_GRAPH = "t1mri/folds"
GOOD_SUBJECTS = ("sub-a", "sub-c")
BAD_SUBJECT = "sub-bad"
NO_GRAPH_SUBJECT = "sub-nograph"
BAD_MESSAGE = "cannot compute the ICBM2009c transform"
ALL_SUBJECTS = -1


class TransformError(Exception):
    """Raised by the fake aims reader for BAD_SUBJECT (module level, so it pickles across processes)."""


def _raised(exc):
    """Returns ``exc`` after raising it, so it carries a traceback."""
    try:
        raise exc
    except Exception as caught:
        return caught


def _make_graphs(src, subjects, side=SIDE):
    """Creates one empty placeholder graph per subject under ``src``."""
    for subject in subjects:
        folder = src / subject / PATH_TO_GRAPH
        folder.mkdir(parents=True)
        (folder / f"{side}{subject}.arg").write_text("")


@pytest.fixture
def fake_aims(monkeypatch):
    """Replaces the module's aims: read raises TransformError (from an OSError) for BAD_SUBJECT."""
    state = types.SimpleNamespace(reads=[], raised=[])

    def read(path):
        state.reads.append(path)
        if BAD_SUBJECT in path:
            try:
                raise OSError("graph file is truncated")
            except OSError as cause:
                error = TransformError(BAD_MESSAGE)
                state.raised.append(error)
                raise error from cause
        return path

    def write(transform, path):
        Path(path).write_text(f"transform of {transform}\n")

    graph_manip = types.SimpleNamespace(getICBM2009cTemplateTransform=lambda graph: graph)
    monkeypatch.setattr(gt, "aims", types.SimpleNamespace(read=read, write=write, GraphManip=graph_manip))
    return state


@pytest.fixture
def layout(tmp_path):
    """<output_dir>/transforms, graphs elsewhere, and a skip-list directory apart from both."""
    output_dir = tmp_path / "derivatives"
    output_dir.mkdir()
    src = tmp_path / "graphs"
    src.mkdir()
    return types.SimpleNamespace(
        output_dir=output_dir,
        transform_dir=output_dir / "transforms",
        src=src,
        skip_dir=tmp_path / "skip_lists",
    )


def _run(layout, parallel=False):
    run_transforms(
        src_dir=str(layout.src),
        transform_dir=str(layout.transform_dir),
        path_to_graph=PATH_TO_GRAPH,
        side=SIDE,
        bids=False,
        parallel=parallel,
        nb_subjects=ALL_SUBJECTS,
        qc_path="",
        skip_dir=str(layout.skip_dir),
    )


def _transform_file(layout, subject):
    return layout.transform_dir / SIDE / f"{SIDE}transform_to_ICBM2009c_{subject}.trm"


def _read_subjects(fake_aims):
    """Subjects whose graph the fake aims reader was asked for."""
    return {Path(path).parent.parent.parent.name for path in fake_aims.reads}


def test_generate_ICBM2009c_transforms_help():
    args = "--help"
    argv = args.split(" ")
    generate_ICBM2009c_transforms.main(argv)


def test_generate_ICBM2009c_transforms_n_0(tmp_path):
    """Tests the function when number of subjects is 0"""

    generate_ICBM2009c_transforms.generate_ICBM2009c_transforms(transform_dir=str(tmp_path), nb_subjects=0)


def test_failing_subject_is_recorded_in_skip_dir_with_its_exception_type(layout, fake_aims):
    """A subject whose transform generation raises is recorded in the skip list of the side under the
    skip_dir argument, with the step, the real exception type and its message."""
    _make_graphs(layout.src, (*GOOD_SUBJECTS, BAD_SUBJECT))

    _run(layout)

    entries = load_skip_entries(layout.skip_dir, SIDE)
    assert [(e.subject, e.step, e.exc_type, e.message) for e in entries] == [
        (BAD_SUBJECT, STEP, "TransformError", BAD_MESSAGE)
    ]
    assert load_skip_entries(layout.output_dir, SIDE) == []
    assert load_skip_entries(layout.transform_dir, SIDE) == []


def test_failing_subject_error_record_carries_the_raised_exception_and_its_cause(layout, fake_aims, caplog):
    """record_skip receives the raised exception object itself (not its str), so the ERROR record
    carries that exception, its traceback and its __cause__."""
    _make_graphs(layout.src, (*GOOD_SUBJECTS, BAD_SUBJECT))

    with caplog.at_level(logging.DEBUG):
        _run(layout)

    assert len(fake_aims.raised) == 1
    raised = fake_aims.raised[0]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.exc_info and r.exc_info[1] is raised]
    assert len(errors) == 1, [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    record = errors[0]
    assert BAD_SUBJECT in record.getMessage()
    assert record.exc_info[2] is not None
    assert isinstance(record.exc_info[1].__cause__, OSError)


def test_parallel_worker_records_failing_subject_in_its_own_process(layout, fake_aims, monkeypatch, tmp_path):
    """In parallel mode, record_skip is called inside the worker process that caught the exception,
    with the exception still carrying its traceback (a result sent back to the parent loses it)."""
    _make_graphs(layout.src, (*GOOD_SUBJECTS, BAD_SUBJECT))
    calls_file = tmp_path / "record_skip_calls.jsonl"
    real_record_skip = skip_list.record_skip
    signature = inspect.signature(real_record_skip)

    def spy(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        exc = bound["exc"]
        call = {
            "pid": os.getpid(),
            "subject": bound["subject"],
            "step": bound["step"],
            "exc_type": type(exc).__name__,
            "has_traceback": exc.__traceback__ is not None,
        }
        with open(calls_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(call) + "\n")
        return real_record_skip(*args, **kwargs)

    monkeypatch.setattr(skip_list, "record_skip", spy)
    monkeypatch.setattr(gt, "record_skip", spy, raising=False)
    monkeypatch.setattr(gt, "define_njobs", lambda *args, **kwargs: 2)

    _run(layout, parallel=True)

    calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
    assert [(c["subject"], c["step"], c["exc_type"]) for c in calls] == [(BAD_SUBJECT, STEP, "TransformError")]
    assert calls[0]["pid"] != os.getpid()
    assert calls[0]["has_traceback"] is True
    assert [e.subject for e in load_skip_entries(layout.skip_dir, SIDE)] == [BAD_SUBJECT]


def test_subject_without_graph_file_is_recorded_as_file_not_found(layout, fake_aims):
    """A subject with no graph file of the side is recorded with a FileNotFoundError naming the
    searched graph path pattern."""
    _make_graphs(layout.src, GOOD_SUBJECTS)
    (layout.src / NO_GRAPH_SUBJECT / PATH_TO_GRAPH).mkdir(parents=True)

    _run(layout)

    entries = load_skip_entries(layout.skip_dir, SIDE)
    assert [(e.subject, e.step, e.exc_type) for e in entries] == [(NO_GRAPH_SUBJECT, STEP, "FileNotFoundError")]
    assert f"{NO_GRAPH_SUBJECT}*/{PATH_TO_GRAPH}/{SIDE}*.arg" in entries[0].message


def test_other_subjects_get_their_transform_when_one_subject_fails(layout, fake_aims):
    """One failing subject does not stop the step: each other selected subject gets its transform."""
    _make_graphs(layout.src, (*GOOD_SUBJECTS, BAD_SUBJECT))

    _run(layout)

    for subject in GOOD_SUBJECTS:
        assert _transform_file(layout, subject).is_file(), subject
    assert not _transform_file(layout, BAD_SUBJECT).exists()


def test_subject_in_side_skip_list_is_not_processed(layout, fake_aims):
    """A subject already in the skip list of the side (recorded by an earlier step) is left out:
    its graph is not read and no transform is written for it."""
    _make_graphs(layout.src, GOOD_SUBJECTS)
    skipped = GOOD_SUBJECTS[1]
    record_skip(layout.skip_dir, SIDE, "generate_skeletons", skipped, _raised(RuntimeError("no skeleton")))

    _run(layout)

    assert skipped not in _read_subjects(fake_aims)
    assert not _transform_file(layout, skipped).exists()
    assert _transform_file(layout, GOOD_SUBJECTS[0]).is_file()


def test_subject_in_opposite_side_skip_list_is_processed(layout, fake_aims):
    """Only the skip list of the step's own side excludes subjects."""
    _make_graphs(layout.src, GOOD_SUBJECTS)
    other_side_only = GOOD_SUBJECTS[1]
    record_skip(
        layout.skip_dir, OPPOSITE_SIDE, "generate_skeletons", other_side_only, _raised(RuntimeError("no skeleton"))
    )

    _run(layout)

    assert _transform_file(layout, other_side_only).is_file()


def test_end_of_step_summary_is_one_info_record_of_the_module_logger(layout, fake_aims, caplog):
    """The end-of-step summary is one INFO record on the module logger: 'Succeeded: 2/3'."""
    _make_graphs(layout.src, (*GOOD_SUBJECTS, BAD_SUBJECT))

    with caplog.at_level(logging.INFO):
        _run(layout)

    summaries = [
        r
        for r in caplog.records
        if r.name == gt.log.name and r.levelno == logging.INFO and "Succeeded: 2/3" in r.getMessage()
    ]
    assert len(summaries) == 1, [r.getMessage() for r in caplog.records if r.name == gt.log.name]


def test_transform_generation_writes_nothing_to_stdout(layout, fake_aims, capsys):
    """The step writes no character to standard output (the summary went to stdout before)."""
    _make_graphs(layout.src, (*GOOD_SUBJECTS, BAD_SUBJECT))
    capsys.readouterr()

    _run(layout)

    assert capsys.readouterr().out == ""


# --- REQ-CTFAIL-BDRABCZUK-F28F1306A455: without skip_dir the skip-list directory is transform_dir ------------


def _run_without_skip_dir(layout):
    """Runs the step as a standalone call that passes no skip_dir."""
    run_transforms(
        src_dir=str(layout.src),
        transform_dir=str(layout.transform_dir),
        path_to_graph=PATH_TO_GRAPH,
        side=SIDE,
        bids=False,
        parallel=False,
        nb_subjects=ALL_SUBJECTS,
        qc_path="",
    )


def test_without_skip_dir_failing_subject_is_recorded_in_transform_dir(layout, fake_aims):
    """Called without skip_dir, the step records a failing subject in the side's skip list under the
    transform_dir argument, and writes no skip-list file anywhere else."""
    _make_graphs(layout.src, (*GOOD_SUBJECTS, BAD_SUBJECT))

    _run_without_skip_dir(layout)

    entries = load_skip_entries(layout.transform_dir, SIDE)
    assert [(e.subject, e.step, e.exc_type) for e in entries] == [(BAD_SUBJECT, STEP, "TransformError")]
    assert sorted(layout.output_dir.parent.rglob("skip_list_*.jsonl")) == [
        layout.transform_dir / f"skip_list_{SIDE}.jsonl"
    ]


def test_without_skip_dir_subject_in_transform_dir_skip_list_is_not_processed(layout, fake_aims):
    """Called without skip_dir, a subject in the side's skip list under the transform_dir argument is left
    out: its graph is not read and no transform is written for it."""
    _make_graphs(layout.src, GOOD_SUBJECTS)
    skipped = GOOD_SUBJECTS[1]
    record_skip(layout.transform_dir, SIDE, "generate_skeletons", skipped, _raised(RuntimeError("no skeleton")))

    _run_without_skip_dir(layout)

    assert skipped not in _read_subjects(fake_aims)
    assert not _transform_file(layout, skipped).exists()
    assert _transform_file(layout, GOOD_SUBJECTS[0]).is_file()
