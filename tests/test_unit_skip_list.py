"""Unit tests for cortical_tiles.brainvisa.utils.skip_list (TASK-CTFAIL-BDRABCZUK-CED106F2F9AA).

The skip list is the durable, per-side record of subjects that failed a cortical_tiles step.
``record_skip`` logs the failure at ERROR with the exception type and the full traceback
(``__cause__`` chain included) and appends one entry to the side's list; ``load_skipped``
returns the recorded subjects of one side and ``load_skip_entries`` the entries themselves;
``clear_skip_list`` empties the list of one side (the run driver calls it at run start).

Each test names the Requirement it verifies. Cross-process tests start fresh Python
interpreters on the same skip-list directory, so "durable" means "on disk", not "in memory".
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from cortical_tiles.brainvisa.utils import skip_list
from cortical_tiles.brainvisa.utils.skip_list import clear_skip_list, load_skip_entries, load_skipped, record_skip
from hypothesis import given, settings
from hypothesis import strategies as st

STEP_SET = (
    "generate_skeletons",
    "generate_foldlabels",
    "generate_ICBM2009c_transforms",
    "resample_files",
    "mask_resampled_foldlabels",
    "generate_crops",
)
STEP = "generate_foldlabels"
PACKAGE_ROOT = Path(skip_list.__file__).resolve().parents[3]
SUBPROCESS_TIMEOUT_S = 120

CHILD_SCRIPT = """
import sys
from cortical_tiles.brainvisa.utils.skip_list import record_skip
skip_dir, side, step, pad = sys.argv[1:5]
for subject in sys.argv[5:]:
    try:
        raise RuntimeError(subject + " " + "x" * int(pad))
    except RuntimeError as exc:
        record_skip(skip_dir=skip_dir, side=side, step=step, subject=subject, exc=exc)
"""


def _raised(exc):
    """Returns ``exc`` after raising it, so it carries a traceback."""
    try:
        raise exc
    except Exception as caught:
        return caught


def _chained_error():
    """RuntimeError raised from KeyError raised from FileNotFoundError, each with a traceback."""
    try:
        try:
            try:
                raise FileNotFoundError("missing graph Lsub-01.arg")
            except FileNotFoundError as root:
                raise KeyError("vertex 42 has no label") from root
        except KeyError as middle:
            raise RuntimeError("foldlabel extraction failed") from middle
    except RuntimeError as top:
        return top


def _child_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(PACKAGE_ROOT), env.get("PYTHONPATH", "")) if p)
    return env


def _start_child(skip_dir, side, subjects, pad=0):
    return subprocess.Popen(
        [sys.executable, "-c", CHILD_SCRIPT, str(skip_dir), side, STEP, str(pad), *subjects],
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_child(proc):
    _, stderr = proc.communicate(timeout=SUBPROCESS_TIMEOUT_S)
    assert proc.returncode == 0, f"child record_skip process failed:\n{stderr[-2000:]}"


def _record_in_subprocess(skip_dir, side, subjects):
    _wait_child(_start_child(skip_dir, side, subjects))


def _error_records(caplog):
    return [record for record in caplog.records if record.levelno == logging.ERROR]


def test_record_skip_logs_one_error_naming_subject_side_step_and_type(tmp_path, caplog):
    """REQ-CTFAIL-BDRABCZUK-110C3035AE16: one ERROR record naming subject, side, step and exception type."""
    with caplog.at_level(logging.DEBUG):
        record_skip(skip_dir=tmp_path, side="R", step=STEP, subject="sub-0042", exc=_raised(ValueError("bad bucket")))

    errors = _error_records(caplog)
    assert len(errors) == 1, f"expected one ERROR record, got {[r.getMessage() for r in errors]}"
    message = errors[0].getMessage()
    assert "sub-0042" in message
    assert re.search(r"\bR\b", message), f"side R not named in {message!r}"
    assert STEP in message
    assert "ValueError" in message


def test_record_skip_error_record_contains_traceback_with_cause_chain(tmp_path, caplog):
    """REQ-CTFAIL-BDRABCZUK-23B9B9C8EEB2: ERROR record holds the traceback of each exception of the __cause__ chain."""
    with caplog.at_level(logging.DEBUG):
        record_skip(skip_dir=tmp_path, side="L", step=STEP, subject="sub-01", exc=_chained_error())

    errors = _error_records(caplog)
    assert len(errors) == 1
    rendered = logging.Formatter("%(message)s").format(errors[0])
    assert "Traceback (most recent call last)" in rendered
    assert "FileNotFoundError: missing graph Lsub-01.arg" in rendered
    assert "KeyError: 'vertex 42 has no label'" in rendered
    assert "RuntimeError: foldlabel extraction failed" in rendered
    assert rendered.count("The above exception was the direct cause of the following exception") == 2


def test_record_skip_entry_holds_subject_step_type_and_message(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-826AB670054E: one entry with subject, step, exception type name and message."""
    exc = _raised(ValueError("graph has no vertex, see log; line 2\nline 3"))
    record_skip(skip_dir=tmp_path, side="L", step=STEP, subject="sub-01", exc=exc)

    entries = load_skip_entries(skip_dir=tmp_path, side="L")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.subject == "sub-01"
    assert entry.step == STEP
    assert entry.exc_type == "ValueError"
    assert entry.message == str(exc)


TEXT = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=40)


@settings(max_examples=50, deadline=None)
@given(
    subject=TEXT,
    message=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=80),
    step=st.sampled_from(STEP_SET),
    side=st.sampled_from(("L", "R")),
)
def test_entry_fields_round_trip_for_arbitrary_text(subject, message, step, side):
    """REQ-CTFAIL-BDRABCZUK-826AB670054E (property): fields survive commas, quotes, newlines and non-ASCII text."""
    with tempfile.TemporaryDirectory() as skip_dir:
        record_skip(skip_dir=skip_dir, side=side, step=step, subject=subject, exc=_raised(OSError(message)))

        entries = load_skip_entries(skip_dir=skip_dir, side=side)
        assert [(e.subject, e.step, e.exc_type, e.message) for e in entries] == [
            (subject, step, "OSError", str(OSError(message)))
        ]
        assert load_skipped(skip_dir=skip_dir, side=side) == {subject}


def test_load_skipped_returns_subject_recorded_by_another_process(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-1EDB190FDD61: a subject recorded in another process is read back here."""
    _record_in_subprocess(tmp_path, "L", ["sub-07"])

    assert load_skipped(skip_dir=tmp_path, side="L") == {"sub-07"}


def test_subject_recorded_for_one_side_is_absent_from_other_side(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-9CEBAEDC9265: exclusion is per side (a subject failing on L stays in R)."""
    record_skip(skip_dir=tmp_path, side="L", step=STEP, subject="sub-01", exc=_raised(ValueError("L only")))
    assert "sub-01" not in load_skipped(skip_dir=tmp_path, side="R")

    record_skip(skip_dir=tmp_path, side="R", step=STEP, subject="sub-02", exc=_raised(ValueError("R only")))
    assert load_skipped(skip_dir=tmp_path, side="L") == {"sub-01"}
    assert load_skipped(skip_dir=tmp_path, side="R") == {"sub-02"}


def test_entries_from_earlier_runs_of_both_sides_are_preserved(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-96F82E51D966: L and R runs, one after the other, never overwrite each other."""
    _record_in_subprocess(tmp_path, "L", ["sub-01"])
    _record_in_subprocess(tmp_path, "R", ["sub-02"])
    _record_in_subprocess(tmp_path, "L", ["sub-03"])
    record_skip(skip_dir=tmp_path, side="R", step=STEP, subject="sub-04", exc=_raised(ValueError("last")))

    assert load_skipped(skip_dir=tmp_path, side="L") == {"sub-01", "sub-03"}
    assert load_skipped(skip_dir=tmp_path, side="R") == {"sub-02", "sub-04"}


def test_concurrent_workers_recording_same_side_lose_no_entry(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-96FF2204FD4C: concurrent processes on one side give one complete entry per call."""
    n_workers, per_worker, pad = 8, 25, 8000  # pad > PIPE_BUF so a torn write would show
    batches = [[f"sub-w{w}-{i:02d}" for i in range(per_worker)] for w in range(n_workers)]
    procs = [_start_child(tmp_path, "L", batch, pad=pad) for batch in batches]
    for proc in procs:
        _wait_child(proc)

    entries = load_skip_entries(skip_dir=tmp_path, side="L")
    expected = {subject for batch in batches for subject in batch}
    assert len(entries) == n_workers * per_worker
    assert {e.subject for e in entries} == expected
    for entry in entries:
        assert entry.step == STEP
        assert entry.exc_type == "RuntimeError"
        assert entry.message == f"{entry.subject} " + "x" * pad


@pytest.mark.parametrize("where", ["existing", "never_created"])
def test_load_skipped_is_empty_when_nothing_recorded(tmp_path, where):
    """REQ-CTFAIL-BDRABCZUK-AFBE20C33221: no entry recorded for a side -> empty set."""
    skip_dir = tmp_path if where == "existing" else tmp_path / "never_created"
    for side in ("L", "R"):
        assert load_skipped(skip_dir=skip_dir, side=side) == set()


BAD_SIDES = ["", "l", "r", "LR", "left", "B", None]


@pytest.mark.parametrize("side", BAD_SIDES)
def test_record_skip_rejects_unknown_side_without_writing(tmp_path, side):
    """REQ-CTFAIL-BDRABCZUK-A3252589BEF7 (was -B1023690091E): side outside {L, R} -> ValueError, no entry written."""
    with pytest.raises(ValueError):
        record_skip(skip_dir=tmp_path, side=side, step=STEP, subject="sub-01", exc=_raised(ValueError("x")))

    assert load_skip_entries(skip_dir=tmp_path, side="L") == []
    assert load_skip_entries(skip_dir=tmp_path, side="R") == []


@pytest.mark.parametrize("side", BAD_SIDES)
def test_load_skipped_rejects_unknown_side(tmp_path, side):
    """REQ-CTFAIL-BDRABCZUK-A3252589BEF7 (was -8F1FFAC9B83E): side outside {L, R} -> ValueError."""
    with pytest.raises(ValueError):
        load_skipped(skip_dir=tmp_path, side=side)


@pytest.mark.parametrize("side", BAD_SIDES)
def test_load_skip_entries_rejects_unknown_side(tmp_path, side):
    """REQ-CTFAIL-BDRABCZUK-A3252589BEF7: load_skip_entries with a side outside {L, R} -> ValueError."""
    _record_both_sides(tmp_path)
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError):
        load_skip_entries(skip_dir=tmp_path, side=side)

    assert _snapshot(tmp_path) == before


class _UntouchableDir:
    """A skip_dir that fails the test as soon as anything turns it into a filesystem path."""

    def __fspath__(self):
        raise AssertionError("skip_dir was accessed before the side guard")

    def __str__(self):
        raise AssertionError("skip_dir was accessed before the side guard")


SIDE_GUARDED_CALLS = {
    "record_skip": lambda d, side: record_skip(
        skip_dir=d, side=side, step=STEP, subject="sub-01", exc=_raised(ValueError("x"))
    ),
    "load_skipped": lambda d, side: load_skipped(skip_dir=d, side=side),
    "load_skip_entries": lambda d, side: load_skip_entries(skip_dir=d, side=side),
    "clear_skip_list": lambda d, side: clear_skip_list(skip_dir=d, side=side),
}


@pytest.mark.parametrize("side", ["left", None])
@pytest.mark.parametrize("call", list(SIDE_GUARDED_CALLS.values()), ids=list(SIDE_GUARDED_CALLS))
def test_side_guard_runs_before_any_filesystem_access(call, side):
    """REQ-CTFAIL-BDRABCZUK-A3252589BEF7: a bad side raises ValueError without touching the skip-list directory."""
    with pytest.raises(ValueError):
        call(_UntouchableDir(), side)


@pytest.mark.parametrize("step", ["", "generate_distmaps", "Generate_crops", "crops", None])
def test_record_skip_rejects_unknown_step_without_writing(tmp_path, step):
    """REQ-CTFAIL-BDRABCZUK-2FD43CD5952E: step outside the cortical_tiles step set -> ValueError, no entry."""
    with pytest.raises(ValueError):
        record_skip(skip_dir=tmp_path, side="L", step=step, subject="sub-01", exc=_raised(ValueError("x")))

    assert load_skip_entries(skip_dir=tmp_path, side="L") == []


@pytest.mark.parametrize("step", STEP_SET)
def test_record_skip_accepts_each_step_of_the_set(tmp_path, step):
    """REQ-CTFAIL-BDRABCZUK-2FD43CD5952E (control): each step of the set is recorded."""
    record_skip(skip_dir=tmp_path, side="L", step=step, subject="sub-01", exc=_raised(ValueError("x")))

    assert [e.step for e in load_skip_entries(skip_dir=tmp_path, side="L")] == [step]


def _snapshot(directory):
    """Sorted (relative path, bytes or None for a directory) under ``directory``; None when it does not exist."""
    if not directory.exists():
        return None
    return sorted(
        (str(p.relative_to(directory)), p.read_bytes() if p.is_file() else None) for p in directory.rglob("*")
    )


def _record_both_sides(skip_dir):
    for side, subjects in (("L", ["sub-01", "sub-02"]), ("R", ["sub-03", "sub-04", "sub-05"])):
        for subject in subjects:
            record_skip(skip_dir=skip_dir, side=side, step=STEP, subject=subject, exc=_raised(ValueError(subject)))


@pytest.mark.parametrize("side", ["L", "R"])
def test_clear_skip_list_empties_the_given_side(tmp_path, side):
    """REQ-CTFAIL-BDRABCZUK-22849FBEDCA4: after clear_skip_list for a side, load_skip_entries of that side is []."""
    _record_both_sides(tmp_path)
    assert load_skip_entries(skip_dir=tmp_path, side=side) != []

    clear_skip_list(skip_dir=tmp_path, side=side)

    assert load_skip_entries(skip_dir=tmp_path, side=side) == []
    assert load_skipped(skip_dir=tmp_path, side=side) == set()


@pytest.mark.parametrize(("side", "opposite"), [("L", "R"), ("R", "L")])
def test_clear_skip_list_preserves_the_opposite_side(tmp_path, side, opposite):
    """REQ-CTFAIL-BDRABCZUK-42C440D22D23: clearing one side keeps each entry of the opposite side."""
    _record_both_sides(tmp_path)
    before = load_skip_entries(skip_dir=tmp_path, side=opposite)
    assert before

    clear_skip_list(skip_dir=tmp_path, side=side)

    assert load_skip_entries(skip_dir=tmp_path, side=opposite) == before


@pytest.mark.parametrize("where", ["never_created", "empty_dir", "other_side_only"])
def test_clear_skip_list_creates_nothing_when_no_list_exists(tmp_path, where):
    """REQ-CTFAIL-BDRABCZUK-742A2A0960E9: no skip-list file for the side -> directory left unchanged."""
    skip_dir = tmp_path / "skips"
    if where != "never_created":
        skip_dir.mkdir()
    if where == "other_side_only":
        record_skip(skip_dir=skip_dir, side="R", step=STEP, subject="sub-01", exc=_raised(ValueError("R only")))
    before = _snapshot(skip_dir)

    clear_skip_list(skip_dir=skip_dir, side="L")

    assert _snapshot(skip_dir) == before


@pytest.mark.parametrize("side", BAD_SIDES)
def test_clear_skip_list_rejects_unknown_side(tmp_path, side):
    """REQ-CTFAIL-BDRABCZUK-A3252589BEF7 (was -8FFC2FB3827A): side outside {L, R} -> ValueError, no list truncated."""
    _record_both_sides(tmp_path)
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError):
        clear_skip_list(skip_dir=tmp_path, side=side)

    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "subject", [None, 42, b"sub-01", Path("sub-01"), ["sub-01"]], ids=["none", "int", "bytes", "path", "list"]
)
def test_record_skip_rejects_non_str_subject_before_any_write(tmp_path, subject):
    """REQ-CTFAIL-BDRABCZUK-46AEF5CD6328: subject not a str -> TypeError, nothing written to the skip-list directory."""
    skip_dir = tmp_path / "skips"
    with pytest.raises(TypeError):
        record_skip(skip_dir=skip_dir, side="L", step=STEP, subject=subject, exc=_raised(ValueError("x")))

    assert not skip_dir.exists()


@pytest.mark.parametrize("side", ["L", "R"])
def test_record_skip_rejects_empty_subject_before_any_write(tmp_path, side):
    """REQ-CTFAIL-BDRABCZUK-ADE28340210B: subject "" -> ValueError, nothing written to the skip-list directory."""
    skip_dir = tmp_path / "skips"
    with pytest.raises(ValueError):
        record_skip(skip_dir=skip_dir, side=side, step=STEP, subject="", exc=_raised(ValueError("x")))

    assert not skip_dir.exists()


@pytest.mark.parametrize(
    "exc", [None, "boom", ValueError, 42, ["ValueError"]], ids=["none", "str", "exception_class", "int", "list"]
)
def test_record_skip_rejects_non_exception_exc_before_any_write(tmp_path, exc):
    """REQ-CTFAIL-BDRABCZUK-9EBAEC493D01: exc not a BaseException instance -> TypeError, nothing written."""
    skip_dir = tmp_path / "skips"
    with pytest.raises(TypeError):
        record_skip(skip_dir=skip_dir, side="L", step=STEP, subject="sub-01", exc=exc)

    assert not skip_dir.exists()


def _entry_line(subject):
    return json.dumps({"subject": subject, "step": STEP, "exc_type": "ValueError", "message": "x"}, ensure_ascii=True)


MALFORMED_LINES = {
    "not_json": "{not json",
    "array": "[1, 2]",
    "string": json.dumps("sub-01"),
    "null": "null",
    "missing_key": json.dumps({"subject": "sub-01", "step": STEP, "exc_type": "ValueError"}),
    "extra_key": json.dumps({"subject": "sub-01", "step": STEP, "exc_type": "E", "message": "x", "region": "S.C."}),
    "non_str_step": json.dumps({"subject": "sub-01", "step": 3, "exc_type": "ValueError", "message": "x"}),
    "null_subject": json.dumps({"subject": None, "step": STEP, "exc_type": "ValueError", "message": "x"}),
}


@pytest.mark.parametrize("line_number", [1, 3])
@pytest.mark.parametrize("bad_line", list(MALFORMED_LINES.values()), ids=list(MALFORMED_LINES))
def test_load_skip_entries_rejects_malformed_line_naming_file_and_line(tmp_path, bad_line, line_number):
    """REQ-CTFAIL-BDRABCZUK-ED8932B45EB6: a malformed line -> ValueError naming the file path and line number."""
    lines = [_entry_line(f"sub-{i:02d}") for i in range(1, 5)]
    lines[line_number - 1] = bad_line
    path = tmp_path / "skip_list_L.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        load_skip_entries(skip_dir=tmp_path, side="L")

    message = str(raised.value)
    assert str(path) in message, f"file path not named in {message!r}"
    rest = message.replace(str(path), "")
    assert re.search(rf"(?<!\d){line_number}(?!\d)", rest), f"line {line_number} not named in {message!r}"
