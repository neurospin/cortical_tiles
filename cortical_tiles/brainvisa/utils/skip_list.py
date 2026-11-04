"""Durable per-side skip list of subjects that failed a cortical_tiles step.

``record_skip`` logs a per-subject failure at ERROR (exception type and full
traceback, ``__cause__`` chain included) and appends it to the skip list of
its side; ``load_skipped`` and ``load_skip_entries`` read that list back and
``clear_skip_list`` empties the list of one side.

Storage: one JSON Lines file per side, ``<skip_dir>/skip_list_<side>.jsonl``,
one ASCII-escaped JSON object per line. Writers append one line under an
exclusive ``fcntl.flock``; readers read under a shared lock, so concurrent
workers on one node never tear or lose an entry.
"""

import fcntl
import json
import os
from dataclasses import dataclass

from cortical_tiles.config.logs import set_file_logger

log = set_file_logger(__file__)

# Named set "cortical_tiles step set": the steps whose per-subject failures are skip-listed.
SKIP_STEPS = (
    "generate_skeletons",
    "generate_foldlabels",
    "generate_ICBM2009c_transforms",
    "resample_files",
    "mask_resampled_foldlabels",
    "generate_crops",
)
SKIP_SIDES = ("L", "R")
SKIP_LIST_FILENAME = "skip_list_{side}.jsonl"

_ENTRY_KEYS = frozenset(("subject", "step", "exc_type", "message"))


@dataclass(frozen=True)
class SkipEntry:
    """One recorded failure of one subject in one step, for one side."""

    subject: str
    step: str
    exc_type: str
    message: str


def _check_side(side):
    """Raises ValueError when ``side`` is not one of SKIP_SIDES."""
    if not isinstance(side, str) or side not in SKIP_SIDES:
        raise ValueError(f"side must be one of {SKIP_SIDES}, got {side!r}")


def _check_record_args(step, subject, exc):
    """Checks the step, subject and exc arguments of record_skip before any filesystem access."""
    if not isinstance(step, str) or step not in SKIP_STEPS:
        raise ValueError(f"step must be one of {SKIP_STEPS}, got {step!r}")
    if not isinstance(subject, str):
        raise TypeError(f"subject must be a str, got {type(subject).__name__}")
    if subject == "":
        raise ValueError("subject must be a non-empty str")
    if not isinstance(exc, BaseException):
        raise TypeError(f"exc must be a BaseException instance, got {type(exc).__name__}")


def _skip_list_path(skip_dir, side):
    """Returns the path of the skip-list file of ``side`` under ``skip_dir``."""
    return os.path.join(os.fspath(skip_dir), SKIP_LIST_FILENAME.format(side=side))


def _write_all(fd, data):
    """Writes every byte of ``data`` to ``fd``, looping over partial writes."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _append_line(path, line):
    """Appends ``line`` to ``path`` under an exclusive lock held for this write only."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            _write_all(fd, line.encode("ascii"))
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def record_skip(skip_dir, side, step, subject, exc):
    """Logs the failure of ``subject`` in ``step`` and appends it to the skip list of ``side``.

    Args:
        skip_dir: str or os.PathLike, directory holding the skip-list files
        side: "L" or "R"
        step: one of SKIP_STEPS
        subject: non-empty subject name
        exc: the exception the step raised for that subject

    Raises:
        ValueError: side or step outside its set, or empty subject
        TypeError: subject not a str, or exc not a BaseException instance
    """
    _check_side(side)
    _check_record_args(step, subject, exc)
    exc_type = type(exc).__name__
    message = str(exc)
    log.error(
        "Skipping subject %s on side %s: step %s failed with %s: %s",
        subject,
        side,
        step,
        exc_type,
        message,
        exc_info=exc,
    )
    entry = {"subject": subject, "step": step, "exc_type": exc_type, "message": message}
    os.makedirs(skip_dir, exist_ok=True)
    _append_line(_skip_list_path(skip_dir, side), json.dumps(entry, ensure_ascii=True) + "\n")


def _parse_line(path, line_number, line):
    """Returns the SkipEntry of one line; ValueError naming ``path`` and ``line_number`` otherwise."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: line {line_number}: not valid JSON ({error.msg})") from error
    if not isinstance(obj, dict) or set(obj) != _ENTRY_KEYS:
        raise ValueError(f"{path}: line {line_number}: expected an object with keys {sorted(_ENTRY_KEYS)}")
    if not all(isinstance(value, str) for value in obj.values()):
        raise ValueError(f"{path}: line {line_number}: every field must be a string")
    return SkipEntry(subject=obj["subject"], step=obj["step"], exc_type=obj["exc_type"], message=obj["message"])


def _read_locked(path):
    """Returns the text of ``path`` read under a shared lock; None when the file does not exist.

    Only ``open`` can raise FileNotFoundError here; reading an open file never does.
    """
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return handle.read()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except FileNotFoundError:
        return None


def load_skip_entries(skip_dir, side):
    """Returns the list of SkipEntry recorded in the skip list of ``side``, in file order.

    A missing directory or file gives an empty list and creates nothing.

    Raises:
        ValueError: side outside SKIP_SIDES, or a malformed line (file path and 1-based line number named)
    """
    _check_side(side)
    path = _skip_list_path(skip_dir, side)
    text = _read_locked(path)
    if not text:
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return [_parse_line(path, number, line) for number, line in enumerate(lines, start=1)]


def load_skipped(skip_dir, side):
    """Returns the set of subjects recorded in the skip list of ``side``."""
    _check_side(side)
    return {entry.subject for entry in load_skip_entries(skip_dir, side)}


def clear_skip_list(skip_dir, side):
    """Empties the skip list of ``side``, leaving the other side's list untouched.

    A missing directory or file is left as it is: nothing is created.

    Raises:
        ValueError: side outside SKIP_SIDES (checked before any filesystem access)
    """
    _check_side(side)
    try:
        fd = os.open(_skip_list_path(skip_dir, side), os.O_WRONLY)
    except FileNotFoundError:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            os.ftruncate(fd, 0)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
