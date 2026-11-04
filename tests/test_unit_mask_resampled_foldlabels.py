"""Unit tests for cortical_tiles.brainvisa.mask_resampled_foldlabels (REQ-CTILESTEST-115..126).

After resampling, champollion_pipeline stage 2 masks each 2 mm foldlabel with its re-skeletonized
2 mm skeleton (``mask_foldlabel_files`` -> ``FoldLabelMasker``), so that crops of both volumes have
the same support. Inputs here are tiny synthetic volumes laid out as the pipeline does:

- ``<src>/<side>_before_masking/<side>resampled_foldlabel_<subject>.nii.gz``
- ``<skeletons>/<side>/<side>resampled_skeleton_<subject>.nii.gz``
- output ``<masked>/<side>/<side>resampled_foldlabel_<subject>.nii.gz``

all under pytest tmp dirs whose names never contain the source file prefix (see defect 11 of
TASK-136, pinned in test_unit_quality_checks_not_processed.py).

Characterization tests: behaviour recorded on cortical_tiles dev 4141bd8. Tests flagged "DEFECT"
pin current behaviour that looks wrong; it is recorded, not fixed (tests-only rule of TASK-114).
"""

import csv
import logging
import os
import subprocess
import sys

import numpy as np
import pytest
from cortical_tiles.brainvisa import mask_resampled_foldlabels as mrf
from cortical_tiles.brainvisa.utils.skip_list import SkipEntry, load_skip_entries, load_skipped, record_skip
from pathos import multiprocessing as pathos_mp
from soma import aims

SIDE = "R"
SHAPE = (10, 11, 12, 1)
SKELETON_VOXELS = [(2, 2, 2), (2, 2, 3), (5, 6, 7), (8, 8, 8)]


def _write(path, arr, voxel_size=2.0):
    vol = aims.Volume(np.ascontiguousarray(arr))
    vol.header()["voxel_size"] = [voxel_size, voxel_size, voxel_size, 1.0]
    aims.write(vol, str(path))


def _skeleton():
    arr = np.zeros(SHAPE, dtype=np.int16)
    for voxel in SKELETON_VOXELS:
        arr[voxel + (0,)] = 60
    return arr


def _foldlabel():
    """Labels on every skeleton voxel plus two voxels outside the skeleton."""
    arr = np.zeros(SHAPE, dtype=np.int16)
    for value, voxel in enumerate(SKELETON_VOXELS, start=1001):
        arr[voxel + (0,)] = value
    arr[0, 0, 0, 0] = 2001
    arr[9, 10, 11, 0] = 2002
    return arr


@pytest.fixture
def tree(tmp_path):
    """Builds the pipeline layout for subjects s1, s2; returns dict of the three root dirs."""
    roots = {name: tmp_path / name for name in ("src", "skel", "masked")}
    (roots["src"] / f"{SIDE}_before_masking").mkdir(parents=True)
    (roots["skel"] / SIDE).mkdir(parents=True)
    for subject in ("s1", "s2"):
        _write(roots["src"] / f"{SIDE}_before_masking" / f"{SIDE}resampled_foldlabel_{subject}.nii.gz", _foldlabel())
        _write(roots["skel"] / SIDE / f"{SIDE}resampled_skeleton_{subject}.nii.gz", _skeleton())
    return roots


def _masker(tree, parallel=False):
    return mrf.FoldLabelMasker(
        src_dir=str(tree["src"]),
        skeleton_dir=str(tree["skel"]),
        masked_dir=str(tree["masked"]),
        side=SIDE,
        parallel=parallel,
    )


def _masked_path(tree, subject):
    return tree["masked"] / SIDE / f"{SIDE}resampled_foldlabel_{subject}.nii.gz"


def _read(path):
    vol = aims.read(str(path))
    return np.asarray(vol).copy(), list(vol.header()["voxel_size"])[:3]


def _expected_masked():
    arr = _foldlabel()
    arr[_skeleton() == 0] = 0
    return arr


def _clear_pathos_pools():
    state = pathos_mp._ProcessPool__STATE
    for pool in list(state.values()):
        pool.terminate()
        pool.join()
    state.clear()


# --- REQ-CTILESTEST-115: nearest_nonzero_idx -------------------------------------------------------


def test_nearest_nonzero_idx_excludes_query_voxel():
    """REQ-CTILESTEST-115: returns the nearest other non-zero voxel; the array is left unchanged."""
    arr = np.zeros((6, 6, 6), dtype=np.int16)
    arr[2, 2, 2] = 5  # query voxel itself, non-zero
    arr[2, 2, 4] = 7  # distance 2
    arr[5, 5, 5] = 9  # distance > 2
    before = arr.copy()

    assert tuple(int(c) for c in mrf.nearest_nonzero_idx(arr, 2, 2, 2)) == (2, 2, 4)
    np.testing.assert_array_equal(arr, before)


# --- REQ-CTILESTEST-116..117: mask_one_file -------------------------------------------------------


def test_mask_one_file_zeroes_labels_outside_skeleton(tree):
    """REQ-CTILESTEST-116: written foldlabel = source foldlabel zeroed where the skeleton is 0, same voxel size."""
    (tree["masked"] / SIDE).mkdir(parents=True)
    _masker(tree).mask_one_file("s1")

    arr, voxel_size = _read(_masked_path(tree, "s1"))
    np.testing.assert_array_equal(arr, _expected_masked())
    assert voxel_size == [2.0, 2.0, 2.0]


def test_mask_one_file_fills_unlabelled_skeleton_voxels(tree):
    """REQ-CTILESTEST-117: skeleton voxels with foldlabel 0 (<= 40) get the nearest non-zero label."""
    foldlabel = _foldlabel()
    foldlabel[2, 2, 3, 0] = 0  # nearest labelled neighbour: (2,2,2) = 1001
    foldlabel[8, 8, 8, 0] = 0  # nearest labelled neighbour: (5,6,7) = 1003
    _write(tree["src"] / f"{SIDE}_before_masking" / f"{SIDE}resampled_foldlabel_s1.nii.gz", foldlabel)
    (tree["masked"] / SIDE).mkdir(parents=True)

    with pytest.warns(UserWarning, match="incompatible foldlabel and skeleton"):
        _masker(tree).mask_one_file("s1")

    arr, _ = _read(_masked_path(tree, "s1"))
    expected = _expected_masked()
    expected[2, 2, 3, 0] = 1001
    expected[8, 8, 8, 0] = 1003
    np.testing.assert_array_equal(arr, expected)


# --- REQ-CTILESTEST-119..124: compute -------------------------------------------------------------


def test_compute_masks_every_subject(tree):
    """REQ-CTILESTEST-119: serial compute masks s1, s2 and writes an empty <masked>/not_processed_files.csv."""
    _masker(tree).compute()

    for subject in ("s1", "s2"):
        arr, _ = _read(_masked_path(tree, subject))
        np.testing.assert_array_equal(arr, _expected_masked())
    assert (tree["masked"] / "not_processed_files.csv").read_text() == ""


def test_compute_skips_already_masked_subjects(tree):
    """REQ-CTILESTEST-120: a subject whose masked file exists is not processed again."""
    (tree["masked"] / SIDE).mkdir(parents=True)
    sentinel = np.full(SHAPE, 5, dtype=np.int16)
    _write(_masked_path(tree, "s1"), sentinel)

    _masker(tree).compute()

    np.testing.assert_array_equal(_read(_masked_path(tree, "s1"))[0], sentinel)
    np.testing.assert_array_equal(_read(_masked_path(tree, "s2"))[0], _expected_masked())


def test_compute_parallel_matches_serial(tree):
    """REQ-CTILESTEST-121: parallel=True writes the same masked files as the serial mode."""
    _clear_pathos_pools()
    try:
        _masker(tree, parallel=True).compute()
    finally:
        _clear_pathos_pools()

    for subject in ("s1", "s2"):
        np.testing.assert_array_equal(_read(_masked_path(tree, subject))[0], _expected_masked())


def test_compute_missing_source_dir_raises(tmp_path):
    """REQ-CTILESTEST-122: missing <src>/<side>_before_masking -> NotADirectoryError."""
    masker = mrf.FoldLabelMasker(str(tmp_path / "nowhere"), str(tmp_path), str(tmp_path / "m"), SIDE, False)
    with pytest.raises(NotADirectoryError):
        masker.compute()


def test_compute_empty_source_dir_returns_without_error(tree):
    """REQ-CTDEFECTS-17 (inverts REQ-CTILESTEST-123): <src>/<side>_before_masking without .nii.gz -> no exception."""
    for path in (tree["src"] / f"{SIDE}_before_masking").iterdir():
        path.unlink()
    _masker(tree).compute()


def test_compute_zero_subjects_does_nothing(tree):
    """REQ-CTILESTEST-124: nb_subjects=0 -> no output directory, no file."""
    _masker(tree).compute(nb_subjects=0)
    assert not tree["masked"].exists()


# --- REQ-CTILESTEST-125 / 126: parse_args and main -------------------------------------------------


def test_parse_args_maps_output_dir_and_subjects(tmp_path):
    """REQ-CTILESTEST-125: -o -> masked_dir (and output_dir), -n all -> -1, -n 3 -> 3."""
    out = str(tmp_path / "out")
    params = mrf.parse_args(["-s", "S", "-k", "K", "-o", out, "-i", "L", "-n", "all"])
    assert params["masked_dir"] == out and params["output_dir"] == out
    assert (params["src_dir"], params["skeleton_dir"], params["side"], params["nb_subjects"]) == ("S", "K", "L", -1)
    assert mrf.parse_args(["-o", out, "-n", "3"])["nb_subjects"] == 3


def test_main_masks_files(tree):
    """REQ-CTILESTEST-126: main(argv) masks the files described by its command line."""
    argv = ["-s", str(tree["src"]), "-k", str(tree["skel"]), "-o", str(tree["masked"]), "-i", SIDE, "-n", "1"]
    mrf.main(argv)

    np.testing.assert_array_equal(_read(_masked_path(tree, "s1"))[0], _expected_masked())
    assert not _masked_path(tree, "s2").exists()
    with open(tree["masked"] / "not_processed_files.csv") as handle:
        rows = list(csv.reader(handle))
    assert [os.path.basename(row[0]) for row in rows] == [f"{SIDE}resampled_foldlabel_s2.nii.gz"]


# --- REQ-CTFAIL-BDRABCZUK-AD7AD519AD6D / B9ADD12BFC75 / 2745F3650346 / 0CD18CCC3E7C / 284E1EE37476 /
# --- 0527B98AAC49 / B14AD6CF9D1F / 728340959F1D: per-subject failures go to the side's skip list -------
# --- REQ-CTILESTEST-BDRABCZUK-06B45F699FCB (supersedes REQ-CTILESTEST-118, formerly AssertionError).
# The failure rule itself (ERROR record, traceback, skip-list format) belongs to skip_list.py
# (REQ-CTFAIL-BDRABCZUK-826AB670054E and siblings); these tests pin only what the masking step adds.

STEP = "mask_resampled_foldlabels"
BAD_SUBJECT = "s2"
GOOD_SUBJECT = "s1"


def _skip_dir(tree):
    """The shared skip-list directory passed to mask_foldlabel_files (one per cortical_tiles output root)."""
    return tree["src"].parent / "skip"


def _mask(tree, parallel=False, **kwargs):
    """Runs mask_foldlabel_files on the tree; skip_dir defaults to _skip_dir(tree), skip_dir=None omits it."""
    kwargs.setdefault("skip_dir", str(_skip_dir(tree)))
    if kwargs["skip_dir"] is None:
        del kwargs["skip_dir"]
    mrf.mask_foldlabel_files(
        src_dir=str(tree["src"]),
        skeleton_dir=str(tree["skel"]),
        masked_dir=str(tree["masked"]),
        side=SIDE,
        parallel=parallel,
        **kwargs,
    )


def _make_mismatched(tree, subject=BAD_SUBJECT):
    """Gives ``subject`` a skeleton with 45 unlabelled voxels more than its foldlabel (over the 40-voxel tolerance)."""
    skeleton = _skeleton()
    skeleton[0:5, 0:3, 9:12, 0] = 60
    _write(tree["skel"] / SIDE / f"{SIDE}resampled_skeleton_{subject}.nii.gz", skeleton)


def _masked_subjects(tree):
    folder = tree["masked"] / SIDE
    prefix = f"{SIDE}resampled_foldlabel_"
    if not folder.is_dir():
        return []
    return sorted(path.name[len(prefix) : -len(".nii.gz")] for path in folder.glob(f"{prefix}*.nii.gz"))


@pytest.fixture
def raising_masker(monkeypatch):
    """FoldLabelMasker.mask_one_file raises RuntimeError('boom') from KeyError for BAD_SUBJECT and masks the
    others; the raised exceptions are collected."""
    raised = []
    original = mrf.FoldLabelMasker.mask_one_file

    def mask_one_file(self, subject):
        if subject == BAD_SUBJECT:
            try:
                raise KeyError("voxel_size")
            except KeyError as cause:
                exc = RuntimeError("boom")
                raised.append(exc)
                raise exc from cause
        original(self, subject)

    monkeypatch.setattr(mrf.FoldLabelMasker, "mask_one_file", mask_one_file)
    return raised


@pytest.fixture
def masked_calls(monkeypatch):
    """Records the subject of each mask_one_file call (serial mode only)."""
    calls = []
    original = mrf.FoldLabelMasker.mask_one_file

    def mask_one_file(self, subject):
        calls.append(subject)
        original(self, subject)

    monkeypatch.setattr(mrf.FoldLabelMasker, "mask_one_file", mask_one_file)
    return calls


def test_failing_subject_is_recorded_in_side_skip_list(tree, raising_masker):
    """REQ-CTFAIL-BDRABCZUK-AD7AD519AD6D: a subject whose masking raises is in the side's skip list, with step
    mask_resampled_foldlabels and the type and message of the raised exception; the other side's list stays empty."""
    _mask(tree)

    assert load_skip_entries(_skip_dir(tree), SIDE) == [
        SkipEntry(subject=BAD_SUBJECT, step=STEP, exc_type="RuntimeError", message="boom")
    ]
    assert load_skip_entries(_skip_dir(tree), "L") == []


def test_failing_subject_error_record_carries_the_raised_exception(tree, raising_masker, caplog):
    """REQ-CTFAIL-BDRABCZUK-AD7AD519AD6D: record_skip receives the raised exception object itself, so the ERROR
    record carries its traceback and __cause__ chain."""
    caplog.set_level(logging.ERROR)
    _mask(tree)

    assert len(raising_masker) == 1
    carried = [record for record in caplog.records if record.exc_info and record.exc_info[1] is raising_masker[0]]
    assert len(carried) == 1
    assert carried[0].levelno == logging.ERROR
    assert isinstance(carried[0].exc_info[1].__cause__, KeyError)


def test_mismatched_subject_is_recorded_with_its_mismatch_message(tree):
    """REQ-CTFAIL-BDRABCZUK-AD7AD519AD6D: a subject over the 40-voxel tolerance (the former bare assert) is
    recorded once, with step mask_resampled_foldlabels and the mismatch message naming the subject."""
    _make_mismatched(tree)

    _mask(tree)

    entries = load_skip_entries(_skip_dir(tree), SIDE)
    assert [(entry.subject, entry.step) for entry in entries] == [(BAD_SUBJECT, STEP)]
    assert "incompatible foldlabel and skeleton" in entries[0].message
    assert BAD_SUBJECT in entries[0].message


def test_subject_over_40_mismatches_is_recorded_through_record_skip(tree):
    """REQ-CTILESTEST-BDRABCZUK-06B45F699FCB (supersedes REQ-CTILESTEST-118): a subject with more than 40 skeleton
    voxels without a foldlabel no longer aborts the run with AssertionError; mask_foldlabel_files returns, the subject
    is recorded through record_skip in the side's skip list with step mask_resampled_foldlabels, and no masked
    foldlabel file is written for it."""
    _make_mismatched(tree)

    _mask(tree)

    entries = load_skip_entries(_skip_dir(tree), SIDE)
    assert [(entry.subject, entry.step) for entry in entries] == [(BAD_SUBJECT, STEP)]
    assert not _masked_path(tree, BAD_SUBJECT).exists()


def test_failing_subject_does_not_stop_the_other_subjects(tree):
    """REQ-CTFAIL-BDRABCZUK-B9ADD12BFC75: compute returns although one subject fails; the other subject's masked
    foldlabel is written and equals the expected masked volume, the failing subject gets none."""
    _make_mismatched(tree)

    _mask(tree)

    assert _masked_subjects(tree) == [GOOD_SUBJECT]
    np.testing.assert_array_equal(_read(_masked_path(tree, GOOD_SUBJECT))[0], _expected_masked())


def test_parallel_failing_subject_is_recorded_in_side_skip_list(tree):
    """REQ-CTFAIL-BDRABCZUK-2745F3650346: in parallel mode (p_map workers) a subject whose masking raises is
    recorded in the side's skip list, and only that subject; the other subject is masked."""
    _make_mismatched(tree)
    _clear_pathos_pools()
    try:
        _mask(tree, parallel=True)
    finally:
        _clear_pathos_pools()

    entries = load_skip_entries(_skip_dir(tree), SIDE)
    assert [(entry.subject, entry.step) for entry in entries] == [(BAD_SUBJECT, STEP)]
    assert _masked_subjects(tree) == [GOOD_SUBJECT]


def test_skip_listed_subject_is_not_masked(tree, masked_calls):
    """REQ-CTFAIL-BDRABCZUK-0CD18CCC3E7C: a subject already in the side's skip list (here from an earlier step)
    gets no masked file and is never handed to mask_one_file; the other subject is masked."""
    record_skip(_skip_dir(tree), SIDE, "resample_files", BAD_SUBJECT, RuntimeError("no resampled file"))

    _mask(tree)

    assert _masked_subjects(tree) == [GOOD_SUBJECT]
    assert masked_calls == [GOOD_SUBJECT]


def test_mask_foldlabel_files_records_failures_in_skip_dir_argument(tree):
    """REQ-CTFAIL-BDRABCZUK-284E1EE37476: the failure is recorded in the skip list under the skip_dir argument,
    not under a folder derived from masked_dir (masked_dir, its side folder or its parent)."""
    _make_mismatched(tree)
    skip_dir = tree["src"].parent / "shared_skip"

    _mask(tree, skip_dir=str(skip_dir))

    assert load_skipped(skip_dir, SIDE) == {BAD_SUBJECT}
    for derived in (tree["masked"], tree["masked"] / SIDE, tree["masked"].parent):
        assert load_skipped(derived, SIDE) == set()


def test_mask_foldlabel_files_reads_skip_list_of_skip_dir_argument(tree, masked_calls):
    """REQ-CTFAIL-BDRABCZUK-284E1EE37476: a subject in the skip list under the skip_dir argument is not masked;
    a skip list under masked_dir (the fallback directory) has no effect when skip_dir is given."""
    skip_dir = tree["src"].parent / "shared_skip"
    record_skip(skip_dir, SIDE, "resample_files", BAD_SUBJECT, RuntimeError("no resampled file"))
    record_skip(tree["masked"], SIDE, "resample_files", GOOD_SUBJECT, RuntimeError("elsewhere"))

    _mask(tree, skip_dir=str(skip_dir))

    assert _masked_subjects(tree) == [GOOD_SUBJECT]
    assert masked_calls == [GOOD_SUBJECT]


def test_mask_foldlabel_files_without_skip_dir_records_failures_in_masked_dir(tree):
    """REQ-CTFAIL-BDRABCZUK-0527B98AAC49: called without skip_dir, mask_foldlabel_files records the failure in the
    skip list under its masked_dir argument (the step's own output folder)."""
    _make_mismatched(tree)

    _mask(tree, skip_dir=None)

    assert load_skipped(tree["masked"], SIDE) == {BAD_SUBJECT}
    assert load_skipped(tree["masked"] / SIDE, SIDE) == set()


def test_mask_foldlabel_files_without_skip_dir_reads_skip_list_of_masked_dir(tree, masked_calls):
    """REQ-CTFAIL-BDRABCZUK-0527B98AAC49: called without skip_dir, a subject in the skip list under masked_dir is
    not masked."""
    record_skip(tree["masked"], SIDE, "resample_files", BAD_SUBJECT, RuntimeError("no resampled file"))

    _mask(tree, skip_dir=None)

    assert _masked_subjects(tree) == [GOOD_SUBJECT]
    assert masked_calls == [GOOD_SUBJECT]


def test_main_passes_skip_dir_option(tmp_path, monkeypatch):
    """REQ-CTFAIL-BDRABCZUK-B14AD6CF9D1F: --skip_dir DIR on the command line reaches mask_foldlabel_files as
    skip_dir=DIR."""
    received = []
    monkeypatch.setattr(mrf, "mask_foldlabel_files", lambda **kwargs: received.append(kwargs))
    shared = str(tmp_path / "shared")

    mrf.main(["-s", str(tmp_path), "-o", str(tmp_path / "masked"), "--skip_dir", shared])

    assert [kwargs.get("skip_dir") for kwargs in received] == [shared]


_OPTIMISED_MASK_SCRIPT = """
import sys
from cortical_tiles.brainvisa import mask_resampled_foldlabels as mrf
print("DEBUG", __debug__)
masker = mrf.FoldLabelMasker(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], False)
try:
    masker.mask_one_file(sys.argv[5])
except Exception as exc:
    print("RAISED", type(exc).__name__)
else:
    print("RETURNED")
"""


def test_mask_one_file_rejects_more_than_40_mismatches_with_assertions_disabled(tree):
    """REQ-CTFAIL-BDRABCZUK-728340959F1D: under python -O (assert statements stripped) mask_one_file still raises
    for a subject whose skeleton has more than 40 voxels without a foldlabel."""
    _make_mismatched(tree)
    (tree["masked"] / SIDE).mkdir(parents=True)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(mrf.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.dirname(package_root), env.get("PYTHONPATH")]))
    argv = [str(tree["src"]), str(tree["skel"]), str(tree["masked"]), SIDE, BAD_SUBJECT]

    result = subprocess.run(
        [sys.executable, "-O", "-c", _OPTIMISED_MASK_SCRIPT, *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "DEBUG False" in lines, result.stdout
    assert any(line.startswith("RAISED ") for line in lines), result.stdout
