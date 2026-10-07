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
import os

import numpy as np
import pytest
from cortical_tiles.brainvisa import mask_resampled_foldlabels as mrf
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


# --- REQ-CTILESTEST-116..118: mask_one_file -------------------------------------------------------


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


def test_mask_one_file_rejects_more_than_40_mismatches(tree):
    """REQ-CTILESTEST-118: more than 40 skeleton voxels without label -> AssertionError, nothing written."""
    skeleton = _skeleton()
    skeleton[0:5, 0:3, 9:12, 0] = 60  # 45 more skeleton voxels, none of them labelled
    _write(tree["skel"] / SIDE / f"{SIDE}resampled_skeleton_s1.nii.gz", skeleton)
    (tree["masked"] / SIDE).mkdir(parents=True)

    with pytest.raises(AssertionError, match="incompatible foldlabel and skeleton"):
        _masker(tree).mask_one_file("s1")
    assert not _masked_path(tree, "s1").exists()


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
