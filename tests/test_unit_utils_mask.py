"""Unit tests for cortical_tiles.brainvisa.utils.mask (REQ-CTILESTEST-07..13).

champollion_pipeline stage 2 crops every region with ``cropping_type='mask'`` and
``combine_type=False`` (``pipeline_loop_2mm.json``: threshold 0, dilation 5), so
``generate_crops`` calls ``compute_simple_mask`` then ``compute_bbox_mask``. These tests
drive both on tiny synthetic sulcus masks written to a pytest tmp dir; nothing is read
from or written to the checkout. ``compute_intersection_mask`` (``cropping_type=
'mask_intersect'``) is covered too. ``compute_centered_mask`` is not: it is only used for
the CINGULATE. region (not in the pipeline's region list) and writes fixed /tmp files.

The morphological dilation is a chamfer approximation of a Euclidean ball, so its exact
voxel set is not pinned; the tests bound it instead: every seed voxel is kept and no
voxel farther than the dilation radius (in mm) from a seed is set.
"""

import os

import numpy as np
import pytest
from scipy import ndimage
from soma import aims

from cortical_tiles.brainvisa.utils.mask import (
    compute_bbox_mask,
    compute_intersection_mask,
    compute_simple_mask,
)

SHAPE = (12, 13, 14)
VOXEL_SIZE = (2.0, 2.0, 2.0)
SIDE = "R"


def _write_mask(mask_dir, name, arr, voxel_size=VOXEL_SIZE):
    """Writes a 3D int16 array as ``<mask_dir>/R/<name>.nii.gz`` (the layout the functions read)."""
    vol = aims.Volume(*arr.shape, 1, dtype="S16")
    vol.header()["voxel_size"] = list(voxel_size) + [1.0]
    np.asarray(vol)[..., 0] = arr
    os.makedirs(os.path.join(mask_dir, SIDE), exist_ok=True)
    aims.write(vol, os.path.join(mask_dir, SIDE, f"{name}.nii.gz"))


def _distance_mm_to(seed, voxel_size=VOXEL_SIZE):
    """Euclidean distance in mm from every voxel to the nearest True voxel of ``seed``."""
    return ndimage.distance_transform_edt(~seed, sampling=voxel_size)


def _mask3d(vol):
    return np.asarray(vol)[..., 0]


# --- REQ-CTILESTEST-07 / 08: compute_bbox_mask ------------------------------------------------


def test_bbox_mask_returns_start_and_exclusive_stop_of_ones():
    """REQ-CTILESTEST-07: bbmin = first index holding a 1, bbmax = last such index + 1, per axis."""
    arr = np.zeros((10, 11, 12, 1), dtype=np.int16)
    arr[2, 5, 7, 0] = 1
    arr[6, 3, 9, 0] = 1
    arr[4, 8, 10, 0] = 1

    bbmin, bbmax = compute_bbox_mask(arr)

    np.testing.assert_array_equal(bbmin, [2, 3, 7, 0])
    np.testing.assert_array_equal(bbmax, [7, 9, 11, 1])


def test_bbox_mask_rejects_all_zero_array():
    """REQ-CTILESTEST-08: an all-zero array raises ValueError."""
    with pytest.raises(ValueError):
        compute_bbox_mask(np.zeros((4, 4, 4, 1), dtype=np.int16))


# --- REQ-CTDEFECTS128-01: compute_bbox_mask on non-binary input (TASK-128 defect a) ----------


def test_bbox_mask_covers_every_nonzero_value():
    """REQ-CTDEFECTS128-01: the box spans all non-zero voxels whatever their value, not only the 1s."""
    arr = np.zeros((10, 11, 12, 1), dtype=np.int16)
    arr[4, 4, 4, 0] = 1
    arr[1, 6, 9, 0] = 3
    arr[8, 2, 5, 0] = 7
    arr[5, 9, 11, 0] = -2

    bbmin, bbmax = compute_bbox_mask(arr)

    np.testing.assert_array_equal(bbmin, [1, 2, 4, 0])
    np.testing.assert_array_equal(bbmax, [9, 10, 12, 1])


def test_bbox_mask_handles_array_without_any_value_one():
    """REQ-CTDEFECTS128-01: an array whose non-zero values are all != 1 still gets its full box."""
    arr = np.zeros((6, 7, 8, 1), dtype=np.int16)
    arr[1, 2, 3, 0] = 12
    arr[4, 5, 6, 0] = 20

    bbmin, bbmax = compute_bbox_mask(arr)

    np.testing.assert_array_equal(bbmin, [1, 2, 3, 0])
    np.testing.assert_array_equal(bbmax, [5, 6, 7, 1])


# --- REQ-CTILESTEST-09..11: compute_simple_mask -----------------------------------------------


@pytest.fixture
def single_sulcus(tmp_path):
    """One sulcus mask: two voxels above threshold 2 (values 3, 9), two at or below it (values 1, 2)."""
    arr = np.zeros(SHAPE, dtype=np.int16)
    arr[3, 4, 5] = 3
    arr[8, 8, 8] = 9
    arr[6, 2, 11] = 1
    arr[1, 10, 2] = 2
    _write_mask(str(tmp_path), "S.A._right", arr)
    seed = arr > 2
    return str(tmp_path), seed


@pytest.mark.parametrize("dilation", [1.0, 5.0])
def test_simple_mask_single_sulcus_is_thresholded_then_dilated(single_sulcus, dilation):
    """REQ-CTILESTEST-09: 0/1 mask, 1 on every voxel > threshold, 0 farther than dilation mm from those."""
    mask_dir, seed = single_sulcus

    vol, _, _ = compute_simple_mask(["S.A._right"], SIDE, mask_dir=mask_dir, dilation=dilation, threshold=2)
    result = _mask3d(vol)

    assert set(np.unique(result)) <= {0, 1}
    assert result[seed].all(), "a voxel above threshold was dropped"
    assert not result[_distance_mm_to(seed) > dilation].any(), "a voxel beyond the dilation radius was set"
    # Discriminates "no dilation": at 5 mm the face neighbours (2 mm away) of each seed are set.
    if dilation >= 2.0:
        assert result[4, 4, 5] == 1 and result[7, 8, 8] == 1


def test_simple_mask_keeps_first_mask_geometry(tmp_path):
    """REQ-CTILESTEST-10: result has the first sulcus mask's shape and voxel size."""
    arr = np.zeros((9, 10, 11), dtype=np.int16)
    arr[4, 5, 6] = 1
    _write_mask(str(tmp_path), "S.A._right", arr, voxel_size=(1.5, 1.5, 1.5))
    _write_mask(str(tmp_path), "S.B._right", arr, voxel_size=(1.5, 1.5, 1.5))

    vol, _, _ = compute_simple_mask(
        ["S.A._right", "S.B._right"], SIDE, mask_dir=str(tmp_path), dilation=5.0, threshold=0
    )

    assert tuple(vol.shape) == (9, 10, 11, 1)
    assert list(vol.header()["voxel_size"])[:3] == [1.5, 1.5, 1.5]
    assert np.asarray(vol).dtype == np.int16


def test_simple_mask_bbox_matches_returned_mask(single_sulcus):
    """REQ-CTILESTEST-11: returned bbmin/bbmax equal compute_bbox_mask of the returned mask."""
    mask_dir, _ = single_sulcus

    vol, bbmin, bbmax = compute_simple_mask(["S.A._right"], SIDE, mask_dir=mask_dir, dilation=5.0, threshold=2)

    nonzero = np.argwhere(np.asarray(vol) > 0)
    np.testing.assert_array_equal(bbmin, nonzero.min(axis=0))
    np.testing.assert_array_equal(bbmax, nonzero.max(axis=0) + 1)


def test_simple_mask_several_sulci_thresholds_the_sum(tmp_path):
    """REQ-CTILESTEST-12: with several sulci, seeds are the voxels where the SUM of mask values > threshold."""
    a = np.zeros(SHAPE, dtype=np.int16)
    b = np.zeros(SHAPE, dtype=np.int16)
    a[3, 3, 3] = 1  # shared voxel: sum 2 > 1 -> kept
    b[3, 3, 3] = 1
    a[9, 9, 9] = 1  # only in a: sum 1 <= 1 -> dropped
    b[2, 10, 11] = 2  # only in b but value 2 > 1 -> kept
    _write_mask(str(tmp_path), "S.A._right", a)
    _write_mask(str(tmp_path), "S.B._right", b)

    # dilation 1 mm < 2 mm voxel: the dilation adds nothing, so result == seeds exactly
    vol, _, _ = compute_simple_mask(
        ["S.A._right", "S.B._right"], SIDE, mask_dir=str(tmp_path), dilation=1.0, threshold=1
    )
    result = _mask3d(vol)

    assert result[3, 3, 3] == 1
    assert result[2, 10, 11] == 1
    assert result[9, 9, 9] == 0
    assert result.sum() == 2


# --- REQ-CTILESTEST-13: compute_intersection_mask ---------------------------------------------


def test_intersection_mask_keeps_voxels_nonzero_in_every_sulcus(tmp_path):
    """REQ-CTILESTEST-13: seeds are the voxels non-zero in every sulcus mask (threshold 0)."""
    a = np.zeros(SHAPE, dtype=np.int16)
    b = np.zeros(SHAPE, dtype=np.int16)
    c = np.zeros(SHAPE, dtype=np.int16)
    a[3, 3, 3] = b[3, 3, 3] = c[3, 3, 3] = 4  # in all three -> kept
    a[5, 6, 7] = 7
    b[5, 6, 7] = 1
    c[5, 6, 7] = 2  # in all three, other values -> kept
    a[9, 9, 9] = b[9, 9, 9] = 5  # missing from c -> dropped
    c[1, 1, 1] = 9  # only in c -> dropped
    for name, arr in (("S.A._right", a), ("S.B._right", b), ("S.C._right", c)):
        _write_mask(str(tmp_path), name, arr)

    vol, bbmin, bbmax = compute_intersection_mask(
        ["S.A._right", "S.B._right", "S.C._right"], SIDE, mask_dir=str(tmp_path), dilation=1.0, threshold=0
    )
    result = _mask3d(vol)

    expected = np.zeros(SHAPE, dtype=np.int16)
    expected[3, 3, 3] = expected[5, 6, 7] = 1
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(bbmin, [3, 3, 3, 0])
    np.testing.assert_array_equal(bbmax, [6, 7, 8, 1])
