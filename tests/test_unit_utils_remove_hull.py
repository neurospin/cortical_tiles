"""Unit tests for cortical_tiles.brainvisa.utils.remove_hull (REQ-CTILESTEST-22, 23).

``remove_hull`` and ``threshold_and_binarize`` are the pure-numpy steps of the hull-free
mesh generation (``create_one_mesh``). Not part of champollion_pipeline stage 2, but the
module was 0% covered. Arrays are 4D (x, y, z, t) as the functions index them.
"""

import numpy as np
import pytest

from cortical_tiles.brainvisa.utils.remove_hull import remove_hull, threshold_and_binarize

EXTERNAL = 11


def _base():
    """7x7x7x1 array, z=0 plane is external (11), everything else internal (0)."""
    arr = np.zeros((7, 7, 7, 1), dtype=np.int16)
    arr[:, :, 0, 0] = EXTERNAL
    return arr


def test_remove_hull_zeroes_voxels_touching_both_internal_and_external():
    """REQ-CTILESTEST-22: a voxel > 11 with both a 0 and an 11 in its 3x3x3 neighbourhood is set to 0."""
    arr = _base()
    arr[3, 3, 1, 0] = 60  # touches z=0 (11) and interior (0) -> removed
    arr[3, 3, 3, 0] = 60  # no 11 within 1 voxel -> kept
    arr[2, 2, 2, 0] = 30  # diagonal-free of z=0 (z=2): no 11 neighbour -> kept

    remove_hull(arr)

    assert arr[3, 3, 1, 0] == 0
    assert arr[3, 3, 3, 0] == 60
    assert arr[2, 2, 2, 0] == 30


def test_remove_hull_keeps_voxels_not_touching_internal():
    """REQ-CTILESTEST-22: a voxel > 11 whose neighbourhood holds 11 but no 0 is kept."""
    arr = np.full((5, 5, 5, 1), EXTERNAL, dtype=np.int16)
    arr[1:4, 1:4, 1:4, 0] = 60  # 3x3x3 block of 60 inside external space: faces see 11 but no 0 -> kept
    arr[2, 2, 2, 0] = 100

    before = arr.copy()
    remove_hull(arr)

    np.testing.assert_array_equal(arr, before)


def test_remove_hull_leaves_values_at_or_below_11_unchanged():
    """REQ-CTILESTEST-22: voxels with value <= 11 are never modified."""
    arr = _base()
    arr[3, 3, 1, 0] = 5  # touches 0 and 11 but not > 11
    before = arr.copy()

    remove_hull(arr)

    np.testing.assert_array_equal(arr, before)


@pytest.mark.parametrize("threshold, expected", [(12, [0, 0, 0, 32767, 32767]), (60, [0, 0, 0, 0, 32767])])
def test_threshold_and_binarize(threshold, expected):
    """REQ-CTILESTEST-23: in place, values < threshold -> 0, values >= threshold -> 32767."""
    arr = np.array([0, 5, 11, 30, 60], dtype=np.int16)

    threshold_and_binarize(arr, threshold)

    np.testing.assert_array_equal(arr, expected)


def test_threshold_and_binarize_default_threshold_is_12():
    """REQ-CTILESTEST-23: default threshold is 12 (11 -> 0, 12 -> 32767)."""
    arr = np.array([11, 12], dtype=np.int16)

    threshold_and_binarize(arr)

    np.testing.assert_array_equal(arr, [0, 32767])
