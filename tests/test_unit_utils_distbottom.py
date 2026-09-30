"""Unit tests for cortical_tiles.brainvisa.utils.distbottom (REQ-CTILESTEST-19, 20).

champollion_pipeline stage 2 writes a distbottom crop next to every skeleton crop
(``skip_distbottom`` false by default); ``generate_distbottom_crops`` computes each subject's
distbottom volume with ``generate_distbottom(skeleton_file, distbottom_file)`` on the
resampled skeleton. These tests feed it a tiny synthetic skeleton file written to a
pytest tmp dir, so the (nondeterministic, TASK-126) resampling step is not involved.
"""

import os

import numpy as np
import pytest
from soma import aims

from cortical_tiles.brainvisa.utils.distbottom import generate_distbottom

BACKGROUND = 32501


def _run(tmp_path, arr, voxel_size):
    vol = aims.Volume(*arr.shape[:3], 1, dtype="S16")
    vol.header()["voxel_size"] = [voxel_size] * 3 + [1.0]
    np.asarray(vol)[:] = arr
    src = os.path.join(str(tmp_path), "skeleton.nii.gz")
    dst = os.path.join(str(tmp_path), "distbottom.nii.gz")
    aims.write(vol, src)
    generate_distbottom(src, dst)
    return aims.read(dst)


def _chain_skeleton():
    """Line along x at (y, z) = (1, 1): bottom at x=1, then 5 simple-surface voxels, a junction voxel,
    an outside (11) voxel at x=8; x=0 is background (0)."""
    arr = np.zeros((9, 3, 3, 1), dtype=np.int16)
    arr[1, 1, 1, 0] = 30
    arr[2:7, 1, 1, 0] = 60
    arr[7, 1, 1, 0] = 110
    arr[8, 1, 1, 0] = 11
    return arr


def test_distbottom_bottom_is_zero_and_outside_is_background(tmp_path):
    """REQ-CTILESTEST-19: bottom (30) -> 0; skeleton values 0 and 11 -> 32501; int16 file."""
    arr = _chain_skeleton()
    out = np.asarray(_run(tmp_path, arr, 2.0))

    assert out.dtype == np.int16
    assert out.shape == arr.shape
    assert out[1, 1, 1, 0] == 0
    assert (out[arr == 0] == BACKGROUND).all()
    assert out[8, 1, 1, 0] == BACKGROUND


@pytest.mark.parametrize("voxel_size", [1.0, 2.0])
def test_distbottom_is_50_times_distance_in_mm_along_a_chain(tmp_path, voxel_size):
    """REQ-CTILESTEST-20: other skeleton voxels -> 50 x distance (mm) to the bottom through the skeleton."""
    out = np.asarray(_run(tmp_path, _chain_skeleton(), voxel_size))

    steps = np.arange(1, 7)  # x = 2..7 (5 simple-surface voxels and the junction voxel)
    np.testing.assert_array_equal(out[2:8, 1, 1, 0], (50 * voxel_size * steps).astype(np.int16))
