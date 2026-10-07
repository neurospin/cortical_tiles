"""Unit tests for cortical_tiles.brainvisa.utils.distmap (REQ-CTILESTEST-86..89).

Both functions shell out to ``AimsChamferDistanceMap`` (available in the pixi env). Inputs are
tiny synthetic 2 mm skeletons written to pytest tmp dirs; ``tempfile.tempdir`` is redirected to
the test's tmp dir so the scratch directory created by ``generate_distmap_from_skeleton_file``
does not land in the system temp dir. The padded case (208x209x210 voxels) runs once per module.

Characterization tests: behaviour recorded on cortical_tiles dev 4141bd8. Tests flagged "DEFECT"
pin current behaviour that looks wrong; it is recorded, not fixed (tests-only rule of TASK-114).
"""

import os
import tempfile

import numpy as np
import pytest
from cortical_tiles.brainvisa.utils import distmap as dm
from soma import aims

SHAPE = (8, 9, 10, 1)
SKELETON_VOXELS = [(4, 4, 4), (2, 3, 5)]
TRANSLATION = (1.0, 2.0, 3.0)
PADDING = 100  # half of the 200 voxels added on each axis


def _write_skeleton(path):
    arr = np.zeros(SHAPE, dtype=np.int16)
    for voxel in SKELETON_VOXELS:
        arr[voxel + (0,)] = 60
    vol = aims.Volume(arr)
    vol.header()["voxel_size"] = [2.0, 2.0, 2.0, 1.0]
    trm = aims.AffineTransformation3d()
    trm.setTranslation(TRANSLATION)
    vol.header()["transformations"] = [trm.toVector()]
    vol.header()["referentials"] = ["Talairach-MNI template-SPM"]
    aims.write(vol, str(path))


@pytest.fixture(scope="module")
def padded(tmp_path_factory):
    """Runs generate_distmap_from_skeleton_file once; returns (skeleton path, distmap path, scratch dir)."""
    tmp = tmp_path_factory.mktemp("distmap_padded")
    skeleton = tmp / "skel.nii.gz"
    _write_skeleton(skeleton)
    scratch = tmp / "scratch"
    scratch.mkdir()
    distmap = tmp / "distmap.nii.gz"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tempfile, "tempdir", str(scratch))
        dm.generate_distmap_from_skeleton_file(str(skeleton), str(distmap))
    return skeleton, distmap, scratch


# --- REQ-CTILESTEST-86: generate_distmap_from_resampled_skeleton -----------------------------------


def test_resampled_distmap_is_distance_to_skeleton(tmp_path):
    """REQ-CTILESTEST-86: float32 map, skeleton shape, ~0 on skeleton voxels, mm distance elsewhere."""
    skeleton = tmp_path / "skel.nii.gz"
    _write_skeleton(skeleton)
    out = tmp_path / "distmap.nii.gz"

    dm.generate_distmap_from_resampled_skeleton(str(skeleton), str(out))

    arr = np.asarray(aims.read(str(out)))
    assert arr.dtype == np.float32
    assert arr.shape == SHAPE
    for voxel in SKELETON_VOXELS:
        assert abs(arr[voxel + (0,)]) < 1e-6
    assert arr[4, 4, 5, 0] == pytest.approx(2.0)  # 6-neighbour at 2 mm
    assert arr[4, 4, 6, 0] == pytest.approx(4.0)
    assert (arr >= 0).all()


# --- REQ-CTILESTEST-87: generate_distmap_from_skeleton_file geometry -------------------------------


def test_padded_distmap_geometry(padded):
    """REQ-CTILESTEST-87: 200 voxels added per axis; skeleton voxel v sits at v+100 with distance 0."""
    _, distmap, _ = padded
    arr = np.asarray(aims.read(str(distmap)))
    assert arr.shape == (SHAPE[0] + 200, SHAPE[1] + 200, SHAPE[2] + 200, 1)
    for x, y, z in SKELETON_VOXELS:
        assert arr[x + PADDING, y + PADDING, z + PADDING, 0] == 0
    assert arr[4 + PADDING, 4 + PADDING, 5 + PADDING, 0] == pytest.approx(2.0)
    assert arr[0, 0, 0, 0] > 300


# --- REQ-CTILESTEST-88: generate_distmap_from_skeleton_file header transformations ----------------


def test_padded_distmap_transformations_are_elementwise_products(padded):
    """REQ-CTILESTEST-88: each transform = element-wise product with inverse padding translation (DEFECT)."""
    skeleton, distmap, _ = padded
    padding = aims.AffineTransformation3d()
    padding.setTranslation([PADDING * 2.0] * 3)
    inverse = np.asarray(padding.inverse().toVector())
    # list(...) copies: a numpy view on a header of a temporary volume would dangle.
    expected = [np.array(list(t)) * inverse for t in aims.read(str(skeleton)).header()["transformations"]]

    written = [np.array(list(t)) for t in aims.read(str(distmap)).header()["transformations"]]

    assert len(written) == len(expected)
    for got, want in zip(written, expected):
        np.testing.assert_allclose(got, want)
    # The Talairach translation (1, 2, 3) becomes (-200, -400, -600), not a composition (-199, -198, -197).
    np.testing.assert_allclose(written[-1][[3, 7, 11]], [-200.0, -400.0, -600.0])


# --- REQ-CTILESTEST-89: generate_distmap_from_skeleton_file scratch dir ----------------------------


def test_padded_distmap_removes_scratch_directory(padded):
    """REQ-CTDEFECTS-11 (inverts REQ-CTILESTEST-89): the mkdtemp scratch dir no longer exists after the call."""
    _, _, scratch = padded
    assert os.listdir(scratch) == []
