"""Unit tests for cortical_tiles.brainvisa.utils.mask (single-sulcus intersection,
compute_centered_mask) and utils.bbox.compute_max_box (REQ-CTILESTEST-148..153).

``generate_crops.CropGenerator.compute_bounding_box_or_mask`` reaches
``compute_intersection_mask`` (cropping_type 'mask_intersect'), ``compute_centered_mask``
(cropping_type 'mask' with combine_type, set for region CINGULATE.) and ``compute_max_box``
(cropping_type 'bbox'). Inputs are tiny synthetic masks / json files in a pytest tmp dir.
``compute_centered_mask`` writes fixed /tmp files (TASK-128 defect d); ``aims.write`` is
replaced by a recorder during those calls so nothing outside tmp_path is written.
"""

import json
import os

import cortical_tiles.brainvisa.utils.mask as mk
import numpy as np
import pytest
from cortical_tiles.brainvisa.utils.bbox import compute_max_box
from soma import aims

SIDE = "R"
SHAPE = (20, 20, 20)


def _write_mask(mask_dir, name, arr, voxel_size=(1.0, 1.0, 1.0)):
    vol = aims.Volume(*arr.shape, 1, dtype="S16")
    vol.header()["voxel_size"] = list(voxel_size) + [1.0]
    np.asarray(vol)[..., 0] = arr
    os.makedirs(os.path.join(mask_dir, SIDE), exist_ok=True)
    aims.write(vol, os.path.join(mask_dir, SIDE, f"{name}.nii.gz"))


@pytest.fixture
def mask_dir(tmp_path):
    d = str(tmp_path / "masks")
    a = np.zeros(SHAPE, np.int16)
    a[5:8, 5:8, 5:8] = 12
    a[15, 15, 15] = 5  # below the threshold used below
    b = np.zeros(SHAPE, np.int16)
    b[7:10, 7:10, 7:10] = 20
    b[2, 2, 2] = 3
    _write_mask(d, "A", a)
    _write_mask(d, "B", b)
    return d


# --- REQ-CTILESTEST-148: single-sulcus intersection ---------------------------------------------


@pytest.mark.parametrize("dilation, threshold", [(2, 10), (0.5, 0), (3, 4)])
def test_single_sulcus_intersection_equals_simple_mask(mask_dir, dilation, threshold):
    """REQ-CTILESTEST-148: one sulcus -> compute_intersection_mask == compute_simple_mask (array, bbmin, bbmax)."""
    inter, imin, imax = mk.compute_intersection_mask(
        ["A"], SIDE, mask_dir=mask_dir, dilation=dilation, threshold=threshold
    )
    simple, smin, smax = mk.compute_simple_mask(["A"], SIDE, mask_dir=mask_dir, dilation=dilation, threshold=threshold)
    np.testing.assert_array_equal(np.asarray(inter), np.asarray(simple))
    np.testing.assert_array_equal(imin, smin)
    np.testing.assert_array_equal(imax, smax)


def test_single_sulcus_intersection_drops_voxels_at_or_below_threshold(mask_dir):
    """REQ-CTILESTEST-148 (guard): the sub-threshold voxel of A does not survive."""
    inter, _, _ = mk.compute_intersection_mask(["A"], SIDE, mask_dir=mask_dir, dilation=0.5, threshold=10)
    arr = np.asarray(inter)[..., 0]
    assert arr[15, 15, 15] == 0 and arr[6, 6, 6] != 0


# --- REQ-CTILESTEST-149..151: compute_centered_mask ----------------------------------------------


@pytest.fixture
def centered(mask_dir, monkeypatch):
    written = []
    monkeypatch.setattr(mk.aims, "write", lambda obj, path, *args, **kwargs: written.append(path))
    result = mk.compute_centered_mask(["A", "B"], SIDE, mask_dir=mask_dir)
    monkeypatch.undo()
    return result, written


def test_centered_mask_writes_intermediates_to_fixed_tmp_paths(centered):
    """REQ-CTILESTEST-149 (DEFECT): intermediate/final volumes are written to five fixed /tmp paths."""
    _, written = centered
    assert written == [
        "/tmp/eligible_mask_1.nii.gz",
        "/tmp/eligible_mask_2.nii.gz",
        "/tmp/intersec_mask.nii.gz",
        "/tmp/intersec_mask_dilated.nii.gz",
        "/tmp/mask_result.nii.gz",
    ]


def test_centered_mask_voxel_size_is_hardcoded_2mm(centered):
    """REQ-CTILESTEST-150 (DEFECT): returned mask has voxel_size 2 mm although the inputs are 1 mm."""
    (mask, _, _), _ = centered
    assert list(mask.header()["voxel_size"])[:3] == [2.0, 2.0, 2.0]


def test_centered_mask_is_binary_and_bbox_matches_it(centered):
    """REQ-CTILESTEST-151: returned mask is non-empty, values in {0,1}; bbox == compute_bbox_mask(mask)."""
    (mask, bbmin, bbmax), _ = centered
    arr = np.asarray(mask)
    assert set(np.unique(arr).tolist()) == {0, 1}
    exp_min, exp_max = mk.compute_bbox_mask(arr)
    np.testing.assert_array_equal(bbmin, exp_min)
    np.testing.assert_array_equal(bbmax, exp_max)


# --- REQ-CTILESTEST-152 / 153: compute_max_box ---------------------------------------------------


@pytest.fixture
def bbox_dir(tmp_path):
    d = tmp_path / "bbox"
    (d / "L").mkdir(parents=True)
    boxes = {
        "S1": {
            "bbmin_voxel": [10, -10, -100],
            "bbmax_voxel": [20, 0, -50],
            "bbmin_AIMS_Talairach": [1.5, 2.5, 3.5],
            "bbmax_AIMS_Talairach": [4.0, 5.0, 6.0],
        },
        "S2": {
            "bbmin_voxel": [0, 100, 0],
            "bbmax_voxel": [10, 110, 10],
            "bbmin_AIMS_Talairach": [-1.0, 3.0, 0.5],
            "bbmax_AIMS_Talairach": [7.0, 2.0, 6.5],
        },
    }
    for name, content in boxes.items():
        (d / "L" / f"{name}.json").write_text(json.dumps(content))
    return str(d)


def test_max_box_voxel_is_union_of_sulcus_boxes(bbox_dir):
    """REQ-CTILESTEST-152: voxel box = per-axis min of bbmin_voxel and max of bbmax_voxel over the sulci."""
    bbmin, bbmax = compute_max_box(["S1", "S2"], "L", src_dir=bbox_dir)
    assert bbmin.tolist() == [0, -10, -100]
    assert bbmax.tolist() == [20, 110, 10]


def test_max_box_talairach_uses_talairach_keys(bbox_dir):
    """REQ-CTILESTEST-153: talairach_box=True reads bbmin/bbmax_AIMS_Talairach instead."""
    bbmin, bbmax = compute_max_box(["S1", "S2"], "L", talairach_box=True, src_dir=bbox_dir)
    assert bbmin.tolist() == [-1.0, 2.5, 0.5]
    assert bbmax.tolist() == [7.0, 5.0, 6.5]


def test_max_box_single_sulcus_returns_its_box(bbox_dir):
    """REQ-CTILESTEST-152: a one-sulcus list returns that sulcus' own voxel box."""
    bbmin, bbmax = compute_max_box(["S2"], "L", src_dir=bbox_dir)
    assert bbmin.tolist() == [0, 100, 0]
    assert bbmax.tolist() == [10, 110, 10]
