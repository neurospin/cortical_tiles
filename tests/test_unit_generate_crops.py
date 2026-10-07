"""Unit tests for cortical_tiles.brainvisa.generate_crops (REQ-CTILESTEST-51..65).

champollion_pipeline stage 2 crops every resampled 2 mm volume to each region with
``generate_crops`` (cropping_type 'mask', combine_type False, threshold 0, dilation 5 in
``pipeline_loop_2mm.json``). Inputs here are tiny synthetic volumes and masks written to pytest tmp
dirs; nothing is read from or written to the checkout.

Characterization tests: behaviour recorded on cortical_tiles dev 1b2bd48. Tests flagged "DEFECT"
or "QUIRK" pin current behaviour that looks wrong or surprising; it is recorded, not fixed
(tests-only rule of TASK-114).
"""

import os

import numpy as np
import pandas as pd
import pytest
from cortical_tiles.brainvisa import generate_crops as gc
from soma import aims

SIDE = "R"
SHAPE = (10, 12, 14, 1)
SUBJECTS = ("b2", "a1")  # written out of order on purpose


def _write_volume(path, arr, voxel_size=2.0):
    vol = aims.Volume(np.asarray(arr).astype(np.int16))
    vol.header()["voxel_size"] = [voxel_size, voxel_size, voxel_size, 1.0]
    aims.write(vol, str(path))


def _read(path):
    return np.asarray(aims.read(str(path))).copy()


def _source(seed):
    arr = np.zeros(SHAPE, dtype=np.int16)
    arr[1:8, 2:9, 3:11, 0] = 60
    arr[3, 4, 5 + seed, 0] = 30
    arr[8, 10, 12, 0] = 100  # outside the mask and its dilation
    return arr


@pytest.fixture
def crop_inputs(tmp_path):
    """mask/R/S.C._right.nii.gz + skel/R/Rresampled_skeleton_<s>.nii.gz for two subjects."""
    mask = np.zeros(SHAPE, dtype=np.int16)
    mask[2:5, 3:7, 4:9, 0] = 1
    (tmp_path / "mask" / SIDE).mkdir(parents=True)
    _write_volume(tmp_path / "mask" / SIDE / "S.C._right.nii.gz", mask)
    (tmp_path / "skel" / SIDE).mkdir(parents=True)
    for seed, subject in enumerate(SUBJECTS):
        _write_volume(tmp_path / "skel" / SIDE / f"{SIDE}resampled_skeleton_{subject}.nii.gz", _source(seed))
    return tmp_path


def _generate(root, crop_dir, **kwargs):
    gc.generate_crops(
        src_dir=str(root / "skel"),
        input_type="skeleton",
        crop_dir=str(crop_dir),
        mask_dir=str(root / "mask"),
        side=SIDE,
        list_sulci=["S.C."],
        threshold=0,
        dilation=2,
        **kwargs,
    )


def _bbox(mask):
    nz = np.argwhere(mask[..., 0])
    return nz.min(axis=0), nz.max(axis=0) + 1


# --- REQ-CTILESTEST-51: crop_bbox -------------------------------------------------------------------


def test_crop_bbox_writes_half_open_box(tmp_path):
    """REQ-CTILESTEST-51: the written volume is source[bbmin:bbmax] with values unchanged."""
    src = np.arange(np.prod(SHAPE), dtype=np.int16).reshape(SHAPE)
    _write_volume(tmp_path / "src.nii.gz", src)
    bbmin, bbmax = np.array([1, 2, 3]), np.array([4, 7, 5])

    gc.crop_bbox(str(tmp_path / "src.nii.gz"), str(tmp_path / "out.nii.gz"), bbmin, bbmax)

    np.testing.assert_array_equal(_read(tmp_path / "out.nii.gz"), src[1:4, 2:7, 3:5, :])


# --- REQ-CTILESTEST-52 / 53: crop_mask --------------------------------------------------------------


@pytest.fixture
def masked_case(tmp_path):
    src = np.full(SHAPE, 60, dtype=np.int16)
    _write_volume(tmp_path / "src.nii.gz", src)
    mask_arr = np.zeros(SHAPE, dtype=np.int16)
    mask_arr[2:5, 3:6, 4:7, 0] = 1
    mask_arr[3, 4, 5, 0] = 0  # hole inside the box
    mask = aims.Volume(mask_arr)
    return tmp_path, src, mask_arr, mask


def test_crop_mask_zeroes_voxels_outside_mask(masked_case):
    """REQ-CTILESTEST-52: no_mask False -> source[bbmin:bbmax] with voxels where mask == 0 set to 0."""
    tmp_path, src, mask_arr, mask = masked_case
    bbmin, bbmax = np.array([1, 2, 3]), np.array([6, 7, 8])

    gc.crop_mask(str(tmp_path / "src.nii.gz"), str(tmp_path / "out.nii.gz"), mask, bbmin, bbmax, SIDE, no_mask=False)

    expected = np.where(mask_arr != 0, src, 0)[1:6, 2:7, 3:8, :]
    np.testing.assert_array_equal(_read(tmp_path / "out.nii.gz"), expected)


def test_crop_mask_no_mask_keeps_all_voxels(masked_case):
    """REQ-CTILESTEST-53: no_mask True -> source[bbmin:bbmax] unchanged, mask ignored."""
    tmp_path, src, _, mask = masked_case
    bbmin, bbmax = np.array([1, 2, 3]), np.array([6, 7, 8])

    gc.crop_mask(str(tmp_path / "src.nii.gz"), str(tmp_path / "out.nii.gz"), mask, bbmin, bbmax, SIDE, no_mask=True)

    np.testing.assert_array_equal(_read(tmp_path / "out.nii.gz"), src[1:6, 2:7, 3:8, :])


# --- REQ-CTILESTEST-54 / 55: generator naming -------------------------------------------------------

GENERATORS = [
    (gc.SkeletonCropGenerator, "skeleton", "crops", "skeleton"),
    (gc.FoldLabelCropGenerator, "foldlabel", "labels", "label"),
    (gc.ExtremitiesCropGenerator, "extremities", "extremities", "extremities"),
    (gc.DistMapCropGenerator, "distmap", "distmaps", "distmap"),
]


@pytest.mark.parametrize("cls, kind, subdir, _npy", GENERATORS)
def test_crop_generator_file_naming(tmp_path, cls, kind, subdir, _npy):
    """REQ-CTILESTEST-54: source <src>/<side>/<side>resampled_<t>_<s>, crop <crop>/<side><dir>/<s>_cropped_<t>."""
    gen = cls(src_dir="/src", crop_dir=str(tmp_path), side=SIDE, list_sulci="S.C.")
    subject = {"subject": "sub", "side": SIDE}

    assert gen.src_file % subject == f"/src/R/Rresampled_{kind}_sub.nii.gz"
    assert gen.cropped_samples_dir == os.path.join(str(tmp_path), f"R{subdir}")
    assert gen.cropped_file % subject == f"sub_cropped_{kind}.nii.gz"
    assert gen.list_sulci == ["S.C._right"]


@pytest.mark.parametrize("cls, _kind, _subdir, npy", GENERATORS)
def test_crop_generator_stacked_array_name(tmp_path, cls, _kind, _subdir, npy):
    """REQ-CTILESTEST-55: stacked array basename is <side>skeleton/label/extremities/distmap."""
    gen = cls(src_dir="/src", crop_dir=str(tmp_path), side=SIDE, list_sulci="S.C.")
    assert gen.file_basename_npy == f"R{npy}"


# --- REQ-CTILESTEST-56 / 57: argument validation ----------------------------------------------------


@pytest.mark.parametrize("input_type", ["skeletons", "label", ""])
def test_generate_crops_rejects_unknown_input_type(tmp_path, input_type):
    """REQ-CTILESTEST-56: unknown input_type -> ValueError."""
    with pytest.raises(ValueError):
        gc.generate_crops(src_dir=str(tmp_path), input_type=input_type, crop_dir=str(tmp_path))


def test_unknown_cropping_type_raises_value_error(tmp_path):
    """REQ-CTILESTEST-57: compute_bounding_box_or_mask with an unknown cropping_type -> ValueError."""
    gen = gc.SkeletonCropGenerator(src_dir=str(tmp_path), crop_dir=str(tmp_path), cropping_type="sphere")
    with pytest.raises(ValueError):
        gen.compute_bounding_box_or_mask(nb_subjects=-1)


# --- REQ-CTILESTEST-58 / 59 / 60: source errors -----------------------------------------------------


def test_crop_files_missing_source_dir_raises_not_a_directory(tmp_path):
    """REQ-CTILESTEST-58: <src_dir>/<side> missing -> NotADirectoryError."""
    gen = gc.SkeletonCropGenerator(src_dir=str(tmp_path / "nope"), crop_dir=str(tmp_path / "crops"), side=SIDE)
    with pytest.raises(NotADirectoryError):
        gen.crop_files(nb_subjects=-1)


def test_crop_files_without_nifti_raises_value_error(tmp_path):
    """REQ-CTDEFECTS-07 (inverts REQ-CTILESTEST-59): <src_dir>/<side> without .nii.gz -> ValueError 'no nifti files'."""
    (tmp_path / "skel" / SIDE).mkdir(parents=True)
    (tmp_path / "skel" / SIDE / "readme.txt").write_text("")
    gen = gc.SkeletonCropGenerator(src_dir=str(tmp_path / "skel"), crop_dir=str(tmp_path / "crops"), side=SIDE)
    with pytest.raises(ValueError, match="no nifti files"):
        gen.crop_files(nb_subjects=-1)


def test_crop_one_file_missing_source_raises_file_not_found(tmp_path):
    """REQ-CTILESTEST-60: cropping a subject without source file -> FileNotFoundError."""
    gen = gc.SkeletonCropGenerator(src_dir=str(tmp_path), crop_dir=str(tmp_path / "crops"), side=SIDE)
    with pytest.raises(FileNotFoundError):
        gen.crop_one_file("absent")


# --- REQ-CTILESTEST-61 / 62 / 63: end-to-end mask cropping -----------------------------------------


def test_generate_crops_masks_and_crops_each_subject(crop_inputs):
    """REQ-CTILESTEST-61: crop = source within the bbox of the computed mask, 0 where that mask is 0."""
    crop_dir = crop_inputs / "crops"
    _generate(crop_inputs, crop_dir, no_mask=False)

    mask = _read(crop_dir / f"{SIDE}mask_skeleton.nii.gz")
    bbmin, bbmax = _bbox(mask)
    box = tuple(slice(lo, hi) for lo, hi in zip(bbmin[:3], bbmax[:3]))
    for seed, subject in enumerate(SUBJECTS):
        crop = _read(crop_dir / f"{SIDE}crops" / f"{subject}_cropped_skeleton.nii.gz")
        expected = np.where(mask != 0, _source(seed), 0)[box]
        np.testing.assert_array_equal(crop, expected)
    np.testing.assert_array_equal(_read(crop_dir / f"{SIDE}mask_cropped.nii.gz"), mask[box])


def test_generate_crops_stacks_subjects_in_sorted_order(crop_inputs):
    """REQ-CTILESTEST-62: <side>skeleton.npy stacks crops in ascending subject order, as listed in the CSV."""
    crop_dir = crop_inputs / "crops"
    _generate(crop_inputs, crop_dir, no_mask=False)

    stacked = np.load(crop_dir / f"{SIDE}skeleton.npy")
    subjects = pd.read_csv(crop_dir / f"{SIDE}skeleton_subject.csv", dtype=str)["Subject"].tolist()
    assert subjects == sorted(SUBJECTS)
    assert np.load(crop_dir / "sub_id.npy").tolist() == sorted(SUBJECTS)
    assert stacked.shape[0] == len(SUBJECTS)
    for index, subject in enumerate(subjects):
        np.testing.assert_array_equal(
            stacked[index], _read(crop_dir / f"{SIDE}crops" / f"{subject}_cropped_skeleton.nii.gz")
        )


def test_generate_crops_default_does_not_apply_mask(crop_inputs):
    """REQ-CTILESTEST-63: without no_mask, generate_crops keeps voxels outside the mask (QUIRK: default True)."""
    crop_dir = crop_inputs / "crops"
    _generate(crop_inputs, crop_dir)

    mask = _read(crop_dir / f"{SIDE}mask_skeleton.nii.gz")
    bbmin, bbmax = _bbox(mask)
    box = tuple(slice(lo, hi) for lo, hi in zip(bbmin[:3], bbmax[:3]))
    crop = _read(crop_dir / f"{SIDE}crops" / f"{SUBJECTS[0]}_cropped_skeleton.nii.gz")
    np.testing.assert_array_equal(crop, _source(0)[box])
    assert np.count_nonzero(crop[mask[box] == 0]) > 0


# --- REQ-CTILESTEST-64 / 65: quality checks ---------------------------------------------------------


def _write_qc_case(crop_dir, kind, skeleton, other, skeleton_subjects, other_subjects):
    os.makedirs(crop_dir, exist_ok=True)
    np.save(os.path.join(crop_dir, f"{SIDE}skeleton.npy"), skeleton)
    np.save(os.path.join(crop_dir, f"{SIDE}{kind}.npy"), other)
    pd.DataFrame({"Subject": skeleton_subjects}).to_csv(
        os.path.join(crop_dir, f"{SIDE}skeleton_subject.csv"), index=False
    )
    pd.DataFrame({"Subject": other_subjects}).to_csv(os.path.join(crop_dir, f"{SIDE}{kind}_subject.csv"), index=False)


def _qc_arrays():
    skeleton = np.zeros((2, 3, 3, 3, 1), dtype=np.int16)
    skeleton[:, 1, 1, 1, 0] = 60
    label = np.where(skeleton != 0, 1003, 0).astype(np.int16)
    return skeleton, label


def test_foldlabel_quality_checks_accept_consistent_arrays(tmp_path):
    """REQ-CTILESTEST-64: same shape, same non-zero support, equal CSVs of matching length -> no error."""
    skeleton, label = _qc_arrays()
    _write_qc_case(str(tmp_path), "label", skeleton, label, ["a", "b"], ["a", "b"])
    gc.quality_checks(str(tmp_path), SIDE)


@pytest.mark.parametrize("defect", ["shape", "label_only_voxel", "skeleton_only_voxel", "csv", "count"])
def test_foldlabel_quality_checks_reject_inconsistent_arrays(tmp_path, defect):
    """REQ-CTILESTEST-64: any shape, non-zero support, CSV or count mismatch -> AssertionError."""
    skeleton, label = _qc_arrays()
    skeleton_subjects, label_subjects = ["a", "b"], ["a", "b"]
    if defect == "shape":
        label = label[:, :2]
    elif defect == "label_only_voxel":
        label[0, 0, 0, 0, 0] = 5
    elif defect == "skeleton_only_voxel":
        skeleton[0, 0, 0, 0, 0] = 60
    elif defect == "csv":
        label_subjects = ["a", "c"]
    elif defect == "count":
        skeleton_subjects = label_subjects = ["a", "b", "c"]
    _write_qc_case(str(tmp_path), "label", skeleton, label, skeleton_subjects, label_subjects)
    with pytest.raises(AssertionError):
        gc.quality_checks(str(tmp_path), SIDE)


def test_extremities_quality_checks_accept_consistent_arrays(tmp_path):
    """REQ-CTILESTEST-65: same shape and equal CSVs of matching length -> no error (supports may differ)."""
    skeleton, _ = _qc_arrays()
    extremities = np.zeros_like(skeleton)
    extremities[0, 0, 0, 0, 0] = 1
    _write_qc_case(str(tmp_path), "extremities", skeleton, extremities, ["a", "b"], ["a", "b"])
    gc.quality_checks_extremities(str(tmp_path), SIDE)


@pytest.mark.parametrize("defect", ["shape", "csv", "count"])
def test_extremities_quality_checks_reject_inconsistent_arrays(tmp_path, defect):
    """REQ-CTILESTEST-65: shape, CSV or count mismatch -> AssertionError."""
    skeleton, _ = _qc_arrays()
    extremities = np.zeros_like(skeleton)
    skeleton_subjects, extremities_subjects = ["a", "b"], ["a", "b"]
    if defect == "shape":
        extremities = extremities[:, :2]
    elif defect == "csv":
        extremities_subjects = ["a", "c"]
    elif defect == "count":
        skeleton_subjects = extremities_subjects = ["a", "b", "c"]
    _write_qc_case(str(tmp_path), "extremities", skeleton, extremities, skeleton_subjects, extremities_subjects)
    with pytest.raises(AssertionError):
        gc.quality_checks_extremities(str(tmp_path), SIDE)
