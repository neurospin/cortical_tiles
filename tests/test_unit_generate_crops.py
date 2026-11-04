"""Unit tests for cortical_tiles.brainvisa.generate_crops (REQ-CTILESTEST-51..65).

champollion_pipeline stage 2 crops every resampled 2 mm volume to each region with
``generate_crops`` (cropping_type 'mask', combine_type False, threshold 0, dilation 5 in
``pipeline_loop_2mm.json``). Inputs here are tiny synthetic volumes and masks written to pytest tmp
dirs; nothing is read from or written to the checkout.

Characterization tests: behaviour recorded on cortical_tiles dev 1b2bd48. Tests flagged "DEFECT"
or "QUIRK" pin current behaviour that looks wrong or surprising; it is recorded, not fixed
(tests-only rule of TASK-114).
"""

import logging
import os

import numpy as np
import pandas as pd
import pytest
from cortical_tiles.brainvisa import generate_crops as gc
from cortical_tiles.brainvisa.utils import skip_list
from pathos import multiprocessing as pathos_mp
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


# --- TASK-CTFAIL-BDRABCZUK-8B4393E2F8A2: crop subject set of a side -------------------------------
# Layout mirrors run_with_params: <out>/<kind dir>/2mm/<side>/<side>resampled_<kind>_<s>.nii.gz; the one
# shared per-side skip list lies in the skip_dir passed to generate_crops (<out> unless a test says otherwise).
# Each test crops into one crop_dir, skeleton pass first, as generate_sulcal_regions does for one region and side.

SEEDS = {"a1": 0, "b2": 1, "c3": 2, "d4": 3}
KIND_DIRS = {"skeleton": "skeletons", "foldlabel": "foldlabels", "extremities": "extremities", "distmap": "distmaps"}
NPY_BASENAMES = {"skeleton": "skeleton", "foldlabel": "label", "extremities": "extremities", "distmap": "distmap"}


def _subject_set_layout(tmp_path, **subjects_by_kind):
    """Writes <tmp>/out/<kind dir>/2mm/R/... volumes for the given subjects and the S.C. mask; returns out."""
    out = tmp_path / "out"
    for kind, subjects in subjects_by_kind.items():
        side_dir = out / KIND_DIRS[kind] / "2mm" / SIDE
        side_dir.mkdir(parents=True)
        for subject in subjects:
            arr = _source(SEEDS[subject])
            if kind == "foldlabel":
                arr = np.where(arr != 0, 1003, 0)
            _write_volume(side_dir / f"{SIDE}resampled_{kind}_{subject}.nii.gz", arr)
    mask = np.zeros(SHAPE, dtype=np.int16)
    mask[2:5, 3:7, 4:9, 0] = 1
    (tmp_path / "mask" / SIDE).mkdir(parents=True)
    _write_volume(tmp_path / "mask" / SIDE / "S.C._right.nii.gz", mask)
    return out


def _crop(tmp_path, out, input_type, skip_dir=None, parallel=False):
    gc.generate_crops(
        src_dir=str(out / KIND_DIRS[input_type] / "2mm"),
        skip_dir=str(out if skip_dir is None else skip_dir),
        input_type=input_type,
        crop_dir=str(tmp_path / "crops"),
        mask_dir=str(tmp_path / "mask"),
        side=SIDE,
        list_sulci=["S.C."],
        threshold=0,
        dilation=2,
        parallel=parallel,
    )


def _stacked_subjects(tmp_path, input_type):
    """Subjects of <side><basename>_subject.csv, after checking <side><basename>.npy has one row per subject."""
    basename = NPY_BASENAMES[input_type]
    subjects = pd.read_csv(tmp_path / "crops" / f"{SIDE}{basename}_subject.csv", dtype=str)["Subject"].tolist()
    assert np.load(tmp_path / "crops" / f"{SIDE}{basename}.npy").shape[0] == len(subjects)
    return subjects


def _skip(out, subject, side=SIDE, step="generate_foldlabels"):
    skip_list.record_skip(
        skip_dir=str(out), side=side, step=step, subject=subject, exc=RuntimeError("graph without fold labels")
    )


def test_crop_subject_set_skip_list_lies_in_skip_dir_argument(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-86ED30A24CAB: skip lists are read from and recorded in the skip_dir argument."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2", "c3"), foldlabel=("b2", "c3"))
    skip_dir = tmp_path / "shared_skip_lists"
    _skip(skip_dir, "c3")

    _crop(tmp_path, out, "skeleton", skip_dir=skip_dir)
    _crop(tmp_path, out, "foldlabel", skip_dir=skip_dir)

    assert "c3" not in _stacked_subjects(tmp_path, "skeleton")
    assert sorted(tmp_path.rglob("skip_list_*.jsonl")) == [skip_dir / f"skip_list_{SIDE}.jsonl"]
    assert "a1" in skip_list.load_skipped(skip_dir=str(skip_dir), side=SIDE)


def test_crop_subject_set_is_two_sided_subjects_minus_skip_list(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-3BB00563E231: skeleton and label CSVs list subjects with both files, minus the skip list."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2", "c3"), foldlabel=("b2", "c3", "d4"))
    _skip(out, "c3")

    _crop(tmp_path, out, "skeleton")
    _crop(tmp_path, out, "foldlabel")

    assert _stacked_subjects(tmp_path, "skeleton") == ["b2"]
    assert _stacked_subjects(tmp_path, "foldlabel") == ["b2"]


@pytest.mark.parametrize("missing", ["foldlabel", "skeleton"])
def test_crop_subject_set_records_one_sided_subject(tmp_path, missing):
    """REQ-CTFAIL-BDRABCZUK-E8172C85579C: a subject with one of skeleton/foldlabel is recorded, step generate_crops."""
    present = "skeleton" if missing == "foldlabel" else "foldlabel"
    out = _subject_set_layout(tmp_path, **{present: ("a1", "b2"), missing: ("b2",)})

    _crop(tmp_path, out, "skeleton")
    _crop(tmp_path, out, "foldlabel")

    entries = skip_list.load_skip_entries(skip_dir=str(out), side=SIDE)
    assert [(entry.subject, entry.step) for entry in entries] == [("a1", "generate_crops")]


@pytest.mark.parametrize("missing", ["foldlabel", "skeleton"])
def test_crop_subject_set_entry_names_missing_input_type(tmp_path, missing):
    """REQ-CTFAIL-BDRABCZUK-A8CAEF729DBF: the recorded entry's message names the missing input type."""
    present = "skeleton" if missing == "foldlabel" else "foldlabel"
    out = _subject_set_layout(tmp_path, **{present: ("a1", "b2"), missing: ("b2",)})

    _crop(tmp_path, out, "skeleton")
    _crop(tmp_path, out, "foldlabel")

    entries = skip_list.load_skip_entries(skip_dir=str(out), side=SIDE)
    assert len(entries) == 1
    assert missing in entries[0].message


def test_crop_subject_set_does_not_record_skip_listed_subject_again(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-D961CE0D63E8: a one-sided subject already skip-listed adds no entry."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2"), foldlabel=("b2",))
    _crop(tmp_path, out, "skeleton")  # a1 stacked before its foldlabel failure is known, as in region 1
    _skip(out, "a1")
    before = skip_list.load_skip_entries(skip_dir=str(out), side=SIDE)

    _crop(tmp_path, out, "foldlabel")

    assert skip_list.load_skip_entries(skip_dir=str(out), side=SIDE) == before


@pytest.mark.parametrize("input_type", ["skeleton", "extremities", "distmap"])
def test_crop_subject_set_leaves_skip_listed_subject_out(tmp_path, input_type):
    """REQ-CTFAIL-BDRABCZUK-637B5474B61E: skeleton, extremities and distmap crops leave skip-listed subjects out."""
    out = _subject_set_layout(tmp_path, **{kind: ("a1", "b2") for kind in {"skeleton", input_type}})
    _skip(out, "a1")

    if input_type == "extremities":
        _crop(tmp_path, out, "skeleton")  # quality_checks_extremities compares with <side>skeleton
    _crop(tmp_path, out, input_type)

    assert _stacked_subjects(tmp_path, input_type) == ["b2"]


def test_crop_subject_set_keeps_subject_skip_listed_on_opposite_side(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-AB3D7CB06EAC: a subject in the L skip list only stays in the R crops (per side)."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2"))
    _skip(out, "b2", side="L")
    _skip(out, "a1", side=SIDE)

    _crop(tmp_path, out, "skeleton")

    assert _stacked_subjects(tmp_path, "skeleton") == ["b2"]


def test_crop_subject_set_regression_missing_foldlabel_does_not_abort(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-D16162F8ED6A (DEF CB59D8EDF4AF): a subject without foldlabel aborts nothing."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2", "c3"), foldlabel=("b2", "c3"))

    _crop(tmp_path, out, "skeleton")
    _crop(tmp_path, out, "foldlabel")  # raised AssertionError in quality_checks (skeleton 3 != label 2)

    assert _stacked_subjects(tmp_path, "foldlabel") == ["b2", "c3"]


# --- REQ-CTFAIL-BDRABCZUK-8FA3023B0E25: without skip_dir the skip-list directory is crop_dir ----------------


def _crop_without_skip_dir(tmp_path, out, input_type):
    """Crops as a standalone call that passes no skip_dir."""
    gc.generate_crops(
        src_dir=str(out / KIND_DIRS[input_type] / "2mm"),
        input_type=input_type,
        crop_dir=str(tmp_path / "crops"),
        mask_dir=str(tmp_path / "mask"),
        side=SIDE,
        list_sulci=["S.C."],
        threshold=0,
        dilation=2,
    )


def test_crop_subject_set_without_skip_dir_records_in_crop_dir(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-8FA3023B0E25: called without skip_dir, a one-sided subject is recorded in the side's
    skip list under the crop_dir argument, and no skip-list file is written anywhere else."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2"), foldlabel=("b2",))

    _crop_without_skip_dir(tmp_path, out, "skeleton")
    _crop_without_skip_dir(tmp_path, out, "foldlabel")

    crop_dir = tmp_path / "crops"
    assert skip_list.load_skipped(skip_dir=str(crop_dir), side=SIDE) == {"a1"}
    assert sorted(tmp_path.rglob("skip_list_*.jsonl")) == [crop_dir / f"skip_list_{SIDE}.jsonl"]


def test_crop_subject_set_without_skip_dir_reads_skip_list_of_crop_dir(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-8FA3023B0E25: called without skip_dir, a subject in the side's skip list under the
    crop_dir argument is left out of the stacked crop array."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2"))
    _skip(tmp_path / "crops", "a1")

    _crop_without_skip_dir(tmp_path, out, "skeleton")

    assert _stacked_subjects(tmp_path, "skeleton") == ["b2"]


# --- TASK-CTFAIL-BDRABCZUK-6524B672025E: a subject whose crop fails, serial runs and p_map workers ----------
# Failures are real unless a test patches crop_mask: "missing" is a dangling link, which the glob lists and
# crop_one_file refuses with FileNotFoundError; "unreadable" is a file aims.read refuses with OSError. crop_one_file
# itself keeps raising (REQ-CTILESTEST-60); these tests go through generate_crops.

BAD = "b2"
STEP = "generate_crops"
MODES = [pytest.param(False, id="serial"), pytest.param(True, id="parallel")]
HOW = [("missing", "FileNotFoundError"), ("unreadable", "OSError")]
SAMPLE_DIRS = {"skeleton": "crops", "foldlabel": "labels"}


def _break_source(out, input_type, subject, how="unreadable"):
    """Replaces the resampled file of the subject so that its crop fails."""
    path = out / KIND_DIRS[input_type] / "2mm" / SIDE / f"{SIDE}resampled_{input_type}_{subject}.nii.gz"
    path.unlink()
    if how == "missing":
        path.symlink_to(path.with_name("absent.nii.gz"))
    else:
        path.write_bytes(b"not a nifti volume")


def _cropped_subjects(tmp_path, input_type):
    """Subjects having a cropped file of input_type under crops/<side><sample dir>."""
    suffix = f"_cropped_{input_type}.nii.gz"
    sample_dir = tmp_path / "crops" / f"{SIDE}{SAMPLE_DIRS[input_type]}"
    return sorted(path.name[: -len(suffix)] for path in sample_dir.glob(f"*{suffix}"))


def _recorded(skip_dir):
    return [(entry.subject, entry.step) for entry in skip_list.load_skip_entries(skip_dir=str(skip_dir), side=SIDE)]


def _clear_pathos_pools():
    state = pathos_mp._ProcessPool__STATE
    for pool in list(state.values()):
        pool.terminate()
        pool.join()
    state.clear()


def _crop_in_mode(tmp_path, out, input_type, parallel, skip_dir=None):
    """Crops serially or in p_map workers; a parallel run gets pools of its own, as no worker may outlive the test."""
    _clear_pathos_pools()
    try:
        _crop(tmp_path, out, input_type, skip_dir=skip_dir, parallel=parallel)
    finally:
        _clear_pathos_pools()


@pytest.fixture
def raising_crop(monkeypatch):
    """crop_mask raises RuntimeError('boom') from KeyError for BAD and crops the others; collects what it raised."""
    raised = []
    original = gc.crop_mask

    def crop_mask(file_src, file_cropped, *args):
        if file_src.endswith(f"_{BAD}.nii.gz"):
            try:
                raise KeyError("voxel_size")
            except KeyError as cause:
                exc = RuntimeError("boom")
                raised.append(exc)
                raise exc from cause
        original(file_src, file_cropped, *args)

    monkeypatch.setattr(gc, "crop_mask", crop_mask)
    return raised


@pytest.mark.parametrize("how, exc_type", HOW)
def test_crop_failure_is_recorded_in_side_skip_list(tmp_path, how, exc_type):
    """REQ-CTFAIL-BDRABCZUK-96A28A1F9AF7: one entry of the failing subject, step generate_crops, in the list of its
    side only, with the type and the message of the raised exception."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", BAD))
    _break_source(out, "skeleton", BAD, how)

    _crop(tmp_path, out, "skeleton")

    entries = skip_list.load_skip_entries(skip_dir=str(out), side=SIDE)
    assert [(entry.subject, entry.step, entry.exc_type) for entry in entries] == [(BAD, STEP, exc_type)]
    assert f"{SIDE}resampled_skeleton_{BAD}.nii.gz" in entries[0].message
    assert skip_list.load_skip_entries(skip_dir=str(out), side="L") == []


def test_crop_failure_error_record_carries_the_raised_exception(tmp_path, raising_crop, caplog):
    """REQ-CTFAIL-BDRABCZUK-96A28A1F9AF7: record_skip receives the raised exception object itself, so the ERROR
    record carries its traceback and __cause__ chain."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", BAD))
    caplog.set_level(logging.ERROR)

    _crop(tmp_path, out, "skeleton")

    assert len(raising_crop) == 1
    carried = [record for record in caplog.records if record.exc_info and record.exc_info[1] is raising_crop[0]]
    assert len(carried) == 1
    assert carried[0].levelno == logging.ERROR
    assert isinstance(carried[0].exc_info[1].__cause__, KeyError)


def test_crop_failure_without_skip_dir_is_recorded_in_crop_dir(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-96A28A1F9AF7, REQ-CTFAIL-BDRABCZUK-8FA3023B0E25: called without skip_dir, the crop failure
    lands in the skip list under the crop_dir argument, and nowhere else."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", BAD))
    _break_source(out, "skeleton", BAD)

    _crop_without_skip_dir(tmp_path, out, "skeleton")

    crop_dir = tmp_path / "crops"
    assert _recorded(crop_dir) == [(BAD, STEP)]
    assert sorted(tmp_path.rglob("skip_list_*.jsonl")) == [crop_dir / f"skip_list_{SIDE}.jsonl"]


@pytest.mark.parametrize("parallel", MODES)
def test_crop_failure_does_not_stop_the_other_subjects(tmp_path, parallel):
    """REQ-CTFAIL-BDRABCZUK-370125387C1E: generate_crops returns and the subjects before and after the failing one
    each get their cropped file; the failing subject gets none."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", BAD, "c3"))
    _break_source(out, "skeleton", BAD)

    _crop_in_mode(tmp_path, out, "skeleton", parallel)

    assert _cropped_subjects(tmp_path, "skeleton") == ["a1", "c3"]


@pytest.mark.parametrize("how, _exc_type", HOW)
def test_parallel_crop_failure_is_recorded_in_side_skip_list(tmp_path, how, _exc_type):
    """REQ-CTFAIL-BDRABCZUK-2A0D318568AD: in a parallel run, the subject failing in a p_map worker is recorded in the
    side's skip list with step generate_crops, and only that subject."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", BAD))
    _break_source(out, "skeleton", BAD, how)

    _crop_in_mode(tmp_path, out, "skeleton", parallel=True)

    assert _recorded(out) == [(BAD, STEP)]


@pytest.mark.parametrize("input_type", ["skeleton", "distmap"])
def test_crop_failure_leaves_subject_out_of_the_stacked_array(tmp_path, input_type):
    """REQ-CTFAIL-BDRABCZUK-5EB6D1D0FF16: the stacked array and subject CSV of the pass hold the other subjects only."""
    out = _subject_set_layout(tmp_path, **{kind: ("a1", BAD, "c3") for kind in {"skeleton", input_type}})
    _break_source(out, input_type, BAD)

    _crop(tmp_path, out, input_type)

    assert _stacked_subjects(tmp_path, input_type) == ["a1", "c3"]


def test_skeleton_crop_failure_leaves_subject_out_of_the_foldlabel_pass(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-5EB6D1D0FF16: a failure of the skeleton pass reaches the foldlabel pass through the skip
    list: both stacks agree, quality_checks passes and the subject is recorded once."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", BAD), foldlabel=("a1", BAD))
    _break_source(out, "skeleton", BAD)

    _crop(tmp_path, out, "skeleton")
    _crop(tmp_path, out, "foldlabel")

    assert _stacked_subjects(tmp_path, "skeleton") == ["a1"]
    assert _stacked_subjects(tmp_path, "foldlabel") == ["a1"]
    assert _recorded(out) == [(BAD, STEP)]


def test_crop_failure_deletes_the_file_left_at_the_cropped_path(tmp_path, monkeypatch):
    """REQ-CTFAIL-BDRABCZUK-440FF32C507C: a write that dies half way leaves a file save_to_numpy could not read; the
    file is deleted, so the other subjects are stacked."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", BAD))
    original = gc.crop_mask

    def crop_mask(file_src, file_cropped, *args):
        if file_src.endswith(f"_{BAD}.nii.gz"):
            with open(file_cropped, "wb") as truncated:
                truncated.write(b"truncated")
            raise OSError("Disk quota exceeded")
        original(file_src, file_cropped, *args)

    monkeypatch.setattr(gc, "crop_mask", crop_mask)

    _crop(tmp_path, out, "skeleton")

    assert _cropped_subjects(tmp_path, "skeleton") == ["a1"]
    assert _stacked_subjects(tmp_path, "skeleton") == ["a1"]


@pytest.mark.parametrize("parallel", MODES)
def test_foldlabel_crop_failure_leaves_subject_out_of_the_skeleton_stack(tmp_path, parallel):
    """REQ-CTFAIL-BDRABCZUK-6301BEC3C3C3: the skeleton pass stacked the subject before its foldlabel crop failed; the
    subject leaves the skeleton stack, so quality_checks passes and the foldlabel pass returns."""
    subjects = ("a1", BAD, "c3")
    out = _subject_set_layout(tmp_path, skeleton=subjects, foldlabel=subjects)
    _crop_in_mode(tmp_path, out, "skeleton", parallel)
    _break_source(out, "foldlabel", BAD)

    _crop_in_mode(tmp_path, out, "foldlabel", parallel)

    assert _stacked_subjects(tmp_path, "skeleton") == ["a1", "c3"]
    assert _stacked_subjects(tmp_path, "foldlabel") == ["a1", "c3"]


def test_extremities_crop_failure_leaves_subject_out_of_the_skeleton_and_foldlabel_stacks(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-68E295289926: run order of generate_sulcal_regions; both earlier passes stacked the subject
    before its extremities crop failed; the three stacks agree and quality_checks_extremities passes."""
    subjects = ("a1", BAD, "c3")
    out = _subject_set_layout(tmp_path, skeleton=subjects, foldlabel=subjects, extremities=subjects)
    _crop(tmp_path, out, "skeleton")
    _crop(tmp_path, out, "foldlabel")
    _break_source(out, "extremities", BAD)

    _crop(tmp_path, out, "extremities")

    assert _stacked_subjects(tmp_path, "skeleton") == ["a1", "c3"]
    assert _stacked_subjects(tmp_path, "foldlabel") == ["a1", "c3"]
    assert _stacked_subjects(tmp_path, "extremities") == ["a1", "c3"]


def test_failure_returned_by_a_p_map_worker_is_not_ignored(tmp_path, monkeypatch):
    """REQ-CTFAIL-BDRABCZUK-828B578D1652: a failure known only from the p_map result, a (subject, reason) pair of
    protocol P with no skip-list entry behind it, still takes the subject out of the skeleton stack."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", BAD), foldlabel=("a1", BAD))
    _crop(tmp_path, out, "skeleton")

    def p_map(worker, subjects, **_):
        return [(BAD, "worker failure") if subject == BAD else worker(subject) for subject in subjects]

    monkeypatch.setattr(gc, "p_map", p_map)

    _crop_in_mode(tmp_path, out, "foldlabel", parallel=True)

    assert _stacked_subjects(tmp_path, "skeleton") == ["a1"]


@pytest.mark.parametrize("parallel", MODES)
def test_crop_failure_of_every_subject_raises_runtime_error(tmp_path, parallel):
    """REQ-CTFAIL-BDRABCZUK-1A3EB88ED676: no valid subject left on the side: RuntimeError naming it, after each subject
    has its own skip-list entry (QC rule: report per subject, fail only if no valid subject remains)."""
    subjects = ("a1", BAD)
    out = _subject_set_layout(tmp_path, skeleton=subjects)
    for subject in subjects:
        _break_source(out, "skeleton", subject)

    with pytest.raises(RuntimeError, match=rf"\b{SIDE}\b"):
        _crop_in_mode(tmp_path, out, "skeleton", parallel)

    assert sorted(_recorded(out)) == [("a1", STEP), (BAD, STEP)]


def test_foldlabel_crop_failure_of_every_subject_raises_runtime_error(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-1A3EB88ED676: the foldlabel pass emptying the side raises although the skeleton pass
    stacked its subjects."""
    subjects = ("a1", BAD)
    out = _subject_set_layout(tmp_path, skeleton=subjects, foldlabel=subjects)
    _crop(tmp_path, out, "skeleton")
    for subject in subjects:
        _break_source(out, "foldlabel", subject)

    with pytest.raises(RuntimeError, match=rf"\b{SIDE}\b"):
        _crop(tmp_path, out, "foldlabel")


def test_skip_list_covering_every_subject_raises_runtime_error(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-1A3EB88ED676: subjects all skip-listed by earlier steps leave an empty stack: same error."""
    subjects = ("a1", BAD)
    out = _subject_set_layout(tmp_path, skeleton=subjects)
    for subject in subjects:
        _skip(out, subject)

    with pytest.raises(RuntimeError, match=rf"\b{SIDE}\b"):
        _crop(tmp_path, out, "skeleton")


@pytest.mark.parametrize("parallel", MODES)
def test_record_skip_failure_is_propagated(tmp_path, parallel):
    """REQ-CTFAIL-BDRABCZUK-28E451D54A18: a skip list that cannot be written stops the run with the record_skip
    exception: a subject is never dropped without its entry."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", BAD))
    _break_source(out, "skeleton", BAD)
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            _crop_in_mode(tmp_path, out, "skeleton", parallel, skip_dir=read_only)
    finally:
        read_only.chmod(0o700)
