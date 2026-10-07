"""Unit tests for cortical_tiles.brainvisa.utils.save_data (REQ-CTILESTEST-90..98).

champollion_pipeline stage 2 packs every crop directory into a numpy array plus a subject CSV with
``save_to_numpy`` (serial path already exercised by the crop tests), then re-reads every crop to
check the array order (``quality_checks`` -> ``compare_array_aims_files``). This module pins the
parallel path, the comparison failures and the pickle/dataframe helpers.

Inputs are tiny synthetic crops ``<subject>_cropped_skeleton.nii.gz`` written to pytest tmp dirs.

The parallel path goes through p_tqdm.p_map, i.e. a pathos ProcessPool that pathos caches per
process and that p_map only clears after a map that completed. Workers read the module global
``list_basename`` as it was when they were forked, so every parallel test starts and ends with an
empty pathos pool cache (``fresh_process_pool``) to stay independent of test order.

Characterization tests: behaviour recorded on cortical_tiles dev 4141bd8. Tests flagged "DEFECT"
pin current behaviour that looks wrong; it is recorded, not fixed (tests-only rule of TASK-114).
"""

import os

import numpy as np
import pandas as pd
import pytest
from cortical_tiles.brainvisa.utils import save_data as sd
from pathos import multiprocessing as pathos_mp
from soma import aims

SUBJECTS = ("s1", "s2", "s3")


def _crop_array(seed):
    arr = np.zeros((4, 5, 6, 1), dtype=np.int16)
    arr[seed % 4, seed % 5, seed % 6, 0] = 60
    arr[1, 1, 1, 0] = 30 + seed
    return arr


def _write_crop(path, arr):
    vol = aims.Volume(arr)
    vol.header()["voxel_size"] = [2.0, 2.0, 2.0, 1.0]
    aims.write(vol, str(path))


@pytest.fixture
def crops(tmp_path):
    """<tmp>/crops/<s>_cropped_skeleton.nii.gz for s1..s3; returns (crop dir, {subject: array})."""
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    arrays = {}
    for seed, subject in enumerate(SUBJECTS, start=1):
        arrays[subject] = _crop_array(seed)
        _write_crop(crop_dir / f"{subject}_cropped_skeleton.nii.gz", arrays[subject])
    return crop_dir, arrays


def _clear_pathos_pools():
    state = pathos_mp._ProcessPool__STATE
    for pool in list(state.values()):
        pool.terminate()
        pool.join()
    state.clear()


@pytest.fixture
def fresh_process_pool():
    _clear_pathos_pools()
    yield
    _clear_pathos_pools()


def _subjects(names):
    return pd.DataFrame({"Subject": list(names)})


def _stack(arrays, names):
    return np.array([arrays[name] for name in names])


# --- REQ-CTILESTEST-90: serial mismatch ------------------------------------------------------------


def test_serial_compare_raises_on_mismatching_array(crops):
    """REQ-CTILESTEST-90: an array row that differs from its subject's crop -> ValueError 'arrays do not match'."""
    crop_dir, arrays = crops
    arr = _stack(arrays, SUBJECTS)
    arr[1, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="subject = s2 and index = 1\narrays do not match"):
        sd.compare_array_aims_files(_subjects(SUBJECTS), arr, str(crop_dir), parallel=False)


def test_serial_compare_accepts_matching_arrays(crops):
    """REQ-CTILESTEST-90: arrays equal to the crops, in CSV order -> no error."""
    crop_dir, arrays = crops
    order = ("s3", "s1", "s2")
    assert sd.compare_array_aims_files(_subjects(order), _stack(arrays, order), str(crop_dir)) is None


# --- REQ-CTILESTEST-91 / 92: subject-to-crop matching ----------------------------------------------


@pytest.mark.parametrize("parallel", [False, True])
def test_compare_raises_when_subject_has_no_crop(crops, parallel, fresh_process_pool):
    """REQ-CTILESTEST-91: a subject without '<subject>_cropped*' file -> ValueError 'not in cropped files'."""
    crop_dir, arrays = crops
    arr = _stack(arrays, SUBJECTS)
    with pytest.raises(ValueError, match="Subject s9 not in cropped files"):
        sd.compare_array_aims_files(_subjects(("s1", "s9", "s3")), arr, str(crop_dir), parallel=parallel)


@pytest.mark.parametrize("parallel", [False, True])
def test_compare_raises_when_subject_matches_several_crops(crops, parallel, fresh_process_pool):
    """REQ-CTILESTEST-92: two files starting with '<subject>_cropped' -> ValueError 'several crops'."""
    crop_dir, arrays = crops
    _write_crop(crop_dir / "s2_cropped_foldlabel.nii.gz", arrays["s2"])
    arr = _stack(arrays, SUBJECTS)
    with pytest.raises(ValueError, match="Subject s2: several crops are matched"):
        sd.compare_array_aims_files(_subjects(SUBJECTS), arr, str(crop_dir), parallel=parallel)


# --- REQ-CTILESTEST-93: parallel comparison --------------------------------------------------------


def test_parallel_compare_accepts_matching_arrays(crops, fresh_process_pool):
    """REQ-CTILESTEST-93: parallel mode, arrays equal to the crops -> no error."""
    crop_dir, arrays = crops
    order = ("s2", "s3", "s1")
    assert sd.compare_array_aims_files(_subjects(order), _stack(arrays, order), str(crop_dir), parallel=True) is None


def test_parallel_compare_raises_on_mismatching_array(crops, fresh_process_pool):
    """REQ-CTILESTEST-93: parallel mode, a differing row -> ValueError 'arrays do not match'."""
    crop_dir, arrays = crops
    arr = _stack(arrays, SUBJECTS)
    arr[2, 1, 1, 1, 0] = 0
    with pytest.raises(ValueError, match="index = 2\narrays do not match"):
        sd.compare_array_aims_files(_subjects(SUBJECTS), arr, str(crop_dir), parallel=True)


# --- REQ-CTILESTEST-94: save_to_numpy parallel -----------------------------------------------------


def test_save_to_numpy_parallel_matches_serial(crops, tmp_path, fresh_process_pool):
    """REQ-CTILESTEST-94: parallel=True writes the same sorted npy, subject CSV and sub_id.npy as serial."""
    crop_dir, arrays = crops
    outputs = {}
    for parallel in (False, True):
        tgt = tmp_path / f"tgt_{parallel}"
        tgt.mkdir()
        ids, samples = sd.save_to_numpy(str(crop_dir), str(tgt), "Rskeleton", parallel=parallel)
        outputs[parallel] = (
            ids,
            np.load(tgt / "Rskeleton.npy"),
            pd.read_csv(tgt / "Rskeleton_subject.csv", dtype=str)["Subject"].tolist(),
            np.load(tgt / "sub_id.npy").tolist(),
        )

    ids, arr, csv_subjects, sub_id = outputs[True]
    assert ids == csv_subjects == sub_id == list(SUBJECTS)
    np.testing.assert_array_equal(arr, _stack(arrays, SUBJECTS))
    np.testing.assert_array_equal(arr, outputs[False][1])
    assert outputs[False][0] == ids


# --- REQ-CTILESTEST-95: get_one_numpy_array --------------------------------------------------------


def test_get_one_numpy_array_rejects_non_nifti(crops):
    """REQ-CTILESTEST-95: a name that is not an existing .nii file -> ValueError 'does not look like a nifti'."""
    crop_dir, _ = crops
    (crop_dir / "notes.txt").write_text("x")
    for name in ("notes.txt", "absent_cropped_skeleton.nii.gz", "s1_cropped_skeleton.nii.gz.minf"):
        with pytest.raises(ValueError, match="does not look like a nifti file"):
            sd.get_one_numpy_array(name, str(crop_dir))


def test_get_one_numpy_array_returns_basename_id(crops):
    """REQ-CTILESTEST-95 (complement): a crop file -> (subject id, its array)."""
    crop_dir, arrays = crops
    subject, sample = sd.get_one_numpy_array("s2_cropped_skeleton.nii.gz", str(crop_dir))
    assert subject == "s2"
    np.testing.assert_array_equal(sample, arrays["s2"])


# --- REQ-CTILESTEST-96: save_to_pickle -------------------------------------------------------------


def test_save_to_pickle_keys_columns_by_subject_id(crops, tmp_path):
    """REQ-CTDEFECTS-13 (inverts REQ-CTILESTEST-96): one column per crop, keyed by the basename before '_cropped_'."""
    crop_dir, arrays = crops
    (crop_dir / "notes.txt").write_text("not a crop")

    sd.save_to_pickle(str(crop_dir), str(tmp_path), "Rskeleton")

    frame = pd.read_pickle(tmp_path / "Rskeleton.pkl")
    assert set(frame.columns) == set(SUBJECTS)
    assert frame.shape == (1, len(SUBJECTS))
    for subject in SUBJECTS:
        np.testing.assert_array_equal(frame[subject][0], arrays[subject])


# --- REQ-CTILESTEST-97: save_to_dataframe_format_from_list -----------------------------------------


def test_save_to_dataframe_format_from_list(tmp_path):
    """REQ-CTILESTEST-97: column list_sample_id[i] holds list_sample_file[i]; written to <tgt>/<basename>.pkl."""
    samples = [np.full((2, 2), i) for i in range(3)]
    sd.save_to_dataframe_format_from_list(
        "unused", str(tmp_path), "Rlabels", list_sample_id=["a", "b", "c"], list_sample_file=samples
    )

    frame = pd.read_pickle(tmp_path / "Rlabels.pkl")
    assert list(frame.columns) == ["a", "b", "c"]
    for name, sample in zip("abc", samples):
        np.testing.assert_array_equal(frame[name][0], sample)


# --- REQ-CTILESTEST-98: parallel comparison after a failed parallel comparison ---------------------


def test_parallel_compare_after_failure_uses_current_crop_list(crops, tmp_path, fresh_process_pool):
    """REQ-CTDEFECTS-12 (inverts REQ-CTILESTEST-98): after a failed parallel call, the next call reads its own dir."""
    crop_dir, arrays = crops
    arr = _stack(arrays, SUBJECTS)
    with pytest.raises(ValueError, match="not in cropped files"):
        sd.compare_array_aims_files(_subjects(("s1", "s9", "s3")), arr, str(crop_dir), parallel=True)

    other_dir = tmp_path / "other_crops"
    other_dir.mkdir()
    for subject in SUBJECTS:
        _write_crop(other_dir / f"{subject}_cropped_skeleton.nii.gz", arrays[subject])
    _write_crop(other_dir / "s2_cropped_foldlabel.nii.gz", arrays["s2"])

    # other_dir holds two s2 crops: matching against other_dir's own files raises (as REQ-CTILESTEST-92).
    with pytest.raises(ValueError, match="several crops are matched"):
        sd.compare_array_aims_files(_subjects(SUBJECTS), arr, str(other_dir), parallel=True)
