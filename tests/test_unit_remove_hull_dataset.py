"""Unit tests for the dataset/CLI layer of cortical_tiles.brainvisa.utils.remove_hull (REQ-CTILESTEST-73..80).

The pure-numpy functions ``remove_hull`` / ``threshold_and_binarize`` are pinned by
test_unit_utils_remove_hull.py (REQ-CTILESTEST-22, 23). This file covers the mesh generation built on
top of them (``create_one_mesh``, ``DatasetHullRemoved``) and the command line parsing. Inputs are
tiny synthetic crops written to pytest tmp dirs; ``define_njobs`` is forced to 1 so ``pqdm`` runs
in-process.

Characterization tests: behaviour recorded on cortical_tiles dev 1b2bd48. Tests flagged "DEFECT"
pin current behaviour that looks wrong; it is recorded, not fixed (tests-only rule of TASK-114).
"""

import numpy as np
import pytest
from cortical_tiles.brainvisa.utils import remove_hull as rh
from soma import aims

SIDE = "R"
SUBJECTS = ("100206", "100307")
# Voxels expected to survive hull removal + thresholding (>= 12) in _crop().
KEPT = {(2, 2, 2)} | {(i, j, k) for i in (4, 5) for j in (4, 5) for k in (4, 5)}


def _crop():
    arr = np.zeros((9, 9, 9, 1), dtype=np.int16)
    arr[:, :, 0, 0] = 11  # external plane
    arr[3, 3, 1, 0] = 60  # touches external and internal -> removed as hull
    arr[4:6, 4:6, 4:6, 0] = 60  # interior block -> kept
    arr[2, 2, 2, 0] = 30  # >= 12 -> kept
    arr[6, 6, 6, 0] = 5  # < 12 -> thresholded away
    return arr


def _write_volume(path, arr):
    vol = aims.Volume(arr.astype(np.int16))
    vol.header()["voxel_size"] = [2.0, 2.0, 2.0, 1.0]
    aims.write(vol, str(path))


@pytest.fixture
def crops(tmp_path):
    """<tmp>/src/Rcrops/<subject>_normalized.nii.gz for two subjects."""
    crop_dir = tmp_path / "src" / f"{SIDE}crops"
    crop_dir.mkdir(parents=True)
    for subject in SUBJECTS:
        _write_volume(crop_dir / f"{subject}_normalized.nii.gz", _crop())
    return tmp_path


def _dataset(root, **kwargs):
    return rh.DatasetHullRemoved(src_dir=str(root / "src"), tgt_dir=str(root / "meshes"), side=SIDE, **kwargs)


def _coords(bucket):
    return {tuple(int(c) for c in row) for row in np.asarray(bucket).tolist()}


# --- REQ-CTILESTEST-73 / 74 / 75 / 76: subject listing ---------------------------------------------


def test_dataset_lists_first_six_characters_of_crop_entries(crops):
    """REQ-CTILESTEST-73: subjects = first 6 characters of each <src>/<side>crops entry without 'minf'."""
    (crops / "src" / f"{SIDE}crops" / "999999_qc_minf").mkdir()
    assert sorted(_dataset(crops).list_subjects) == sorted(SUBJECTS)


def test_dataset_number_subjects_keeps_that_many(crops):
    """REQ-CTILESTEST-74: number_subjects N > 0 keeps N listed subjects (DEFECT: os.listdir order, unsorted)."""
    subjects = _dataset(crops, number_subjects=1).list_subjects
    assert len(subjects) == 1
    assert set(subjects) <= set(SUBJECTS)


def test_dataset_explicit_list_subjects_is_used_unchanged(crops):
    """REQ-CTILESTEST-75: list_subjects, when given, is the subject list as is."""
    assert _dataset(crops, list_subjects=["x", "y"]).list_subjects == ["x", "y"]


def test_dataset_zero_subjects_creates_no_mesh(crops, monkeypatch):
    """REQ-CTDEFECTS-08 (inverts REQ-CTILESTEST-76): number_subjects 0 -> create_meshes returns {} and writes no mesh."""
    monkeypatch.setattr(rh, "define_njobs", lambda: 1)
    dataset = _dataset(crops, number_subjects=0)

    assert dataset.create_meshes() == {}
    meshes = crops / "meshes"
    assert not meshes.exists() or not list(meshes.glob("*.gii"))


# --- REQ-CTILESTEST-77 / 78: mesh creation ----------------------------------------------------------


def test_create_one_mesh_function_returns_bucket_of_kept_voxels(tmp_path):
    """REQ-CTILESTEST-77: bucket = voxels >= 12 left after hull removal; a mesh is returned."""
    _write_volume(tmp_path / "crop.nii.gz", _crop())
    vol = aims.read(str(tmp_path / "crop.nii.gz"))

    _, bucket, mesh = rh.create_one_mesh(vol)

    assert _coords(bucket) == KEPT
    assert mesh.vertex().size() > 0


def test_dataset_create_meshes_writes_one_mesh_per_subject(crops, monkeypatch):
    """REQ-CTILESTEST-78: tgt_dir created, mesh_<subject>.gii written, {subject: bucket} returned."""
    monkeypatch.setattr(rh, "define_njobs", lambda: 1)

    buckets = _dataset(crops).create_meshes()

    assert sorted(buckets) == sorted(SUBJECTS)
    for subject in SUBJECTS:
        assert _coords(buckets[subject]) == KEPT
        assert (crops / "meshes" / f"mesh_{subject}.gii").is_file()


# --- REQ-CTILESTEST-79 / 80: command line ----------------------------------------------------------


@pytest.mark.parametrize("value, expected", [("all", -1), ("0", 0), ("7", 7)])
def test_parse_args_number_of_subjects(value, expected):
    """REQ-CTILESTEST-79: -n 'all' -> -1, a non-negative integer -> that integer."""
    params = rh.parse_args(["-s", "S", "-t", "T", "-i", "L", "-n", value])
    assert params == {"src_dir": "S", "tgt_dir": "T", "side": "L", "nb_subjects": expected}


@pytest.mark.parametrize("value", ["-1", "x", "1.5"])
def test_parse_args_rejects_bad_number_of_subjects(value):
    """REQ-CTILESTEST-80: -n negative or non-integer -> ValueError."""
    with pytest.raises(ValueError):
        rh.parse_args(["-n", value])
