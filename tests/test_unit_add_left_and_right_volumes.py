"""Unit tests for cortical_tiles.brainvisa.add_left_and_right_volumes (REQ-CTILESTEST-158..161).

champollion_pipeline's whole-brain path (run_cortical_tiles.py) fuses the L and R resampled
skeletons into F/ with ``add_left_and_right_volumes``. The voxel-wise priority fusion is
covered by test_full_brain.py; these tests pin the CLI, the missing-directory error and the
two sanity-check warnings, on tiny synthetic volumes in a pytest tmp dir.
"""

import glob
import logging
import os
from os.path import basename, join

import numpy as np
import pytest
from cortical_tiles.brainvisa import add_left_and_right_volumes as alr
from pathos import multiprocessing as pathos_mp
from soma import aims


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


def _write(arr, path):
    vol = aims.Volume(arr.astype(np.int16))
    vol.header()["voxel_size"] = [2.0, 2.0, 2.0, 1.0]
    aims.write(vol, path)


def _tree(src_dir, subjects, left=None, right=None):
    os.makedirs(join(src_dir, "L"), exist_ok=True)
    os.makedirs(join(src_dir, "R"), exist_ok=True)
    if left is None:
        left = np.zeros((4, 4, 4), np.int16)
        left[0, 0, 0] = 60
    if right is None:
        right = np.zeros((4, 4, 4), np.int16)
        right[3, 3, 3] = 30
    for s in subjects:
        _write(left, join(src_dir, "L", f"Lresampled_skeleton_{s}.nii.gz"))
        _write(right, join(src_dir, "R", f"Rresampled_skeleton_{s}.nii.gz"))
    return left, right


def _fused(src_dir):
    return sorted(basename(p) for p in glob.glob(join(src_dir, "F", "*.nii.gz")))


# --- REQ-CTILESTEST-158: main ----------------------------------------------------------------------


@pytest.mark.parametrize("parallel", [[], ["-a"]], ids=["serial", "parallel"])
def test_main_fuses_requested_number_of_subjects(tmp_path, fresh_process_pool, parallel):
    """REQ-CTILESTEST-158: main(['-s', d, '-n', '2']) writes F/F<output>_<subject>.nii.gz for exactly 2 subjects."""
    src = str(tmp_path / "sk")
    left, right = _tree(src, ["s1", "s2", "s3"])

    alr.main(["-s", src, "-e", "fused", "-n", "2", *parallel])

    written = _fused(src)
    assert len(written) == 2
    assert set(written) <= {f"Ffused_{s}.nii.gz" for s in ("s1", "s2", "s3")}
    out = np.asarray(aims.read(join(src, "F", written[0])))[..., 0]
    np.testing.assert_array_equal(out, left + right)


def test_main_default_fuses_all_subjects(tmp_path):
    """REQ-CTILESTEST-158: default '-n all' fuses every subject, named after the default output filename."""
    src = str(tmp_path / "sk")
    _tree(src, ["s1", "s2"])
    alr.main(["-s", src])
    assert _fused(src) == ["Fresampled_skeleton_s1.nii.gz", "Fresampled_skeleton_s2.nii.gz"]


# --- REQ-CTILESTEST-159: missing source directory --------------------------------------------------


def test_compute_raises_not_a_directory_when_src_dir_vanished(tmp_path):
    """REQ-CTILESTEST-159: compute raises NotADirectoryError when src_dir is not a directory."""
    src = tmp_path / "sk"
    adder = alr.AddLeftandRightVolumes(
        src_dir=str(src), src_filename="resampled_skeleton", output_filename="resampled_skeleton", parallel=False
    )
    # The constructor creates <src>/F; remove the whole tree before compute.
    os.rmdir(src / "F")
    os.rmdir(src)
    with pytest.raises(NotADirectoryError):
        adder.compute(number_subjects="all")


# --- REQ-CTILESTEST-160: non-skeleton value warning ------------------------------------------------


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_non_skeleton_fused_value_logs_warning(tmp_path, caplog):
    """REQ-CTILESTEST-160: a fused value outside the skeleton set (e.g. 40) logs 'unexpected skeleton values'."""
    src = str(tmp_path / "sk")
    left = np.zeros((4, 4, 4), np.int16)
    left[1, 1, 1] = 40
    _tree(src, ["s1"], left=left)
    with caplog.at_level(logging.WARNING):
        alr.add_left_and_right_volumes(src_dir=src)
    assert any("unexpected skeleton values" in m for m in _warnings(caplog))


def test_skeleton_fused_values_log_no_value_warning(tmp_path, caplog):
    """REQ-CTILESTEST-160 (guard): skeleton-only values log no 'unexpected skeleton values' warning."""
    src = str(tmp_path / "sk")
    _tree(src, ["s1"])
    with caplog.at_level(logging.WARNING):
        alr.add_left_and_right_volumes(src_dir=src)
    assert not any("unexpected skeleton values" in m for m in _warnings(caplog))


# --- REQ-CTILESTEST-161: contentious-voxel warning -------------------------------------------------


@pytest.mark.parametrize("n_both, warned", [(201, True), (200, False)])
def test_contentious_voxel_warning_above_200_names_left_file_twice(tmp_path, caplog, n_both, warned):
    """REQ-CTILESTEST-161 (DEFECT): >200 voxels non-zero on both sides warns, naming the LEFT file twice."""
    src = str(tmp_path / "sk")
    left = np.zeros((10, 10, 10), np.int16)
    right = np.zeros((10, 10, 10), np.int16)
    left.reshape(-1)[:n_both] = 60
    right.reshape(-1)[:n_both] = 60  # identical values still count as "contentious"
    _tree(src, ["s1"], left=left, right=right)

    with caplog.at_level(logging.WARNING):
        alr.add_left_and_right_volumes(src_dir=src)

    hits = [m for m in _warnings(caplog) if "voxels with different values" in m]
    assert len(hits) == int(warned)
    if warned:
        left_file = join(src, "L", "Lresampled_skeleton_s1.nii.gz")
        right_file = join(src, "R", "Rresampled_skeleton_s1.nii.gz")
        assert f"{n_both} voxels" in hits[0]
        assert hits[0].count(left_file) == 2
        assert right_file not in hits[0]
