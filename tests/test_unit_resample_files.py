"""Unit tests for cortical_tiles.brainvisa.resample_files (REQ-CTILESTEST-41..50).

champollion_pipeline stage 2 resamples each raw skeleton / foldlabel volume to the 2 mm ICBM2009c
grid with ``resample_files`` (called by generate_one_sulcal_region.run_with_params). Inputs here are
tiny synthetic volumes written to pytest tmp dirs, with identity ``.trm`` transforms, so nothing is
read from or written to the checkout. Foldlabel resampling is used for the end-to-end checks because
it is deterministic and fast; the skeleton path (VipSkeleton, nondeterministic, TASK-126) is only
checked for naming.

Characterization tests: behaviour recorded on cortical_tiles dev 1b2bd48. Tests flagged "DEFECT"
pin current behaviour that looks wrong; it is recorded, not fixed (tests-only rule of TASK-114).
"""

import csv
import os

import numpy as np
import pytest
from cortical_tiles.brainvisa import resample_files as rf
from cortical_tiles.brainvisa.utils.referentials import generate_ref_volume_ICBM2009c
from soma import aims

SIDE = "R"
LABELS = (1003, 7005)


def _write_volume(path, arr, voxel_size=1.0):
    vol = aims.Volume(arr.astype(np.int16))
    vol.header()["voxel_size"] = [voxel_size, voxel_size, voxel_size, 1.0]
    aims.write(vol, str(path))


def _foldlabel_array():
    arr = np.zeros((20, 20, 20, 1), dtype=np.int16)
    arr[5:9, 5:9, 5:9, 0] = LABELS[0]
    arr[10:12, 10:12, 10:12, 0] = LABELS[1]
    return arr


@pytest.fixture
def foldlabel_tree(tmp_path):
    """raw/R/Rfoldlabel_<s>.nii.gz and transforms/R/Rtransform_to_ICBM2009c_<s>.trm for s1, s2."""
    raw = tmp_path / "raw" / SIDE
    raw.mkdir(parents=True)
    transforms = tmp_path / "transforms" / SIDE
    transforms.mkdir(parents=True)
    for subject in ("s1", "s2"):
        _write_volume(raw / f"{SIDE}foldlabel_{subject}.nii.gz", _foldlabel_array())
        aims.write(aims.AffineTransformation3d(), str(transforms / f"{SIDE}transform_to_ICBM2009c_{subject}.trm"))
    return tmp_path


def _resample_foldlabels(root):
    rf.resample_files(
        src_dir=str(root / "raw"),
        input_type="foldlabel",
        resampled_dir=str(root / "out"),
        transform_dir=str(root / "transforms"),
        side=SIDE,
        out_voxel_size=2.0,
        src_filename="foldlabel_",
        output_filename="resampled_foldlabel_",
    )
    return root / "out" / f"{SIDE}_before_masking"


@pytest.fixture
def resamplers(monkeypatch):
    """Makes FileResampler.compute record the resampler instead of running; returns the record list."""
    built = []
    monkeypatch.setattr(rf.FileResampler, "compute", lambda self, nb_subjects=-1: built.append(self))
    return built


def _build(resamplers, input_type, src_filename="src_", output_filename="out_"):
    rf.resample_files(
        src_dir="/src",
        input_type=input_type,
        resampled_dir="/res",
        transform_dir="/trm",
        side=SIDE,
        out_voxel_size=2.0,
        src_filename=src_filename,
        output_filename=output_filename,
    )
    return resamplers[-1]


# --- REQ-CTILESTEST-41: unknown input type ----------------------------------------------------------


@pytest.mark.parametrize("input_type", ["skeletons", "label", ""])
def test_resample_files_rejects_unknown_input_type(tmp_path, input_type):
    """REQ-CTILESTEST-41: an input_type outside the four supported ones raises ValueError."""
    with pytest.raises(ValueError):
        rf.resample_files(src_dir=str(tmp_path), input_type=input_type, resampled_dir=str(tmp_path))


# --- REQ-CTILESTEST-42: file naming -----------------------------------------------------------------


@pytest.mark.parametrize(
    "input_type, out_subdir",
    [("skeleton", "R"), ("foldlabel", "R_before_masking"), ("extremities", "R_before_masking"), ("distmap", "R")],
)
def test_resampler_file_naming(resamplers, input_type, out_subdir):
    """REQ-CTILESTEST-42: source, transform and output file names for subject 'sub' and side R."""
    resampler = _build(resamplers, input_type)
    subject = {"subject": "sub", "side": SIDE}

    assert resampler.src_file % subject == "/src/R/Rsrc_sub.nii.gz"
    assert resampler.transform_file % subject == "/trm/R/Rtransform_to_ICBM2009c_sub.trm"
    assert resampler.resampled_file % subject == f"/res/{out_subdir}/Rout_sub.nii.gz"
    assert resampler.out_voxel_size == (2.0, 2.0, 2.0)


# --- REQ-CTILESTEST-43: None filename fallbacks -----------------------------------------------------


@pytest.mark.parametrize(
    "input_type, src_name, out_name",
    [
        ("skeleton", "skeleton_generated_", "resampled_skeleton_"),
        ("foldlabel", "foldlabel_", "resampled_foldlabel_"),
        # DEFECT: extremities fall back to the foldlabel output name, not "resampled_extremities_".
        ("extremities", "extremities_", "resampled_foldlabel_"),
    ],
)
def test_resample_files_none_filenames_fall_back_per_input_type(resamplers, input_type, src_name, out_name):
    """REQ-CTILESTEST-43: None src/output filenames are replaced by per-input-type defaults."""
    resampler = _build(resamplers, input_type, src_filename=None, output_filename=None)
    subject = {"subject": "sub", "side": SIDE}

    assert os.path.basename(resampler.src_file % subject) == f"R{src_name}sub.nii.gz"
    assert os.path.basename(resampler.resampled_file % subject) == f"R{out_name}sub.nii.gz"


# --- REQ-CTILESTEST-44: foldlabel resampling output -------------------------------------------------


def test_resampled_labels_lie_on_icbm_grid_with_source_values_only(foldlabel_tree):
    """REQ-CTILESTEST-44: per subject, voxel size 2 mm, ICBM2009c 2 mm shape, values within the source labels."""
    out_dir = _resample_foldlabels(foldlabel_tree)
    ref_shape = tuple(generate_ref_volume_ICBM2009c((2.0, 2.0, 2.0)).np.shape)

    for subject in ("s1", "s2"):
        vol = aims.read(str(out_dir / f"{SIDE}resampled_foldlabel_{subject}.nii.gz"))
        arr = np.asarray(vol)
        assert tuple(vol.header()["voxel_size"][:3]) == (2.0, 2.0, 2.0)
        assert arr.shape == ref_shape
        assert set(np.unique(arr).tolist()) <= {0, *LABELS}
        assert np.count_nonzero(arr) > 0


# --- REQ-CTILESTEST-45: resume ----------------------------------------------------------------------


def test_resampling_only_processes_subjects_without_output(foldlabel_tree):
    """REQ-CTILESTEST-45: a second run resamples only the subject whose output is missing."""
    out_dir = _resample_foldlabels(foldlabel_tree)
    kept = out_dir / f"{SIDE}resampled_foldlabel_s1.nii.gz"
    removed = out_dir / f"{SIDE}resampled_foldlabel_s2.nii.gz"
    removed.unlink()
    sentinel = 1_000_000_000
    os.utime(kept, (sentinel, sentinel))

    _resample_foldlabels(foldlabel_tree)

    assert removed.is_file()
    assert os.stat(kept).st_mtime == sentinel


# --- REQ-CTILESTEST-46: failing subject is skipped --------------------------------------------------


def test_failing_subject_is_skipped_and_listed_as_not_processed(foldlabel_tree, caplog):
    """REQ-CTILESTEST-46: a subject whose resampling raises gets no output, a warning, and a CSV row."""
    (foldlabel_tree / "transforms" / SIDE / f"{SIDE}transform_to_ICBM2009c_s2.trm").unlink()

    with caplog.at_level("WARNING"):
        out_dir = _resample_foldlabels(foldlabel_tree)

    assert (out_dir / f"{SIDE}resampled_foldlabel_s1.nii.gz").is_file()
    assert not (out_dir / f"{SIDE}resampled_foldlabel_s2.nii.gz").exists()
    assert any("[skip] subject s2" in record.getMessage() for record in caplog.records)
    with open(foldlabel_tree / "out" / "not_processed_files.csv") as f:
        rows = [row[0] for row in csv.reader(f)]
    assert rows == [str(foldlabel_tree / "raw" / SIDE / f"{SIDE}foldlabel_s2.nii.gz")]


# --- REQ-CTILESTEST-47 / 48 / 49: source errors -----------------------------------------------------


def test_missing_side_source_dir_raises_not_a_directory(tmp_path):
    """REQ-CTILESTEST-47: <src_dir>/<side> missing -> NotADirectoryError."""
    with pytest.raises(NotADirectoryError):
        rf.resample_files(
            src_dir=str(tmp_path / "nope"), input_type="foldlabel", resampled_dir=str(tmp_path / "out"), side=SIDE
        )


def test_empty_side_source_dir_raises_index_error(tmp_path):
    """REQ-CTILESTEST-48: <src_dir>/<side> without .nii.gz -> IndexError (DEFECT: no explicit error)."""
    (tmp_path / "raw" / SIDE).mkdir(parents=True)
    with pytest.raises(IndexError):
        rf.resample_files(
            src_dir=str(tmp_path / "raw"), input_type="foldlabel", resampled_dir=str(tmp_path / "out"), side=SIDE
        )


def test_wrapper_raises_file_not_found_for_missing_source(foldlabel_tree):
    """REQ-CTILESTEST-49: resampling a subject without a source file raises FileNotFoundError."""
    resampler = rf.FoldLabelResampler(
        src_dir=str(foldlabel_tree / "raw"),
        resampled_dir=str(foldlabel_tree / "out"),
        transform_dir=str(foldlabel_tree / "transforms"),
        side=SIDE,
        out_voxel_size=2.0,
        parallel=False,
        src_filename="foldlabel_",
        output_filename="resampled_foldlabel_",
        do_skel=False,
        immortals=[],
    )
    with pytest.raises(FileNotFoundError):
        resampler.resample_one_subject_wrapper("absent")


# --- REQ-CTILESTEST-50: parse_args ------------------------------------------------------------------


@pytest.mark.parametrize("extra", [[], ["-y", "foldlabel", "-n", "3"]])
def test_parse_args_raises_key_error_for_valid_arguments(tmp_path, extra):
    """REQ-CTILESTEST-50: parse_args raises KeyError 'output_dir' on valid arguments (DEFECT: CLI unusable)."""
    with pytest.raises(KeyError, match="output_dir"):
        rf.parse_args(["-o", str(tmp_path / "out"), *extra])
