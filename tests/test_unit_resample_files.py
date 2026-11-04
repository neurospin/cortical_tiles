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
import inspect
import logging
import os

import numpy as np
import pytest
from cortical_tiles.brainvisa import resample_files as rf
from cortical_tiles.brainvisa.utils.referentials import generate_ref_volume_ICBM2009c
from cortical_tiles.brainvisa.utils.skip_list import SkipEntry, load_skip_entries, load_skipped, record_skip
from pathos import multiprocessing as pathos_mp
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


def _skip_dir(root):
    """The shared skip-list directory passed to resample_files (one per cortical_tiles output root)."""
    return root / "skip"


def _resample_foldlabels(root, resampled_dir=None, parallel=False, skip_dir=None):
    """Resamples the foldlabel tree of ``root`` into resampled_dir (default <root>/foldlabels/2mm, as in
    generate_one_sulcal_region) with skip_dir (default <root>/skip)."""
    resampled_dir = root / "foldlabels" / "2mm" if resampled_dir is None else resampled_dir
    skip_dir = _skip_dir(root) if skip_dir is None else skip_dir
    rf.resample_files(
        src_dir=str(root / "raw"),
        input_type="foldlabel",
        resampled_dir=str(resampled_dir),
        transform_dir=str(root / "transforms"),
        side=SIDE,
        out_voxel_size=2.0,
        parallel=parallel,
        src_filename="foldlabel_",
        output_filename="resampled_foldlabel_",
        skip_dir=str(skip_dir),
    )
    return resampled_dir / f"{SIDE}_before_masking"


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
        skip_dir="/skip",
    )
    return resamplers[-1]


# --- REQ-CTILESTEST-41: unknown input type ----------------------------------------------------------


@pytest.mark.parametrize("input_type", ["skeletons", "label", ""])
def test_resample_files_rejects_unknown_input_type(tmp_path, input_type):
    """REQ-CTILESTEST-41: an input_type outside the four supported ones raises ValueError."""
    with pytest.raises(ValueError):
        rf.resample_files(
            src_dir=str(tmp_path), input_type=input_type, resampled_dir=str(tmp_path), skip_dir=str(tmp_path / "skip")
        )


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
    ],
)
def test_resample_files_none_filenames_fall_back_per_input_type(resamplers, input_type, src_name, out_name):
    """REQ-CTILESTEST-43: None src/output filenames are replaced by per-input-type defaults."""
    resampler = _build(resamplers, input_type, src_filename=None, output_filename=None)
    subject = {"subject": "sub", "side": SIDE}

    assert os.path.basename(resampler.src_file % subject) == f"R{src_name}sub.nii.gz"
    assert os.path.basename(resampler.resampled_file % subject) == f"R{out_name}sub.nii.gz"


def test_resample_files_extremities_none_filenames_fall_back_to_extremities_names(resamplers):
    """REQ-CTDEFECTS-05 (inverts the extremities case of REQ-CTILESTEST-43): output <side>resampled_extremities_<subject>."""
    resampler = _build(resamplers, "extremities", src_filename=None, output_filename=None)
    subject = {"subject": "sub", "side": SIDE}

    assert os.path.basename(resampler.src_file % subject) == "Rextremities_sub.nii.gz"
    assert os.path.basename(resampler.resampled_file % subject) == "Rresampled_extremities_sub.nii.gz"


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
    """REQ-CTILESTEST-46, ERROR level per TASK-CTFAIL-BDRABCZUK-7B965D6726C4: a subject whose resampling
    raises gets no output, an ERROR record naming it (no longer a file-log warning), and a CSV row."""
    (foldlabel_tree / "transforms" / SIDE / f"{SIDE}transform_to_ICBM2009c_s2.trm").unlink()

    with caplog.at_level("WARNING"):
        out_dir = _resample_foldlabels(foldlabel_tree)

    assert (out_dir / f"{SIDE}resampled_foldlabel_s1.nii.gz").is_file()
    assert not (out_dir / f"{SIDE}resampled_foldlabel_s2.nii.gz").exists()
    assert any(record.levelno == logging.ERROR and "s2" in record.getMessage() for record in caplog.records)
    with open(foldlabel_tree / "foldlabels" / "2mm" / "not_processed_files.csv") as f:
        rows = [row[0] for row in csv.reader(f)]
    assert rows == [str(foldlabel_tree / "raw" / SIDE / f"{SIDE}foldlabel_s2.nii.gz")]


# --- REQ-CTILESTEST-47 / 48 / 49: source errors -----------------------------------------------------


def test_missing_side_source_dir_raises_not_a_directory(tmp_path):
    """REQ-CTILESTEST-47: <src_dir>/<side> missing -> NotADirectoryError."""
    with pytest.raises(NotADirectoryError):
        rf.resample_files(
            src_dir=str(tmp_path / "nope"),
            input_type="foldlabel",
            resampled_dir=str(tmp_path / "out"),
            side=SIDE,
            skip_dir=str(tmp_path / "skip"),
        )


def test_empty_side_source_dir_returns_without_error(tmp_path):
    """REQ-CTDEFECTS-06 (inverts REQ-CTILESTEST-48): <src_dir>/<side> without .nii.gz -> returns, no exception."""
    (tmp_path / "raw" / SIDE).mkdir(parents=True)
    rf.resample_files(
        src_dir=str(tmp_path / "raw"),
        input_type="foldlabel",
        resampled_dir=str(tmp_path / "out"),
        side=SIDE,
        skip_dir=str(tmp_path / "skip"),
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
def test_parse_args_returns_resample_files_keyword_arguments(tmp_path, extra):
    """REQ-CTDEFECTS-04 (inverts REQ-CTILESTEST-50): keys == resample_files parameters; resampled_dir = -o value."""
    out = str(tmp_path / "out")
    params = rf.parse_args(["-o", out, *extra])

    assert set(params) == set(inspect.signature(rf.resample_files).parameters)
    assert params["resampled_dir"] == out


# --- REQ-CTFAIL-BDRABCZUK-9D9A5EB278C7 / 1C0311FBAACC / BF96DCDA5C82 / E980205A894B -----------------
# The failure rule itself (ERROR record, traceback, skip-list format) belongs to skip_list.py
# (REQ-CTFAIL-BDRABCZUK-826AB670054E and siblings); these tests pin only what resample_files adds.

BAD_SUBJECT = "s2"


def _clear_pathos_pools():
    """Terminates pathos' cached process pools, so the next p_map forks workers from the current state."""
    state = pathos_mp._ProcessPool__STATE
    for pool in list(state.values()):
        pool.terminate()
        pool.join()
    state.clear()


def _niftis(folder):
    return sorted(path.name for path in folder.glob("*.nii.gz")) if folder.is_dir() else []


@pytest.fixture
def raising_resampler(monkeypatch):
    """FoldLabelResampler.resample_one_subject raises RuntimeError('boom') from KeyError for BAD_SUBJECT's
    source and resamples the others; the raised exceptions are collected."""
    raised = []
    original = rf.FoldLabelResampler.resample_one_subject

    def resample_one_subject(src_file, **kwargs):
        if os.path.basename(src_file) == f"{SIDE}foldlabel_{BAD_SUBJECT}.nii.gz":
            try:
                raise KeyError("voxel_size")
            except KeyError as cause:
                exc = RuntimeError("boom")
                raised.append(exc)
                raise exc from cause
        original(src_file=src_file, **kwargs)

    monkeypatch.setattr(rf.FoldLabelResampler, "resample_one_subject", staticmethod(resample_one_subject))
    return raised


@pytest.fixture
def resampled_sources(monkeypatch):
    """Records the source file of each resample_one_subject call (serial mode only)."""
    sources = []
    original = rf.FoldLabelResampler.resample_one_subject

    def resample_one_subject(src_file, **kwargs):
        sources.append(os.path.basename(src_file))
        original(src_file=src_file, **kwargs)

    monkeypatch.setattr(rf.FoldLabelResampler, "resample_one_subject", staticmethod(resample_one_subject))
    return sources


def test_failing_subject_is_recorded_in_side_skip_list(foldlabel_tree, raising_resampler):
    """REQ-CTFAIL-BDRABCZUK-9D9A5EB278C7: a subject whose resampling raises is in the side's skip list, with
    step resample_files and the type and message of the raised exception; the other side's list stays empty."""
    _resample_foldlabels(foldlabel_tree)

    assert load_skip_entries(_skip_dir(foldlabel_tree), SIDE) == [
        SkipEntry(subject=BAD_SUBJECT, step="resample_files", exc_type="RuntimeError", message="boom")
    ]
    assert load_skip_entries(_skip_dir(foldlabel_tree), "L") == []


def test_failing_subject_error_record_carries_the_raised_exception(foldlabel_tree, raising_resampler, caplog):
    """REQ-CTFAIL-BDRABCZUK-9D9A5EB278C7: record_skip receives the raised exception object itself, so the
    ERROR record carries its traceback and __cause__ chain."""
    caplog.set_level(logging.ERROR)
    _resample_foldlabels(foldlabel_tree)

    assert len(raising_resampler) == 1
    carried = [record for record in caplog.records if record.exc_info and record.exc_info[1] is raising_resampler[0]]
    assert len(carried) == 1
    assert carried[0].levelno == logging.ERROR
    assert isinstance(carried[0].exc_info[1].__cause__, KeyError)


def test_parallel_failing_subject_is_recorded_in_side_skip_list(foldlabel_tree):
    """REQ-CTFAIL-BDRABCZUK-1C0311FBAACC: in parallel mode (p_map workers) a subject whose resampling raises
    (here: its transform file is missing) is recorded in the side's skip list, and only that subject."""
    (foldlabel_tree / "transforms" / SIDE / f"{SIDE}transform_to_ICBM2009c_{BAD_SUBJECT}.trm").unlink()
    _clear_pathos_pools()
    try:
        out_dir = _resample_foldlabels(foldlabel_tree, parallel=True)
    finally:
        _clear_pathos_pools()

    entries = load_skip_entries(_skip_dir(foldlabel_tree), SIDE)
    assert [(entry.subject, entry.step) for entry in entries] == [(BAD_SUBJECT, "resample_files")]
    assert _niftis(out_dir) == [f"{SIDE}resampled_foldlabel_s1.nii.gz"]


def test_skip_listed_subject_is_not_resampled(foldlabel_tree, resampled_sources):
    """REQ-CTFAIL-BDRABCZUK-BF96DCDA5C82: a subject already in the side's skip list (here from an earlier
    step) gets no resampled file and is never handed to resample_one_subject; the other subject is resampled."""
    record_skip(_skip_dir(foldlabel_tree), SIDE, "generate_skeletons", BAD_SUBJECT, RuntimeError("no skeleton"))

    out_dir = _resample_foldlabels(foldlabel_tree)

    assert _niftis(out_dir) == [f"{SIDE}resampled_foldlabel_s1.nii.gz"]
    assert resampled_sources == [f"{SIDE}foldlabel_s1.nii.gz"]


def test_resample_files_records_failures_in_skip_dir_argument(foldlabel_tree, raising_resampler):
    """REQ-CTFAIL-BDRABCZUK-E980205A894B: the failure is recorded in the skip list under the skip_dir argument,
    not under a folder derived from resampled_dir (neither resampled_dir nor the two levels above it)."""
    output_dir = foldlabel_tree / "derivatives"
    resampled_dir = output_dir / "foldlabels" / "2mm"
    skip_dir = foldlabel_tree / "shared_skip"

    _resample_foldlabels(foldlabel_tree, resampled_dir=resampled_dir, skip_dir=skip_dir)

    assert load_skipped(skip_dir, SIDE) == {BAD_SUBJECT}
    for derived in (resampled_dir, resampled_dir.parent, output_dir):
        assert load_skipped(derived, SIDE) == set()


def test_resample_files_reads_skip_list_of_skip_dir_argument(foldlabel_tree, resampled_sources):
    """REQ-CTFAIL-BDRABCZUK-E980205A894B: a subject in the skip list under the skip_dir argument is not
    resampled; a skip list in a folder derived from resampled_dir (two levels above it) has no effect."""
    output_dir = foldlabel_tree / "derivatives"
    resampled_dir = output_dir / "foldlabels" / "2mm"
    skip_dir = foldlabel_tree / "shared_skip"
    record_skip(skip_dir, SIDE, "generate_skeletons", BAD_SUBJECT, RuntimeError("no skeleton"))
    record_skip(output_dir, SIDE, "generate_foldlabels", "s1", RuntimeError("elsewhere"))

    out_dir = _resample_foldlabels(foldlabel_tree, resampled_dir=resampled_dir, skip_dir=skip_dir)

    assert _niftis(out_dir) == [f"{SIDE}resampled_foldlabel_s1.nii.gz"]
    assert resampled_sources == [f"{SIDE}foldlabel_s1.nii.gz"]


# --- REQ-CTFAIL-BDRABCZUK-D42920AB80B0: without skip_dir the skip-list directory is resampled_dir --------


def _resample_foldlabels_without_skip_dir(root, resampled_dir):
    """Resamples the foldlabel tree of ``root`` into resampled_dir as a standalone call that passes no skip_dir."""
    rf.resample_files(
        src_dir=str(root / "raw"),
        input_type="foldlabel",
        resampled_dir=str(resampled_dir),
        transform_dir=str(root / "transforms"),
        side=SIDE,
        out_voxel_size=2.0,
        parallel=False,
        src_filename="foldlabel_",
        output_filename="resampled_foldlabel_",
    )
    return resampled_dir / f"{SIDE}_before_masking"


def test_resample_files_without_skip_dir_records_failures_in_resampled_dir(foldlabel_tree, raising_resampler):
    """REQ-CTFAIL-BDRABCZUK-D42920AB80B0: called without skip_dir, a failing subject is recorded in the side's
    skip list under the resampled_dir argument, and no skip-list file is written anywhere else."""
    resampled_dir = foldlabel_tree / "derivatives" / "foldlabels" / "2mm"

    _resample_foldlabels_without_skip_dir(foldlabel_tree, resampled_dir)

    assert load_skipped(resampled_dir, SIDE) == {BAD_SUBJECT}
    assert sorted(foldlabel_tree.rglob("skip_list_*.jsonl")) == [resampled_dir / f"skip_list_{SIDE}.jsonl"]


def test_resample_files_without_skip_dir_reads_skip_list_of_resampled_dir(foldlabel_tree, resampled_sources):
    """REQ-CTFAIL-BDRABCZUK-D42920AB80B0: called without skip_dir, a subject in the side's skip list under the
    resampled_dir argument gets no resampled file and is never handed to resample_one_subject."""
    resampled_dir = foldlabel_tree / "derivatives" / "foldlabels" / "2mm"
    record_skip(resampled_dir, SIDE, "generate_skeletons", BAD_SUBJECT, RuntimeError("no skeleton"))

    out_dir = _resample_foldlabels_without_skip_dir(foldlabel_tree, resampled_dir)

    assert _niftis(out_dir) == [f"{SIDE}resampled_foldlabel_s1.nii.gz"]
    assert resampled_sources == [f"{SIDE}foldlabel_s1.nii.gz"]
