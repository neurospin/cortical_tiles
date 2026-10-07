"""Unit test for cortical_tiles.brainvisa.utils.quality_checks.get_not_processed_files (REQ-CTILESTEST-138).

``get_not_processed_files`` is how resample_files and mask_resampled_foldlabels decide which source
files still need processing. It splits each full source path on ``src_filename`` (the file prefix)
and rebuilds the not-processed paths from the text before the *first* occurrence. This pins
defect 11 of TASK-136: when a parent directory name contains the prefix, the rebuilt paths are
wrong (they drop everything after that directory name). Inputs are empty placeholder files in a
pytest tmp dir.

Characterization test: behaviour recorded on cortical_tiles dev 4141bd8. "DEFECT": current
behaviour that looks wrong is recorded, not fixed (tests-only rule of TASK-114).
"""

from cortical_tiles.brainvisa.utils.quality_checks import get_not_processed_files

SRC_FILENAME = "foldlabel_"


def _source_dir(root, name):
    src = root / name / "R"
    src.mkdir(parents=True)
    for subject in ("s1", "s2"):
        (src / f"R{SRC_FILENAME}{subject}.nii.gz").write_text("")
    tgt = root / "target"
    tgt.mkdir(exist_ok=True)
    return src, tgt


def test_not_processed_paths_are_source_files_for_neutral_dir(tmp_path):
    """REQ-CTILESTEST-138 (control): without the prefix in a parent dir, the actual source paths are returned."""
    src, tgt = _source_dir(tmp_path, "raw")
    result = get_not_processed_files(str(src), str(tgt), SRC_FILENAME)
    assert sorted(result) == sorted(str(src / f"R{SRC_FILENAME}{s}.nii.gz") for s in ("s1", "s2"))


def test_not_processed_paths_are_source_files_when_parent_dir_contains_prefix(tmp_path):
    """REQ-CTDEFECTS-09 (inverts REQ-CTILESTEST-138): a parent dir containing src_filename -> actual source paths."""
    src, tgt = _source_dir(tmp_path, f"my{SRC_FILENAME}data")
    result = get_not_processed_files(str(src), str(tgt), SRC_FILENAME)

    assert sorted(result) == sorted(str(src / f"R{SRC_FILENAME}{s}.nii.gz") for s in ("s1", "s2"))
