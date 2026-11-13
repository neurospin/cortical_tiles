"""Extremities are removed from the crop and resample steps of cortical_tiles (epic TASK-NODISTB-BDRABCZUK-5DDC4636377C).

USER DECISION 2026-11-09: "Remove both, everywhere". These tests pin the removal (TASK-NODISTB-BDRABCZUK-D8697465DB43),
the second half after test_unit_no_extremities.py (run_with_params and generate_sulcal_regions):

- REQ "generate_crops shall raise ValueError when input_type is not skeleton, foldlabel or distmap": extremities is
  refused like any unknown input type, before a file is written under crop_dir.
- REQ "resample_files shall raise ValueError when input_type is not skeleton, foldlabel or distmap": the same for the
  resample step, before a file is written under resampled_dir.
- REQ "the crop and resample sources name no extremities": generate_crops.py, resample_files.py and
  utils/crop_subjects.py hold no ExtremitiesCropGenerator, quality_checks_extremities, ExtremitiesResampler,
  resample_one_extremities, extremities filename constant, extremities row of the run stacks or extremities help text.

The source layouts are empty: before the removal the extremities branches run on them instead of refusing.
"""

from pathlib import Path

import cortical_tiles
import pytest
from cortical_tiles.brainvisa import generate_crops as gc
from cortical_tiles.brainvisa import resample_files as rf

SIDE = "R"
BRAINVISA_DIR = Path(cortical_tiles.__file__).resolve().parent / "brainvisa"
CLEANED_SOURCES = ("generate_crops.py", "resample_files.py", "utils/crop_subjects.py")


def _files_under(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _empty_layout(tmp_path):
    src = tmp_path / "src"
    (src / SIDE).mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    return src, out


# --- generate_crops refuses input_type extremities ----------------------------------------------------


def test_generate_crops_refuses_extremities_input_type(tmp_path):
    """input_type extremities -> ValueError naming input_type, and no file under crop_dir."""
    src, crop_dir = _empty_layout(tmp_path)

    with pytest.raises(ValueError, match="input_type"):
        gc.generate_crops(
            src_dir=str(src),
            input_type="extremities",
            crop_dir=str(crop_dir),
            mask_dir=str(tmp_path / "mask"),
            side=SIDE,
            skip_dir=str(crop_dir),
        )

    assert _files_under(crop_dir) == []


# --- resample_files refuses input_type extremities ----------------------------------------------------


def test_resample_files_refuses_extremities_input_type(tmp_path):
    """input_type extremities -> ValueError naming input_type, and no file under resampled_dir."""
    src, resampled_dir = _empty_layout(tmp_path)

    with pytest.raises(ValueError, match="input_type"):
        rf.resample_files(
            src_dir=str(src),
            input_type="extremities",
            resampled_dir=str(resampled_dir),
            transform_dir=str(tmp_path / "transforms"),
            side=SIDE,
            skip_dir=str(tmp_path / "skip"),
        )

    assert _files_under(resampled_dir) == []


# --- the crop and resample sources name no extremities ------------------------------------------------


@pytest.mark.parametrize("source", CLEANED_SOURCES)
def test_crop_and_resample_sources_name_no_extremities(source):
    """The source file holds no 'extremities', in any letter case."""
    lines = (BRAINVISA_DIR / source).read_text(encoding="utf-8").splitlines()

    assert [number for number, line in enumerate(lines, 1) if "extremities" in line.lower()] == []
