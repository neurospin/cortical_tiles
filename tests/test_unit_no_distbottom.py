"""Distbottom crop generation is removed from cortical_tiles (epic TASK-NODISTB-BDRABCZUK-5DDC4636377C).

USER DECISION 2026-11-09: "Remove both, everywhere". These tests pin the removal:

- REQ-NODISTB-BDRABCZUK-562DF1BFC99B: run_with_params creates no <side>distbottom crop directory.
- REQ-NODISTB-BDRABCZUK-55C7A8CEEE78: a legacy skip_distbottom params key is ignored, never an error
  (the pipeline wrote it until its own cleanup Task lands).
- REQ-NODISTB-BDRABCZUK-73DBBBD28F3A: the package holds no distbottom module.
- REQ-NODISTB-BDRABCZUK-1C28A0E37B2B: the package source names no distbottom.

As in test_unit_generate_one_sulcal_region.py, every step function of run_with_params is replaced by
a recorder on the imported module object, so no image is processed. A recorder stands in for the
real step's directory: generate_crops creates <crop_dir>/<side>crops, and a distbottom step, while
the module still has one, creates <crop_dir>/<side>distbottom as the real generate_distbottom_crops
does.
"""

import importlib.util
import json
import os
from pathlib import Path

import cortical_tiles
import pytest
from cortical_tiles.brainvisa import generate_one_sulcal_region as gosr

REGION = "S.Or."
SIDE = "R"

# Step functions run_with_params calls; those the module no longer has are left alone.
STEP_FUNCTIONS = (
    "compute_mask",
    "generate_skeletons",
    "generate_distmaps",
    "generate_extremities",
    "generate_foldlabels",
    "generate_ICBM2009c_transforms",
    "resample_files",
    "mask_foldlabel_files",
    "mask_extremities_files",
    "generate_crops",
)

PACKAGE_DIR = Path(cortical_tiles.__file__).resolve().parent
REMOVED_MODULES = (
    "cortical_tiles.brainvisa.generate_distbottom_crops",
    "cortical_tiles.brainvisa.utils.distbottom",
)


def _recorder(name, recorded):
    def record(**kwargs):
        recorded.append((name, kwargs))
        if name == "generate_crops":
            os.makedirs(os.path.join(kwargs["crop_dir"], kwargs["side"] + "crops"), exist_ok=True)
        elif "distbottom" in name:
            os.makedirs(os.path.join(kwargs["crop_dir"], kwargs["side"] + "distbottom"), exist_ok=True)

    return record


@pytest.fixture
def calls(monkeypatch):
    """Replaces each step function of the module, distbottom ones included, by a recorder."""
    recorded = []
    names = list(STEP_FUNCTIONS) + [name for name in dir(gosr) if "distbottom" in name.lower()]
    for name in names:
        if callable(getattr(gosr, name, None)):
            monkeypatch.setattr(gosr, name, _recorder(name, recorded))
    monkeypatch.setattr(gosr, "setup_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(gosr, "check_if_number_skeletons_equals_number_foldlabels", lambda *args, **kwargs: None)
    return recorded


def _params(root, **overrides):
    root.mkdir(parents=True, exist_ok=True)
    regions_json = root / "regions.json"
    regions_json.write_text(json.dumps({"brain": {"S.Or._right": {"S.Or._right": 1}, "S.Or._left": {"S.Or._left": 1}}}))
    params = {
        "save_behavior": "best",
        "side": SIDE,
        "out_voxel_size": 2.0,
        "region_name": REGION,
        "brain_regions_json": str(regions_json),
        "parallel": False,
        "nb_subjects": -1,
        "input_type": "skeleton",
        "labeled_subjects_dir": "labeled",
        "path_to_graph_supervised": "pgs",
        "supervised_output_dir": str(root / "supervised"),
        "nb_subjects_mask": -1,
        "graphs_dir": "graphs",
        "path_to_graph": "pg",
        "path_to_skeleton_with_hull": "ph",
        "skel_qc_path": "",
        "output_dir": str(root / "derivatives"),
        "junction": "thin",
        "bids": False,
        "new_sulcus": None,
        "resampled_skel": False,
        "cropping_type": "mask",
        "combine_type": False,
        "no_mask": False,
        "threshold": 0,
        "dilation": 5,
        "njobs": 1,
        "masks_version": "canonical_25",
    }
    params.update(overrides)
    return params


def _crop_dir(calls):
    (crops,) = [kwargs for name, kwargs in calls if name == "generate_crops"]
    return crops["crop_dir"]


def _paths_under(directory):
    return sorted(str(path.relative_to(directory)) for path in Path(directory).rglob("*"))


# --- REQ-NODISTB-BDRABCZUK-562DF1BFC99B -------------------------------------------------------------


@pytest.mark.parametrize("save_behavior", ["best", "clear_and_compute"])
def test_run_with_params_writes_no_distbottom_directory(tmp_path, calls, save_behavior):
    """REQ-NODISTB-BDRABCZUK-562DF1BFC99B: skeleton run -> no <crop_dir>/<side>distbottom."""
    gosr.run_with_params(_params(tmp_path, save_behavior=save_behavior))

    crop_dir = _crop_dir(calls)
    assert not os.path.exists(os.path.join(crop_dir, SIDE + "distbottom")), (
        f"run_with_params created {SIDE}distbottom under {crop_dir}; steps called: {[name for name, _ in calls]}"
    )


# --- REQ-NODISTB-BDRABCZUK-55C7A8CEEE78 -------------------------------------------------------------


@pytest.mark.parametrize("skip_distbottom", [False, True])
def test_run_with_params_ignores_skip_distbottom_key(tmp_path, calls, skip_distbottom):
    """REQ-NODISTB-BDRABCZUK-55C7A8CEEE78: a skip_distbottom key leaves the crop tree as without it."""
    gosr.run_with_params(_params(tmp_path / "without"))
    without = _crop_dir(calls)
    calls.clear()

    gosr.run_with_params(_params(tmp_path / "with", skip_distbottom=skip_distbottom))
    with_key = _crop_dir(calls)

    assert _paths_under(with_key) == _paths_under(without)


# --- REQ-NODISTB-BDRABCZUK-73DBBBD28F3A -------------------------------------------------------------


def test_package_holds_no_distbottom_module():
    """REQ-NODISTB-BDRABCZUK-73DBBBD28F3A: no module of cortical_tiles has distbottom in its name."""
    named = sorted(
        str(path.relative_to(PACKAGE_DIR)) for path in PACKAGE_DIR.rglob("*.py") if "distbottom" in path.stem.lower()
    )
    importable = [name for name in REMOVED_MODULES if importlib.util.find_spec(name) is not None]

    assert (named, importable) == ([], [])


# --- REQ-NODISTB-BDRABCZUK-1C28A0E37B2B -------------------------------------------------------------


def test_package_source_names_no_distbottom():
    """REQ-NODISTB-BDRABCZUK-1C28A0E37B2B: no cortical_tiles source file holds the text distbottom."""
    offenders = sorted(
        str(path.relative_to(PACKAGE_DIR))
        for path in PACKAGE_DIR.rglob("*.py")
        if "distbottom" in path.read_text(encoding="utf-8").lower()
    )

    assert offenders == []
