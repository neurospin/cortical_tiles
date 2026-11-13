"""Extremities generation is removed from the cortical_tiles region pipeline (epic TASK-NODISTB-BDRABCZUK-5DDC4636377C).

USER DECISION 2026-11-09: "Remove both, everywhere". These tests pin the removal (TASK-NODISTB-BDRABCZUK-6D169C37D06A):

- REQ "cortical_tiles holds no extremities module": no module of the package has extremities in its name.
- REQ "run_with_params refuses input_type extremities": ValueError before any step runs.
- REQ "generate_sulcal_regions -y extremities is a usage error": argparse exit status 2.
- REQ "generate_sulcal_regions default input types are skeleton then foldlabel".
- REQ "generate_sulcal_regions -y help names no extremities".
- REQ "brainvisa README names no extremities".

An explicit request for extremities is refused, not ignored (unlike the legacy skip_distbottom key, which the
pipeline wrote): the pipeline never asks for extremities, and a run asked for them must not end without them.

As in test_unit_no_distbottom.py, every step function of run_with_params is replaced by a recorder on the imported
module object, so no image is processed. generate_sulcal_regions imports its sibling as a top-level module, so its
tests put the brainvisa directory on sys.path, as test_unit_generate_sulcal_regions.py does.
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
BRAINVISA_DIR = PACKAGE_DIR / "brainvisa"
README = BRAINVISA_DIR / "README.rst"
REMOVED_MODULES = (
    "cortical_tiles.brainvisa.generate_extremities",
    "cortical_tiles.brainvisa.mask_resampled_extremities",
    "cortical_tiles.brainvisa.utils.skeleton_extremities",
)
REQUIRED_ARGS = ["-d", "D", "--path_to_graph", "G", "--path_sk_with_hull", "H"]


def _recorder(name, recorded):
    def record(**kwargs):
        recorded.append((name, kwargs))
        if name == "generate_crops":
            os.makedirs(os.path.join(kwargs["crop_dir"], kwargs["side"] + "crops"), exist_ok=True)

    return record


@pytest.fixture
def calls(monkeypatch):
    """Replaces each step function of the module, extremities ones included, by a recorder."""
    recorded = []
    names = list(STEP_FUNCTIONS) + [name for name in dir(gosr) if "extremit" in name.lower()]
    for name in names:
        if callable(getattr(gosr, name, None)):
            monkeypatch.setattr(gosr, name, _recorder(name, recorded))
    monkeypatch.setattr(gosr, "setup_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(gosr, "check_if_number_skeletons_equals_number_foldlabels", lambda *args, **kwargs: None)
    return recorded


@pytest.fixture
def gsr(monkeypatch):
    monkeypatch.syspath_prepend(str(BRAINVISA_DIR))
    import generate_sulcal_regions

    return generate_sulcal_regions


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


# --- cortical_tiles holds no extremities module -------------------------------------------------------


def test_package_holds_no_extremities_module():
    """No module of cortical_tiles has extremities in its name, and none of the three removed ones imports."""
    named = sorted(
        str(path.relative_to(PACKAGE_DIR)) for path in PACKAGE_DIR.rglob("*.py") if "extremities" in path.stem.lower()
    )
    importable = [name for name in REMOVED_MODULES if importlib.util.find_spec(name) is not None]

    assert (named, importable) == ([], [])


# --- run_with_params refuses input_type extremities ---------------------------------------------------


@pytest.mark.parametrize("save_behavior", ["best", "clear_and_compute"])
def test_run_with_params_refuses_extremities_input_type(tmp_path, calls, save_behavior):
    """input_type extremities -> ValueError, and no step function was called before it."""
    with pytest.raises(ValueError):
        gosr.run_with_params(_params(tmp_path, input_type="extremities", save_behavior=save_behavior))

    assert [name for name, _ in calls] == []


# --- generate_sulcal_regions -y extremities is a usage error ------------------------------------------


@pytest.mark.parametrize("input_types", [["extremities"], ["skeleton", "extremities"]])
def test_input_types_option_rejects_extremities(gsr, input_types):
    """-y with extremities among its values -> argparse usage error, exit status 2."""
    with pytest.raises(SystemExit) as stopped:
        gsr.parse_args([*REQUIRED_ARGS, "-y", *input_types])

    assert stopped.value.code == 2


# --- generate_sulcal_regions default input types are skeleton then foldlabel --------------------------


def test_input_types_default_is_skeleton_then_foldlabel(gsr, monkeypatch, tmp_path):
    """Without -y, each region and side runs skeleton, then foldlabel, and nothing else."""
    recorded = []
    monkeypatch.setattr(gsr, "run_with_params", lambda cfg: recorded.append((cfg["side"], cfg["input_type"])))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "pipeline_loop_2mm.json").write_text(json.dumps({"output_dir": str(tmp_path / "out")}))

    params = gsr.parse_args(["-d", str(dataset), "--path_to_graph", "G", "--path_sk_with_hull", "H", "-r", REGION])
    gsr.generate_sulcal_regions(**params)

    assert recorded == [("L", "skeleton"), ("L", "foldlabel"), ("R", "skeleton"), ("R", "foldlabel")]


# --- generate_sulcal_regions -y help names no extremities ---------------------------------------------


def test_input_types_help_names_no_extremities(gsr, capsys):
    """The -h output of the script holds no 'extremities' (the -y help is the only place it named one)."""
    with pytest.raises(SystemExit):
        gsr.parse_args(["-h"])

    assert "extremities" not in capsys.readouterr().out.lower()


# --- brainvisa README names no extremities ------------------------------------------------------------


def test_brainvisa_readme_names_no_extremities():
    """cortical_tiles/brainvisa/README.rst shows no extremities example."""
    assert "extremities" not in README.read_text(encoding="utf-8").lower()
