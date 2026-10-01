"""Unit tests for cortical_tiles.brainvisa.generate_one_sulcal_region (REQ-CTILESTEST-27..40).

``run_with_params`` is the per-region orchestrator that champollion_pipeline stage 2 reaches through
``generate_sulcal_regions`` -> ``RegionPipelineRunner``. The end-to-end smoke test
(test_generate_sulcal_regions_smoke.py) runs it once on real data; these tests pin its path
composition and step selection instead. Every step function it calls (compute_mask,
generate_skeletons, resample_files, generate_crops, ...) is replaced, on the imported module object
only, by a recorder, so no image is processed and each test runs in milliseconds. ``setup_log`` is
stubbed too, so no file log handler is attached to the shared cortical_tiles logger.

Characterization tests: behaviour recorded on cortical_tiles dev 1b2bd48. Some tests pin current
behaviour that looks defective (flagged "DEFECT" below); they are recorded, not fixed (tests-only
rule of TASK-114).
"""

import json
import os

import pytest
from cortical_tiles.brainvisa import generate_one_sulcal_region as gosr

REGION = "S.Or."
SULCI = ["S.Or.", "S.Pe.C."]

STEP_FUNCTIONS = [
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
    "generate_distbottom_crops",
]

CROP_SUBDIR = {"skeleton": "crops", "foldlabel": "labels", "distmap": "distmaps", "extremities": "extremitiess"}


@pytest.fixture
def regions_json(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text(
        json.dumps(
            {
                "brain": {
                    "S.Or._right": {"S.Or._right": 1, "S.Pe.C._right": 1},
                    "S.Or._left": {"S.Or._left": 1},
                }
            }
        )
    )
    return str(path)


@pytest.fixture
def calls(monkeypatch):
    """Replaces every step function of the module by a recorder; returns the list of (name, kwargs)."""
    recorded = []

    def make(name):
        def recorder(**kwargs):
            recorded.append((name, kwargs))
            if name == "generate_crops":
                # The real generate_crops creates <crop_dir>/<side><subdir>; later steps test for it.
                subdir = CROP_SUBDIR[kwargs["input_type"]]
                os.makedirs(os.path.join(kwargs["crop_dir"], kwargs["side"] + subdir), exist_ok=True)

        return recorder

    for name in STEP_FUNCTIONS:
        monkeypatch.setattr(gosr, name, make(name))
    monkeypatch.setattr(gosr, "setup_log", lambda *args, **kwargs: None)
    return recorded


def _params(tmp_path, regions_json, **overrides):
    params = {
        "save_behavior": "best",
        "side": "R",
        "out_voxel_size": 2.0,
        "region_name": REGION,
        "brain_regions_json": regions_json,
        "parallel": False,
        "nb_subjects": -1,
        "input_type": "skeleton",
        "labeled_subjects_dir": "labeled",
        "path_to_graph_supervised": "pgs",
        "supervised_output_dir": str(tmp_path / "supervised"),
        "nb_subjects_mask": -1,
        "graphs_dir": "graphs",
        "path_to_graph": "pg",
        "path_to_skeleton_with_hull": "ph",
        "skel_qc_path": "",
        "output_dir": str(tmp_path / "derivatives"),
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


def _called(calls, name):
    return [kwargs for called, kwargs in calls if called == name]


# --- REQ-CTILESTEST-27 / 28 / 29: get_sulci_list ----------------------------------------------------


@pytest.mark.parametrize("side, expected", [("R", ["S.Or.", "S.Pe.C."]), ("L", ["S.Or."])])
def test_get_sulci_list_strips_side_suffix(regions_json, side, expected):
    """REQ-CTILESTEST-27: sulci of brain/<region>_<right|left>, side suffix removed, JSON order kept."""
    assert gosr.get_sulci_list(REGION, side, json_path=regions_json) == expected


@pytest.mark.parametrize("side", ["X", "right", ""])
def test_get_sulci_list_rejects_unknown_side(regions_json, side):
    """REQ-CTILESTEST-28: a side other than 'R' or 'L' raises ValueError."""
    with pytest.raises(ValueError):
        gosr.get_sulci_list(REGION, side, json_path=regions_json)


def test_get_sulci_list_unknown_region_raises_key_error(regions_json):
    """REQ-CTILESTEST-29: unknown region -> KeyError (DEFECT: its `except ValueError` handler is dead)."""
    with pytest.raises(KeyError):
        gosr.get_sulci_list("Not.A.Region.", "R", json_path=regions_json)


# --- REQ-CTILESTEST-30 / 31: is_step_to_be_computed -------------------------------------------------


@pytest.fixture
def step_paths(tmp_path):
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()
    full = tmp_path / "full"
    full.mkdir()
    (full / "x.nii.gz").write_text("")
    a_file = tmp_path / "mask.nii.gz"
    a_file.write_text("")
    return {"missing": str(missing), "empty": str(empty), "full": str(full), "file": str(a_file)}


@pytest.mark.parametrize("save_behavior", ["best", "minimal", "clear_and_compute"])
@pytest.mark.parametrize("kind", ["missing", "empty", "full", "file"])
def test_is_step_to_be_computed_truth_table(step_paths, kind, save_behavior):
    """REQ-CTILESTEST-30: False exactly for an existing non-empty-dir-or-file path with 'minimal'."""
    expected = not (save_behavior == "minimal" and kind in ("full", "file"))
    assert gosr.is_step_to_be_computed(step_paths[kind], "step", save_behavior=save_behavior) is expected


@pytest.mark.parametrize("kind", ["full", "file"])
def test_is_step_to_be_computed_rejects_unknown_save_behavior(step_paths, kind):
    """REQ-CTILESTEST-31: unknown save_behavior on an existing, non-empty path raises ValueError."""
    with pytest.raises(ValueError):
        gosr.is_step_to_be_computed(step_paths[kind], "step", save_behavior="always")


# --- REQ-CTILESTEST-32 / 33: generate_crops path composition ---------------------------------------


@pytest.mark.parametrize("out_voxel_size, vs", [(2.0, "2mm"), (2, "2mm"), (1.5, "1.5mm")])
def test_run_with_params_skeleton_crop_paths(tmp_path, regions_json, calls, out_voxel_size, vs):
    """REQ-CTILESTEST-32: generate_crops paths: skeletons/<vs>, crops/<version>/<vs>/<region>/mask, mask/..."""
    params = _params(tmp_path, regions_json, out_voxel_size=out_voxel_size, masks_version="v9")
    gosr.run_with_params(params)

    (crops,) = _called(calls, "generate_crops")
    out = params["output_dir"]
    sup = params["supervised_output_dir"]
    assert crops["src_dir"] == os.path.join(out, "skeletons", vs)
    assert crops["crop_dir"] == os.path.join(out, "crops", "v9", vs, REGION, "mask")
    assert crops["mask_dir"] == os.path.join(sup, "mask", "v9") + "/" + vs
    assert crops["list_sulci"] == SULCI
    assert crops["input_type"] == "skeleton"


def test_run_with_params_masks_version_defaults_to_canonical_25(tmp_path, regions_json, calls):
    """REQ-CTILESTEST-33: without a masks_version key, 'canonical_25' is used for masks and crops."""
    params = _params(tmp_path, regions_json)
    del params["masks_version"]
    gosr.run_with_params(params)

    (crops,) = _called(calls, "generate_crops")
    assert crops["crop_dir"] == os.path.join(params["output_dir"], "crops", "canonical_25", "2mm", REGION, "mask")
    assert crops["mask_dir"].startswith(os.path.join(params["supervised_output_dir"], "mask", "canonical_25"))


# --- REQ-CTILESTEST-34 / 35: out_voxel_size 'raw' ---------------------------------------------------


def test_run_with_params_raw_skips_transform_and_resampling(tmp_path, regions_json, calls):
    """REQ-CTILESTEST-34: out_voxel_size 'raw' calls neither generate_ICBM2009c_transforms nor resample_files."""
    gosr.run_with_params(_params(tmp_path, regions_json, out_voxel_size="raw"))

    names = [name for name, _ in calls]
    assert "generate_ICBM2009c_transforms" not in names
    assert "resample_files" not in names
    assert "generate_crops" in names


def test_run_with_params_raw_crop_source_dir_lacks_separator(tmp_path, regions_json, calls):
    """REQ-CTILESTEST-35: 'raw' skeleton crops read <output_dir>/skeletonsraw (DEFECT: missing '/')."""
    params = _params(tmp_path, regions_json, out_voxel_size="raw")
    gosr.run_with_params(params)

    (crops,) = _called(calls, "generate_crops")
    assert crops["src_dir"] == os.path.join(params["output_dir"], "skeletons") + "raw"


# --- REQ-CTILESTEST-36: mask generation only for missing masks --------------------------------------


@pytest.mark.parametrize("save_behavior", ["best", "minimal", "clear_and_compute"])
def test_run_with_params_computes_only_missing_masks(tmp_path, regions_json, calls, save_behavior):
    """REQ-CTILESTEST-36: compute_mask runs for each sulcus whose mask file is absent, never for an existing one."""
    params = _params(tmp_path, regions_json, save_behavior=save_behavior)
    mask_side_dir = os.path.join(params["supervised_output_dir"], "mask", "canonical_25", "2mm", "R")
    os.makedirs(mask_side_dir)
    open(os.path.join(mask_side_dir, "S.Or._right.nii.gz"), "w").close()
    if save_behavior == "clear_and_compute":
        # Pre-create the distbottom dir so the REQ-CTILESTEST-40 defect does not interrupt the run.
        os.makedirs(os.path.join(params["output_dir"], "crops", "canonical_25", "2mm", REGION, "mask", "Rdistbottom"))

    gosr.run_with_params(params)

    masks = _called(calls, "compute_mask")
    assert [m["sulcus"] for m in masks] == ["S.Pe.C."]
    assert masks[0]["mask_dir"] == os.path.join(params["supervised_output_dir"], "mask", "canonical_25", "2mm")
    assert masks[0]["side"] == "R"
    assert os.path.isfile(os.path.join(mask_side_dir, "S.Or._right.nii.gz"))


# --- REQ-CTILESTEST-37: distbottom crops ------------------------------------------------------------


@pytest.mark.parametrize("skip", [False, True])
def test_run_with_params_distbottom_follows_skip_flag(tmp_path, regions_json, calls, skip):
    """REQ-CTILESTEST-37: distbottom crops use generate_crops' crop_dir as src and target, unless skip_distbottom."""
    gosr.run_with_params(_params(tmp_path, regions_json, skip_distbottom=skip))

    (crops,) = _called(calls, "generate_crops")
    distbottom = _called(calls, "generate_distbottom_crops")
    if skip:
        assert distbottom == []
    else:
        assert len(distbottom) == 1
        assert distbottom[0]["src_dir"] == crops["crop_dir"]
        assert distbottom[0]["crop_dir"] == crops["crop_dir"]
        assert distbottom[0]["side"] == "R"


# --- REQ-CTILESTEST-38: foldlabel masking -----------------------------------------------------------


def test_run_with_params_foldlabel_masked_with_resampled_skeletons(tmp_path, regions_json, calls):
    """REQ-CTILESTEST-38: foldlabel -> mask_foldlabel_files(foldlabels/<vs>, skeletons/<vs>) after resampling."""
    params = _params(tmp_path, regions_json, input_type="foldlabel")
    gosr.run_with_params(params)

    names = [name for name, _ in calls]
    assert names.index("resample_files") < names.index("mask_foldlabel_files") < names.index("generate_crops")
    (masked,) = _called(calls, "mask_foldlabel_files")
    out = params["output_dir"]
    assert masked["src_dir"] == os.path.join(out, "foldlabels", "2mm")
    assert masked["masked_dir"] == os.path.join(out, "foldlabels", "2mm")
    assert masked["skeleton_dir"] == os.path.join(out, "skeletons", "2mm")
    (crops,) = _called(calls, "generate_crops")
    assert crops["src_dir"] == os.path.join(out, "foldlabels", "2mm")


# --- REQ-CTILESTEST-39: pipeline params JSON --------------------------------------------------------


@pytest.mark.parametrize(
    "input_type, filename",
    [
        ("skeleton", "pipeline_params_Rcrops.json"),
        ("foldlabel", "pipeline_params_Rlabels.json"),
        ("distmap", "pipeline_params_Rdistmaps.json"),
        ("extremities", "pipeline_params_Rextremities.json"),
    ],
)
def test_run_with_params_writes_params_json_next_to_crops(tmp_path, regions_json, calls, input_type, filename):
    """REQ-CTILESTEST-39: the params dict is dumped as JSON in crop_dir under a per-input-type name."""
    params = _params(tmp_path, regions_json, input_type=input_type)
    gosr.run_with_params(params)

    (crops,) = _called(calls, "generate_crops")
    written = os.path.join(crops["crop_dir"], filename)
    assert os.path.isfile(written)
    with open(written) as f:
        dumped = json.load(f)
    assert dumped["input_type"] == input_type
    assert dumped["region_name"] == REGION
    assert dumped["crops_dir"] == os.path.join(params["output_dir"], "crops", "canonical_25")


# --- REQ-CTILESTEST-40: clear_and_compute and the distbottom directory ------------------------------


def test_run_with_params_clear_and_compute_without_distbottom_dir_raises(tmp_path, regions_json, calls):
    """REQ-CTILESTEST-40: clear_and_compute + no <crop_dir>/Rdistbottom -> FileNotFoundError (DEFECT)."""
    with pytest.raises(FileNotFoundError):
        gosr.run_with_params(_params(tmp_path, regions_json, save_behavior="clear_and_compute"))

    names = [name for name, _ in calls]
    assert "generate_crops" in names
    assert "generate_distbottom_crops" not in names
