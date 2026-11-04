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

import inspect
import json
import logging
import os
import re

import pytest
from cortical_tiles.brainvisa import generate_one_sulcal_region as gosr
from cortical_tiles.brainvisa.utils.skip_list import record_skip

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

# The real functions, kept before the `calls` fixture replaces them, for their signatures.
SKIP_DIR_STEPS = {
    name: getattr(gosr, name)
    for name in (
        "generate_skeletons",
        "generate_foldlabels",
        "generate_ICBM2009c_transforms",
        "resample_files",
        "mask_foldlabel_files",
        "generate_crops",
    )
}

# The real check: `calls` replaces it, as an empty tree has no subject with both volumes.
COUNT_CHECK = gosr.check_if_number_skeletons_equals_number_foldlabels

# (input_type, out_voxel_size) -> the steps of SKIP_DIR_STEPS that run_with_params calls for it.
SKIP_DIR_RUNS = {
    ("skeleton", 2.0): ("generate_skeletons", "generate_ICBM2009c_transforms", "resample_files", "generate_crops"),
    ("distmap", 2.0): ("generate_skeletons", "generate_ICBM2009c_transforms", "resample_files", "generate_crops"),
    ("foldlabel", 2.0): (
        "generate_foldlabels",
        "generate_ICBM2009c_transforms",
        "resample_files",
        "mask_foldlabel_files",
        "generate_crops",
    ),
    ("extremities", 2.0): ("generate_ICBM2009c_transforms", "resample_files", "generate_crops"),
    ("skeleton", "raw"): ("generate_skeletons", "generate_crops"),
    ("foldlabel", "raw"): ("generate_foldlabels", "generate_crops"),
}


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
    monkeypatch.setattr(gosr, "check_if_number_skeletons_equals_number_foldlabels", lambda *args, **kwargs: None)
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


def test_get_sulci_list_unknown_region_raises_value_error_naming_region_and_json(regions_json):
    """REQ-CTDEFECTS-03 (inverts REQ-CTILESTEST-29): unknown region -> ValueError naming region and json_path."""
    with pytest.raises(ValueError) as excinfo:
        gosr.get_sulci_list("Not.A.Region.", "R", json_path=regions_json)
    assert "Not.A.Region." in str(excinfo.value)
    assert regions_json in str(excinfo.value)


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


def test_run_with_params_raw_crop_source_dir_is_raw_subdir(tmp_path, regions_json, calls):
    """REQ-CTDEFECTS-01 (inverts REQ-CTILESTEST-35): 'raw' skeleton crops read <output_dir>/skeletons/raw."""
    params = _params(tmp_path, regions_json, out_voxel_size="raw")
    gosr.run_with_params(params)

    (crops,) = _called(calls, "generate_crops")
    assert crops["src_dir"] == os.path.join(params["output_dir"], "skeletons", "raw")


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


def test_run_with_params_clear_and_compute_without_distbottom_dir_computes_distbottom(tmp_path, regions_json, calls):
    """REQ-CTDEFECTS-02 (inverts REQ-CTILESTEST-40): clear_and_compute + no <crop_dir>/Rdistbottom -> distbottom computed."""
    gosr.run_with_params(_params(tmp_path, regions_json, save_behavior="clear_and_compute"))

    names = [name for name, _ in calls]
    assert "generate_crops" in names
    assert "generate_distbottom_crops" in names


# --- REQ-CTFAIL-BDRABCZUK-7062EA551CE0: one shared per-side skip list ------------------------------


@pytest.mark.parametrize("side", ["R", "L"])
@pytest.mark.parametrize(
    "input_type, out_voxel_size, step",
    [(input_type, size, step) for (input_type, size), steps in SKIP_DIR_RUNS.items() for step in steps],
)
def test_run_with_params_passes_output_dir_as_skip_dir(
    tmp_path, regions_json, calls, input_type, out_voxel_size, step, side
):
    """REQ-CTFAIL-BDRABCZUK-7062EA551CE0: a step of the skip-list set receives skip_dir=<output_dir>."""
    params = _params(tmp_path, regions_json, input_type=input_type, out_voxel_size=out_voxel_size, side=side)
    gosr.run_with_params(params)

    (kwargs,) = _called(calls, step)
    assert kwargs.get("skip_dir") == params["output_dir"]
    # The recorders take any keyword: the real signature must accept what run_with_params passes.
    inspect.signature(SKIP_DIR_STEPS[step]).bind(**kwargs)


# --- skeleton / foldlabel count check: report per subject, raise when no subject is usable ----------


def _write_resampled(output_dir, side, skeletons=(), foldlabels=()):
    """Writes resampled volumes with their .minf sidecars; returns (resampled_dir, skeleton_dir)."""
    skeleton_dir = os.path.join(output_dir, "skeletons", "2mm")
    resampled_dir = os.path.join(output_dir, "foldlabels", "2mm")
    folders = (
        (os.path.join(skeleton_dir, side), "skeleton", skeletons),
        (os.path.join(resampled_dir, f"{side}_before_masking"), "foldlabel", foldlabels),
    )
    for folder, kind, subjects in folders:
        for subject in subjects:
            os.makedirs(folder, exist_ok=True)
            volume = os.path.join(folder, f"{side}resampled_{kind}_{subject}.nii.gz")
            for path in (volume, volume + ".minf"):
                with open(path, "w", encoding="utf-8"):
                    pass
    return resampled_dir, skeleton_dir


def _skip(skip_dir, side, subjects):
    for subject in subjects:
        record_skip(skip_dir, side, "generate_foldlabels", subject, RuntimeError("no graph"))


def _spy_count_check(monkeypatch, calls):
    """Replaces the count check by a recorder of its bound arguments, appended to `calls`."""
    signature = inspect.signature(COUNT_CHECK)

    def spy(*args, **kwargs):
        calls.append(("count_check", dict(signature.bind(*args, **kwargs).arguments)))

    monkeypatch.setattr(gosr, "check_if_number_skeletons_equals_number_foldlabels", spy)


def _assert_no_valid_subject(raised):
    # By name: NoValidSubjectError (crop guard, utils/crop_subjects.py) is not on this branch's base yet.
    assert type(raised).__name__ == "NoValidSubjectError"


def _no_valid_subject_error(resampled_dir, skeleton_dir, side, skip_dir):
    """Returns the error of a check that must find no subject with both volumes."""
    with pytest.raises(RuntimeError) as excinfo:
        COUNT_CHECK(resampled_dir, skeleton_dir, side, skip_dir)
    _assert_no_valid_subject(excinfo.value)
    return excinfo.value


def _warned(caplog, subject=None):
    """Returns the WARNING messages, those naming `subject` exactly when one is given."""
    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    if subject is None:
        return messages
    return [message for message in messages if re.search(rf"\b{subject}\b", message)]


@pytest.mark.parametrize(
    "skeletons, foldlabels",
    [
        ([], []),
        (["sub-1"], []),
        ([], ["sub-1"]),
        (["sub-1"], ["sub-2"]),
    ],
)
def test_count_check_raises_when_no_subject_has_both(tmp_path, skeletons, foldlabels):
    """REQ-CTFAIL-BDRABCZUK-93A3589FEB57: nothing usable, equal numbers of disjoint subjects included."""
    resampled_dir, skeleton_dir = _write_resampled(str(tmp_path), "R", skeletons, foldlabels)

    _no_valid_subject_error(resampled_dir, skeleton_dir, "R", str(tmp_path))


@pytest.mark.parametrize(
    "skeletons, foldlabels",
    [
        (["sub-1"], ["sub-1"]),
        (["sub-1", "sub-2"], ["sub-1"]),
    ],
)
def test_count_check_raises_when_only_skip_listed_subjects_have_both(tmp_path, skeletons, foldlabels):
    """REQ-CTFAIL-BDRABCZUK-93A3589FEB57: a skip-listed subject with both volumes is not usable."""
    resampled_dir, skeleton_dir = _write_resampled(str(tmp_path), "R", skeletons, foldlabels)
    _skip(str(tmp_path), "R", ["sub-1"])

    _no_valid_subject_error(resampled_dir, skeleton_dir, "R", str(tmp_path))


def test_count_check_warns_once_per_one_sided_subject(tmp_path, caplog):
    """REQ-CTFAIL-BDRABCZUK-8124960D2098: one record per subject, naming the missing input type only."""
    caplog.set_level(logging.WARNING)
    resampled_dir, skeleton_dir = _write_resampled(
        str(tmp_path), "R", ["sub-1", "sub-2", "sub-3"], ["sub-1", "sub-4"]
    )

    COUNT_CHECK(resampled_dir, skeleton_dir, "R", str(tmp_path))

    assert len(_warned(caplog)) == 3
    for subject, missing, present in (
        ("sub-2", "foldlabel", "skeleton"),
        ("sub-3", "foldlabel", "skeleton"),
        ("sub-4", "skeleton", "foldlabel"),
    ):
        (message,) = _warned(caplog, subject)
        assert missing in message
        assert present not in message


@pytest.mark.parametrize(
    "skeletons, foldlabels",
    [
        (["sub-1", "sub-2"], ["sub-1"]),
        (["sub-1"], ["sub-1", "sub-2"]),
        (["sub-1", "sub-2"], ["sub-1", "sub-3"]),
    ],
)
def test_count_check_returns_none_when_a_subject_has_both(tmp_path, skeletons, foldlabels):
    """REQ-CTFAIL-BDRABCZUK-FDEE30DE671F: a mismatch leaves the run going while one subject is usable."""
    resampled_dir, skeleton_dir = _write_resampled(str(tmp_path), "R", skeletons, foldlabels)

    assert COUNT_CHECK(resampled_dir, skeleton_dir, "R", str(tmp_path)) is None


@pytest.mark.parametrize("subjects", [["sub-1"], ["sub-1", "sub-2"]])
def test_count_check_returns_none_when_numbers_equal(tmp_path, caplog, subjects):
    """REQ-CTFAIL-BDRABCZUK-FDEE30DE671F: sidecars, the other side and masked foldlabels do not count."""
    caplog.set_level(logging.WARNING)
    resampled_dir, skeleton_dir = _write_resampled(str(tmp_path), "R", subjects, subjects)
    _write_resampled(str(tmp_path), "L", skeletons=["sub-9"])
    masked = os.path.join(resampled_dir, "R")
    os.makedirs(masked)
    with open(os.path.join(masked, "Rresampled_foldlabel_sub-9.nii.gz"), "w", encoding="utf-8"):
        pass

    assert COUNT_CHECK(resampled_dir, skeleton_dir, "R", str(tmp_path)) is None
    assert not _warned(caplog)


@pytest.mark.parametrize(
    "skeletons, foldlabels",
    [
        (["sub-1", "sub-2"], ["sub-2"]),
        (["sub-2"], ["sub-1", "sub-2"]),
        (["sub-1", "sub-2"], ["sub-1", "sub-2"]),
    ],
)
def test_count_check_leaves_skip_listed_subject_out(tmp_path, caplog, skeletons, foldlabels):
    """REQ-CTFAIL-BDRABCZUK-1BAC19089172: a volume left behind by a skip-listed subject is not reported."""
    caplog.set_level(logging.WARNING)
    resampled_dir, skeleton_dir = _write_resampled(str(tmp_path), "R", skeletons, foldlabels)
    _skip(str(tmp_path), "R", ["sub-1"])
    caplog.clear()

    assert COUNT_CHECK(resampled_dir, skeleton_dir, "R", str(tmp_path)) is None
    assert not _warned(caplog)


def test_count_check_ignores_skip_list_of_other_side(tmp_path, caplog):
    """REQ-CTFAIL-BDRABCZUK-1BAC19089172: only the skip list of the checked side excludes a subject."""
    caplog.set_level(logging.WARNING)
    resampled_dir, skeleton_dir = _write_resampled(str(tmp_path), "R", ["sub-1", "sub-2"], ["sub-2"])
    _skip(str(tmp_path), "L", ["sub-1"])
    caplog.clear()

    COUNT_CHECK(resampled_dir, skeleton_dir, "R", str(tmp_path))

    assert len(_warned(caplog, "sub-1")) == 1


def test_count_check_matches_skip_listed_subject_exactly(tmp_path, caplog):
    """REQ-CTFAIL-BDRABCZUK-1BAC19089172: skip-listed sub-1 does not exclude sub-10."""
    caplog.set_level(logging.WARNING)
    resampled_dir, skeleton_dir = _write_resampled(str(tmp_path), "R", ["sub-1", "sub-10", "sub-2"], ["sub-2"])
    _skip(str(tmp_path), "R", ["sub-1"])
    caplog.clear()

    COUNT_CHECK(resampled_dir, skeleton_dir, "R", str(tmp_path))

    assert len(_warned(caplog, "sub-10")) == 1
    assert not _warned(caplog, "sub-1")


def test_count_check_error_states_both_numbers(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-1A9EA2403DB5: the numbers compared, skip-listed subjects left out."""
    resampled_dir, skeleton_dir = _write_resampled(
        str(tmp_path), "R", ["sub-1", "sub-2", "sub-3", "sub-4"], ["sub-1"]
    )
    _skip(str(tmp_path), "R", ["sub-1"])

    raised = _no_valid_subject_error(resampled_dir, skeleton_dir, "R", str(tmp_path))

    # The temporary directory's own digits must not satisfy the assertion.
    message = str(raised).replace(str(tmp_path), "")
    assert {"3", "0"} <= set(re.findall(r"\b\d+\b", message))


@pytest.mark.parametrize("side", ["R", "L"])
def test_count_check_error_names_skip_list_path(tmp_path, side):
    """REQ-CTFAIL-BDRABCZUK-163609EF0EA9: the message points at <skip_dir>/skip_list_<side>.jsonl."""
    resampled_dir, skeleton_dir = _write_resampled(str(tmp_path), side, ["sub-1"], ["sub-2"])
    skip_dir = str(tmp_path / "skips")

    raised = _no_valid_subject_error(resampled_dir, skeleton_dir, side, skip_dir)

    assert os.path.join(skip_dir, f"skip_list_{side}.jsonl") in str(raised)


@pytest.mark.parametrize("side", ["R", "L"])
def test_run_with_params_calls_count_check_between_resample_and_mask(
    tmp_path, regions_json, calls, monkeypatch, side
):
    """REQ-CTFAIL-BDRABCZUK-F134A975CFB1: the check sees foldlabels/<vs>, skeletons/<vs>, the side and output_dir."""
    _spy_count_check(monkeypatch, calls)
    params = _params(tmp_path, regions_json, input_type="foldlabel", side=side)
    gosr.run_with_params(params)

    names = [name for name, _ in calls]
    assert "count_check" in names
    assert names.index("resample_files") < names.index("count_check") < names.index("mask_foldlabel_files")
    (checked,) = _called(calls, "count_check")
    out = params["output_dir"]
    assert checked == {
        "resampled_dir": os.path.join(out, "foldlabels", "2mm"),
        "skeleton_dir": os.path.join(out, "skeletons", "2mm"),
        "side": side,
        "skip_dir": out,
    }


def test_run_with_params_subject_limit_skips_count_check(tmp_path, regions_json, calls, monkeypatch):
    """REQ-CTFAIL-BDRABCZUK-4EF049A7A611: the first N subjects of two steps may differ once a subject is skipped."""
    _spy_count_check(monkeypatch, calls)
    params = _params(tmp_path, regions_json, input_type="foldlabel", nb_subjects=1)
    gosr.run_with_params(params)

    assert not _called(calls, "count_check")


def test_run_with_params_no_common_subject_stops_before_masking(tmp_path, regions_json, calls, monkeypatch):
    """REQ-CTFAIL-BDRABCZUK-881A3A936D24: the real check's error leaves run_with_params; nothing is masked."""
    monkeypatch.setattr(gosr, "check_if_number_skeletons_equals_number_foldlabels", COUNT_CHECK)
    params = _params(tmp_path, regions_json, input_type="foldlabel")
    _write_resampled(params["output_dir"], "R", ["sub-1"], ["sub-2"])

    with pytest.raises(RuntimeError) as excinfo:
        gosr.run_with_params(params)

    _assert_no_valid_subject(excinfo.value)
    assert not _called(calls, "mask_foldlabel_files")


def test_run_with_params_one_sided_subject_continues_to_masking(tmp_path, regions_json, calls, monkeypatch, caplog):
    """REQ-CTFAIL-BDRABCZUK-FDEE30DE671F: one one-sided subject is reported; the pass goes on."""
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(gosr, "check_if_number_skeletons_equals_number_foldlabels", COUNT_CHECK)
    params = _params(tmp_path, regions_json, input_type="foldlabel")
    _write_resampled(params["output_dir"], "R", ["sub-1", "sub-2"], ["sub-1"])

    gosr.run_with_params(params)

    assert len(_warned(caplog, "sub-2")) == 1
    assert len(_called(calls, "mask_foldlabel_files")) == 1
