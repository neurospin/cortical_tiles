"""Characterization smoke test of the crop generation used by champollion_pipeline stage 2.

champollion_pipeline's run_cortical_tiles.py writes a ``pipeline_loop_2mm.json`` into a
dataset directory and runs ``generate_sulcal_regions.py -d <that dir>`` from the
``cortical_tiles/brainvisa`` directory. This test drives the same entry point in-process
(``generate_sulcal_regions`` -> ``RegionPipelineRunner`` -> ``generate_one_sulcal_region.run_with_params``)
for one region (S.Or.), one side (R), one input type (skeleton) and one subject (100206),
using only repository-tracked inputs:

- sulcal graph: data/source/unsupervised/ANALYSIS/3T_morphologist/100206 (also used by test_benchmark)
- 2mm region mask: data/mask/canonical_25/2mm/R/S.Or._right.nii.gz

No tracked reference crop exists for this configuration, so the expectations below are a
characterization recorded on 2026-10-01 (pixi default env of champollion_pipeline, cortical_tiles
dev 3fdc2ef). The 2mm resampling step is not reproducible run to run: the raw skeleton and the
transform are byte-identical across runs but the resampled skeleton is not, and over 10 runs the
S.Or. crop held 550-572 non-zero voxels with value set within {0, 30, 35, 40, 60, 70, 80}. The test
therefore pins what is stable (crop geometry, dtype, masking, skeleton/distbottom consistency) and
bounds the voxel count with a band instead of an exact reference.

Every path is anchored on this file (not the cwd) and every output goes to a pytest tmp dir.
"""

import json
import os

import numpy as np
import pytest
from soma import aims

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAINVISA_DIR = os.path.join(REPO_DIR, "cortical_tiles", "brainvisa")
SUBJECT_DIR = os.path.join(REPO_DIR, "data/source/unsupervised/ANALYSIS/3T_morphologist/100206")
SUPERVISED_OUTPUT_DIR = os.path.join(REPO_DIR, "data")  # holds mask/canonical_25/2mm

SUBJECT = "100206"
REGION = "S.Or."
SIDE = "R"
PATH_TO_GRAPH = "t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto"
PATH_SK_WITH_HULL = "t1mri/default_acquisition/default_analysis/segmentation"

# Recorded characterization (see module docstring).
EXPECTED_CROP_SHAPE = (26, 34, 23, 1)  # bounding box of the 2mm S.Or._right mask
EXPECTED_VOXEL_SIZE = (2.0, 2.0, 2.0)
SKELETON_NONZERO_BAND = (500, 620)  # observed 550-572 over 10 runs
SKELETON_ALLOWED_VALUES = {0, 30, 35, 40, 60, 70, 80}
DISTBOTTOM_BACKGROUND = 32501  # value written wherever the skeleton crop is 0


def _pipeline_config(subjects_dir, output_dir, regions_json, labeled_dir):
    """Mirror of champollion_pipeline's pipeline_loop_2mm.json after run_cortical_tiles.py fills it in."""
    return {
        "save_behavior": "best",
        "side": SIDE,
        "out_voxel_size": 2.0,
        "region_name": REGION,
        "brain_regions_json": regions_json,
        "parallel": False,
        "nb_subjects": -1,
        "input_type": "skeleton",
        # Only read if a mask is missing: point it at an empty dir so a missing mask fails loudly.
        "labeled_subjects_dir": labeled_dir,
        "path_to_graph_supervised": "t1mri/t1/default_analysis/folds/3.3/base2018_manual",
        "supervised_output_dir": SUPERVISED_OUTPUT_DIR,
        "nb_subjects_mask": -1,
        "graphs_dir": subjects_dir,
        "path_to_graph": PATH_TO_GRAPH,
        "path_to_skeleton_with_hull": PATH_SK_WITH_HULL,
        "skel_qc_path": "",
        "output_dir": output_dir,
        "junction": "thin",
        "bids": False,
        "new_sulcus": None,
        "resampled_skel": False,
        "cropping_type": "mask",
        "combine_type": False,
        "no_mask": False,
        "threshold": 0,
        "dilation": 5,
        "skip_distbottom": False,
        "masks_version": "canonical_25",
    }


@pytest.fixture(scope="module")
def crop_dir(tmp_path_factory):
    """Runs generate_sulcal_regions once for S.Or./R/skeleton on subject 100206; returns the crop dir."""
    tmp = tmp_path_factory.mktemp("sulcal_region_smoke")
    subjects_dir = tmp / "subjects"
    subjects_dir.mkdir()
    # One-subject graphs dir: the tracked 3T_morphologist dir also holds 100307.
    os.symlink(SUBJECT_DIR, subjects_dir / SUBJECT)
    labeled_dir = tmp / "labeled_subjects_unused"
    labeled_dir.mkdir()
    regions_json = tmp / "sulci_regions.json"
    regions_json.write_text(json.dumps({"brain": {"S.Or._right": {"S.Or._right": 1}}}))
    dataset_dir = tmp / "dataset"
    dataset_dir.mkdir()
    output_dir = tmp / "derivatives"
    config = _pipeline_config(str(subjects_dir), str(output_dir), str(regions_json), str(labeled_dir))
    (dataset_dir / "pipeline_loop_2mm.json").write_text(json.dumps(config))

    # generate_sulcal_regions.py imports its sibling as a top-level module
    # ("from generate_one_sulcal_region import ..."), which is why the pipeline runs it
    # from the brainvisa dir; putting that dir on sys.path reproduces this without a chdir.
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(BRAINVISA_DIR)
        from generate_sulcal_regions import generate_sulcal_regions

        generate_sulcal_regions(
            regions=[REGION],
            sides=[SIDE],
            input_types=["skeleton"],
            path_dataset=str(dataset_dir),
            verbose="",
            output_dir=None,
            path_to_graph=PATH_TO_GRAPH,
            path_sk_with_hull=PATH_SK_WITH_HULL,
            sk_qc_path="",
            njobs=1,
        )
    return os.path.join(str(output_dir), "crops", "canonical_25", "2mm", REGION, "mask")


def _read_volume(path):
    assert os.path.isfile(path), f"missing crop file {path}"
    volume = aims.read(path)
    return np.asarray(volume).copy(), tuple(volume.header()["voxel_size"][:3])


def test_skeleton_crop_matches_characterization(crop_dir):
    """REQ-CTILESTEST-05: skeleton crop of S.Or./R for 100206 matches the recorded characterization."""
    skeleton, voxel_size = _read_volume(os.path.join(crop_dir, f"{SIDE}crops", f"{SUBJECT}_cropped_skeleton.nii.gz"))
    assert skeleton.shape == EXPECTED_CROP_SHAPE
    assert skeleton.dtype == np.int16
    assert voxel_size == EXPECTED_VOXEL_SIZE
    assert set(np.unique(skeleton).tolist()) <= SKELETON_ALLOWED_VALUES
    nonzero = int(np.count_nonzero(skeleton))
    low, high = SKELETON_NONZERO_BAND
    assert low <= nonzero <= high, f"{nonzero} non-zero skeleton voxels, expected {low}-{high}"

    # cropping_type "mask": nothing survives outside the region mask, cropped to the same box.
    mask, _ = _read_volume(os.path.join(crop_dir, f"{SIDE}mask_cropped.nii.gz"))
    assert mask.shape == EXPECTED_CROP_SHAPE
    assert int(np.count_nonzero(skeleton[mask == 0])) == 0

    # The stacked array consumed downstream holds exactly this one subject's crop.
    stacked = np.load(os.path.join(crop_dir, f"{SIDE}skeleton.npy"))
    assert stacked.shape == (1, *EXPECTED_CROP_SHAPE)
    np.testing.assert_array_equal(stacked[0], skeleton)
    assert np.load(os.path.join(crop_dir, "sub_id.npy")).tolist() == [SUBJECT]


def test_distbottom_crop_matches_skeleton_crop(crop_dir):
    """REQ-CTILESTEST-06: distbottom crop is background exactly where the skeleton crop is 0."""
    skeleton, _ = _read_volume(os.path.join(crop_dir, f"{SIDE}crops", f"{SUBJECT}_cropped_skeleton.nii.gz"))
    distbottom, voxel_size = _read_volume(
        os.path.join(crop_dir, f"{SIDE}distbottom", f"{SUBJECT}_cropped_distbottom.nii.gz")
    )
    assert distbottom.shape == EXPECTED_CROP_SHAPE
    assert distbottom.dtype == np.int16
    assert voxel_size == EXPECTED_VOXEL_SIZE
    np.testing.assert_array_equal(distbottom == DISTBOTTOM_BACKGROUND, skeleton == 0)
    assert int((distbottom[skeleton != 0] < 0).sum()) == 0

    stacked = np.load(os.path.join(crop_dir, f"{SIDE}distbottom.npy"))
    assert stacked.shape == (1, *EXPECTED_CROP_SHAPE)
    np.testing.assert_array_equal(stacked[0], distbottom)
