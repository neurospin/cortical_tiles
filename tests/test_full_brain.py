"""Regression tests for the full-brain (F) preprocessing path.

Covers:
- REQ-FULLBRAIN-01: add_left_and_right_volumes fuses L/ + R/ into F/
  voxel-wise by AddLeftandRightVolumes.priority_order.
- REQ-FULLBRAIN-02: remove_ventricle(side="F") strips the ventricle using
  both the L and the R labelled graphs.
- REQ-FULLBRAIN-03: add_left_and_right_volumes processes every subject
  when called without number_subjects.

Requirements are tracked in champollion_pipeline's elm/REQUIREMENTS.md.
"""

import glob
import os
from os.path import dirname, join

import numpy as np
from cortical_tiles.brainvisa.add_left_and_right_volumes import (
    AddLeftandRightVolumes,
    add_left_and_right_volumes,
)
from cortical_tiles.brainvisa.remove_ventricle import remove_ventricle
from soma import aims

_DATA_DIR = join(dirname(dirname(os.path.abspath(__file__))), "data")
_MORPHO_DIR = join(_DATA_DIR, "source", "unsupervised", "ANALYSIS", "3T_morphologist")
_PATH_TO_GRAPH = "t1mri/default_acquisition/default_analysis/folds/3.1"
_LABELLING_SESSION = "default_session_auto"
_SUBJECT = "100206"
_NATIVE_SKELETON = join(
    _MORPHO_DIR,
    _SUBJECT,
    "t1mri",
    "default_acquisition",
    "default_analysis",
    "segmentation",
    f"Lskeleton_{_SUBJECT}.nii.gz",
)

# (left value, right value) pairs covering: one side empty, identical
# values, and every direction of a genuine conflict.
_VOXEL_PAIRS = [
    (0, 0),
    (0, 30),
    (30, 0),
    (60, 60),
    (60, 30),
    (30, 60),
    (100, 30),
    (30, 100),
    (35, 110),
    (110, 35),
    (80, 60),
    (60, 80),
]


def _write_volume(array, path):
    vol = aims.Volume(array.astype(np.int16))
    vol.header()["voxel_size"] = [2.0, 2.0, 2.0, 1.0]
    aims.write(vol, path)


def _expected_fused_value(left, right):
    rank = AddLeftandRightVolumes.priority_order
    return left if rank[left] <= rank[right] else right


def _make_left_right_tree(src_dir, subjects):
    os.makedirs(join(src_dir, "L"))
    os.makedirs(join(src_dir, "R"))
    n = len(_VOXEL_PAIRS)
    left = np.zeros((n, 1, 1), dtype=np.int16)
    right = np.zeros((n, 1, 1), dtype=np.int16)
    for i, (lv, rv) in enumerate(_VOXEL_PAIRS):
        left[i, 0, 0] = lv
        right[i, 0, 0] = rv
    for subject in subjects:
        _write_volume(left, join(src_dir, "L", f"Lresampled_skeleton_{subject}.nii.gz"))
        _write_volume(right, join(src_dir, "R", f"Rresampled_skeleton_{subject}.nii.gz"))


def _ventricle_voxels(graph_path):
    """Voxels zeroed by remove_ventricle's contract: every bucket of a
    ventricle-labelled vertex and of each of its incident edges."""
    graph = aims.read(graph_path)
    voxels = set()
    for vertex in graph.vertices():
        if not vertex.get("label", "unknown").startswith("ventricle"):
            continue
        for name in ("aims_bottom", "aims_other", "aims_ss"):
            bucket = vertex.get(name)
            if bucket is not None:
                voxels.update(tuple(int(c) for c in p) for p in bucket[0].keys())
        for edge in vertex.edges():
            for name in ("aims_plidepassage", "aims_junction"):
                if name in edge:
                    voxels.update(tuple(int(c) for c in p) for p in edge[name][0].keys())
    return voxels


def _graph_path(side):
    return join(_MORPHO_DIR, _SUBJECT, _PATH_TO_GRAPH, _LABELLING_SESSION, f"{side}{_SUBJECT}_{_LABELLING_SESSION}.arg")


# ---------------------------------------------------------------------------
# REQ-FULLBRAIN-01
# ---------------------------------------------------------------------------


class TestAddLeftAndRightVolumesFusion:
    def test_fused_voxels_follow_priority_order(self, tmp_path):
        src_dir = str(tmp_path / "skeletons")
        _make_left_right_tree(src_dir, ["sub01"])

        add_left_and_right_volumes(src_dir=src_dir, number_subjects="all")

        out = join(src_dir, "F", "Fresampled_skeleton_sub01.nii.gz")
        assert os.path.exists(out), f"fused volume not written: {out}"
        fused = np.asarray(aims.read(out))[:, 0, 0, 0]
        expected = [_expected_fused_value(lv, rv) for lv, rv in _VOXEL_PAIRS]
        assert fused.tolist() == expected


# ---------------------------------------------------------------------------
# REQ-FULLBRAIN-02
# ---------------------------------------------------------------------------


class TestRemoveVentricleFullBrain:
    def test_side_f_strips_ventricle_of_both_graphs(self, tmp_path):
        left_voxels = _ventricle_voxels(_graph_path("L"))
        right_voxels = _ventricle_voxels(_graph_path("R"))
        # Sanity: the fixture really exercises both hemispheres.
        assert left_voxels and right_voxels
        assert left_voxels - right_voxels and right_voxels - left_voxels

        # Native Morphologist grid the graphs' bucket voxels index into.
        shape = np.asarray(aims.read(_NATIVE_SKELETON)).shape[:3]
        src_dir = tmp_path / "skeletons"
        (src_dir / "F").mkdir(parents=True)
        _write_volume(np.full(shape, 30, dtype=np.int16), str(src_dir / "F" / f"Fresampled_skeleton_{_SUBJECT}.nii.gz"))
        output_dir = tmp_path / "whole_brain"

        remove_ventricle(
            side="F",
            src_dir=str(src_dir),
            output_dir=str(output_dir),
            morpho_dir=_MORPHO_DIR,
            path_to_graph=_PATH_TO_GRAPH,
            labelling_session=_LABELLING_SESSION,
            src_filename="resampled_skeleton",
        )

        outputs = glob.glob(str(output_dir / "F" / "*.nii.gz"))
        assert len(outputs) == 1, f"expected one F output, got {outputs}"
        result = np.asarray(aims.read(outputs[0]))[..., 0]

        expected = np.full(shape, 30, dtype=np.int16)
        for i, j, k in left_voxels | right_voxels:
            expected[i, j, k] = 0
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# REQ-FULLBRAIN-03
# ---------------------------------------------------------------------------


class TestAddLeftAndRightVolumesDefaultSubjects:
    def test_default_number_subjects_processes_every_subject(self, tmp_path):
        src_dir = str(tmp_path / "skeletons")
        subjects = ["sub01", "sub02"]
        _make_left_right_tree(src_dir, subjects)

        add_left_and_right_volumes(src_dir=src_dir)

        written = sorted(os.path.basename(p) for p in glob.glob(join(src_dir, "F", "*.nii.gz")))
        assert written == [f"Fresampled_skeleton_{s}.nii.gz" for s in subjects]
