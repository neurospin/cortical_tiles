"""Unit tests for cortical_tiles.brainvisa.utils.skeleton and .foldlabel (REQ-CTILESTEST-14..18).

champollion_pipeline stage 2 builds the raw skeleton and foldlabel volumes of every subject
from its Morphologist graph (``generate_skeletons`` / ``generate_foldlabels``, junction
``'thin'`` in ``pipeline_loop_2mm.json``) before any resampling, so these functions are
deterministic (the VipSkeleton nondeterminism of TASK-126 happens later, in resample.py).

The tests use a tiny synthetic ``CorticalFoldArg`` graph built in memory: two fold vertices
with simple-surface / bottom / other buckets, one ``junction`` edge, one ``hull_junction``
edge and one ``plidepassage`` edge, with deliberately overlapping voxels so that the
write-order precedence of each junction mode is observable.
"""

import numpy as np
import pytest
from soma import aims

from cortical_tiles.brainvisa.utils.foldlabel import check_if_valid_foldlabel, generate_foldlabel_from_graph
from cortical_tiles.brainvisa.utils.skeleton import generate_skeleton_from_graph, is_skeleton

BBOX_MAX = [6, 7, 8]

# Voxels of the synthetic graph
V1_SS = [(1, 1, 1), (2, 2, 2)]
V1_BOTTOM = [(3, 3, 3)]
V2_SS = [(1, 2, 3)]
V2_OTHER = [(4, 4, 4)]
JUNCTION = [(2, 2, 2), (0, 0, 0)]  # (2,2,2) overlaps V1 simple surface
HULL_JUNCTION = [(5, 5, 5)]
PLIDEPASSAGE = [(0, 1, 0), (3, 3, 3)]  # (3,3,3) overlaps V1 bottom


def _bucket(voxels):
    bucket = aims.BucketMap_VOID()
    bucket.setSizeXYZT(1.0, 1.0, 1.0, 1.0)
    for voxel in voxels:
        bucket[0][aims.Point3d(*voxel)] = 0
    return aims.rc_ptr_BucketMap_VOID(bucket)


@pytest.fixture
def graph():
    g = aims.Graph("CorticalFoldArg")
    g["voxel_size"] = [2.0, 2.0, 2.0, 1.0]
    g["boundingbox_min"] = [0, 0, 0]
    g["boundingbox_max"] = BBOX_MAX
    v1 = g.addVertex("fold")
    v1["aims_ss"] = _bucket(V1_SS)
    v1["aims_bottom"] = _bucket(V1_BOTTOM)
    v2 = g.addVertex("fold")
    v2["aims_ss"] = _bucket(V2_SS)
    v2["aims_other"] = _bucket(V2_OTHER)
    g.addEdge(v1, v2, "junction")["aims_junction"] = _bucket(JUNCTION)
    g.addEdge(v1, v2, "hull_junction")["aims_junction"] = _bucket(HULL_JUNCTION)
    g.addEdge(v1, v2, "plidepassage")["aims_plidepassage"] = _bucket(PLIDEPASSAGE)
    return g


def _as_dict(vol):
    arr = np.asarray(vol)
    return {tuple(int(c) for c in idx[:3]): int(arr[tuple(idx)]) for idx in np.argwhere(arr)}


# --- REQ-CTILESTEST-14: is_skeleton ------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 30, 35, 60, 100, 110, 120])
def test_is_skeleton_accepts_each_skeleton_value(value):
    """REQ-CTILESTEST-14: arrays made only of skeleton values are skeletons."""
    arr = np.zeros((3, 3, 3, 1), dtype=np.int16)
    arr[1, 1, 1, 0] = value
    assert is_skeleton(arr)


@pytest.mark.parametrize("value", [1, 11, 40, 70, 80, 32501, -30])
def test_is_skeleton_rejects_any_other_value(value):
    """REQ-CTILESTEST-14: a single value outside {0,30,35,60,100,110,120} makes it not a skeleton."""
    arr = np.full((3, 3, 3, 1), 60, dtype=np.int16)
    arr[0, 0, 0, 0] = value
    assert not is_skeleton(arr)


# --- REQ-CTILESTEST-15 / 16: generate_skeleton_from_graph --------------------------------------


def test_thin_skeleton_values_and_precedence(graph):
    """REQ-CTILESTEST-15: thin junction -> vertex voxels (60/30/100) override edge voxels (110/35/120)."""
    vol = generate_skeleton_from_graph(graph, junction="thin")

    assert tuple(vol.shape) == (BBOX_MAX[0] + 1, BBOX_MAX[1] + 1, BBOX_MAX[2] + 1, 1)
    assert np.asarray(vol).dtype == np.int16
    assert _as_dict(vol) == {
        (1, 1, 1): 60,
        (2, 2, 2): 60,  # simple surface over junction
        (1, 2, 3): 60,
        (3, 3, 3): 30,  # bottom over pli de passage
        (4, 4, 4): 100,
        (0, 0, 0): 110,
        (5, 5, 5): 35,  # hull junction
        (0, 1, 0): 120,
    }


def test_wide_skeleton_edges_override_vertices(graph):
    """REQ-CTILESTEST-16: wide junction -> junction (110) and pli-de-passage (120) override vertex voxels."""
    vol = generate_skeleton_from_graph(graph, junction="wide")

    assert _as_dict(vol) == {
        (1, 1, 1): 60,
        (2, 2, 2): 110,  # junction over simple surface
        (1, 2, 3): 60,
        (3, 3, 3): 120,  # pli de passage over bottom
        (4, 4, 4): 100,
        (0, 0, 0): 110,
        (5, 5, 5): 110,  # hull junction buckets are plain junctions in wide mode
        (0, 1, 0): 120,
    }


# --- REQ-CTILESTEST-17: generate_foldlabel_from_graph (thin) ------------------------------------


def test_thin_foldlabel_encodes_vertex_identity(graph):
    """REQ-CTILESTEST-17: distinct n >= 1 per vertex; ss -> 1000+n, bottom -> 7000+n, other -> n."""
    labels = _as_dict(generate_foldlabel_from_graph(graph, junction="thin"))

    n1 = labels[V1_SS[0]] - 1000
    n2 = labels[V2_SS[0]] - 1000
    assert n1 >= 1 and n2 >= 1 and n1 != n2
    for voxel in V1_SS:
        assert labels[voxel] == 1000 + n1
    for voxel in V2_SS:
        assert labels[voxel] == 1000 + n2
    assert labels[V2_OTHER[0]] == n2


def test_thin_foldlabel_bottom_label(graph):
    """REQ-CTILESTEST-17: a bottom voxel not shared with any edge is labelled 7000+n."""
    graph_vertices = list(graph.vertices())
    extra = [(6, 6, 6)]
    for vertex in graph_vertices:
        if "aims_bottom" in vertex:
            vertex["aims_bottom"] = _bucket(V1_BOTTOM + extra)
    labels = _as_dict(generate_foldlabel_from_graph(graph, junction="thin"))

    n1 = labels[V1_SS[0]] - 1000
    assert labels[extra[0]] == 7000 + n1


# --- REQ-CTILESTEST-18: check_if_valid_foldlabel ------------------------------------------------


# REQ-CTILESTEST-18 used to pin "any non-zero multiple of 999 raises" (TASK-128 defect b:
# 7992 is bottom label 7000 + fold 992). Inverted by REQ-CTDEFECTS128-02.
@pytest.mark.parametrize("value", [999, 1998, 6993, 7992])
def test_foldlabel_check_accepts_nonzero_multiple_of_999(value):
    """REQ-CTDEFECTS128-02: a valid label that happens to be a multiple of 999 does not raise."""
    vol = aims.Volume(3, 3, 3, 1, dtype="S16")
    np.asarray(vol)[1, 1, 1, 0] = value
    assert check_if_valid_foldlabel(vol) is None


def test_foldlabel_check_accepts_other_labels():
    """REQ-CTILESTEST-18: a volume with 0 and no multiple of 999 passes (returns None)."""
    vol = aims.Volume(3, 3, 3, 1, dtype="S16")
    np.asarray(vol)[..., 0] = np.arange(27).reshape(3, 3, 3) + 1000
    np.asarray(vol)[0, 0, 0, 0] = 0
    assert check_if_valid_foldlabel(vol) is None


# --- REQ-CTDEFECTS128-03: real label overflow (TASK-128 defect b) -------------------------------


@pytest.mark.parametrize("junction", ["thin", "wide"])
def test_foldlabel_generation_rejects_graph_over_999_vertices(junction):
    """REQ-CTDEFECTS128-03: a graph with 1000 vertices raises ValueError (fold 1000 spills into the next label range)."""
    g = aims.Graph("CorticalFoldArg")
    g["voxel_size"] = [2.0, 2.0, 2.0, 1.0]
    g["boundingbox_min"] = [0, 0, 0]
    g["boundingbox_max"] = BBOX_MAX
    # Bucket-less vertices: the volume stays all-zero, so only the vertex count can reveal the
    # overflow (vertex iteration order is not insertion order, so per-vertex labels are unstable).
    for _ in range(1000):
        g.addVertex("fold")
    with pytest.raises(ValueError):
        generate_foldlabel_from_graph(g, junction=junction)
