"""Unit tests for cortical_tiles.brainvisa.utils.foldlabel, wide junction and file API (REQ-CTILESTEST-81..85).

The thin-junction builder used by champollion_pipeline stage 2 is covered by
test_unit_utils_skeleton_foldlabel.py (REQ-CTILESTEST-17/18). This module pins the
``junction='wide'`` builder and ``generate_foldlabel_from_graph_file``.

A tiny ``CorticalFoldArg`` graph is built in memory. aims cannot write such a synthetic graph
with buckets to disk (it lacks the Morphologist object table), so the file API is exercised with
the module's ``aims`` reference replaced by a reader that serves the in-memory graph for one path
and delegates writing to the real ``aims.write``; outputs go to pytest tmp dirs.

The iteration order of graph vertices and edges is not insertion order, so labels are checked
through the relationships they encode, not through fixed per-vertex numbers.

Characterization tests: behaviour recorded on cortical_tiles dev 4141bd8.
"""

import types

import numpy as np
import pytest
from cortical_tiles.brainvisa.utils import foldlabel as fl
from soma import aims

BBOX_MAX = [6, 7, 8]

V1_SS = [(1, 1, 1), (2, 2, 2)]
V1_BOTTOM = [(3, 3, 3), (6, 6, 6)]  # (3,3,3) overlaps the pli de passage
V1_OTHER = [(5, 0, 0)]
V2_SS = [(1, 2, 3)]
V2_OTHER = [(4, 4, 4)]
V2_BOTTOM = [(6, 7, 8)]
JUNCTION = [(2, 2, 2), (0, 0, 0)]  # (2,2,2) overlaps V1 simple surface
HULL_JUNCTION = [(5, 5, 5)]
PLIDEPASSAGE = [(0, 1, 0), (3, 3, 3)]


def _bucket(voxels):
    bucket = aims.BucketMap_VOID()
    bucket.setSizeXYZT(1.0, 1.0, 1.0, 1.0)
    for voxel in voxels:
        bucket[0][aims.Point3d(*voxel)] = 0
    return aims.rc_ptr_BucketMap_VOID(bucket)


def _graph(v1_ss=V1_SS):
    g = aims.Graph("CorticalFoldArg")
    g["voxel_size"] = [2.0, 2.0, 2.0, 1.0]
    g["boundingbox_min"] = [0, 0, 0]
    g["boundingbox_max"] = BBOX_MAX
    v1 = g.addVertex("fold")
    v1["aims_ss"] = _bucket(v1_ss)
    v1["aims_bottom"] = _bucket(V1_BOTTOM)
    v1["aims_other"] = _bucket(V1_OTHER)
    v2 = g.addVertex("fold")
    v2["aims_ss"] = _bucket(V2_SS)
    v2["aims_other"] = _bucket(V2_OTHER)
    v2["aims_bottom"] = _bucket(V2_BOTTOM)
    g.addEdge(v1, v2, "junction")["aims_junction"] = _bucket(JUNCTION)
    g.addEdge(v1, v2, "hull_junction")["aims_junction"] = _bucket(HULL_JUNCTION)
    g.addEdge(v1, v2, "plidepassage")["aims_plidepassage"] = _bucket(PLIDEPASSAGE)
    return g


def _as_dict(vol):
    arr = np.asarray(vol)
    return {tuple(int(c) for c in idx[:3]): int(arr[tuple(idx)]) for idx in np.argwhere(arr)}


@pytest.fixture
def wide_labels():
    return _as_dict(fl.generate_foldlabel_from_graph(_graph(), junction="wide"))


# --- REQ-CTILESTEST-81: wide junction vertex labels ------------------------------------------------


def test_wide_foldlabel_vertex_labels(wide_labels):
    """REQ-CTILESTEST-81: vertex n (distinct, 1-based) -> other 1+n, simple surface 1000+n, bottom 2000+n."""
    n1 = wide_labels[V1_SS[0]] - 1000
    n2 = wide_labels[V2_SS[0]] - 1000
    assert {n1, n2} == {1, 2}
    assert wide_labels[V1_OTHER[0]] == 1 + n1
    assert wide_labels[V1_BOTTOM[1]] == 2000 + n1
    assert wide_labels[V2_OTHER[0]] == 1 + n2
    assert wide_labels[V2_BOTTOM[0]] == 2000 + n2


# --- REQ-CTILESTEST-82: wide junction edge labels --------------------------------------------------


def test_wide_foldlabel_edge_labels(wide_labels):
    """REQ-CTILESTEST-82: edge m (distinct, 1-based over all edges) -> junction 3000+m, pli de passage 4000+m."""
    m_junction = wide_labels[JUNCTION[1]] - 3000
    m_hull = wide_labels[HULL_JUNCTION[0]] - 3000
    m_pdp = wide_labels[PLIDEPASSAGE[0]] - 4000
    assert {m_junction, m_hull, m_pdp} == {1, 2, 3}
    assert wide_labels[JUNCTION[0]] == 3000 + m_junction


# --- REQ-CTILESTEST-83: wide junction precedence ---------------------------------------------------


def test_wide_foldlabel_edges_override_vertices(wide_labels):
    """REQ-CTILESTEST-83: voxels in both a vertex and an edge bucket carry the edge label."""
    assert 3001 <= wide_labels[(2, 2, 2)] <= 3003  # junction over simple surface
    assert 4001 <= wide_labels[(3, 3, 3)] <= 4003  # pli de passage over bottom


# --- REQ-CTILESTEST-84 / 85: generate_foldlabel_from_graph_file ------------------------------------


def _serve_graph(monkeypatch, graph_path, graph):
    def read(path):
        assert path == graph_path
        return graph

    monkeypatch.setattr(fl, "aims", types.SimpleNamespace(read=read, write=aims.write))


@pytest.mark.parametrize("junction", ["thin", "wide"])
def test_graph_file_writes_generated_foldlabel(tmp_path, monkeypatch, junction):
    """REQ-CTILESTEST-84: the written file equals generate_foldlabel_from_graph(graph read from file, junction)."""
    graph_path = str(tmp_path / "Rgraph.arg")
    out = str(tmp_path / "Rfoldlabel_s1.nii.gz")
    # One graph instance for both sides: aims vertex/edge iteration order differs between two instances
    # built identically, and in thin mode the junction/hull-junction label depends on which vertex is
    # visited first, so only the same instance gives a reproducible expected volume.
    graph = _graph()
    _serve_graph(monkeypatch, graph_path, graph)

    fl.generate_foldlabel_from_graph_file(graph_path, out, junction)

    written = aims.read(out)
    expected = fl.generate_foldlabel_from_graph(graph, junction)
    np.testing.assert_array_equal(np.asarray(written), np.asarray(expected))
    assert list(written.header()["voxel_size"])[:3] == [2.0, 2.0, 2.0]


def test_graph_file_wraps_generation_error(tmp_path, monkeypatch):
    """REQ-CTILESTEST-85: a generation error is re-raised as Exception naming the graph file, chained, no output."""
    graph_path = str(tmp_path / "Rgraph.arg")
    out = tmp_path / "Rfoldlabel_s1.nii.gz"
    _serve_graph(monkeypatch, graph_path, _graph(v1_ss=[(50, 50, 50)]))  # outside the bounding box

    with pytest.raises(Exception, match=f"Error generating foldlabel from {graph_path}") as info:
        fl.generate_foldlabel_from_graph_file(graph_path, str(out), "thin")
    assert type(info.value) is Exception
    assert isinstance(info.value.__cause__, IndexError)
    assert not out.exists()
