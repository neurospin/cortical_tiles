"""Unit tests for cortical_tiles.brainvisa.compute_mask (REQ-CTILESTEST-66..72).

``compute_mask`` builds a region mask by counting, per ICBM2009c voxel, how many manually labelled
subjects have the sulcus there. generate_one_sulcal_region.run_with_params calls it for every sulcus
whose mask file is missing. The tests use tiny in-memory ``CorticalFoldArg`` graphs: aims cannot write
synthetic graphs with buckets to .arg without a full data directory, so empty ``.arg`` placeholder
files are created for the subject/graph discovery (``glob``) and ``compute_mask.aims`` is replaced, on
the imported module only, by a proxy whose ``read`` returns the in-memory graph for those paths and
delegates everything else to ``soma.aims``. Outputs go to pytest tmp dirs.

Characterization tests: behaviour recorded on cortical_tiles dev 1b2bd48. Tests flagged "DEFECT"
pin current behaviour that looks wrong; it is recorded, not fixed (tests-only rule of TASK-114).
"""

import os

import numpy as np
import pytest
from cortical_tiles.brainvisa import compute_mask as cm
from soma import aims

SIDE = "R"
SULCUS = "S.C."
PATH_TO_GRAPH = "t1mri/default/folds"
OUT_VS = (2.0, 2.0, 2.0)

# Voxels (1 mm graph) of S.C._right per subject: subA's two voxels fall in the same 2 mm voxel.
SUBJECT_VOXELS = {
    "subA": [(10, 10, 10), (11, 10, 10)],
    "subB": [(10, 10, 10), (20, 20, 20)],
}
OTHER_SULCUS_VOXELS = [(5, 5, 5)]


def _bucket(voxels):
    bucket = aims.BucketMap_VOID()
    bucket.setSizeXYZT(1.0, 1.0, 1.0, 1.0)
    for voxel in voxels:
        bucket[0][aims.Point3d(*voxel)] = 0
    return aims.rc_ptr_BucketMap_VOID(bucket)


def _graph(voxels):
    g = aims.Graph("CorticalFoldArg")
    g["voxel_size"] = [1.0, 1.0, 1.0, 1.0]
    g["boundingbox_min"] = [0, 0, 0]
    g["boundingbox_max"] = [30, 30, 30]
    sulcus = g.addVertex("fold")
    sulcus["name"] = "S.C._right"
    sulcus["aims_ss"] = _bucket(voxels[:1])
    sulcus["aims_bottom"] = _bucket(voxels[1:])
    other = g.addVertex("fold")
    other["name"] = "F.C.M._right"
    other["aims_other"] = _bucket(OTHER_SULCUS_VOXELS)
    return g


def _expected_index(graph, voxel):
    """Mask index of a 1 mm graph voxel: ICBM2009c position / output voxel size, rounded."""
    transform = aims.GraphManip.getICBM2009cTemplateTransform(graph)
    position = np.asarray(transform.transform(np.asarray(voxel, dtype=float)))
    return tuple(np.round(position / np.asarray(OUT_VS)).astype(int))


class _AimsProxy:
    def __init__(self, graphs):
        self._graphs = graphs

    def read(self, filename, *args, **kwargs):
        if filename in self._graphs:
            return self._graphs[filename]
        return aims.read(filename, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(aims, name)


@pytest.fixture
def database(tmp_path, monkeypatch):
    """<src>/<subject>/<PATH_TO_GRAPH>/R<subject>_auto.arg placeholders backed by in-memory graphs."""
    graphs = {}
    src = tmp_path / "src"
    for subject, voxels in SUBJECT_VOXELS.items():
        graph_dir = src / subject / PATH_TO_GRAPH
        graph_dir.mkdir(parents=True)
        graph_file = graph_dir / f"{SIDE}{subject}_auto.arg"
        graph_file.write_text("")
        graphs[str(graph_file)] = _graph(voxels)
    monkeypatch.setattr(cm, "aims", _AimsProxy(graphs))
    return src, graphs


def _compute(src, mask_dir):
    return cm.compute_mask(
        src_dir=str(src),
        mask_dir=str(mask_dir),
        path_to_graph=PATH_TO_GRAPH,
        sulcus=SULCUS,
        side=SIDE,
        out_voxel_size=OUT_VS[0],
    )


def _expected_counts(graphs):
    counts = {}
    for graph in graphs.values():
        indices = set()
        for vertex in graph.vertices():
            if vertex["name"] != "S.C._right":
                continue
            for key in ("aims_ss", "aims_bottom"):
                indices |= {_expected_index(graph, tuple(v)) for v in vertex[key][0].keys()}
        for index in indices:
            counts[index] = counts.get(index, 0) + 1
    return counts


def _nonzero(arr):
    return {tuple(int(c) for c in idx[:3]): int(arr[tuple(idx)]) for idx in np.argwhere(arr)}


# --- REQ-CTILESTEST-66: file naming -----------------------------------------------------------------


@pytest.mark.parametrize("side, new_sulcus, name", [("R", None, "S.C._right"), ("L", "S.X.", "S.X._left")])
def test_mask_file_and_sample_dir_naming(tmp_path, side, new_sulcus, name):
    """REQ-CTILESTEST-66: mask <mask_dir>/<side>/<name>.nii.gz, samples in <mask_dir>/<side>/<name>."""
    mask = cm.MaskAroundSulcus(src_dir="/src", mask_dir=str(tmp_path), sulcus=SULCUS, new_sulcus=new_sulcus, side=side)
    assert mask.mask_file == os.path.join(str(tmp_path), side, f"{name}.nii.gz")
    assert mask.mask_sample_dir == f"{tmp_path}/{side}/{name}"


# --- REQ-CTILESTEST-67 / 68: increment_one_mask -----------------------------------------------------


def test_increment_one_mask_adds_one_per_covered_voxel(database):
    """REQ-CTILESTEST-67: +1 once per mask voxel covered by the named sulcus; other sulci ignored."""
    _, graphs = database
    graph_file, graph = next(iter(graphs.items()))
    mask = cm.initialize_mask(OUT_VS)
    np.asarray(mask)[...] = 3

    cm.increment_one_mask(graph_file, mask, "S.C._right", OUT_VS)

    covered = set(_expected_counts({graph_file: graph}))
    arr = np.asarray(mask)
    assert {tuple(int(c) for c in idx[:3]) for idx in np.argwhere(arr == 4)} == covered
    assert np.count_nonzero(arr == 3) == arr.size - len(covered)


def test_increment_one_mask_returns_binary_subject_volume(database):
    """REQ-CTILESTEST-68: returns a 0/1 volume of the covered voxels with the mask's voxel size."""
    _, graphs = database
    graph_file, graph = next(iter(graphs.items()))
    mask = cm.initialize_mask(OUT_VS)

    vol_one = cm.increment_one_mask(graph_file, mask, "S.C._right", OUT_VS)

    assert _nonzero(np.asarray(vol_one)) == {idx: 1 for idx in _expected_counts({graph_file: graph})}
    assert list(vol_one.header()["voxel_size"])[:3] == list(OUT_VS)


# --- REQ-CTILESTEST-69 / 70: compute ----------------------------------------------------------------


def test_compute_mask_counts_subjects_per_voxel(tmp_path, database):
    """REQ-CTILESTEST-69: final mask voxel = number of subjects covering it; one 0/1 sample per subject."""
    src, graphs = database
    mask_dir = tmp_path / "mask"
    _compute(src, mask_dir)

    final = aims.read(str(mask_dir / SIDE / "S.C._right.nii.gz"))
    assert _nonzero(np.asarray(final)) == _expected_counts(graphs)
    assert 2 in _expected_counts(graphs).values()  # the shared (10,10,10) voxel
    for subject in SUBJECT_VOXELS:
        sample = np.asarray(aims.read(str(mask_dir / SIDE / "S.C._right" / f"{subject}.nii.gz")))
        assert set(np.unique(sample).tolist()) == {0, 1}


def test_compute_mask_resume_rewrites_mask_from_remaining_subjects_only(tmp_path, database):
    """REQ-CTILESTEST-70: resume overwrites the mask with the remaining subjects' counts (DEFECT)."""
    src, graphs = database
    mask_dir = tmp_path / "mask"
    _compute(src, mask_dir)
    (mask_dir / SIDE / "S.C._right" / "subB.nii.gz").unlink()

    _compute(src, mask_dir)

    final = aims.read(str(mask_dir / SIDE / "S.C._right.nii.gz"))
    sub_b = {k: v for k, v in graphs.items() if "subB" in k}
    assert _nonzero(np.asarray(final)) == _expected_counts(sub_b)


# --- REQ-CTILESTEST-71: missing graph ---------------------------------------------------------------


def test_compute_mask_subject_without_graph_raises_runtime_error(tmp_path, database):
    """REQ-CTILESTEST-71: a subject directory without matching .arg graph -> RuntimeError."""
    src, _ = database
    (src / "subC").mkdir()
    with pytest.raises(RuntimeError, match="No graph file"):
        _compute(src, tmp_path / "mask")


# --- REQ-CTILESTEST-72: parse_args ------------------------------------------------------------------


def test_parse_args_maps_command_line_to_compute_mask_arguments(tmp_path):
    """REQ-CTILESTEST-72: -o -> mask_dir, -s -> src_dir list, -n all -> -1."""
    out = str(tmp_path / "out")
    params = cm.parse_args(["-s", "A", "B", "-o", out, "-u", "S.C.", "-i", "L", "-p", "PG", "-x", "2", "-n", "all"])

    assert params == {
        "src_dir": ["A", "B"],
        "path_to_graph": "PG",
        "mask_dir": out,
        "sulcus": "S.C.",
        "new_sulcus": None,
        "side": "L",
        "out_voxel_size": 2.0,
        "nb_subjects": -1,
    }
