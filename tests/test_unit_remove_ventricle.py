"""Unit tests for cortical_tiles.brainvisa.remove_ventricle (REQ-CTILESTEST-140..147).

champollion_pipeline's whole-brain path (run_cortical_tiles.py) calls
``remove_ventricle(side="F", ...)`` after fusing L and R skeletons. The end-to-end path
on the tracked subject is covered by test_full_brain.py; these tests pin the remaining
branches with a tiny in-memory ``CorticalFoldArg`` graph and volumes written to a pytest
tmp dir (nothing read from or written to the checkout, cwd-independent).
"""

import os

import numpy as np
import pytest
from cortical_tiles.brainvisa import remove_ventricle as rv
from soma import aims

LABELLING_SESSION = "deepcnn_session_auto"


def _bucket(voxels):
    bucket = aims.BucketMap_VOID()
    bucket.setSizeXYZT(1.0, 1.0, 1.0, 1.0)
    for voxel in voxels:
        bucket[0][aims.Point3d(*voxel)] = 0
    return aims.rc_ptr_BucketMap_VOID(bucket)


def _graph(ventricle=True, ventricle_ss=((1, 1, 1), (2, 3, 4))):
    """Graph with one sulcus vertex and (optionally) one ventricle vertex.

    The ventricle vertex has a non-empty simple-surface bucket, a non-empty bottom
    bucket, an EMPTY other bucket, and a junction edge to the sulcus vertex.
    """
    g = aims.Graph("CorticalFoldArg")
    g["voxel_size"] = [1.0, 1.0, 1.0, 1.0]
    sulcus = g.addVertex("fold")
    sulcus["label"] = "S.C._left"
    sulcus["aims_ss"] = _bucket([(0, 0, 0), (5, 5, 5)])
    if ventricle:
        vent = g.addVertex("fold")
        vent["label"] = "ventricle_left"
        vent["aims_ss"] = _bucket(list(ventricle_ss))
        vent["aims_bottom"] = _bucket([(4, 4, 4)])
        vent["aims_other"] = _bucket([])
        g.addEdge(vent, sulcus, "junction")["aims_junction"] = _bucket([(3, 3, 3)])
        g.addEdge(vent, sulcus, "plidepassage")["aims_plidepassage"] = _bucket([(0, 5, 0)])
    return g


def _volume(shape=(6, 6, 6), value=30):
    vol = aims.Volume(list(shape), dtype="S16")
    vol.header()["voxel_size"] = [1.0, 1.0, 1.0, 1.0]
    vol.fill(value)
    return vol


# --- REQ-CTILESTEST-140: native-grid removal -----------------------------------------------------


def test_native_removal_zeroes_exactly_ventricle_vertex_and_edge_buckets():
    """REQ-CTILESTEST-140: transform=None zeroes every non-empty ventricle bucket voxel, nothing else."""
    vol = _volume()
    out = rv.remove_ventricle_from_graph(vol, _graph(), background=0)

    expected = np.full((6, 6, 6), 30, dtype=np.int16)
    for voxel in [(1, 1, 1), (2, 3, 4), (4, 4, 4), (3, 3, 3), (0, 5, 0)]:
        expected[voxel] = 0
    np.testing.assert_array_equal(np.asarray(out)[..., 0], expected)


def test_native_removal_uses_given_background_value():
    """REQ-CTILESTEST-140: removed voxels take the ``background`` value."""
    out = rv.remove_ventricle_from_graph(_volume(), _graph(), background=7)
    arr = np.asarray(out)[..., 0]
    assert arr[1, 1, 1] == 7 and arr[0, 0, 0] == 30


# --- REQ-CTILESTEST-141: no ventricle with a transform -------------------------------------------


def test_transform_without_ventricle_returns_volume_unchanged():
    """REQ-CTILESTEST-141: a graph with no ventricle vertex returns the same, unmodified volume."""
    vol = _volume()
    out = rv.remove_ventricle_from_graph(vol, _graph(ventricle=False), transform=aims.AffineTransformation3d())
    assert out is vol
    assert (np.asarray(out) == 30).all()


# --- REQ-CTILESTEST-142: resampled grid mismatch ------------------------------------------------


def test_transform_grid_mismatch_raises_out_of_bounds_error():
    """REQ-CTILESTEST-142: resampled mask grid (ICBM template) != volume grid -> VentricleVoxelOutOfBoundsError."""
    vol = _volume()
    with pytest.raises(rv.VentricleVoxelOutOfBoundsError, match="does not match target volume grid"):
        rv.remove_ventricle_from_graph(vol, _graph(), transform=aims.AffineTransformation3d())
    assert (np.asarray(vol) == 30).all()


# --- REQ-CTILESTEST-143: negative index with a transform -----------------------------------------


def test_transform_negative_ventricle_index_raises_out_of_bounds_error():
    """REQ-CTILESTEST-143: a negative ventricle voxel index with a transform -> VentricleVoxelOutOfBoundsError."""
    graph = _graph(ventricle_ss=((1, 1, 1), (-1, 2, 2)))
    with pytest.raises(rv.VentricleVoxelOutOfBoundsError, match="negative index"):
        rv.remove_ventricle_from_graph(_volume(), graph, transform=aims.AffineTransformation3d())


# --- REQ-CTILESTEST-144: BIDS labelled-graph path -------------------------------------------------


def _remover(tmp_path, side="L", bids=True, path_to_graph="t1mri/*/default_analysis/folds/3.1"):
    return rv.RemoveVentricleFromVolume(
        src_dir=str(tmp_path / "src"),
        output_dir=str(tmp_path / "out"),
        morpho_dir="/morpho",
        path_to_graph=path_to_graph,
        labelling_session=LABELLING_SESSION,
        src_filename="skeleton_generated_",
        output_filename="skeleton_generated_without_ventricle_",
        side=side,
        bids=bids,
        parallel=False,
    )


@pytest.mark.parametrize(
    "subject, sub_id, keys",
    [
        ("sub-01_ses-1_run-2", "sub-01", "ses-1_run-2"),
        ("_sub-01_ses-1", "sub-01", "ses-1"),  # leading '_' stripped first
        ("sub-02", "sub-02", ""),  # no session/acquisition/run
    ],
)
def test_bids_labelled_graph_path_replaces_star_by_keys(tmp_path, subject, sub_id, keys):
    """REQ-CTILESTEST-144: BIDS graph = <morpho>/<id>/<path_to_graph, '*'->keys>/<session>/<side><id>_<session>.arg."""
    remover = _remover(tmp_path, side="F")
    expected = [
        os.path.join(
            "/morpho",
            sub_id,
            f"t1mri/{keys}/default_analysis/folds/3.1",
            LABELLING_SESSION,
            f"{side}{sub_id}_{LABELLING_SESSION}.arg",
        )
        for side in ("L", "R")
    ]
    assert remover.get_labelled_graph(subject) == expected


# --- REQ-CTILESTEST-145: missing inputs are logged, not raised -----------------------------------


def _write_source(tmp_path, side, subject):
    src = tmp_path / "src" / side
    src.mkdir(parents=True, exist_ok=True)
    path = src / f"{side}skeleton_generated_{subject}.nii.gz"
    aims.write(_volume(), str(path))
    return path


@pytest.mark.parametrize("missing", ["source", "graph"])
def test_missing_source_or_graph_logs_error_and_writes_nothing(tmp_path, caplog, missing):
    """REQ-CTILESTEST-145: missing source volume or labelled graph -> error logged, no output, no exception."""
    if missing == "graph":
        _write_source(tmp_path, "L", "s01")
    remover = _remover(tmp_path, bids=False, path_to_graph="folds")

    with caplog.at_level("ERROR"):
        remover.remove_ventricle_from_one_subject("s01")

    assert os.listdir(tmp_path / "out" / "L") == []
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and errors[0].startswith("s01: FileNotFoundError")
    assert ("Labelled graph not found" if missing == "graph" else "Source file not found") in errors[0]


# --- REQ-CTILESTEST-146: missing source directory ------------------------------------------------


def test_compute_raises_not_a_directory_for_missing_side_dir(tmp_path):
    """REQ-CTILESTEST-146: compute raises NotADirectoryError when <src_dir>/<side> is not a directory."""
    remover = _remover(tmp_path, bids=False)
    with pytest.raises(NotADirectoryError):
        remover.compute(number_subjects=-1)


# --- REQ-CTILESTEST-147: main / parse_args -------------------------------------------------------


def test_main_calls_remove_ventricle_with_parsed_arguments(tmp_path, monkeypatch):
    """REQ-CTILESTEST-147: main(argv) calls remove_ventricle once; dirs absolute, nb_subjects int, rest as parsed."""
    received = []
    monkeypatch.setattr(rv, "remove_ventricle", lambda **kwargs: received.append(kwargs))
    monkeypatch.chdir(tmp_path)

    rv.main(
        [
            "-s",
            "skel",
            "-o",
            "out",
            "-m",
            "morpho",
            "-p",
            "PG",
            "-l",
            "SESS",
            "-f",
            "src_",
            "-e",
            "dst_",
            "-i",
            "F",
            "-b",
            "-a",
            "-n",
            "3",
        ]
    )

    assert received == [
        dict(
            src_dir=str(tmp_path / "skel"),
            output_dir=str(tmp_path / "out"),
            morpho_dir=str(tmp_path / "morpho"),
            path_to_graph="PG",
            labelling_session="SESS",
            src_filename="src_",
            output_filename="dst_",
            side="F",
            bids=True,
            parallel=True,
            number_subjects=3,
        )
    ]


def test_main_all_subjects_maps_to_minus_one(tmp_path, monkeypatch):
    """REQ-CTILESTEST-147: '-n all' (the default) reaches remove_ventricle as -1."""
    received = []
    monkeypatch.setattr(rv, "remove_ventricle", lambda **kwargs: received.append(kwargs))
    rv.main(["-o", str(tmp_path / "out"), "-i", "R"])
    assert received[0]["number_subjects"] == -1
    assert received[0]["side"] == "R"
