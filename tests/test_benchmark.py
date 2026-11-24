import os

import numpy as np
import pandas as pd
from cortical_tiles.brainvisa.benchmark_generation_distmap import Benchmark
from soma import aims

# Every input and reference file is resolved against the repository checkout
# (not the cwd) and is tracked in git, so the test is runnable from a clean
# clone. The 1mm masks were moved to data/mask/canonical_25/1mm in 9afcf1e /
# a2f67e9; S.C._right.nii.gz is byte-identical to the former data/mask/1mm one.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mask_dir = os.path.join(REPO_DIR, "data/mask/canonical_25/1mm")
src_dir = os.path.join(REPO_DIR, "data/source/unsupervised/ANALYSIS/3T_morphologist")
bbox_dir = os.path.join(REPO_DIR, "data/reference/bbox/")
ref_dir = os.path.join(REPO_DIR, "data/reference/benchmark")
# The expected abnormality list was never committed under data/reference; it
# lives with the tests (tests/data is not gitignored, data/reference content is
# upstream-owned).
ref_csv = os.path.join(REPO_DIR, "tests/data/benchmark/abnormality_test_suppr.csv")


def equal_skeletons(skel_ref, skel_target):
    """Returns True if skel1 and skel2 are identical"""
    equal_skeleton = np.array_equal(skel_ref, skel_target)
    return equal_skeleton


def equal_csv_files(csv1, csv2):
    """Returns True if csv1 and csv2 are identical"""
    csv1 = pd.read_csv(csv1)
    csv2 = pd.read_csv(csv2)
    equal_csv = csv1.equals(csv2)
    return equal_csv


def are_arrays_almost_equal(arr1, arr2, epsilon, max_number_different_pixels):
    """Returns True if arrays arr1 and arr2 are almost equal

    arr1 and arr2 are almost equal if at most max_number_different_pixels pixels
    differ by more than epsilon

    Args:
        arr1: first numpy array
        arr2: second numpy array
        epsilon: max allowed difference between pixels values
        max_number_different_pixels: max allowed different number of pixels

    Returns:
        equal_arrays: True if arrays are almost equal
        number_different_pixels: number of pixels differing by more thanepsilon
    """
    difference = abs(arr1 - arr2) >= epsilon
    number_different_pixels = np.count_nonzero(difference)
    equal_arrays = number_different_pixels <= max_number_different_pixels
    return equal_arrays, number_different_pixels


def test_suppr_benchmark(tmp_path):
    """Tests suppr benchmark generation"""
    tgt_dir = str(tmp_path / "benchmark1")
    os.makedirs(tgt_dir)
    sulci_list = ["S.C._right"]

    benchmark = Benchmark(
        1, "R", 200, sulci_list, data_dir=src_dir, saving_dir=tgt_dir, bbox_dir=bbox_dir, mask_dir=mask_dir
    )
    subjects_list = ["100206"]

    abnormality_test = []
    givers = []

    for sub in subjects_list:
        benchmark.get_simple_surfaces(sub)
        if benchmark.surfaces and len(benchmark.surfaces.keys()) > 0:
            benchmark.generate_skeleton(sub)
            save_sub = benchmark.delete_ss(sub)
            abnormality_test.append(save_sub)
            benchmark.save_file(save_sub)
            benchmark.save_lists(abnormality_test, givers, subjects_list)

    skel_target = aims.read(os.path.join(tgt_dir, "modified_skeleton_100206.nii.gz")).arraydata()
    skel_ref = aims.read(os.path.join(ref_dir, "skeleton_100206_suppr.nii.gz")).arraydata()

    equal_skel, nb_different_pixels = are_arrays_almost_equal(skel_ref, skel_target, 1, 0)
    print(f"skel_target: {np.unique(skel_target, return_counts=True)}")
    print(f"skel_ref: {np.unique(skel_ref, return_counts=True)}")
    print(f"nb of different pixels: {nb_different_pixels}")
    assert equal_skel

    tgt_csv = os.path.join(tgt_dir, "abnormality_test.csv")
    equal_csv = equal_csv_files(tgt_csv, ref_csv)
    assert equal_csv


def _simple_surface_indexes(tmp_path, n_prior_allocations):
    """Vertex indexes of Benchmark.surfaces, in key order, after some allocations.

    aims keeps the graph vertices in a set ordered by memory address, so the
    iteration order depends on the allocation history of the process.
    """
    held = [aims.Volume([10 + i, 10, 10], dtype="S16") for i in range(n_prior_allocations)]
    benchmark = Benchmark(
        1, "R", 200, ["S.C._right"], data_dir=src_dir, saving_dir=str(tmp_path), bbox_dir=bbox_dir, mask_dir=mask_dir
    )
    benchmark.get_simple_surfaces("100206")
    assert held is not None
    return [benchmark.surfaces[key]["index"] for key in sorted(benchmark.surfaces)]


def test_simple_surfaces_follow_vertex_index_order(tmp_path):
    """The int keys of Benchmark.surfaces do not depend on heap addresses"""
    for n_prior_allocations in (0, 3, 17):
        indexes = _simple_surface_indexes(tmp_path, n_prior_allocations)
        assert len(indexes) > 1
        assert indexes == sorted(indexes)


def test_delete_ss_leaves_global_random_state(tmp_path):
    """Picking the simple surface to delete does not reseed the global random module"""
    import random

    benchmark = Benchmark(
        1, "R", 200, ["S.C._right"], data_dir=src_dir, saving_dir=str(tmp_path), bbox_dir=bbox_dir, mask_dir=mask_dir
    )
    benchmark.get_simple_surfaces("100206")
    benchmark.generate_skeleton("100206")
    random.seed(1234)
    state = random.getstate()
    benchmark.delete_ss("100206")
    assert random.getstate() == state
