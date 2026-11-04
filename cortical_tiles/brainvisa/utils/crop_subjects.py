"""Subject set of the stacked crop arrays of one side.

``quality_checks`` and ``quality_checks_extremities`` check that the stacked
arrays of a side agree with each other and with their subject CSVs;
``record_one_sided_subjects`` skip-lists the subjects having a resampled file
of one input type only, and ``restrict_stack`` drops excluded subjects from a
stacked array and its CSV.
"""

import os
import re
from os.path import abspath, basename, dirname, join

import numpy as np
import pandas as pd

from cortical_tiles.brainvisa.utils.skip_list import record_skip


def quality_checks(crop_dir, side):
    s = np.load(f"{crop_dir}/{side}skeleton.npy")
    f = np.load(f"{crop_dir}/{side}label.npy")

    # checks if same voxel position
    assert (s.shape == f.shape), (
        f"Skeleton and foldlabel of different shapes: {s.shape} != {f.shape}")
    assert (f[s == 0].sum() == 0), (
        f"Foldlabel and skeleton arrays with different non-zero positions: "
        f"{(f[s == 0] != 0).sum()} different non-zero positions")
    assert (s[f == 0].sum() == 0), (
        f"Foldlabel and skeleton arrays with different non-zero positions: "
        f"{(s[f == 0] != 0).sum()} different non-zero positions")

    # Checks if subjects are equal between foldlabel and skeleton
    dff = pd.read_csv(f"{crop_dir}/{side}label_subject.csv")
    dfs = pd.read_csv(f"{crop_dir}/{side}skeleton_subject.csv")
    assert (dff.equals(dfs)), \
        "List of subjects for foldlabel and skeleton are not equal"

    # Checks if numpy arrays and csvs are consistent
    assert (s.shape[0] == len(dfs)), \
        "Number of skeleton subjects differs between numpy array and csv"
    assert (f.shape[0] == len(dff)), \
        "Number of foldlabel subjects differs between numpy array and csv"


def quality_checks_extremities(crop_dir, side):
    s = np.load(f"{crop_dir}/{side}skeleton.npy")
    f = np.load(f"{crop_dir}/{side}extremities.npy")

    # checks if same voxel position
    assert (s.shape == f.shape), (
        "Skeleton and extremities of different shapes: "
        f"{s.shape} != {f.shape}")

    # Checks if subjects are equal between foldlabel and skeleton
    dff = pd.read_csv(f"{crop_dir}/{side}extremities_subject.csv")
    dfs = pd.read_csv(f"{crop_dir}/{side}skeleton_subject.csv")
    assert (dff.equals(dfs)), \
        "List of subjects for extremities and skeleton are not equal"

    # Checks if numpy arrays and csvs are consistent
    assert (s.shape[0] == len(dfs)), \
        "Number of skeleton subjects differs between numpy array and csv"
    assert (f.shape[0] == len(dff)), \
        "Number of extremities subjects differs between numpy array and csv"


def restrict_stack(crop_dir, file_basename, excluded, write_sub_id=False):
    """Drops excluded subjects' rows from a stacked npy and its csv"""
    csv_file = join(crop_dir, f"{file_basename}_subject.csv")
    if not os.path.isfile(csv_file):
        return
    subjects = pd.read_csv(csv_file, dtype=str)["Subject"]
    keep = ~subjects.isin(list(excluded)).to_numpy()
    if keep.all():
        return
    npy_file = join(crop_dir, f"{file_basename}.npy")
    np.save(npy_file, np.load(npy_file)[keep])
    subjects[keep].to_frame().to_csv(csv_file, index=False)
    if write_sub_id:
        np.save(join(crop_dir, "sub_id.npy"), subjects[keep].tolist())


def record_one_sided_subjects(src_dir, side, skip_dir, step, excluded):
    """Records and returns the one-sided subjects not yet in excluded"""
    src_dir = abspath(src_dir)
    side_dirs = {"foldlabel": join(src_dir, side), "skeleton": join(
        dirname(dirname(src_dir)), "skeletons", basename(src_dir), side)}
    found = {}
    for kind, side_dir in side_dirs.items():
        if not os.path.isdir(side_dir):
            raise NotADirectoryError(f"{side_dir} is not a directory")
        expr = re.compile(rf"^{side}resampled_{kind}_(.*)\.nii\.gz$")
        found[kind] = {m[1] for m in map(expr.search, os.listdir(side_dir))
                       if m}
    one_sided = (found["skeleton"] ^ found["foldlabel"]) - set(excluded)
    for subject in sorted(one_sided):
        missing = "foldlabel" if subject in found["skeleton"] else "skeleton"
        record_skip(skip_dir, side, step, subject, FileNotFoundError(
            f"no resampled {missing} file of side {side} for subject "
            f"{subject} (searched {side_dirs[missing]})"))
    return one_sided
