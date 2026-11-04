"""Subject set of the stacked crop arrays of one side.

``quality_checks`` and ``quality_checks_extremities`` check that the stacked
arrays of a side agree with each other and with their subject CSVs;
``record_one_sided_subjects`` skip-lists the subjects having a resampled file
of one input type only, ``restrict_stack`` drops excluded subjects from a
stacked array and its CSV, ``collect_failed_subjects`` reads the failures out
of the crop worker results, and ``check_stack_has_subjects`` raises
``NoValidSubjectError`` when a stack holds no subject.

A subject with an all-zero skeleton crop leaves the whole run:
``exclude_empty_rows`` finds it in the skeleton stack, ``mirror_empty_crops``
copies its entry to each side of the run, and ``exclude_empty_crops`` takes it
out of the stacks and cropped files of each region and side of the run.

A subject QC removes stays in memory, with no skip-list entry:
``crop_subject_keys`` and ``cropped_subjects`` list the keys a step can meet,
``qc_removed_subjects`` keeps those that pass QC by themselves or by their
``subject_form``. A subject QC keeps with one stale crop and no resampled file
is found by ``stale_crop_subjects`` and skip-listed by
``record_stale_crop_subjects``.
"""

import os
import re
from os.path import abspath, basename, dirname, join

import numpy as np
import pandas as pd

from cortical_tiles.brainvisa.utils.skip_list import \
    SKIP_SIDES, load_skip_entries, record_skip, subject_form
from cortical_tiles.brainvisa.utils.subjects import select_good_qc

# Named set "run stacks": (stack, sample directory, cropped file kind)
_RUN_STACKS = (
    ("skeleton", "crops", "skeleton"),
    ("label", "labels", "foldlabel"),
    ("extremities", "extremities", "extremities"),
    ("distmap", "distmaps", "distmap"),
    ("distbottom", "distbottom", "distbottom"),
)
_SAMPLE_DIRS = {kind: sample_dir for _, sample_dir, kind in _RUN_STACKS}


class NoValidSubjectError(RuntimeError):
    """No subject of a side is left in its stacked crop array"""


class EmptyCropError(ValueError):
    """The skeleton crop of a subject holds only zero voxels"""


def collect_failed_subjects(results):
    """Returns the subjects of the crop worker results that carry a reason"""
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return {subject for subject, reason in results if reason is not None}


def check_stack_has_subjects(crop_dir, file_basename, side):
    """Raises NoValidSubjectError when the subject CSV of a stack is empty"""
    csv_file = join(crop_dir, f"{file_basename}_subject.csv")
    if pd.read_csv(csv_file, dtype=str).empty:
        raise NoValidSubjectError(
            f"no valid subject left on side {side} in {csv_file}")


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


def _subjects_in(folder, expr):
    """Returns the first group of expr over the file names of a folder

    A missing folder holds no subject.
    """
    if not os.path.isdir(folder):
        return set()
    return {m[1] for m in map(expr.search, os.listdir(folder)) if m}


def _resampled_subjects(side_dir, side, kind):
    """Returns the subjects of the resampled files of a kind in side_dir"""
    return _subjects_in(
        side_dir, re.compile(rf"^{side}resampled_{kind}_(.*)\.nii\.gz$"))


def _resampled_side_dirs(src_dir, side):
    """Returns {kind: folder} of the resampled skeletons and foldlabels"""
    src_dir = abspath(src_dir)
    return {"foldlabel": join(src_dir, side), "skeleton": join(
        dirname(dirname(src_dir)), "skeletons", basename(src_dir), side)}


def record_one_sided_subjects(src_dir, side, skip_dir, step, excluded):
    """Records and returns the one-sided subjects not yet in excluded"""
    side_dirs = _resampled_side_dirs(src_dir, side)
    found = {}
    for kind, side_dir in side_dirs.items():
        if not os.path.isdir(side_dir):
            raise NotADirectoryError(f"{side_dir} is not a directory")
        found[kind] = _resampled_subjects(side_dir, side, kind)
    one_sided = (found["skeleton"] ^ found["foldlabel"]) - set(excluded)
    for subject in sorted(one_sided):
        missing = "foldlabel" if subject in found["skeleton"] else "skeleton"
        record_skip(skip_dir, side, step, subject, FileNotFoundError(
            f"no resampled {missing} file of side {side} for subject "
            f"{subject} (searched {side_dirs[missing]})"))
    return one_sided


def qc_removed_subjects(qc_path, keys):
    """Returns the keys that QC keeps neither by themselves nor by their form

    An empty qc_path keeps every key.
    """
    forms = {key: subject_form(key) for key in keys}
    kept = set(select_good_qc(sorted({*forms, *forms.values()}), qc_path))
    return {key for key, form in forms.items()
            if key not in kept and form not in kept}


def cropped_subjects(crop_dir, side, kind):
    """Returns the subjects having a cropped file of a kind on a side"""
    return _subjects_in(join(crop_dir, f"{side}{_SAMPLE_DIRS[kind]}"),
                        re.compile(rf"^(.*)_cropped_{kind}\.nii\.gz$"))


def crop_subject_keys(src_dir, crop_dir, side, input_type, earlier_stacks):
    """Returns each subject key a crop pass of input_type can meet

    Resampled files, cropped files of every kind (a subject whose resampled
    file is gone is named only there) and the CSVs of earlier_stacks.
    """
    keys = _resampled_subjects(join(src_dir, side), side, input_type)
    if input_type == "foldlabel":
        keys |= _resampled_subjects(
            _resampled_side_dirs(src_dir, side)["skeleton"], side, "skeleton")
    for kind in _SAMPLE_DIRS:
        keys |= cropped_subjects(crop_dir, side, kind)
    for stack in earlier_stacks:
        csv_file = join(crop_dir, f"{side}{stack}_subject.csv")
        if os.path.isfile(csv_file):
            keys |= set(pd.read_csv(csv_file, dtype=str)["Subject"])
    return keys


def stale_crop_subjects(src_dir, crop_dir, side):
    """Returns {subject: kind held} of the subjects left with one stale crop

    Such a subject has a cropped file of one kind in {skeleton, foldlabel}
    and no resampled file of these kinds.
    """
    resampled = set()
    for kind, side_dir in _resampled_side_dirs(src_dir, side).items():
        resampled |= _resampled_subjects(side_dir, side, kind)
    skeletons = cropped_subjects(crop_dir, side, "skeleton")
    foldlabels = cropped_subjects(crop_dir, side, "foldlabel")
    return {subject: "skeleton" if subject in skeletons else "foldlabel"
            for subject in (skeletons ^ foldlabels) - resampled}


def record_stale_crop_subjects(stale, side, skip_dir, step, excluded):
    """Records and returns the subjects of stale not yet in excluded"""
    recorded = set(stale) - set(excluded)
    for subject in sorted(recorded):
        record_skip(skip_dir, side, step, subject, FileNotFoundError(
            f"only a cropped {stale[subject]} file of side {side} is left "
            f"for subject {subject}: no resampled file"))
    return recorded


def remove_crop_file(path):
    """Deletes a cropped file and its .minf sidecar, each when present"""
    for name in (path, f"{path}.minf"):
        if os.path.exists(name):
            os.remove(name)


def exclude_empty_rows(crop_dir, side, skip_dir, step, cropped_path):
    """Records, deletes and drops the subjects with an all-zero skeleton row

    Returns the set of those subjects.
    """
    file_basename = f"{side}skeleton"
    subjects = pd.read_csv(
        join(crop_dir, f"{file_basename}_subject.csv"), dtype=str)["Subject"]
    rows = np.load(join(crop_dir, f"{file_basename}.npy"))
    empty = {subject for subject, row in zip(subjects, rows, strict=True)
             if not row.any()}
    for subject in sorted(empty):
        remove_crop_file(cropped_path(subject))
        record_skip(skip_dir, side, step, subject,
                    EmptyCropError(f"empty crop in {crop_dir}"))
    restrict_stack(crop_dir, file_basename, empty, write_sub_id=True)
    return empty


def mirror_empty_crops(skip_dir, sides):
    """Gives each side in sides an entry per empty crop; returns the subjects

    Reads the lists of every side, writes only to the sides in ``sides``.
    """
    found = {}
    for side in SKIP_SIDES:
        for entry in load_skip_entries(skip_dir, side):
            if entry.exc_type == EmptyCropError.__name__:
                found.setdefault(entry.subject, entry)
    for side in sides:
        listed = {(entry.subject, entry.step)
                  for entry in load_skip_entries(skip_dir, side)}
        for subject, entry in sorted(found.items()):
            if (subject, entry.step) not in listed:
                record_skip(skip_dir, side, entry.step, subject,
                            EmptyCropError(f"{entry.message} (opposite side)"))
    return set(found)


def _exclude_from_side(crop_dir, side, subjects):
    """Drops the subjects from the stacks of a side and deletes their files"""
    for stack, sample_dir, kind in _RUN_STACKS:
        restrict_stack(crop_dir, f"{side}{stack}", subjects)
        for subject in subjects:
            remove_crop_file(join(crop_dir, f"{side}{sample_dir}",
                                  f"{subject}_cropped_{kind}.nii.gz"))
    pickle_file = join(crop_dir, f"{side}distbottom.pkl")
    if os.path.isfile(pickle_file):
        pd.read_pickle(pickle_file).drop(
            columns=list(subjects), errors="ignore").to_pickle(pickle_file)


def exclude_empty_crops(crop_dirs, subjects, sides):
    """Drops the subjects from each crop_dir and side, sub_id.npy included"""
    for crop_dir in crop_dirs:
        for side in sides:
            _exclude_from_side(crop_dir, side, subjects)
        sub_id_file = join(crop_dir, "sub_id.npy")
        if os.path.isfile(sub_id_file):
            ids = np.load(sub_id_file)
            np.save(sub_id_file, ids[~np.isin(ids, list(subjects))])
