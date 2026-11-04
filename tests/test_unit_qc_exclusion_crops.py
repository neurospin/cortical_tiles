"""Unit tests for the QC exclusion of cortical_tiles.brainvisa.generate_crops (DEF-CTFAIL-BDRABCZUK-BBE8B66A7B92).

Subjects that ``select_good_qc`` removes under ``--sk_qc_path`` (a qc value of 0, or no row in the QC file) are left
out of the stacked crop arrays of a side, whichever of their intermediate files an earlier run left on disk. The
exclusion stays in memory: such a subject gets no entry in the skip list of the side being cropped, so that it never
turns a run into exit code 3. A subject QC keeps that an earlier run left with one cropped file and no resampled file
cannot be in both stacks: it leaves them and is recorded. generate_distbottom_crops follows the same QC file, so the
distbottom stack stays aligned with the skeleton stack. In BIDS mode a key of the QC file is a folder name (sub-1) and a
crop key a file name (sub-1_ses-1): a key passes QC by itself or by its form without trailing session parts.

Layout and stack helpers come from test_unit_generate_crops, the run_with_params harness from
test_unit_generate_one_sulcal_region. Each crop pass runs in the order generate_sulcal_regions uses.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cortical_tiles.brainvisa import generate_crops as gc
from cortical_tiles.brainvisa import generate_distbottom_crops as gd
from cortical_tiles.brainvisa import generate_one_sulcal_region as gosr
from cortical_tiles.brainvisa.utils import skip_list
from hypothesis import Phase, assume, given, settings
from hypothesis import strategies as st

from tests.test_unit_generate_crops import (
    KIND_DIRS,
    SEEDS,
    SIDE,
    _read,
    _skip,
    _stacked_subjects,
    _subject_set_layout,
    _write_volume,
)
from tests.test_unit_generate_one_sulcal_region import _called, _params, calls, regions_json  # noqa: F401

OPPOSITE_SIDE = "L"
QC_ZERO = "qc_zero"
QC_UNMENTIONED = "unmentioned"
QC_KEPT = "kept"
REMOVAL_CAUSES = (QC_ZERO, QC_UNMENTIONED)
CROP_FOLDERS = {"skeleton": "crops", "foldlabel": "labels"}
SKELETON_FOLDLABEL = ("skeleton", "foldlabel")
INPUT_TYPES = ("skeleton", "foldlabel", "extremities", "distmap")
# The passes that come before an input_type and write stacks its own pass reads
EARLIER_PASSES = {"foldlabel": ("skeleton",), "extremities": ("skeleton",)}
OUTSIDE_SKELETON = 32501  # value of a distbottom voxel off the skeleton, as the distbottom quality_checks expects


def _qc_file(tmp_path, states):
    """Writes <tmp>/qc.tsv from {subject: QC_KEPT | QC_ZERO | QC_UNMENTIONED}; returns its path."""
    rows = [f"{subject}\t{int(state == QC_KEPT)}" for subject, state in states.items() if state != QC_UNMENTIONED]
    path = Path(tmp_path) / "qc.tsv"
    path.write_text("\n".join(["participant_id\tqc", *rows]) + "\n")
    return str(path)


def _crop_with_qc(tmp_path, out, input_type, **kwargs):
    gc.generate_crops(
        src_dir=str(out / KIND_DIRS[input_type] / "2mm"),
        skip_dir=str(out),
        input_type=input_type,
        crop_dir=str(Path(tmp_path) / "crops"),
        mask_dir=str(Path(tmp_path) / "mask"),
        side=SIDE,
        list_sulci=["S.C."],
        threshold=0,
        dilation=2,
        **kwargs,
    )


def _resampled(out, input_type, subject):
    return out / KIND_DIRS[input_type] / "2mm" / SIDE / f"{SIDE}resampled_{input_type}_{subject}.nii.gz"


def _cropped(root, input_type, subject):
    folder = f"{SIDE}{CROP_FOLDERS[input_type]}"
    return Path(root) / "crops" / folder / f"{subject}_cropped_{input_type}.nii.gz"


def _kinds_of(input_type):
    return {"skeleton", input_type}


def _other_kind(input_type):
    return "foldlabel" if input_type == "skeleton" else "skeleton"


def _skip_entries(out):
    return [(entry.subject, entry.step) for entry in skip_list.load_skip_entries(skip_dir=str(out), side=SIDE)]


def _stale_crop_layout(tmp_path, stale_kind, stale_subjects):
    """Crops a1 and stale_subjects, then leaves each stale subject with its stale_kind cropped file and nothing else."""
    subjects = ("a1", *stale_subjects)
    out = _subject_set_layout(tmp_path, skeleton=subjects, foldlabel=subjects)
    _crop_with_qc(tmp_path, out, "skeleton")
    _crop_with_qc(tmp_path, out, "foldlabel")
    for subject in stale_subjects:
        _resampled(out, "skeleton", subject).unlink()
        _resampled(out, "foldlabel", subject).unlink()
        _cropped(tmp_path, _other_kind(stale_kind), subject).unlink()
    return out


def _both_passes(tmp_path, out, qc_path):
    _crop_with_qc(tmp_path, out, "skeleton", qc_path=qc_path)
    _crop_with_qc(tmp_path, out, "foldlabel", qc_path=qc_path)


def _no_stack_check(monkeypatch):
    """Turns quality_checks off: it would stop the pass on split stacks before the test reaches its own assertion.
    test_property_stacks_agree_for_each_set_of_intermediate_files runs the same cases with quality_checks on."""
    monkeypatch.setattr(gc, "quality_checks", lambda *_: None)


@pytest.mark.parametrize("cause", REMOVAL_CAUSES)
@pytest.mark.parametrize("input_type", INPUT_TYPES)
def test_qc_removed_subject_is_left_out_of_its_stack(tmp_path, input_type, cause):
    """REQ-CTFAIL-BDRABCZUK-4EE90A47B94F: a subject removed by select_good_qc is out of the array and CSV of each
    input_type; the subject it keeps stays."""
    out = _subject_set_layout(tmp_path, **{kind: ("a1", "b2") for kind in _kinds_of(input_type)})
    qc_path = _qc_file(tmp_path, {"a1": QC_KEPT, "b2": cause})

    for earlier in EARLIER_PASSES.get(input_type, ()):
        _crop_with_qc(tmp_path, out, earlier, qc_path=qc_path)
    _crop_with_qc(tmp_path, out, input_type, qc_path=qc_path)

    assert _stacked_subjects(tmp_path, input_type) == ["a1"]


@pytest.mark.parametrize("resampled_left", [True, False], ids=["resampled_left", "resampled_gone"])
@pytest.mark.parametrize("input_type", ["skeleton", "distmap"])
def test_cropped_file_of_earlier_run_is_not_stacked(tmp_path, input_type, resampled_left):
    """REQ-CTFAIL-BDRABCZUK-C10B45169DC3: the cropped file an earlier run left for a removed subject is not stacked,
    also once its resampled file is gone, so that only the crop folder still names the subject."""
    out = _subject_set_layout(tmp_path, **{kind: ("a1", "b2") for kind in _kinds_of(input_type)})
    _crop_with_qc(tmp_path, out, input_type)
    if not resampled_left:
        _resampled(out, input_type, "b2").unlink()

    _crop_with_qc(tmp_path, out, input_type, qc_path=_qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_ZERO}))

    assert _stacked_subjects(tmp_path, input_type) == ["a1"]


def test_foldlabel_pass_leaves_removed_subject_out_of_the_skeleton_stack(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-CFC598550719: a skeleton stack written without QC loses the subject in the foldlabel pass;
    the two stacks agree, so quality_checks passes."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2"), foldlabel=("a1", "b2"))
    _crop_with_qc(tmp_path, out, "skeleton")

    _crop_with_qc(tmp_path, out, "foldlabel", qc_path=_qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_ZERO}))

    assert _stacked_subjects(tmp_path, "skeleton") == ["a1"]
    assert _stacked_subjects(tmp_path, "foldlabel") == ["a1"]


def test_extremities_pass_leaves_removed_subject_out_of_the_earlier_stacks(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-B19A3E1314B6: skeleton and foldlabel stacks written without QC lose the subject in the
    extremities pass."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2"), foldlabel=("a1", "b2"), extremities=("a1", "b2"))
    _crop_with_qc(tmp_path, out, "skeleton")
    _crop_with_qc(tmp_path, out, "foldlabel")

    _crop_with_qc(tmp_path, out, "extremities", qc_path=_qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_ZERO}))

    assert _stacked_subjects(tmp_path, "skeleton") == ["a1"]
    assert _stacked_subjects(tmp_path, "foldlabel") == ["a1"]


@pytest.mark.parametrize("kwargs", [{}, {"qc_path": ""}], ids=["default", "empty_string"])
def test_empty_qc_path_excludes_no_subject(tmp_path, kwargs):
    """REQ-CTFAIL-BDRABCZUK-2F32B7827F24: without a QC file the stack holds each subject outside the skip list."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2", "c3"))
    _skip(out, "c3")

    _crop_with_qc(tmp_path, out, "skeleton", **kwargs)

    assert _stacked_subjects(tmp_path, "skeleton") == ["a1", "b2"]


@pytest.mark.parametrize("input_type", INPUT_TYPES)
def test_run_with_params_passes_skel_qc_path_to_generate_crops(tmp_path, regions_json, calls, input_type):  # noqa: F811
    """REQ-CTFAIL-BDRABCZUK-2C566F6580CA: the crop step of a region receives the QC file of --sk_qc_path."""
    params = _params(tmp_path, regions_json, skel_qc_path="/data/qc.tsv", input_type=input_type)

    gosr.run_with_params(params)

    assert [kwargs.get("qc_path") for kwargs in _called(calls, "generate_crops")] == ["/data/qc.tsv"]


def test_qc_exclusion_leaves_the_stacks_of_the_opposite_side_unchanged(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-F188B75C2046: per-side exclusion, a pass on one side rewrites no stack of the other side.
    Holds before the change too: this test pins it."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2"))
    crops = tmp_path / "crops"
    crops.mkdir()
    opposite_npy = crops / f"{OPPOSITE_SIDE}skeleton.npy"
    opposite_csv = crops / f"{OPPOSITE_SIDE}skeleton_subject.csv"
    np.save(opposite_npy, np.zeros((2, 3)))
    opposite_csv.write_text("Subject\na1\nb2\n")
    before = opposite_npy.read_bytes(), opposite_csv.read_bytes()

    _crop_with_qc(tmp_path, out, "skeleton", qc_path=_qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_ZERO}))

    assert (opposite_npy.read_bytes(), opposite_csv.read_bytes()) == before


@pytest.mark.parametrize("earlier_entry", [True, False], ids=["earlier_entry", "no_list"])
def test_qc_exclusion_leaves_the_skip_list_of_the_opposite_side_unchanged(tmp_path, earlier_entry):
    """REQ-CTFAIL-BDRABCZUK-85D239C78F9F: per-side exclusion, a pass on one side writes nothing in the skip list of
    the other side. Holds before the change too: this test pins it."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2"))
    if earlier_entry:
        _skip(out, "a1", side=OPPOSITE_SIDE)
    before = skip_list.load_skip_entries(skip_dir=str(out), side=OPPOSITE_SIDE)

    _crop_with_qc(tmp_path, out, "skeleton", qc_path=_qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_ZERO}))

    assert skip_list.load_skip_entries(skip_dir=str(out), side=OPPOSITE_SIDE) == before


def test_regression_files_left_by_an_earlier_run_do_not_split_the_stacks(tmp_path):
    """INV-1070E8BBFD45, reproduction of DEF-CTFAIL-BDRABCZUK-BBE8B66A7B92: QC removes b2 and c3 but their skeletons
    are still on disk, c3 with its foldlabel, so the two stacks used to differ (779 != 776 on ABIDE I)."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2", "c3"), foldlabel=("a1", "c3"))
    qc_path = _qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_ZERO, "c3": QC_ZERO})

    _crop_with_qc(tmp_path, out, "skeleton", qc_path=qc_path)
    _crop_with_qc(tmp_path, out, "foldlabel", qc_path=qc_path)

    assert _stacked_subjects(tmp_path, "skeleton") == ["a1"]
    assert _stacked_subjects(tmp_path, "foldlabel") == ["a1"]


SUBJECT_CASES = st.tuples(
    st.sampled_from((QC_KEPT, *REMOVAL_CAUSES)),  # what the QC file says of the subject
    st.booleans(),  # a resampled skeleton lies on disk
    st.booleans(),  # a resampled foldlabel lies on disk
)


@settings(max_examples=12, deadline=None, derandomize=True, database=None, phases=(Phase.explicit, Phase.generate))
@given(
    cases=st.fixed_dictionaries({subject: SUBJECT_CASES for subject in ("a1", "b2", "c3", "d4")}),
    earlier_run=st.booleans(),
)
def test_property_stacks_agree_without_removed_subjects(cases, earlier_run):
    """INV-1070E8BBFD45: over generated QC files and files on disk, the skeleton and foldlabel stacks of a side list
    the subjects that QC keeps and that hold both volumes, with or without crops left by an earlier run."""
    two_sided_kept = sorted(
        s for s, (qc, skeleton, foldlabel) in cases.items() if qc == QC_KEPT and skeleton and foldlabel
    )
    assume(two_sided_kept)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = _subject_set_layout(
            root,
            skeleton=[subject for subject, (_, skeleton, _) in cases.items() if skeleton],
            foldlabel=[subject for subject, (_, _, foldlabel) in cases.items() if foldlabel],
        )
        qc_path = _qc_file(root, {subject: qc for subject, (qc, _, _) in cases.items()})

        if earlier_run:
            _crop_with_qc(root, out, "skeleton")
            _crop_with_qc(root, out, "foldlabel")
        _crop_with_qc(root, out, "skeleton", qc_path=qc_path)
        _crop_with_qc(root, out, "foldlabel", qc_path=qc_path)

        assert _stacked_subjects(root, "skeleton") == two_sided_kept
        assert _stacked_subjects(root, "foldlabel") == two_sided_kept


@pytest.mark.parametrize("missing", SKELETON_FOLDLABEL)
def test_one_sided_subject_kept_by_qc_is_recorded_beside_a_removed_one(tmp_path, missing):
    """REQ-CTFAIL-BDRABCZUK-15740CCB6D47: of two one-sided subjects, only the one QC keeps is recorded."""
    out = _subject_set_layout(tmp_path, **{_other_kind(missing): ("a1", "b2", "c3"), missing: ("a1",)})

    _both_passes(tmp_path, out, _qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_KEPT, "c3": QC_ZERO}))

    assert _skip_entries(out) == [("b2", "generate_crops")]


@pytest.mark.parametrize("cause", REMOVAL_CAUSES)
@pytest.mark.parametrize("missing", SKELETON_FOLDLABEL)
def test_qc_removed_subject_gets_no_skip_list_entry(tmp_path, missing, cause):
    """REQ-CTFAIL-BDRABCZUK-128D50705052: neither b2, one-sided, nor c3, with both files, gets an entry on the cropped
    side, so the exclusion stays in memory."""
    out = _subject_set_layout(tmp_path, **{_other_kind(missing): ("a1", "b2", "c3"), missing: ("a1", "c3")})

    _both_passes(tmp_path, out, _qc_file(tmp_path, {"a1": QC_KEPT, "b2": cause, "c3": cause}))

    assert _skip_entries(out) == []


@pytest.mark.parametrize("stale_kind", SKELETON_FOLDLABEL)
def test_qc_removed_subject_with_one_stale_crop_gets_no_entry(tmp_path, monkeypatch, stale_kind):
    """REQ-CTFAIL-BDRABCZUK-128D50705052: QC wins over the stale crop, the subject leaves the stacks with no entry."""
    _no_stack_check(monkeypatch)
    out = _stale_crop_layout(tmp_path, stale_kind, ("b2",))

    _both_passes(tmp_path, out, _qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_ZERO}))

    stacks = _stacked_subjects(tmp_path, "skeleton"), _stacked_subjects(tmp_path, "foldlabel")
    assert (stacks, _skip_entries(out)) == ((["a1"], ["a1"]), [])


@pytest.mark.parametrize("with_qc_file", [True, False], ids=["qc_file", "empty_qc_path"])
@pytest.mark.parametrize("stale_kind", SKELETON_FOLDLABEL)
def test_kept_subject_with_one_stale_crop_leaves_both_stacks(tmp_path, monkeypatch, stale_kind, with_qc_file):
    """REQ-CTFAIL-BDRABCZUK-EB1FB5090CE5: the stale crop of b2 has no other-kind partner: b2 is in no stack."""
    _no_stack_check(monkeypatch)
    out = _stale_crop_layout(tmp_path, stale_kind, ("b2",))

    _both_passes(tmp_path, out, _qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_KEPT}) if with_qc_file else "")

    assert (_stacked_subjects(tmp_path, "skeleton"), _stacked_subjects(tmp_path, "foldlabel")) == (["a1"], ["a1"])


@pytest.mark.parametrize("stale_kind", SKELETON_FOLDLABEL)
def test_kept_subject_with_one_stale_crop_is_recorded(tmp_path, monkeypatch, stale_kind):
    """REQ-CTFAIL-BDRABCZUK-1A5FECDF83CA: b2 is recorded with step generate_crops; c3, already skip-listed, is not
    recorded again."""
    _no_stack_check(monkeypatch)
    out = _stale_crop_layout(tmp_path, stale_kind, ("b2", "c3"))
    _skip(out, "c3")

    _both_passes(tmp_path, out, _qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_KEPT, "c3": QC_KEPT}))

    assert _skip_entries(out) == [("c3", "generate_foldlabels"), ("b2", "generate_crops")]


INTERMEDIATE_FILES = st.tuples(
    st.sampled_from((QC_KEPT, *REMOVAL_CAUSES)),  # what the QC file says of the subject
    st.booleans(),  # a resampled skeleton lies on disk
    st.booleans(),  # a resampled foldlabel lies on disk
    st.booleans(),  # a cropped skeleton lies on disk
    st.booleans(),  # a cropped foldlabel lies on disk
)
ALL_FILES_KEPT = (QC_KEPT, True, True, True, True)


def _subjects_of_both_stacks(cases):
    """Subjects QC keeps that hold both resampled files, or no resampled file and both cropped files."""
    return sorted(
        subject
        for subject, (qc, skeleton, foldlabel, skeleton_crop, foldlabel_crop) in cases.items()
        if qc == QC_KEPT and skeleton == foldlabel and (skeleton or (skeleton_crop and foldlabel_crop))
    )


@settings(max_examples=12, deadline=None, derandomize=True, database=None, phases=(Phase.explicit, Phase.generate))
@given(
    cases=st.fixed_dictionaries(
        {"a1": st.just(ALL_FILES_KEPT), **{subject: INTERMEDIATE_FILES for subject in ("b2", "c3", "d4")}}
    )
)
def test_property_stacks_agree_for_each_set_of_intermediate_files(cases):
    """REQ-INV-BDRABCZUK-B39B20A54BE9, REQ-CTFAIL-BDRABCZUK-EB1FB5090CE5: over generated QC files and each mix of
    resampled and cropped files an earlier run left, quality_checks passes and both stacks list the same subjects."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = _subject_set_layout(root, skeleton=tuple(cases), foldlabel=tuple(cases))
        _crop_with_qc(root, out, "skeleton")
        _crop_with_qc(root, out, "foldlabel")
        for subject, (_, skeleton, foldlabel, skeleton_crop, foldlabel_crop) in cases.items():
            for kind, resampled_left, crop_left in (
                ("skeleton", skeleton, skeleton_crop),
                ("foldlabel", foldlabel, foldlabel_crop),
            ):
                if not resampled_left:
                    _resampled(out, kind, subject).unlink()
                if not crop_left:
                    _cropped(root, kind, subject).unlink()

        _both_passes(root, out, _qc_file(root, {subject: case[0] for subject, case in cases.items()}))

        expected = _subjects_of_both_stacks(cases)
        assert _stacked_subjects(root, "skeleton") == expected
        assert _stacked_subjects(root, "foldlabel") == expected


# --- generate_distbottom_crops under the QC file ---------------------------------------------------------------------
# Test names keep "_cropped" out of tmp_path: DEF-CTFAIL-BDRABCZUK-C930C8933449. No test passes skip_dir: its default,
# the crop directory, holds no list here.


def _write_distbottom(skeleton_file, distbottom_file):
    """Stands in for generate_distbottom: the distance computation is not under test here."""
    skeleton = _read(skeleton_file)
    _write_volume(distbottom_file, np.where(skeleton == 0, OUTSIDE_SKELETON, skeleton).astype(np.int16))


@pytest.fixture
def distbottom(monkeypatch, tmp_path):
    """Runs the real generate_distbottom_crops on tmp_path/crops, as run_with_params does."""
    monkeypatch.setattr(gd, "generate_distbottom", _write_distbottom)
    crop_dir = str(tmp_path / "crops")

    def run(**kwargs):
        gd.generate_distbottom_crops(src_dir=crop_dir, crop_dir=crop_dir, side=SIDE, **kwargs)

    return run


def _cropped_distbottom(root, subject):
    return Path(root) / "crops" / f"{SIDE}distbottom" / f"{subject}_cropped_distbottom.nii.gz"


def _distbottom_subjects(root):
    """Subjects of <side>distbottom_subject.csv, after checking <side>distbottom.npy has one row per subject."""
    crops = Path(root) / "crops"
    subjects = pd.read_csv(crops / f"{SIDE}distbottom_subject.csv", dtype=str)["Subject"].tolist()
    assert np.load(crops / f"{SIDE}distbottom.npy").shape[0] == len(subjects)
    return subjects


@pytest.mark.parametrize("cause", REMOVAL_CAUSES)
def test_distbottom_leaves_qc_removed_subject_out_of_the_stack(tmp_path, distbottom, cause):
    """REQ-CTFAIL-BDRABCZUK-613F12316DE7: b2 is out of the array and CSV of the distbottom stack, though its cropped
    skeleton, the source of that step, stays on disk."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2", "c3"))
    _crop_with_qc(tmp_path, out, "skeleton")
    qc_path = _qc_file(tmp_path, {"a1": QC_KEPT, "b2": cause, "c3": QC_KEPT})

    _crop_with_qc(tmp_path, out, "skeleton", qc_path=qc_path)
    distbottom(qc_path=qc_path)

    assert _distbottom_subjects(tmp_path) == ["a1", "c3"]


@pytest.mark.parametrize("skeleton_crop_left", [True, False], ids=["skeleton_crop_left", "skeleton_crop_gone"])
def test_distbottom_file_of_earlier_run_is_not_stacked(tmp_path, distbottom, skeleton_crop_left):
    """REQ-CTFAIL-BDRABCZUK-6E7E390BC3B4: the distbottom file an earlier run left for b2 is not stacked, also once only
    the distbottom folder still names the subject."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2"))
    _crop_with_qc(tmp_path, out, "skeleton")
    distbottom()
    if not skeleton_crop_left:
        _resampled(out, "skeleton", "b2").unlink()
        _cropped(tmp_path, "skeleton", "b2").unlink()
    qc_path = _qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_ZERO})

    _crop_with_qc(tmp_path, out, "skeleton", qc_path=qc_path)
    distbottom(qc_path=qc_path)

    assert _cropped_distbottom(tmp_path, "b2").exists()
    assert _distbottom_subjects(tmp_path) == ["a1"]


@pytest.mark.parametrize("distbottom_left", [True, False], ids=["distbottom_left", "no_distbottom"])
@pytest.mark.parametrize("resampled_left", [True, False], ids=["resampled_left", "resampled_gone"])
def test_distbottom_stack_stays_aligned_with_the_skeleton_stack(tmp_path, distbottom, resampled_left, distbottom_left):
    """REQ-CTFAIL-BDRABCZUK-F67061A1575D: the design probe. The cropped skeleton of b2 is left over: the step used to
    stop on the shapes, quality_checks here stays on and both stacks hold a1 and c3, in the same order."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", "b2", "c3"))
    _crop_with_qc(tmp_path, out, "skeleton")
    if distbottom_left:
        distbottom()
    if not resampled_left:
        _resampled(out, "skeleton", "b2").unlink()
    qc_path = _qc_file(tmp_path, {"a1": QC_KEPT, "b2": QC_ZERO, "c3": QC_KEPT})

    _crop_with_qc(tmp_path, out, "skeleton", qc_path=qc_path)
    distbottom(qc_path=qc_path)

    assert _cropped(tmp_path, "skeleton", "b2").exists()
    assert _stacked_subjects(tmp_path, "skeleton") == ["a1", "c3"]
    assert _distbottom_subjects(tmp_path) == _stacked_subjects(tmp_path, "skeleton")


def test_run_with_params_passes_skel_qc_path_to_distbottom(tmp_path, regions_json, calls):  # noqa: F811
    """REQ-CTFAIL-BDRABCZUK-A2CC6CE5348E: the distbottom step of a region receives the QC file of --sk_qc_path."""
    params = _params(tmp_path, regions_json, skel_qc_path="/data/qc.tsv", skip_distbottom=False)

    gosr.run_with_params(params)

    assert [kwargs.get("qc_path") for kwargs in _called(calls, "generate_distbottom_crops")] == ["/data/qc.tsv"]


KEPT_BIDS_KEYS = ("sub-1", "sub-1_ses-1", "sub-2_ses-1_run-1", "sub-2_ses-1_acq-hr_run-2", "sub-4_ses-1")
REMOVED_BIDS_KEYS = ("sub-3_ses-1", "sub-5_ses-1")
# Folder names, except sub-4_ses-1; the keys above are file names.
QC_OF_BIDS_KEYS = {"sub-1": QC_KEPT, "sub-2": QC_KEPT, "sub-3": QC_ZERO, "sub-4_ses-1": QC_KEPT}


@pytest.mark.parametrize(
    "stacked_by",
    [lambda root: _stacked_subjects(root, "skeleton"), _distbottom_subjects],
    ids=["generate_crops", "generate_distbottom_crops"],
)
def test_bids_key_passes_qc_by_its_subject_form(tmp_path, monkeypatch, distbottom, stacked_by):
    """REQ-CTFAIL-BDRABCZUK-87634A604945: the probe of the design report. QC lists folder names, so sub-1_ses-1 passes
    by sub-1; sub-4_ses-1 passes by itself, as a folder really named so; sub-3_ses-1 (a qc of 0) and sub-5_ses-1 (no
    row) pass by neither form."""
    keys = (*KEPT_BIDS_KEYS, *REMOVED_BIDS_KEYS)
    for seed, key in enumerate(keys):
        monkeypatch.setitem(SEEDS, key, seed % len(SEEDS))
    out = _subject_set_layout(tmp_path, skeleton=keys)
    qc_path = _qc_file(tmp_path, QC_OF_BIDS_KEYS)

    _crop_with_qc(tmp_path, out, "skeleton", qc_path=qc_path)
    distbottom(qc_path=qc_path)

    assert sorted(stacked_by(tmp_path)) == sorted(KEPT_BIDS_KEYS)
