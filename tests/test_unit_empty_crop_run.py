"""Run-level exclusion of a subject with an empty skeleton crop (TASK-CTFAIL-BDRABCZUK-CADDE061C615).

User decision 2026-11-04: one all-zero skeleton crop in any region and side fails the whole scan, every region and
both sides, regions already cropped included; the other subjects carry on and the run ends with exit 3.
User answers to the design questions, same day: generate_distbottom_crops honours the skip list and the end of the
run deletes the cropped files of the dropped subject; each skeleton stack row is checked, so an empty crop left by an
earlier run counts as well.

``generate_sulcal_regions.py`` runs as a script, as in test_unit_generate_sulcal_regions.py. Only
``run_with_params`` is replaced, by calls of the real ``generate_crops`` and ``generate_distbottom_crops`` on the crop
directory it would use, so the region loop, the skip lists, the crops and the quality checks are real and no graph is
needed. Only the distance computation of a distbottom volume is stubbed: a volume this small has no bottom voxel in
each crop.
"""

import inspect
import json
import logging
import os
import re
import runpy
import sys

import numpy as np
import pandas as pd
import pytest
from cortical_tiles.brainvisa import generate_crops as gc
from cortical_tiles.brainvisa import generate_distbottom_crops as gd
from cortical_tiles.brainvisa import generate_one_sulcal_region as gosr
from cortical_tiles.brainvisa.utils import skip_list

from .test_unit_generate_crops import (
    KIND_DIRS,
    SEEDS,
    SHAPE,
    SIDE,
    _crop,
    _cropped_subjects,
    _empty_crop_layout,
    _read,
    _source,
    _stacked_subjects,
    _subject_set_layout,
    _write_volume,
)
from .test_unit_generate_one_sulcal_region import REGION, STEP_FUNCTIONS, _params

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAINVISA_DIR = os.path.join(REPO_DIR, "cortical_tiles", "brainvisa")
SCRIPT = os.path.join(BRAINVISA_DIR, "generate_sulcal_regions.py")

EMPTY = "b2"
KEPT = ["a1", "c3"]
THIN = "a1"
STEP = "generate_crops"
SIDES = ("L", "R")
INPUT_TYPES = ("skeleton", "foldlabel", "extremities")
SOURCE_KINDS = (*INPUT_TYPES, "distmap")
STACKS = ("skeleton", "label", "extremities", "distbottom")
# Run order: FIRST is cropped before the defect is found in HOLLOW, LAST after it.
FIRST, HOLLOW, LAST = "S.Or.", "S.T.s.", "S.Pe.C."
REGIONS = (FIRST, HOLLOW, LAST)
MASK_BOXES = {
    FIRST: np.s_[2:5, 3:7, 4:9],
    HOLLOW: np.s_[7:9, 9:11, 11:13],
    LAST: np.s_[2:5, 3:7, 4:9],
}
SIDE_NAMES = {"L": "left", "R": "right"}
OUTSIDE_SKELETON = 32501  # value of a distbottom voxel off the skeleton, as quality_checks expects it


class RunStopped(Exception):
    """The run goes down in the region loop, as a killed job does: nothing after the loop runs."""


def _write_distbottom(skeleton_file, distbottom_file):
    """Stands in for generate_distbottom: the distance computation is not under test here."""
    skeleton = _read(skeleton_file)
    _write_volume(distbottom_file, np.where(skeleton == 0, OUTSIDE_SKELETON, skeleton).astype(np.int16))


def _volume(kind, subject, hollow):
    """Volume of the subject; a hollow one only has a voxel inside the FIRST and LAST boxes, none in the HOLLOW box."""
    if hollow:
        arr = np.zeros(SHAPE, dtype=np.int16)
        arr[3, 4, 5, 0] = 60
    else:
        arr = _source(SEEDS.get(subject, SEEDS["d4"]))
    return np.where(arr != 0, 1003, 0) if kind == "foldlabel" else arr


class EmptyCropRun:
    """Runs of the script on tiny volumes under root; hollow holds the (subject, side) pairs of the hollow volumes."""

    def __init__(self, root, monkeypatch, caplog, subjects=("a1", EMPTY, "c3"), hollow=((EMPTY, "R"),)):
        self.root = root
        self.out = root / "out"
        self.subjects = subjects
        self.stop_before = None
        self._monkeypatch = monkeypatch
        self._caplog = caplog
        for side in SIDES:
            for kind in SOURCE_KINDS:
                side_dir = self.out / KIND_DIRS[kind] / "2mm" / side
                side_dir.mkdir(parents=True)
                for subject in subjects:
                    _write_volume(self.source(side, kind, subject), _volume(kind, subject, (subject, side) in hollow))
            for region, box in MASK_BOXES.items():
                mask = np.zeros(SHAPE, dtype=np.int16)
                mask[(*box, 0)] = 1
                (root / "mask" / region / side).mkdir(parents=True)
                _write_volume(root / "mask" / region / side / f"S.C._{SIDE_NAMES[side]}.nii.gz", mask)
        (root / "graphs").mkdir()
        for subject in subjects:
            (root / "graphs" / subject).mkdir()
        (root / "dataset").mkdir()

    def source(self, side, kind, subject):
        return self.out / KIND_DIRS[kind] / "2mm" / side / f"{side}resampled_{kind}_{subject}.nii.gz"

    def thin_out(self, subject):
        """Leaves the subject one skeleton voxel in each mask box and all-zero extremities, on each side."""
        skeleton = np.zeros(SHAPE, dtype=np.int16)
        skeleton[3, 4, 5, 0] = skeleton[7, 9, 11, 0] = 60
        volumes = {
            "skeleton": skeleton,
            "foldlabel": np.where(skeleton != 0, 1003, 0),
            "extremities": np.zeros_like(skeleton),
        }
        for side in SIDES:
            for kind, volume in volumes.items():
                _write_volume(self.source(side, kind, subject), volume)

    def crop_dir(self, region):
        return self.out / "crops" / "canonical_25" / "2mm" / region / "mask"

    def crop_files(self, region, side, subject):
        """Names of the files of the subject in the sample directories of the region and side, sidecars included."""
        return sorted(path.name for path in self.crop_dir(region).glob(f"{side}*/{subject}_cropped_*"))

    def leave_empty_crop(self, region, side, subject):
        """Makes the files an earlier run wrote for the subject the ones of an all-zero skeleton crop."""
        crop_dir = self.crop_dir(region)
        skeleton = crop_dir / f"{side}crops" / f"{subject}_cropped_skeleton.nii.gz"
        zeros = np.zeros_like(_read(skeleton))
        _write_volume(skeleton, zeros)
        _write_volume(crop_dir / f"{side}labels" / f"{subject}_cropped_foldlabel.nii.gz", zeros)
        _write_volume(crop_dir / f"{side}distbottom" / f"{subject}_cropped_distbottom.nii.gz", zeros + OUTSIDE_SKELETON)

    def _run_with_params(self, cfg):
        """generate_crops, then generate_distbottom_crops for a skeleton, as run_with_params calls them."""
        if cfg["region_name"] == self.stop_before:
            raise RunStopped(self.stop_before)
        side, crop_dir = cfg["side"], self.crop_dir(cfg["region_name"])
        gc.generate_crops(
            src_dir=str(self.out / KIND_DIRS[cfg["input_type"]] / "2mm"),
            input_type=cfg["input_type"],
            crop_dir=str(crop_dir),
            mask_dir=str(self.root / "mask" / cfg["region_name"]),
            side=side,
            list_sulci=["S.C."],
            threshold=0,
            dilation=2,
            skip_dir=cfg["output_dir"],
        )
        if cfg["input_type"] == "skeleton":
            gd.generate_distbottom_crops(
                src_dir=str(crop_dir), crop_dir=str(crop_dir), side=side, skip_dir=cfg["output_dir"]
            )

    def run(self, sides=SIDES, regions=REGIONS, input_types=INPUT_TYPES, stop_before=None):
        """Runs the script and returns its exit code; caplog then holds the records of this run."""
        self.stop_before = stop_before
        config = {
            "save_behavior": "best",
            "graphs_dir": str(self.root / "graphs"),
            "output_dir": str(self.out),
            "out_voxel_size": 2,
            "no_mask": False,
            "masks_version": "canonical_25",
        }
        (self.root / "dataset" / "pipeline_loop_2mm.json").write_text(json.dumps(config))
        self._monkeypatch.setattr("generate_one_sulcal_region.run_with_params", self._run_with_params)
        argv = ["-d", str(self.root / "dataset"), "--path_to_graph", "G", "--path_sk_with_hull", "H", "-i", *sides]
        argv += ["-r", *regions, "-y", *input_types]
        self._monkeypatch.setattr(sys, "argv", ["generate_sulcal_regions.py", *argv])
        with self._caplog.at_level(logging.INFO):
            try:
                runpy.run_path(SCRIPT, run_name="__main__")
            except SystemExit as exc:
                return 0 if exc.code is None else exc.code
        return 0

    def stacked(self, region, side, stack):
        crop_dir = self.crop_dir(region)
        subjects = pd.read_csv(crop_dir / f"{side}{stack}_subject.csv", dtype=str)["Subject"].tolist()
        assert np.load(crop_dir / f"{side}{stack}.npy").shape[0] == len(subjects)
        return subjects

    def entries(self, side):
        return [(e.subject, e.step) for e in skip_list.load_skip_entries(skip_dir=str(self.out), side=side)]


@pytest.fixture
def empty_crop_run(monkeypatch, tmp_path_factory, caplog):
    monkeypatch.syspath_prepend(BRAINVISA_DIR)
    monkeypatch.setattr(gd, "generate_distbottom", _write_distbottom)

    def build(root=None, **kwargs):
        # Not tmp_path, which starts with the test name: get_not_processed_distbottom_files cuts whole paths at the
        # first "_cropped", so a test named test_..._cropped_... would see no subject to process.
        return EmptyCropRun(root or tmp_path_factory.mktemp("scan"), monkeypatch, caplog, **kwargs)

    return build


@pytest.mark.parametrize("skip_dir", [None, "crops"], ids=["skip_dir_apart", "skip_dir_is_crop_dir"])
def test_empty_crop_is_returned_by_load_skipped(tmp_path, skip_dir):
    """REQ-CTFAIL-BDRABCZUK-6F254AF80592: every later step reads load_skipped(skip_dir, side), so this entry is what
    fails the whole scan; the wording holds wherever skip_dir lies."""
    out = _empty_crop_layout(tmp_path)
    run_skip_dir = out if skip_dir is None else tmp_path / skip_dir

    _crop(tmp_path, out, "skeleton", skip_dir=run_skip_dir)

    assert skip_list.load_skipped(skip_dir=str(run_skip_dir), side=SIDE) == {EMPTY}


def test_empty_crop_leaves_subject_out_of_a_later_region(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-8CED2524479A: the entry written for the first region is read by the next one, which
    crops the subject without trouble otherwise."""
    out = _empty_crop_layout(tmp_path)
    mask = np.zeros(SHAPE, dtype=np.int16)
    mask[7:9, 9:11, 11:13, 0] = 1
    later = tmp_path / "later"
    (later / "mask" / "R").mkdir(parents=True)
    _write_volume(later / "mask" / "R" / "S.C._right.nii.gz", mask)

    _crop(tmp_path, out, "skeleton")
    _crop(later, out, "skeleton")

    assert _stacked_subjects(later, "skeleton") == KEPT


@pytest.mark.parametrize("hollow_side, other_side", [("R", "L"), ("L", "R")])
def test_empty_crop_reaches_the_skip_list_of_the_opposite_side(empty_crop_run, hollow_side, other_side):
    """REQ-CTFAIL-BDRABCZUK-F8EB4E09EE83: skip lists are per side, so the other side needs an entry of its own to
    leave its later regions without the subject; the defect found on R and the one found on L both cross over."""
    run = empty_crop_run(hollow=((EMPTY, hollow_side),))

    run.run()

    assert run.entries(hollow_side) == [(EMPTY, STEP)]
    assert run.entries(other_side) == [(EMPTY, STEP)]


@pytest.mark.parametrize("stack", STACKS)
def test_empty_crop_removes_subject_from_a_region_already_cropped(empty_crop_run, stack):
    """REQ-CTFAIL-BDRABCZUK-3817C6341B42: FIRST was cropped, in each stack, before HOLLOW showed the defect."""
    run = empty_crop_run()

    run.run()

    assert run.stacked(FIRST, "R", stack) == KEPT


@pytest.mark.parametrize("region", REGIONS)
@pytest.mark.parametrize("side", SIDES)
def test_empty_crop_removes_subject_from_both_sides(empty_crop_run, region, side):
    """REQ-CTFAIL-BDRABCZUK-3817C6341B42: the volumes of the subject on L are sound, and it still leaves that side."""
    run = empty_crop_run()

    run.run()

    assert run.stacked(region, side, "skeleton") == KEPT


def test_empty_crop_keeps_rows_of_the_remaining_subjects(empty_crop_run, tmp_path):
    """REQ-CTFAIL-BDRABCZUK-CEC6D61074CD: a healthy subject's crops are the ones of a run without the dropped subject,
    in the regions rewritten after the fact as well."""
    run = empty_crop_run(root=tmp_path / "with")
    reference = empty_crop_run(root=tmp_path / "without", subjects=tuple(KEPT))

    run.run()
    reference.run()

    for region in REGIONS:
        for side in SIDES:
            for stack in STACKS:
                name = f"{side}{stack}.npy"
                rows = np.load(run.crop_dir(region) / name)
                assert np.array_equal(rows, np.load(reference.crop_dir(region) / name)), (region, side, stack)


def test_empty_crop_run_exits_with_code_3(empty_crop_run):
    """REQ-CTFAIL-BDRABCZUK-84E9B9F3BE3E: a failed scan among valid ones is a partial run."""
    run = empty_crop_run()

    assert run.run() == 3


def test_empty_crop_summary_names_the_subject_on_each_side(empty_crop_run, caplog):
    """REQ-CTFAIL-BDRABCZUK-984F1A251DB1: one summary record per side, not record_skip's own per-subject records."""
    run = empty_crop_run()

    run.run()

    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    summaries = [message for message in messages if not message.startswith("Skipping subject")]
    for side in SIDES:
        side_word = re.compile(rf"\b{side}\b")
        found = [message for message in summaries if STEP in message and side_word.search(message)]
        assert [message for message in found if EMPTY in message], side


def test_empty_crop_leaves_skip_list_outside_sides_unchanged(empty_crop_run):
    """REQ-CTFAIL-BDRABCZUK-4F4A0CA2B9D2: an R-only run has no say on L: the opposite side's entry needs L in sides."""
    run = empty_crop_run()

    run.run(sides=("R",))

    assert run.entries("R") == [(EMPTY, STEP)]
    assert run.entries("L") == []


@pytest.mark.parametrize("subjects", [("a1", EMPTY, "c3"), tuple(KEPT)], ids=["one_subject_dropped", "none_dropped"])
def test_subject_without_empty_crop_stays_in_the_stacks(empty_crop_run, subjects):
    """REQ-CTFAIL-BDRABCZUK-1823CE233AA2: regression guard, passes in red: the drop stays narrow, a single voxel per
    region keeps THIN and its all-zero extremities drop nobody. Membership only: EMPTY leaving is REQ 3817C6341B42's."""
    run = empty_crop_run(subjects=subjects)
    run.thin_out(THIN)

    run.run()

    for side in SIDES:
        assert not {subject for subject, _ in run.entries(side)} & set(KEPT), side
        for region in REGIONS:
            for stack in ("skeleton", "extremities"):
                assert set(KEPT) <= set(run.stacked(region, side, stack)), (region, side, stack)


# --- generate_distbottom_crops and the skip list --------------------------------------------------------------------


@pytest.fixture
def distbottom(monkeypatch, tmp_path):
    """Runs the real generate_distbottom_crops on tmp_path/crops, as run_with_params does, and returns its subjects."""
    monkeypatch.setattr(gd, "generate_distbottom", _write_distbottom)
    crop_dir = tmp_path / "crops"

    def run(**kwargs):
        gd.generate_distbottom_crops(src_dir=str(crop_dir), crop_dir=str(crop_dir), side=SIDE, **kwargs)
        subjects = pd.read_csv(crop_dir / f"{SIDE}distbottom_subject.csv", dtype=str)["Subject"].tolist()
        assert np.load(crop_dir / f"{SIDE}distbottom.npy").shape[0] == len(subjects)
        return subjects

    return run


@pytest.mark.parametrize(
    "cause",
    [ValueError("empty crop"), FileNotFoundError("no resampled foldlabel file")],
    ids=["empty_crop", "one_sided_subject"],
)
def test_distbottom_leaves_skip_listed_subject_out_of_the_stack(tmp_path, distbottom, cause):
    """REQ-CTFAIL-BDRABCZUK-640110C35A06: the rerun crash of the design report: the crop files of the subject are still
    on disk, its skeleton row is gone, and the quality checks stopped the run on the shapes; the one-sided subject,
    whose rerun stack this changes, was accepted by the user."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", EMPTY, "c3"))
    _crop(tmp_path, out, "skeleton")
    distbottom(skip_dir=str(out))
    skip_list.record_skip(str(out), SIDE, STEP, EMPTY, cause)
    _crop(tmp_path, out, "skeleton")

    assert distbottom(skip_dir=str(out)) == KEPT


def test_distbottom_skip_list_lies_in_skip_dir_argument(tmp_path, distbottom):
    """REQ-CTFAIL-BDRABCZUK-4C6F598696FC: the list of skip_dir decides, the one of crop_dir has no say."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", EMPTY, "c3"))
    shared = tmp_path / "shared_skip_lists"
    _crop(tmp_path, out, "skeleton")
    skip_list.record_skip(str(shared), SIDE, STEP, EMPTY, ValueError("empty crop"))
    skip_list.record_skip(str(tmp_path / "crops"), SIDE, STEP, "c3", ValueError("empty crop"))
    _crop(tmp_path, out, "skeleton", skip_dir=shared)

    assert distbottom(skip_dir=str(shared)) == KEPT


def test_distbottom_without_skip_dir_reads_skip_list_of_crop_dir(tmp_path, distbottom):
    """REQ-CTFAIL-BDRABCZUK-C3F1B69868FC: the default of generate_crops, so the two steps of a region read one list."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", EMPTY, "c3"))
    crop_dir = tmp_path / "crops"
    _crop(tmp_path, out, "skeleton", skip_dir=crop_dir)
    skip_list.record_skip(str(crop_dir), SIDE, STEP, EMPTY, ValueError("empty crop"))
    _crop(tmp_path, out, "skeleton", skip_dir=crop_dir)

    assert distbottom() == KEPT


def test_run_with_params_passes_output_dir_to_distbottom_as_skip_dir(tmp_path, monkeypatch):
    """REQ-CTFAIL-BDRABCZUK-DC09EB4D6282: the skip lists of a run lie in output_dir, not in the crop directory."""
    regions_json = tmp_path / "regions.json"
    regions_json.write_text(json.dumps({"brain": {f"{REGION}_right": {f"{REGION}_right": 1}}}))
    params = _params(tmp_path, str(regions_json))
    calls = {}

    def recorder(name):
        def record(**kwargs):
            calls[name] = kwargs
            if name == "generate_crops":
                os.makedirs(os.path.join(kwargs["crop_dir"], f"{kwargs['side']}crops"), exist_ok=True)

        return record

    for name in STEP_FUNCTIONS:
        monkeypatch.setattr(gosr, name, recorder(name))
    monkeypatch.setattr(gosr, "setup_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(gosr, "check_if_number_skeletons_equals_number_foldlabels", lambda *args, **kwargs: None)
    signature = inspect.signature(gd.generate_distbottom_crops)

    gosr.run_with_params(params)

    assert calls["generate_distbottom_crops"].get("skip_dir") == params["output_dir"]
    signature.bind(**calls["generate_distbottom_crops"])


# --- the end of the run deletes the cropped files of the dropped subject, and nothing else --------------------------


@pytest.mark.parametrize("region", REGIONS)
@pytest.mark.parametrize("side", SIDES)
def test_empty_crop_deletes_cropped_files_in_each_region_and_side(empty_crop_run, region, side):
    """REQ-CTFAIL-BDRABCZUK-24A1E69B1C83: nothing of a failed scan stays, sidecars included: FIRST and the sound side
    were cropped in full before the defect showed."""
    run = empty_crop_run()

    run.run()

    assert run.crop_files(region, side, EMPTY) == []


def test_empty_crop_deletes_cropped_files_of_each_input_type(empty_crop_run):
    """REQ-CTFAIL-BDRABCZUK-24A1E69B1C83: the sample directories of a region do not share one name pattern; a kept
    subject shows that each of them was written."""
    run = empty_crop_run()

    run.run(input_types=(*INPUT_TYPES, "distmap"))

    kinds = ("skeleton", "foldlabel", "extremities", "distmap", "distbottom")
    assert {f"{THIN}_cropped_{kind}.nii.gz" for kind in kinds} <= set(run.crop_files(FIRST, "R", THIN))
    assert run.crop_files(FIRST, "R", EMPTY) == []


def test_empty_crop_deletion_passes_over_input_types_not_run(empty_crop_run):
    """REQ-CTFAIL-BDRABCZUK-24A1E69B1C83: a run of the skeleton alone has no labels or extremities directory."""
    run = empty_crop_run()

    assert run.run(input_types=("skeleton",)) == 3
    assert run.crop_files(FIRST, "R", EMPTY) == []


@pytest.mark.parametrize("like", [f"{EMPTY}1", f"x{EMPTY}"], ids=["name_starts_alike", "name_ends_alike"])
def test_empty_crop_keeps_cropped_files_of_a_subject_with_a_like_name(empty_crop_run, like):
    """REQ-CTFAIL-BDRABCZUK-9D4798E8D99B: the deletion goes by the exact subject, never by a part of a file name."""
    run = empty_crop_run(subjects=("a1", EMPTY, like, "c3"))

    run.run()

    assert run.crop_files(FIRST, "R", EMPTY) == []
    for region in REGIONS:
        for side in SIDES:
            assert run.crop_files(region, side, like), (region, side)


def test_empty_crop_keeps_cropped_files_of_a_subject_skipped_for_another_cause(empty_crop_run):
    """REQ-CTFAIL-BDRABCZUK-9D4798E8D99B: a skip entry alone deletes nothing: only an empty crop fails a scan."""
    run = empty_crop_run()
    run.run()
    skip_list.record_skip(str(run.out), "R", STEP, THIN, FileNotFoundError("no resampled foldlabel file"))

    run.run()

    assert run.crop_files(FIRST, "R", EMPTY) == []
    for region in REGIONS:
        assert run.crop_files(region, "R", THIN), region


def test_empty_crop_keeps_cropped_files_on_a_side_outside_sides(empty_crop_run):
    """REQ-CTFAIL-BDRABCZUK-162B00E4CA53: an R-only run has no say on L, as for the skip list of L."""
    run = empty_crop_run()
    run.run(sides=("L",))

    run.run(sides=("R",))

    assert run.crop_files(FIRST, "R", EMPTY) == []
    assert run.crop_files(FIRST, "L", EMPTY)


def test_empty_crop_keeps_cropped_files_in_a_region_outside_the_run(empty_crop_run):
    """REQ-CTFAIL-BDRABCZUK-1889F2A2F2ED: a run names its regions, and the others stay as they are."""
    run = empty_crop_run()
    run.run(regions=(FIRST,))

    run.run(regions=(HOLLOW, LAST))

    assert run.crop_files(HOLLOW, "L", EMPTY) == []
    for side in SIDES:
        assert run.crop_files(FIRST, side, EMPTY), side


# --- a second run on the same output folder -------------------------------------------------------------------------


def _rerun(run, first_run):
    """Runs the script a second time, after a first run that finished or went down in LAST; returns the exit code."""
    if first_run == "killed":
        with pytest.raises(RunStopped):
            run.run(stop_before=LAST)
    else:
        run.run()
    return run.run()


@pytest.mark.parametrize("first_run", ["finished", "killed"])
def test_rerun_on_the_same_output_exits_with_code_3(empty_crop_run, first_run):
    """REQ-CTFAIL-BDRABCZUK-40D77184A7D7: skip lists outlive a run under best: the second run must not crash on them,
    with the cropped files deleted (finished) or still on disk (killed before the end of the run)."""
    run = empty_crop_run()

    assert _rerun(run, first_run) == 3


@pytest.mark.parametrize("first_run", ["finished", "killed"])
@pytest.mark.parametrize("stack", STACKS)
def test_rerun_on_the_same_output_keeps_the_subject_out_of_the_stacks(empty_crop_run, first_run, stack):
    """REQ-CTFAIL-BDRABCZUK-1A29A2C5D864: under best the stacks are built again from the files on disk."""
    run = empty_crop_run()

    _rerun(run, first_run)

    for region in REGIONS:
        for side in SIDES:
            assert run.stacked(region, side, stack) == KEPT, (region, side)


# --- an empty crop left by an earlier run ---------------------------------------------------------------------------


def _earlier_empty_crop_layout(tmp_path):
    """Layout whose skeleton crop of EMPTY, written by an earlier run, is all zeros while its source volume is sound:
    a crop already on disk is stacked without being cropped again."""
    out = _subject_set_layout(tmp_path, skeleton=("a1", EMPTY, "c3"))
    _crop(tmp_path, out, "skeleton")
    cropped = tmp_path / "crops" / f"{SIDE}crops" / f"{EMPTY}_cropped_skeleton.nii.gz"
    _write_volume(cropped, np.zeros_like(_read(cropped)))
    return out


def test_earlier_empty_crop_is_recorded_through_record_skip(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-2C7AEC6771C2: derivative trees made before this fix are where the defect lies."""
    out = _earlier_empty_crop_layout(tmp_path)

    _crop(tmp_path, out, "skeleton")

    entries = skip_list.load_skip_entries(skip_dir=str(out), side=SIDE)
    assert [(entry.subject, entry.step) for entry in entries] == [(EMPTY, STEP)]
    assert "empty crop" in entries[0].message


def test_earlier_empty_crop_leaves_subject_out_of_the_skeleton_stack(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-37ABC12C91DB: the row of the old crop does not reach the stack."""
    out = _earlier_empty_crop_layout(tmp_path)

    _crop(tmp_path, out, "skeleton")

    assert _stacked_subjects(tmp_path, "skeleton") == KEPT


def test_earlier_empty_crop_deletes_the_cropped_file(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-6B0530082EA1: the same end as for a crop of this run."""
    out = _earlier_empty_crop_layout(tmp_path)

    _crop(tmp_path, out, "skeleton")

    assert _cropped_subjects(tmp_path, "skeleton") == KEPT


def test_earlier_empty_crop_fails_the_whole_scan(empty_crop_run):
    """REQ-CTFAIL-BDRABCZUK-9E913FF706E6: the crop and the distbottom file of the first run stay on disk in HOLLOW; the
    run goes through and the subject leaves each region and side, as for a crop of this run."""
    run = empty_crop_run(hollow=())
    run.run()
    run.leave_empty_crop(HOLLOW, "R", EMPTY)

    run.run()

    for region in REGIONS:
        for side in SIDES:
            for stack in STACKS:
                assert run.stacked(region, side, stack) == KEPT, (region, side, stack)


def test_empty_crop_of_each_subject_exits_with_code_1(empty_crop_run):
    """REQ-CTFAIL-BDRABCZUK-45A18695FB71: the all-empty region stays as recorded, user decision: the run leaves no valid
    subject on any side, the rule of the run summary."""
    run = empty_crop_run(hollow=(("a1", "R"), (EMPTY, "R"), ("c3", "R")))

    assert run.run() == 1
