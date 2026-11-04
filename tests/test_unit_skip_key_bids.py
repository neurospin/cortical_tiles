"""Unit tests for the one skip-list key of the region pipeline (TASK-CTFAIL-BDRABCZUK-10C79B65D176).

In BIDS mode the unit of work is the session: a graph file, and the file key its outputs carry (sub-1_ses-1, with
the acq and run parts when the graph path has them). So a failure is recorded under that session key, whichever step
meets it, and a multi-session subject with one failed session keeps its other sessions and stays valid. A subject with
no session to name (no graph file at all) is recorded under its folder name (sub-1), and that entry covers the session
keys of the subject in the steps that work on file keys. A subject whose sessions all failed leaves the valid subjects,
so a run left with no valid subject exits 1, not 3.

Graph steps run on empty placeholder graphs, with the conversion replaced by GraphStep.convert. Layouts and stack
helpers of the file steps come from the test modules of those steps.

The test_non_bids_* guards prove section S12 of COMP-CTFAIL-BDRABCZUK-5CEDC06BA622 leaves non-BIDS runs as they were:
the graph-step and crop guards pass on the base 121ff75 and must still pass after the change.
"""

from pathlib import Path

import pytest
from cortical_tiles.brainvisa import generate_foldlabels as gf
from cortical_tiles.brainvisa import generate_ICBM2009c_transforms as gt
from cortical_tiles.brainvisa import generate_skeletons as gs
from cortical_tiles.brainvisa.utils import skip_list
from cortical_tiles.brainvisa.utils.crop_subjects import NoValidSubjectError
from cortical_tiles.brainvisa.utils.skip_list import load_skip_entries, load_skipped, record_skip
from soma import aims

from tests import test_unit_generate_crops as crop_tests
from tests import test_unit_generate_one_sulcal_region as region_tests
from tests import test_unit_mask_resampled_foldlabels as mask_tests
from tests import test_unit_resample_files as resample_tests
from tests.test_unit_generate_sulcal_regions import gsr, script  # noqa: F401
from tests.test_unit_qc_exclusion_crops import _distbottom_subjects, distbottom  # noqa: F401

SUBJECT = "sub-1"
# One graph file per scan; the folder name of a scan is also what its key adds to the subject name
SCANS = ("ses-1", "ses-2_acq-a", "ses-3_acq-b_run-2")
KEYS = tuple(f"{SUBJECT}_{scan}" for scan in SCANS)

GRAPH_SIDE = "L"
GRAPH_STEPS = ("generate_skeletons", "generate_foldlabels", "generate_ICBM2009c_transforms")
PATH_TO_GRAPH = "t1mri/*/folds"
OUTPUT_NAMES = {
    "generate_skeletons": (f"{GRAPH_SIDE}skeleton_generated_", ".nii.gz"),
    "generate_foldlabels": (f"{GRAPH_SIDE}foldlabel_", ".nii.gz"),
    "generate_ICBM2009c_transforms": (f"{GRAPH_SIDE}transform_to_ICBM2009c_", ".trm"),
}

FILE_SIDE = crop_tests.SIDE
# The subject of the entry is sub-1: sub-10 shares its first characters, not its name
OF_THE_SUBJECT = "sub-1_ses-1"
OF_ANOTHER_SUBJECT = "sub-10_ses-1"
OTHER_SESSION = "sub-1_ses-2"

SCRIPT_GRAPH_FOLDER = "G"  # the --path_to_graph value of ScriptRun.run
SCRIPT_SIDES = ("L", "R")


class GraphError(Exception):
    """Raised by GraphStep.convert for the first graph file of a failing run."""


class GraphStep:
    """One graph step on tmp_path, its conversion replaced by ``convert``; ``fail_first`` makes the first graph raise.

    Failing the first graph met, not a named one, keeps each test independent of the order of ``glob``.
    """

    def __init__(self, name, tmp_path, monkeypatch):
        self.name = name
        self.src = tmp_path / "graphs"
        self.out = tmp_path / "derivatives"
        self.skip_dir = tmp_path / "skip_lists"
        self.bids = True
        self.fail_first = False
        self.failed_graphs = []
        self.convert_calls = 0
        for scan in SCANS:
            folder = self.src / SUBJECT / "t1mri" / scan / "folds"
            folder.mkdir(parents=True)
            (folder / f"{GRAPH_SIDE}{SUBJECT}.arg").write_text("")
        monkeypatch.setattr(gs, "generate_skeleton_from_graph_file", lambda graph, out, _: self.convert(graph, out))
        monkeypatch.setattr(gf, "generate_foldlabel_from_graph_file", lambda graph, out, _: self.convert(graph, out))
        manip = type("GraphManip", (), {"getICBM2009cTemplateTransform": staticmethod(lambda graph: graph)})
        fake_aims = type(
            "Aims", (), {"read": staticmethod(str), "write": staticmethod(self.convert), "GraphManip": manip}
        )
        monkeypatch.setattr(gt, "aims", fake_aims)

    def convert(self, graph_file, output_file):
        self.convert_calls += 1
        if self.fail_first and not self.failed_graphs:
            self.failed_graphs.append(graph_file)
            raise GraphError(f"cannot read {graph_file}")
        Path(output_file).write_text("")

    def run(self):
        arguments = {
            "src_dir": str(self.src),
            "side": GRAPH_SIDE,
            "parallel": False,
            "path_to_graph": PATH_TO_GRAPH,
            "bids": self.bids,
            "qc_path": "",
            "skip_dir": str(self.skip_dir),
        }
        if self.name == "generate_skeletons":
            step = gs.GraphConvert2Skeleton(skeleton_dir=str(self.out), junction="thin", njobs=1, **arguments)
        elif self.name == "generate_foldlabels":
            step = gf.GraphConvert2FoldLabel(foldlabel_dir=str(self.out), junction="thin", **arguments)
        else:
            step = gt.GraphGenerateTransform(transform_dir=str(self.out), **arguments)
        step.compute(nb_subjects=-1)

    def skip(self, key):
        record_skip(self.skip_dir, GRAPH_SIDE, "generate_skeletons", key, RuntimeError("failed in an earlier step"))

    def written_keys(self):
        """Returns the keys of the output files of the step."""
        prefix, suffix = OUTPUT_NAMES[self.name]
        names = (path.name for path in (self.out / GRAPH_SIDE).iterdir())
        return {name[len(prefix) : -len(suffix)] for name in names if name.startswith(prefix) and name.endswith(suffix)}

    def failed_key(self):
        """Returns the key of the graph file that raised."""
        (graph_file,) = self.failed_graphs
        return f"{SUBJECT}_{Path(graph_file).parent.parent.name}"


@pytest.fixture(params=GRAPH_STEPS)
def graph_step(request, tmp_path, monkeypatch):
    return GraphStep(request.param, tmp_path, monkeypatch)


def test_graph_step_records_a_failed_graph_under_its_session_key(graph_step):
    """REQ-CTFAIL-BDRABCZUK-3A5BA85C89D4: the entry names the session that failed, not the folder of the subject."""
    graph_step.fail_first = True

    graph_step.run()

    assert load_skipped(graph_step.skip_dir, GRAPH_SIDE) == {graph_step.failed_key()}


def test_graph_step_writes_the_other_graphs_after_a_failure(graph_step):
    """REQ-CTFAIL-BDRABCZUK-736CEC584198: the failure of one session does not stop the sessions after it."""
    graph_step.fail_first = True

    graph_step.run()

    assert graph_step.written_keys() == set(KEYS) - {graph_step.failed_key()}


def test_graph_step_writes_nothing_for_a_skip_listed_session_key(graph_step):
    """REQ-CTFAIL-BDRABCZUK-CCAC6742FD1D: a session an earlier step recorded gets no output in this step."""
    graph_step.skip(KEYS[1])

    graph_step.run()

    assert KEYS[1] not in graph_step.written_keys()


def test_graph_step_keeps_the_sibling_of_a_skip_listed_session_key(graph_step):
    """REQ-CTFAIL-BDRABCZUK-1B84FD0939BB: the entry of one session leaves the other sessions processed."""
    graph_step.skip(KEYS[1])

    graph_step.run()

    assert graph_step.written_keys() >= {KEYS[0], KEYS[2]}


@pytest.mark.parametrize("fail_first", [False, True], ids=["success", "failure"])
def test_non_bids_graph_step_keeps_one_output_and_the_folder_key(graph_step, fail_first):
    """S12 guard: without bids the first graph alone is converted, under the folder name, whatever its path says."""
    graph_step.bids = False
    graph_step.fail_first = fail_first

    graph_step.run()

    assert graph_step.convert_calls == 1
    assert graph_step.written_keys() == (set() if fail_first else {SUBJECT})
    assert load_skipped(graph_step.skip_dir, GRAPH_SIDE) == ({SUBJECT} if fail_first else set())


def _resampled_keys(tmp_path, keys, entries):
    """Runs resample_files on foldlabels of keys with entries in the skip list; returns the keys of the outputs."""
    raw = tmp_path / "raw" / FILE_SIDE
    transforms = tmp_path / "transforms" / FILE_SIDE
    raw.mkdir(parents=True)
    transforms.mkdir(parents=True)
    for key in keys:
        resample_tests._write_volume(raw / f"{FILE_SIDE}foldlabel_{key}.nii.gz", resample_tests._foldlabel_array())
        aims.write(aims.AffineTransformation3d(), str(transforms / f"{FILE_SIDE}transform_to_ICBM2009c_{key}.trm"))
    for entry in entries:
        record_skip(
            resample_tests._skip_dir(tmp_path), FILE_SIDE, "generate_skeletons", entry, RuntimeError("no graph")
        )
    prefix = f"{FILE_SIDE}resampled_foldlabel_"
    names = resample_tests._niftis(resample_tests._resample_foldlabels(tmp_path))
    return {name[len(prefix) : -len(".nii.gz")] for name in names}


def _masked_keys(tmp_path, keys, entries):
    """Runs mask_foldlabel_files on keys with entries in the skip list; returns the keys of the outputs."""
    tree = {name: tmp_path / name for name in ("src", "skel", "masked")}
    before_masking = tree["src"] / f"{FILE_SIDE}_before_masking"
    skeletons = tree["skel"] / FILE_SIDE
    before_masking.mkdir(parents=True)
    skeletons.mkdir(parents=True)
    for key in keys:
        mask_tests._write(before_masking / f"{FILE_SIDE}resampled_foldlabel_{key}.nii.gz", mask_tests._foldlabel())
        mask_tests._write(skeletons / f"{FILE_SIDE}resampled_skeleton_{key}.nii.gz", mask_tests._skeleton())
    for entry in entries:
        record_skip(mask_tests._skip_dir(tree), FILE_SIDE, "generate_skeletons", entry, RuntimeError("no graph"))
    mask_tests._mask(tree)
    return set(mask_tests._masked_subjects(tree))


def _cropped_layout(tmp_path, monkeypatch, keys):
    """Crops the skeletons of keys, as an earlier run did; returns the output root."""
    for seed, key in enumerate(keys):
        monkeypatch.setitem(crop_tests.SEEDS, key, seed % 4)
    out = crop_tests._subject_set_layout(tmp_path, skeleton=keys)
    crop_tests._crop(tmp_path, out, "skeleton")
    return out


def _stacked_keys(tmp_path, monkeypatch, keys, entries):
    """Reruns generate_crops over the files of an earlier run, with entries in the skip list; returns the stack."""
    out = _cropped_layout(tmp_path, monkeypatch, keys)
    for entry in entries:
        crop_tests._skip(out, entry)
    crop_tests._crop(tmp_path, out, "skeleton")
    return set(crop_tests._stacked_subjects(tmp_path, "skeleton"))


def _distbottom_keys(tmp_path, monkeypatch, distbottom_run, keys, entries):
    """Reruns generate_distbottom_crops over the files of an earlier run, with entries in the skip list."""
    out = _cropped_layout(tmp_path, monkeypatch, keys)
    distbottom_run(skip_dir=str(out))
    for entry in entries:
        crop_tests._skip(out, entry)
    crop_tests._crop(tmp_path, out, "skeleton")
    distbottom_run(skip_dir=str(out))
    return set(_distbottom_subjects(tmp_path))


@pytest.fixture
def file_step_keys(tmp_path, monkeypatch, distbottom):  # noqa: F811
    """Returns run(step, keys, entries): the keys a file step processed, with entries in the skip list."""

    def run(step, keys, entries):
        if step == "resample_files":
            return _resampled_keys(tmp_path, keys, entries)
        if step == "mask_resampled_foldlabels":
            return _masked_keys(tmp_path, keys, entries)
        if step == "generate_crops":
            return _stacked_keys(tmp_path, monkeypatch, keys, entries)
        return _distbottom_keys(tmp_path, monkeypatch, distbottom, keys, entries)

    return run


@pytest.mark.parametrize("step", ["resample_files", "mask_resampled_foldlabels"])
def test_subject_entry_filters_session_files_at_resample_and_mask(file_step_keys, step):
    """REQ-CTFAIL-BDRABCZUK-1EAEFD3B5C49: the entry sub-1 of an earlier step covers the file key sub-1_ses-1."""
    assert file_step_keys(step, (OF_THE_SUBJECT, OF_ANOTHER_SUBJECT), [SUBJECT]) == {OF_ANOTHER_SUBJECT}


@pytest.mark.parametrize("step", ["generate_crops", "generate_distbottom_crops"])
def test_subject_entry_leaves_session_keys_out_of_the_crop_stacks(file_step_keys, step):
    """REQ-CTFAIL-BDRABCZUK-AA25426AE8F1: the files of the subject left by an earlier run leave the stack."""
    assert file_step_keys(step, (OF_THE_SUBJECT, OF_ANOTHER_SUBJECT), [SUBJECT]) == {OF_ANOTHER_SUBJECT}


def test_subject_entry_covers_a_one_sided_session_key(tmp_path, monkeypatch):
    """REQ-CTFAIL-BDRABCZUK-1326D81DBFFF: a one-sided file key of a skip-listed subject gets no entry of its own."""
    for seed, key in enumerate((OF_THE_SUBJECT, OF_ANOTHER_SUBJECT)):
        monkeypatch.setitem(crop_tests.SEEDS, key, seed)
    out = crop_tests._subject_set_layout(
        tmp_path, skeleton=(OF_THE_SUBJECT, OF_ANOTHER_SUBJECT), foldlabel=(OF_ANOTHER_SUBJECT,)
    )
    crop_tests._skip(out, SUBJECT)

    crop_tests._crop(tmp_path, out, "skeleton")
    crop_tests._crop(tmp_path, out, "foldlabel")

    assert [entry.subject for entry in load_skip_entries(str(out), FILE_SIDE)] == [SUBJECT]


def test_subject_entry_leaves_session_keys_out_of_the_count_check(tmp_path):
    """REQ-CTFAIL-BDRABCZUK-7DC8AD2D2401: with its only volumes left out, no subject has both volumes."""
    resampled_dir, skeleton_dir = region_tests._write_resampled(
        str(tmp_path), FILE_SIDE, [OF_THE_SUBJECT], [OF_THE_SUBJECT]
    )
    region_tests._skip(str(tmp_path), FILE_SIDE, [SUBJECT])

    with pytest.raises(NoValidSubjectError):
        region_tests.COUNT_CHECK(resampled_dir, skeleton_dir, FILE_SIDE, str(tmp_path))


@pytest.mark.parametrize(
    "step", ["resample_files", "mask_resampled_foldlabels", "generate_crops", "generate_distbottom_crops"]
)
def test_session_entry_keeps_the_other_sessions_at_file_steps(file_step_keys, step):
    """REQ-CTFAIL-BDRABCZUK-4E7F1BB8E49C: the entry of one session leaves out that session only."""
    assert file_step_keys(step, (OF_THE_SUBJECT, OTHER_SESSION), [OF_THE_SUBJECT]) == {OTHER_SESSION}


def _bids_subject(run, subject, sessions):
    """Writes the graph files of subject in the dataset of the ScriptRun, on each side, in BIDS mode."""
    run.config["bids"] = True
    folder = run.graphs_dir / subject / SCRIPT_GRAPH_FOLDER
    folder.mkdir(parents=True)
    for session in sessions:
        for side in SCRIPT_SIDES:
            (folder / f"{side}{subject}_{session}_auto.arg").write_text("")


@pytest.mark.parametrize("step", ["resample_files", "mask_resampled_foldlabels", "generate_crops"])
def test_subject_failing_in_each_session_leaves_the_valid_subjects(script, step):  # noqa: F811
    """REQ-CTFAIL-BDRABCZUK-F2EF56BB53B2: each session of the only subject failed: exit 1, through valid_subjects."""
    _bids_subject(script, "sub-a", ("ses-1", "ses-2"))
    for side in SCRIPT_SIDES:
        script.skip(side, step, "sub-a_ses-1")
        script.skip(side, step, "sub-a_ses-2")

    assert script.run() == 1


@pytest.mark.parametrize("step", ["resample_files", "mask_resampled_foldlabels", "generate_crops"])
def test_subject_keeping_one_session_stays_valid(script, step):  # noqa: F811
    """REQ-CTFAIL-BDRABCZUK-2F4AB7A11CFE: one failed session of two: the subject stays valid, the run exits 3."""
    _bids_subject(script, "sub-a", ("ses-1", "ses-2"))
    for side in SCRIPT_SIDES:
        script.skip(side, step, "sub-a_ses-1")

    assert script.run() == 3


@pytest.mark.parametrize("key", ["sub-1", "sub-10", "a1", "sub-1_session-2"])
def test_non_bids_key_matches_the_skip_list_exactly(key):
    """S12 guard: a key with no session part is its own subject form, so covered_by is the plain membership test."""
    names = {SUBJECT}

    assert skip_list.subject_form(key) == key
    assert skip_list.covered_by(key, names) == (key in names)


def _cropped_stacks(root, subjects, entries):
    """Runs the skeleton then the foldlabel pass of subjects under root; returns {stack file: bytes}."""
    root.mkdir()
    out = crop_tests._subject_set_layout(root, skeleton=subjects, foldlabel=subjects)
    for entry in entries:
        crop_tests._skip(out, entry)
    crop_tests._crop(root, out, "skeleton")
    crop_tests._crop(root, out, "foldlabel")
    names = (f"{FILE_SIDE}{stack}{suffix}" for stack in ("skeleton", "label") for suffix in (".npy", "_subject.csv"))
    return {name: (root / "crops" / name).read_bytes() for name in names}


def test_non_bids_crop_stacks_equal_a_run_without_the_subject(tmp_path):
    """S12 guard: the stacks of a run with c3 skip-listed are, byte for byte, those of a run that never had c3."""
    with_entry = _cropped_stacks(tmp_path / "with_entry", ("a1", "b2", "c3"), ["c3"])
    without_subject = _cropped_stacks(tmp_path / "without_subject", ("a1", "b2"), [])

    assert with_entry == without_subject
