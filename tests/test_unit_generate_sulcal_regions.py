"""Unit tests for cortical_tiles.brainvisa.generate_sulcal_regions (REQ-CTILESTEST-99..114).

champollion_pipeline stage 2 runs ``generate_sulcal_regions.py -d <dataset>`` from the brainvisa
directory; the script reads ``<dataset>/pipeline_loop_2mm.json``, resolves it, and runs every
region through ``RegionPipelineRunner`` -> ``generate_one_sulcal_region.run_with_params``. The
end-to-end path is covered by test_generate_sulcal_regions_smoke.py; here ``run_with_params`` is
replaced, on the imported module, by a recorder, so each test only checks the configuration
handed to it (milliseconds, no image processed).

The module imports its sibling as a top-level module (``from generate_one_sulcal_region import``),
so the brainvisa directory is put on sys.path, as in the smoke test. The ``$local`` branch derives
two paths from the current working directory: those tests chdir into a pytest tmp dir.

Characterization tests: behaviour recorded on cortical_tiles dev 4141bd8. Tests flagged "DEFECT"
pin current behaviour that looks wrong; it is recorded, not fixed (tests-only rule of TASK-114).
"""

import inspect
import json
import logging
import os
import re
import runpy
import sys

import pytest

from cortical_tiles.brainvisa.utils.crop_subjects import NoValidSubjectError
from cortical_tiles.brainvisa.utils.skip_list import load_skip_entries, record_skip

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAINVISA_DIR = os.path.join(REPO_DIR, "cortical_tiles", "brainvisa")
SCRIPT = os.path.join(BRAINVISA_DIR, "generate_sulcal_regions.py")

INSULA = "F.C.L.p.-subsc.-F.C.L.a.-INSULA."


@pytest.fixture
def gsr(monkeypatch):
    monkeypatch.syspath_prepend(BRAINVISA_DIR)
    import generate_sulcal_regions

    return generate_sulcal_regions


@pytest.fixture
def calls(gsr, monkeypatch):
    """Replaces run_with_params by a recorder of (a copy of) each config it receives."""
    recorded = []
    monkeypatch.setattr(gsr, "run_with_params", lambda cfg: recorded.append(dict(cfg)))
    return recorded


def _dataset(tmp_path, config):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "pipeline_loop_2mm.json").write_text(json.dumps(config))
    return dataset


def _run(gsr, dataset, **overrides):
    kwargs = dict(
        regions=["S.Or."],
        sides=["R"],
        input_types=["skeleton"],
        path_dataset=str(dataset),
        verbose="",
        output_dir=None,
        path_to_graph="",
        path_sk_with_hull="",
        sk_qc_path="",
        njobs=1,
    )
    kwargs.update(overrides)
    gsr.generate_sulcal_regions(**kwargs)


# --- REQ-CTILESTEST-99 / 100: parse_args ---------------------------------------------------------


def test_parse_args_keys_match_generate_sulcal_regions_parameters(gsr):
    """REQ-CTILESTEST-99: parse_args keys == generate_sulcal_regions parameter names (main splats them)."""
    params = gsr.parse_args(["-d", "D", "--path_to_graph", "G", "--path_sk_with_hull", "H"])
    assert set(params) == set(inspect.signature(gsr.generate_sulcal_regions).parameters)


@pytest.mark.parametrize("flags, expected", [([], ""), (["-v"], "-v"), (["-vv"], "-vv"), (["-v", "-v", "-v"], "-vvv")])
def test_parse_args_verbose_string(gsr, flags, expected):
    """REQ-CTILESTEST-100: verbose is '' without -v, otherwise '-' followed by one 'v' per -v."""
    params = gsr.parse_args(["-d", "D", "--path_to_graph", "G", "--path_sk_with_hull", "H", *flags])
    assert params["verbose"] == expected


# --- REQ-CTILESTEST-101..103: RegionPipelineRunner -----------------------------------------------


def test_runner_calls_run_with_params_per_side_and_input_type(gsr, calls):
    """REQ-CTILESTEST-101: one call per (side, input_type), side-major, with side/input_type/njobs/region set."""
    gsr.RegionPipelineRunner({"keep": "me"}, "S.Or.").run(["L", "R"], ["skeleton", "foldlabel"], 3)

    assert [(c["side"], c["input_type"]) for c in calls] == [
        ("L", "skeleton"),
        ("L", "foldlabel"),
        ("R", "skeleton"),
        ("R", "foldlabel"),
    ]
    assert all(c["njobs"] == 3 and c["region_name"] == "S.Or." and c["keep"] == "me" for c in calls)


@pytest.mark.parametrize(
    "region, expected", [("OCCIPITAL", 1), (INSULA, 1), ("S.Or.", 0), ("CINGULATE.", 0), ("S.C.-sylv.", 0)]
)
def test_runner_threshold(gsr, calls, region, expected):
    """REQ-CTILESTEST-102: threshold 1 for OCCIPITAL and the insula region on both sides, else 0."""
    gsr.RegionPipelineRunner({"threshold": 7}, region).run(["L", "R"], ["skeleton"], 1)
    assert [c["threshold"] for c in calls] == [expected, expected]


@pytest.mark.parametrize("region, expected", [("CINGULATE.", True), ("S.Or.", False), ("OCCIPITAL", False)])
def test_runner_combine_type(gsr, calls, region, expected):
    """REQ-CTILESTEST-103: combine_type is True only for region CINGULATE."""
    gsr.RegionPipelineRunner({"combine_type": "x"}, region).run(["R"], ["skeleton"], 1)
    assert calls[0]["combine_type"] is expected


# --- REQ-CTILESTEST-104..106: explicit (non-$local) configuration ---------------------------------


BASE_CONFIG = {
    "masks_version": "canonical_25",
    "path_to_graph": "cfg/graph",
    "path_to_skeleton_with_hull": "cfg/hull",
    "skel_qc_path": "cfg/qc.tsv",
    "output_dir": "/cfg/out",
}


@pytest.mark.parametrize("masks, expected", [("v2", "v2"), (None, "canonical_25"), ("", "canonical_25")])
def test_masks_argument_overrides_masks_version(gsr, calls, tmp_path, masks, expected):
    """REQ-CTILESTEST-104: a non-empty masks argument replaces masks_version; None or '' keeps the config value."""
    _run(gsr, _dataset(tmp_path, BASE_CONFIG), masks=masks)
    assert calls[0]["masks_version"] == expected


def test_non_empty_graph_paths_override_config(gsr, calls, tmp_path):
    """REQ-CTILESTEST-105: non-empty path_to_graph / path_sk_with_hull replace the config values."""
    _run(gsr, _dataset(tmp_path, BASE_CONFIG), path_to_graph="arg/graph", path_sk_with_hull="arg/hull")
    assert calls[0]["path_to_graph"] == "arg/graph"
    assert calls[0]["path_to_skeleton_with_hull"] == "arg/hull"


def test_empty_graph_paths_keep_config(gsr, calls, tmp_path):
    """REQ-CTILESTEST-105: empty path_to_graph / path_sk_with_hull keep the config values."""
    _run(gsr, _dataset(tmp_path, BASE_CONFIG))
    assert calls[0]["path_to_graph"] == "cfg/graph"
    assert calls[0]["path_to_skeleton_with_hull"] == "cfg/hull"


@pytest.mark.parametrize("sk_qc_path", ["", "arg/qc.tsv"])
def test_skel_qc_path_always_overwritten(gsr, calls, tmp_path, sk_qc_path):
    """REQ-CTILESTEST-106: skel_qc_path always becomes the sk_qc_path argument, even ''."""
    _run(gsr, _dataset(tmp_path, BASE_CONFIG), sk_qc_path=sk_qc_path)
    assert calls[0]["skel_qc_path"] == sk_qc_path


def test_pipeline_json_is_not_written_back(gsr, calls, tmp_path):
    """REQ-CTILESTEST-107: pipeline_loop_2mm.json is left byte-identical."""
    dataset = _dataset(tmp_path, BASE_CONFIG)
    before = (dataset / "pipeline_loop_2mm.json").read_bytes()
    _run(gsr, dataset, masks="v2", path_to_graph="arg/graph", sk_qc_path="arg/qc.tsv")
    assert (dataset / "pipeline_loop_2mm.json").read_bytes() == before
    assert len(calls) == 1


# --- REQ-CTILESTEST-108..112: $local configuration ------------------------------------------------


LOCAL_KEYS = (
    "brain_regions_json",
    "supervised_output_dir",
    "graphs_dir",
    "output_dir",
    "path_to_graph",
    "path_to_skeleton_with_hull",
    "skel_qc_path",
)


@pytest.fixture
def local_cwd(tmp_path, monkeypatch):
    """chdir into <tmp>/p1/p2/p3/p4/p5 so that the 3rd and 4th parents stay inside tmp_path."""
    cwd = tmp_path.joinpath("p1", "p2", "p3", "p4", "p5")
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    return tmp_path


def _local_config(tmp_path):
    return _dataset(tmp_path, {key: "$local" for key in LOCAL_KEYS})


# REQ-CTDEFECTS-21: a $local graph path with an empty argument raises, so every all-$local run
# that is expected to succeed passes both graph arguments.
LOCAL_GRAPH_ARGS = dict(path_to_graph="arg/graph", path_sk_with_hull="arg/hull")


def test_local_brain_regions_json(gsr, calls, local_cwd):
    """REQ-CTILESTEST-108: $local brain_regions_json -> <4th parent of cwd>/sulci_regions_champollion_V1.json."""
    _run(gsr, _local_config(local_cwd), output_dir="out", **LOCAL_GRAPH_ARGS)
    assert calls[0]["brain_regions_json"] == os.path.join(str(local_cwd / "p1"), "sulci_regions_champollion_V1.json")


def test_local_supervised_output_dir(gsr, calls, local_cwd):
    """REQ-CTILESTEST-109: $local supervised_output_dir -> <3rd parent of cwd>/cortical_tiles/data."""
    _run(gsr, _local_config(local_cwd), output_dir="out", **LOCAL_GRAPH_ARGS)
    assert calls[0]["supervised_output_dir"] == os.path.join(str(local_cwd / "p1" / "p2"), "cortical_tiles/data")


def test_local_graphs_dir(gsr, calls, local_cwd):
    """REQ-CTILESTEST-110: $local graphs_dir -> <path_dataset>/derivatives/morphologist-6.0."""
    dataset = _local_config(local_cwd)
    _run(gsr, dataset, output_dir="out", **LOCAL_GRAPH_ARGS)
    assert calls[0]["graphs_dir"] == os.path.join(str(dataset), "derivatives/morphologist-6.0")


@pytest.mark.parametrize("output_dir", ["out", "/abs/out"])
def test_local_output_dir_uses_given_output_dir(gsr, calls, local_cwd, output_dir):
    """REQ-CTDEFECTS-14 (inverts REQ-CTILESTEST-111): $local output_dir, output_dir given -> join(dataset, output_dir)."""
    dataset = _local_config(local_cwd)
    _run(gsr, dataset, output_dir=output_dir, **LOCAL_GRAPH_ARGS)
    assert calls[0]["output_dir"] == os.path.join(str(dataset), output_dir)


@pytest.mark.parametrize("output_dir", [None, ""])
def test_local_output_dir_defaults_to_derivatives_when_not_given(gsr, calls, local_cwd, output_dir):
    """REQ-CTDEFECTS-15 (inverts REQ-CTILESTEST-112): $local output_dir, None/'' -> <dataset>/derivatives/cortical_tiles-<v>."""
    dataset = _local_config(local_cwd)
    _run(gsr, dataset, output_dir=output_dir, **LOCAL_GRAPH_ARGS)
    expected = os.path.join(str(dataset), f"derivatives/cortical_tiles-{gsr._CORTICAL_TILES_VERSION}")
    assert calls[0]["output_dir"] == expected


def test_local_graph_paths_and_qc_path(gsr, calls, local_cwd):
    """REQ-CTILESTEST-113: $local graph paths take non-empty arguments; skel_qc_path = argument.

    Its "else stay '$local'" half is inverted by REQ-CTDEFECTS-21 (see the next test).
    """
    _run(gsr, _local_config(local_cwd), output_dir="out", sk_qc_path="arg/qc.tsv", **LOCAL_GRAPH_ARGS)
    assert calls[0]["path_to_graph"] == "arg/graph"
    assert calls[0]["path_to_skeleton_with_hull"] == "arg/hull"
    assert calls[0]["skel_qc_path"] == "arg/qc.tsv"


@pytest.mark.parametrize("empty", ["", None])
@pytest.mark.parametrize(
    "config_key, argument",
    [("path_to_graph", "path_to_graph"), ("path_to_skeleton_with_hull", "path_sk_with_hull")],
)
def test_local_graph_path_with_empty_argument_raises_value_error(gsr, calls, local_cwd, config_key, argument, empty):
    """REQ-CTDEFECTS-21 (inverts REQ-CTILESTEST-113's "stay '$local'"): $local graph path + empty/None argument
    -> ValueError naming the config key and the argument, before any region runs."""
    kwargs = dict(LOCAL_GRAPH_ARGS)
    kwargs[argument] = empty
    with pytest.raises(ValueError) as excinfo:
        _run(gsr, _local_config(local_cwd), output_dir="out", **kwargs)
    message = str(excinfo.value)
    assert config_key in message
    assert argument in message
    assert calls == []


# --- REQ-CTILESTEST-114: main ----------------------------------------------------------------------


def test_main_passes_parsed_arguments(gsr, monkeypatch):
    """REQ-CTILESTEST-114: main(argv) calls generate_sulcal_regions once with the parse_args params."""
    received = []
    monkeypatch.setattr(gsr, "generate_sulcal_regions", lambda **kw: received.append(kw))
    argv = ["-d", "D", "--path_to_graph", "G", "--path_sk_with_hull", "H", "-i", "R", "-r", "S.Or.", "--njobs", "2"]

    gsr.main(argv)

    assert received == [gsr.parse_args(argv)]
    assert received[0]["sides"] == ["R"] and received[0]["regions"] == ["S.Or."] and received[0]["njobs"] == 2


# --- TASK-CTFAIL-BDRABCZUK-A3796AA95383: run summary, exit code, skip-list reset -------------------
#
# These tests drive the script as ``python generate_sulcal_regions.py`` does (runpy, __main__), so
# they pin the process exit code whichever function computes it. ``run_with_params`` is replaced on
# its own module, which the script imports it from. Entries recorded before the run are what a
# normal rerun finds; caplog is cleared after seeding, so the ERROR records left are the summary's.

SKELETONS = "generate_skeletons"
CROPS = "generate_crops"
NO_VALID_SUBJECT = re.compile("no valid subject", re.IGNORECASE)


class ScriptRun:
    """One dataset for the script: subject folders under graphs_dir, skip lists under output_dir."""

    def __init__(self, tmp_path, monkeypatch, sulcal_module, caplog):
        self.graphs_dir = tmp_path / "graphs"
        self.output_dir = tmp_path / "out"
        self.dataset = tmp_path / "dataset"
        self.calls = []
        self.on_call = None
        self.config = {
            "save_behavior": "best",
            "graphs_dir": str(self.graphs_dir),
            "output_dir": str(self.output_dir),
            "path_to_graph": "cfg/graph",
            "path_to_skeleton_with_hull": "cfg/hull",
            "skel_qc_path": "",
        }
        self._monkeypatch = monkeypatch
        self._caplog = caplog
        self.graphs_dir.mkdir()
        self.dataset.mkdir()
        monkeypatch.setattr(sulcal_module, "run_with_params", self._record_call)

    def _record_call(self, cfg):
        self.calls.append((cfg["region_name"], cfg["side"], cfg["input_type"]))
        if self.on_call is not None:
            self.on_call(cfg)

    def subjects(self, *names):
        for name in names:
            (self.graphs_dir / name).mkdir()

    def skip(self, side, step, subject, exc=None, skip_dir=None):
        record_skip(skip_dir or self.output_dir, side, step, subject, exc or ValueError("bad graph"))

    def run(self, sides=("L", "R"), regions=("S.Or.",), extra=()):
        """Runs the script and returns its exit code; caplog then holds the records of this run only."""
        (self.dataset / "pipeline_loop_2mm.json").write_text(json.dumps(self.config))
        argv = ["-d", str(self.dataset), "--path_to_graph", "G", "--path_sk_with_hull", "H"]
        argv += ["-i", *sides, "-r", *regions, "-y", "skeleton", *extra]
        self._monkeypatch.setattr(sys, "argv", ["generate_sulcal_regions.py", *argv])
        self._caplog.clear()
        with self._caplog.at_level(logging.INFO):
            try:
                runpy.run_path(SCRIPT, run_name="__main__")
            except SystemExit as exc:
                return 0 if exc.code is None else exc.code
            except Exception:  # the interpreter turns an uncaught exception into exit 1
                return 1
        return 0

    def summary_records(self):
        """The ERROR messages of the run, record_skip's own per-subject records left out."""
        return [
            record.getMessage()
            for record in self._caplog.records
            if record.levelno == logging.ERROR and not record.getMessage().startswith("Skipping subject")
        ]


@pytest.fixture
def script(gsr, tmp_path, monkeypatch, caplog):
    import generate_one_sulcal_region

    return ScriptRun(tmp_path, monkeypatch, generate_one_sulcal_region, caplog)


def _summary_of(script, side, step):
    """The only summary record naming ``step`` for ``side``."""
    side_word = re.compile(rf"\b{side}\b")
    found = [message for message in script.summary_records() if step in message and side_word.search(message)]
    assert len(found) == 1, f"expected one summary of {side}/{step}, got {script.summary_records()}"
    return found[0]


# --- run summary -----------------------------------------------------------------------------------


def test_run_summary_logs_one_error_record_per_side_and_step(script):
    """REQ-CTFAIL-BDRABCZUK-9EC36FABC64D: one ERROR record per (side in sides, step with an entry)."""
    script.subjects("sub-a", "sub-b", "sub-c")
    script.skip("L", SKELETONS, "sub-a")
    script.skip("L", SKELETONS, "sub-b")
    script.skip("L", CROPS, "sub-b")
    script.skip("R", CROPS, "sub-c")

    script.run()

    assert len(script.summary_records()) == 3
    for side, step in (("L", SKELETONS), ("L", CROPS), ("R", CROPS)):
        _summary_of(script, side, step)


def test_run_summary_logs_no_error_record_without_skip_entry(script):
    """REQ-CTFAIL-BDRABCZUK-9EC36FABC64D: nothing skipped, no summary record."""
    script.subjects("sub-a")

    script.run()

    assert script.summary_records() == []


def test_run_summary_covers_entry_recorded_in_last_region_call(script):
    """REQ-CTFAIL-BDRABCZUK-9EC36FABC64D: the summary follows the region loop, so it sees its last entry."""
    script.subjects("sub-a", "sub-b")

    def skip_on_last_call(_cfg):
        if len(script.calls) == 2:
            script.skip("R", CROPS, "sub-b")

    script.on_call = skip_on_last_call

    script.run()

    assert "sub-b" in _summary_of(script, "R", CROPS)


def test_run_summary_states_distinct_subject_count(script):
    """REQ-CTFAIL-BDRABCZUK-59AE13F7DD6E: the count is of distinct subjects, a repeated entry counts once."""
    script.subjects("sub-a", "sub-b", "sub-c")
    script.skip("L", CROPS, "sub-a")
    script.skip("L", CROPS, "sub-a")
    script.skip("L", CROPS, "sub-b")

    script.run()

    assert re.search(r"\b2\b", _summary_of(script, "L", CROPS))


def test_run_summary_names_each_subject(script):
    """REQ-CTFAIL-BDRABCZUK-7F013865AC3D: each distinct subject of that side and step, and no other."""
    script.subjects("sub-a", "sub-b", "sub-c")
    script.skip("L", CROPS, "sub-a")
    script.skip("L", CROPS, "sub-a")
    script.skip("L", CROPS, "sub-b")
    script.skip("L", SKELETONS, "sub-c")

    script.run()

    message = _summary_of(script, "L", CROPS)
    assert message.count("sub-a") == 1
    assert "sub-b" in message
    assert "sub-c" not in message


def test_run_summary_states_each_cause(script):
    """REQ-CTFAIL-BDRABCZUK-6397D61CF3A7: exc_type with message of each distinct cause, a repeat stated once."""
    script.subjects("sub-a", "sub-b", "sub-c")
    script.skip("R", CROPS, "sub-a", FileNotFoundError("no skeleton file"))
    script.skip("R", CROPS, "sub-b", FileNotFoundError("no skeleton file"))
    script.skip("R", CROPS, "sub-c", AssertionError("empty crop"))

    script.run()

    message = _summary_of(script, "R", CROPS)
    assert "FileNotFoundError" in message
    assert message.count("no skeleton file") == 1
    assert "AssertionError" in message
    assert "empty crop" in message


def _empty_left_side(script):
    """L keeps no valid subject through entries of two steps, R keeps sub-b with an entry on sub-a."""
    script.subjects("sub-a", "sub-b")
    script.skip("L", SKELETONS, "sub-a")
    script.skip("L", CROPS, "sub-b")
    script.skip("R", CROPS, "sub-a")


def test_run_summary_states_empty_side_in_each_step_record(script):
    """REQ-CTFAIL-BDRABCZUK-7D10357175DE: each step record of a side left empty says it keeps no valid subject."""
    _empty_left_side(script)

    script.run()

    for step in (SKELETONS, CROPS):
        assert NO_VALID_SUBJECT.search(_summary_of(script, "L", step))


def test_run_summary_leaves_empty_side_statement_out_of_side_keeping_a_subject(script):
    """REQ-CTFAIL-BDRABCZUK-7D10357175DE: the record of the side keeping a subject does not call it empty."""
    _empty_left_side(script)

    script.run()

    assert not NO_VALID_SUBJECT.search(_summary_of(script, "R", CROPS))


@pytest.mark.parametrize(
    "lost_regions",
    [pytest.param(("S.Or.",), id="lost-in-one-region"), pytest.param(("S.Or.", "OCCIPITAL"), id="lost-in-each-region")],
)
def test_run_summary_names_side_lost_without_skip_entry(script, lost_regions):
    """REQ-CTFAIL-BDRABCZUK-309DB18C6C45: one record names the side, however many regions raised for it."""
    script.subjects("sub-a", "sub-b")

    def raise_on_left_side(cfg):
        if cfg["side"] == "L" and cfg["region_name"] in lost_regions:
            raise NoValidSubjectError("no subject of side L has both resampled volumes")

    script.on_call = raise_on_left_side

    script.run(regions=("S.Or.", "OCCIPITAL"))

    assert len(script.calls) == 4
    records = script.summary_records()
    assert len(records) == 1
    assert re.search(r"\bL\b", records[0])


# --- exit code -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([("L", "sub-a")], id="one-subject-on-one-side"),
        pytest.param([("L", "sub-a"), ("R", "sub-a")], id="one-subject-on-both-sides"),
        pytest.param([("L", "sub-a"), ("L", "sub-b")], id="one-side-emptied-the-other-valid"),
        pytest.param([("L", "sub-a"), ("R", "sub-b")], id="each-side-keeps-a-subject"),
    ],
)
def test_exit_code_3_when_skip_entry_leaves_a_valid_subject(script, entries):
    """REQ-CTFAIL-BDRABCZUK-96FC3110E1A8: some subjects skipped, a valid one left on a side -> exit 3."""
    script.subjects("sub-a", "sub-b")
    for side, subject in entries:
        script.skip(side, SKELETONS, subject)

    assert script.run() == 3


@pytest.mark.parametrize(
    "subjects, entries",
    [
        pytest.param(("sub-a",), [("L", "sub-a"), ("R", "sub-a")], id="only-subject-skipped-on-both-sides"),
        pytest.param(
            ("sub-a", "sub-b"),
            [("L", "sub-a"), ("L", "sub-b"), ("R", "sub-a"), ("R", "sub-b")],
            id="each-subject-skipped-on-both-sides",
        ),
        pytest.param((), [], id="no-input-subject"),
    ],
)
def test_exit_code_1_when_no_valid_subject_left(script, subjects, entries):
    """REQ-CTFAIL-BDRABCZUK-4127B66C1D40: no side keeps a valid subject -> exit 1."""
    script.subjects(*subjects)
    for side, subject in entries:
        script.skip(side, SKELETONS, subject)

    assert script.run() == 1


def test_exit_code_0_when_nothing_skipped(script):
    """REQ-CTFAIL-BDRABCZUK-55C3DAD3DB34: no skip entry, a valid subject -> exit 0."""
    script.subjects("sub-a")

    assert script.run() == 0


def test_exit_code_0_ignores_skip_entry_of_side_outside_sides(script):
    """REQ-CTFAIL-BDRABCZUK-55C3DAD3DB34: an entry of a side the run left out does not count."""
    script.subjects("sub-a")
    script.skip("L", SKELETONS, "sub-a")

    assert script.run(sides=("R",)) == 0


def test_exit_code_1_on_escaping_exception_with_skip_entries(script):
    """REQ-CTFAIL-BDRABCZUK-D42669337BCA: 1 stays the run-level failure, skip entries do not turn it into 3.

    Guard: already true before this Task, it stops the partial code from masking a crash.
    """
    script.subjects("sub-a", "sub-b")
    script.skip("L", SKELETONS, "sub-a")

    def crash(_cfg):
        raise RuntimeError("run-level failure")

    script.on_call = crash

    assert script.run() == 1


def _no_valid_subject_on_call(script, call_number):
    """The given call raises as the count check does: no entry recorded, the side lost."""

    def on_call(_cfg):
        if len(script.calls) == call_number:
            raise NoValidSubjectError("no subject of side L has both resampled volumes")

    return on_call


@pytest.mark.parametrize(
    "raising_call", [pytest.param(1, id="first-region-call"), pytest.param(4, id="last-region-call")]
)
def test_exit_code_3_when_no_valid_subject_error_leaves_no_skip_entry(script, raising_call):
    """REQ-CTFAIL-BDRABCZUK-1D03DBFCB485: the raise alone gives 3, from any region, with no entry on any side."""
    script.subjects("sub-a", "sub-b")
    script.on_call = _no_valid_subject_on_call(script, raising_call)

    exit_code = script.run(regions=("S.Or.", "OCCIPITAL"))

    assert len(script.calls) == 4
    assert load_skip_entries(script.output_dir, "L") == []
    assert load_skip_entries(script.output_dir, "R") == []
    assert exit_code == 3


def test_exit_code_1_when_no_valid_subject_error_and_no_valid_subject_left(script):
    """REQ-CTFAIL-BDRABCZUK-1D03DBFCB485, REQ-CTFAIL-BDRABCZUK-4127B66C1D40: no valid subject left, 1 outranks 3.

    The call count tells the computed 1 from an exception escaping the loop.
    """
    script.subjects("sub-a")
    script.skip("L", SKELETONS, "sub-a")
    script.skip("R", SKELETONS, "sub-a")
    script.on_call = _no_valid_subject_on_call(script, 1)

    exit_code = script.run()

    assert len(script.calls) == 2
    assert exit_code == 1


def test_valid_subject_leaves_out_non_subject_entry(script):
    """REQ-CTFAIL-BDRABCZUK-EC7DF3FC4D5B: a graphs_dir entry refused by is_it_a_subject is no valid subject."""
    script.subjects("sub-a")
    (script.graphs_dir / "sub-b.minf").write_text("")
    script.skip("L", SKELETONS, "sub-a")
    script.skip("R", SKELETONS, "sub-a")

    assert script.run() == 1


def test_valid_subject_leaves_out_qc_rejected_subject(script, tmp_path):
    """REQ-CTFAIL-BDRABCZUK-EC7DF3FC4D5B: a subject dropped by select_good_qc is no valid subject."""
    script.subjects("sub-a", "sub-b")
    qc_file = tmp_path / "qc.tsv"
    qc_file.write_text("participant_id\tqc\nsub-a\t1\nsub-b\t0\n")
    script.skip("L", SKELETONS, "sub-a")
    script.skip("R", SKELETONS, "sub-a")

    assert script.run(extra=("--sk_qc_path", str(qc_file))) == 1


# --- a skip never stops the run --------------------------------------------------------------------


def _skip_entry_on_first_call(script):
    """The first call records one entry and returns."""

    def on_call(_cfg):
        if len(script.calls) == 1:
            script.skip("L", SKELETONS, "sub-a")

    return on_call


def _no_valid_subject_on_left_side(script):
    """Each call of L records the entries of that side, then raises as generate_crops does on an empty stack."""

    def on_call(cfg):
        if cfg["side"] == "L":
            script.skip("L", CROPS, "sub-a")
            script.skip("L", CROPS, "sub-b")
            raise NoValidSubjectError("no subject left in the stack of side L")

    return on_call


@pytest.mark.parametrize(
    "on_call_of",
    [
        pytest.param(_skip_entry_on_first_call, id="entry-recorded"),
        pytest.param(_no_valid_subject_on_left_side, id="entries-recorded-then-no-valid-subject-raised"),
    ],
)
def test_run_finishes_remaining_calls_after_recorded_skip(script, on_call_of):
    """REQ-CTFAIL-BDRABCZUK-E9E61C41A160: a call recording a skip leaves the remaining calls of the run.

    The raising case is the real risk: an uncaught NoValidSubjectError would stop the loop while R keeps a subject.
    """
    script.subjects("sub-a", "sub-b")
    script.on_call = on_call_of(script)

    exit_code = script.run(regions=("S.Or.", "OCCIPITAL"))

    assert script.calls == [
        ("S.Or.", "L", "skeleton"),
        ("S.Or.", "R", "skeleton"),
        ("OCCIPITAL", "L", "skeleton"),
        ("OCCIPITAL", "R", "skeleton"),
    ]
    assert exit_code == 3


# --- skip-list reset -------------------------------------------------------------------------------


def test_clear_and_compute_empties_skip_list_before_first_region(script):
    """REQ-CTFAIL-BDRABCZUK-107B7A11BBE7: clear_and_compute empties the list of each side in sides at run start."""
    script.config["save_behavior"] = "clear_and_compute"
    script.subjects("sub-a", "sub-b")
    script.skip("L", SKELETONS, "sub-a")
    script.skip("R", SKELETONS, "sub-b")
    seen = []
    script.on_call = lambda _cfg: seen.append(
        (load_skip_entries(script.output_dir, "L"), load_skip_entries(script.output_dir, "R"))
    )

    script.run()

    assert seen[0] == ([], [])


def test_clear_and_compute_keeps_entry_recorded_during_the_run(script):
    """REQ-CTFAIL-BDRABCZUK-107B7A11BBE7: the clear happens once per side, so an entry of the first region stays."""
    script.config["save_behavior"] = "clear_and_compute"
    script.subjects("sub-a", "sub-b")
    script.skip("L", SKELETONS, "sub-b")

    def skip_on_first_call(_cfg):
        if len(script.calls) == 1:
            script.skip("L", CROPS, "sub-a")

    script.on_call = skip_on_first_call

    assert script.run(regions=("S.Or.", "OCCIPITAL")) == 3
    assert [entry.subject for entry in load_skip_entries(script.output_dir, "L")] == ["sub-a"]


@pytest.mark.parametrize("save_behavior", ["best", "minimal"])
def test_other_save_behavior_keeps_skip_entries(script, save_behavior):
    """REQ-CTFAIL-BDRABCZUK-A031F9E35FF3: a normal rerun finds and keeps the entries of the previous run.

    Guard: clear_skip_list has no caller yet, it stops the clear call from reaching a normal rerun.
    """
    script.config["save_behavior"] = save_behavior
    script.subjects("sub-a", "sub-b")
    script.skip("L", SKELETONS, "sub-a")

    script.run()

    assert [entry.subject for entry in load_skip_entries(script.output_dir, "L")] == ["sub-a"]


@pytest.mark.parametrize("save_behavior", ["best", "clear_and_compute"])
def test_side_outside_sides_keeps_its_skip_list(script, save_behavior):
    """REQ-CTFAIL-BDRABCZUK-05D40790A955: a run limited to R leaves the list of L as it found it.

    Guard: clear_skip_list has no caller yet, it stops the clear call from reaching a side the run left out.
    """
    script.config["save_behavior"] = save_behavior
    script.subjects("sub-a", "sub-b")
    script.skip("L", SKELETONS, "sub-a")

    script.run(sides=("R",))

    assert [entry.subject for entry in load_skip_entries(script.output_dir, "L")] == ["sub-a"]


def test_skip_dir_is_the_resolved_output_dir(script):
    """REQ-CTFAIL-BDRABCZUK-C77724B08161: a $local output_dir resolves to <dataset>/<-o> before skip lists are read."""
    script.config["output_dir"] = "$local"
    script.subjects("sub-a", "sub-b")
    script.skip("L", SKELETONS, "sub-a", skip_dir=script.dataset / "out")

    assert script.run(extra=("-o", "out")) == 3
