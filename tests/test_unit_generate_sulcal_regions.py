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
import os

import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAINVISA_DIR = os.path.join(REPO_DIR, "cortical_tiles", "brainvisa")

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


def test_local_brain_regions_json(gsr, calls, local_cwd):
    """REQ-CTILESTEST-108: $local brain_regions_json -> <4th parent of cwd>/sulci_regions_champollion_V1.json."""
    _run(gsr, _local_config(local_cwd), output_dir="out")
    assert calls[0]["brain_regions_json"] == os.path.join(str(local_cwd / "p1"), "sulci_regions_champollion_V1.json")


def test_local_supervised_output_dir(gsr, calls, local_cwd):
    """REQ-CTILESTEST-109: $local supervised_output_dir -> <3rd parent of cwd>/cortical_tiles/data."""
    _run(gsr, _local_config(local_cwd), output_dir="out")
    assert calls[0]["supervised_output_dir"] == os.path.join(str(local_cwd / "p1" / "p2"), "cortical_tiles/data")


def test_local_graphs_dir(gsr, calls, local_cwd):
    """REQ-CTILESTEST-110: $local graphs_dir -> <path_dataset>/derivatives/morphologist-6.0."""
    dataset = _local_config(local_cwd)
    _run(gsr, dataset, output_dir="out")
    assert calls[0]["graphs_dir"] == os.path.join(str(dataset), "derivatives/morphologist-6.0")


@pytest.mark.parametrize("output_dir", ["out", "/abs/out"])
def test_local_output_dir_uses_given_output_dir(gsr, calls, local_cwd, output_dir):
    """REQ-CTDEFECTS-14 (inverts REQ-CTILESTEST-111): $local output_dir, output_dir given -> join(dataset, output_dir)."""
    dataset = _local_config(local_cwd)
    _run(gsr, dataset, output_dir=output_dir)
    assert calls[0]["output_dir"] == os.path.join(str(dataset), output_dir)


@pytest.mark.parametrize("output_dir", [None, ""])
def test_local_output_dir_defaults_to_derivatives_when_not_given(gsr, calls, local_cwd, output_dir):
    """REQ-CTDEFECTS-15 (inverts REQ-CTILESTEST-112): $local output_dir, None/'' -> <dataset>/derivatives/cortical_tiles-<v>."""
    dataset = _local_config(local_cwd)
    _run(gsr, dataset, output_dir=output_dir)
    expected = os.path.join(str(dataset), f"derivatives/cortical_tiles-{gsr._CORTICAL_TILES_VERSION}")
    assert calls[0]["output_dir"] == expected


def test_local_graph_paths_and_qc_path(gsr, calls, local_cwd):
    """REQ-CTILESTEST-113: $local graph paths take non-empty arguments, else stay '$local'; skel_qc_path = argument."""
    _run(gsr, _local_config(local_cwd), output_dir="out", path_to_graph="arg/graph", sk_qc_path="arg/qc.tsv")
    assert calls[0]["path_to_graph"] == "arg/graph"
    assert calls[0]["path_to_skeleton_with_hull"] == "$local"
    assert calls[0]["skel_qc_path"] == "arg/qc.tsv"


# --- REQ-CTILESTEST-114: main ----------------------------------------------------------------------


def test_main_passes_parsed_arguments(gsr, monkeypatch):
    """REQ-CTILESTEST-114: main(argv) calls generate_sulcal_regions once with the parse_args params."""
    received = []
    monkeypatch.setattr(gsr, "generate_sulcal_regions", lambda **kw: received.append(kw))
    argv = ["-d", "D", "--path_to_graph", "G", "--path_sk_with_hull", "H", "-i", "R", "-r", "S.Or.", "--njobs", "2"]

    gsr.main(argv)

    assert received == [gsr.parse_args(argv)]
    assert received[0]["sides"] == ["R"] and received[0]["regions"] == ["S.Or."] and received[0]["njobs"] == 2
