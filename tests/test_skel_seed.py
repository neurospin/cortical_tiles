"""Seeded VipSkeleton re-skeletonization (TASK-126 split, SKELSEED tasks 1-3).

The 2 mm skeleton resampling re-skeletonizes with two ``VipSkeleton`` calls whose output depends on a
clock-seeded random generator unless ``-srand <seed>`` is given (TASK-126 evidence: ``-srand 42``
gives identical outputs, unseeded runs gave 4 different md5s). These tests pin the seed plumbing:

* REQ-SKELSEED-BDRABCZUK-BFBCC246C7FA: ``resample(do_skel=True, srand=N)`` adds ``-srand N`` to both
  VipSkeleton commands; ``srand=None`` (the default) leaves the commands unchanged.
* REQ-SKELSEED-BDRABCZUK-04C2A504FC9F: the ``skel_seed`` configuration key reaches ``resample(srand=)``
  through ``run_with_params`` -> ``resample_files(skel_seed=)`` -> SkeletonResampler; without the key
  the module constant ``DEFAULT_SKEL_SEED`` (42) is used.
* REQ-SKELSEED-BDRABCZUK-860C364DCA0B: ``generate_sulcal_regions.py --skel_seed N`` overrides the
  ``skel_seed`` value of ``pipeline_loop_2mm.json`` (flag > config file > DEFAULT_SKEL_SEED).

No BrainVISA binary is run: a fake ``VipSkeleton`` put first on PATH records its arguments and copies
its input to its output, so the commands are checked whatever subprocess API runs them.
"""

import inspect
import json
import os
import sys

import numpy as np
import pytest
from cortical_tiles.brainvisa import generate_one_sulcal_region as gosr
from cortical_tiles.brainvisa import resample_files as rf
from cortical_tiles.brainvisa.utils import resample as rs
from soma import aims

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAINVISA_DIR = os.path.join(REPO_DIR, "cortical_tiles", "brainvisa")

SKELETON_VALUES = [100, 60, 10, 20, 40, 50, 70, 80, 110, 120, 30, 35]
IMMORTALS = [30, 50, 80, 35, 110, 120]


# --- helpers -------------------------------------------------------------------------------------


@pytest.fixture
def vip_calls(tmp_path, monkeypatch):
    """Puts a fake VipSkeleton first on PATH; returns a function reading the recorded argument lists."""
    bindir = tmp_path / "fake_bin"
    bindir.mkdir()
    log = tmp_path / "vip_calls.jsonl"
    fake = tmp_path / "fake_vipskeleton.py"
    fake.write_text(
        "import json, shutil, sys\n"
        "args = sys.argv[1:]\n"
        f"open({str(log)!r}, 'a').write(json.dumps(args) + '\\n')\n"
        "shutil.copyfile(args[args.index('-i') + 1], args[args.index('-so') + 1])\n"
    )
    launcher = bindir / "VipSkeleton"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")

    def read():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    return read


def _skeleton_volume():
    """A 1 mm synthetic skeleton: a simple surface (60) bordered by a bottom (30) and a junction (80)."""
    arr = np.zeros((40, 40, 40, 1), dtype=np.int16)
    arr[10:30, 10:30, 20, 0] = 60
    arr[10:30, 10, 20, 0] = 30
    arr[10, 10:30, 20, 0] = 80
    vol = aims.Volume(arr)
    vol.header()["voxel_size"] = [1.0, 1.0, 1.0, 1.0]
    return vol


def _resample_skeleton(**kwargs):
    return rs.resample(
        _skeleton_volume(),
        aims.AffineTransformation3d(),
        output_vs=(2, 2, 2),
        values=SKELETON_VALUES,
        do_skel=True,
        immortals=IMMORTALS,
        **kwargs,
    )


def _option_value(args, option):
    return args[args.index(option) + 1] if option in args else None


# --- REQ-SKELSEED-BDRABCZUK-BFBCC246C7FA: resample(srand=) -----------------------------------------


def test_resample_srand_defaults_to_none():
    """srand is an optional resample() argument whose default None keeps the legacy behaviour."""
    parameters = inspect.signature(rs.resample).parameters
    assert "srand" in parameters, "resample() has no srand argument"
    assert parameters["srand"].default is None


@pytest.mark.parametrize("seed", [42, 0, 7])
def test_resample_skel_passes_srand_to_both_vipskeleton_commands(vip_calls, seed):
    """do_skel=True with an integer srand: both VipSkeleton commands carry -srand <seed> (0 included)."""
    _resample_skeleton(srand=seed)

    calls = vip_calls()
    assert len(calls) == 2
    assert [_option_value(args, "-srand") for args in calls] == [str(seed), str(seed)]


def test_resample_skel_with_srand_none_omits_srand_option(vip_calls):
    """do_skel=True with srand=None: neither VipSkeleton command has a -srand option (clock seed)."""
    _resample_skeleton(srand=None)

    calls = vip_calls()
    assert len(calls) == 2
    assert all("-srand" not in args for args in calls)


# --- REQ-SKELSEED-BDRABCZUK-04C2A504FC9F: skel_seed config -> resample(srand=) ---------------------


STEP_FUNCTIONS = [
    "compute_mask",
    "generate_skeletons",
    "generate_distmaps",
    "generate_extremities",
    "generate_foldlabels",
    "generate_ICBM2009c_transforms",
    "resample_files",
    "mask_foldlabel_files",
    "mask_extremities_files",
    "generate_crops",
    "generate_distbottom_crops",
]


@pytest.fixture
def step_calls(monkeypatch):
    """Replaces every step function of generate_one_sulcal_region by a recorder of (name, kwargs)."""
    recorded = []

    def make(name):
        def recorder(**kwargs):
            recorded.append((name, kwargs))
            if name == "generate_crops":
                os.makedirs(os.path.join(kwargs["crop_dir"], kwargs["side"] + "crops"), exist_ok=True)

        return recorder

    for name in STEP_FUNCTIONS:
        monkeypatch.setattr(gosr, name, make(name))
    monkeypatch.setattr(gosr, "setup_log", lambda *args, **kwargs: None)
    return recorded


def _region_params(tmp_path, **overrides):
    regions_json = tmp_path / "regions.json"
    regions_json.write_text(json.dumps({"brain": {"S.Or._right": {"S.Or._right": 1}}}))
    params = {
        "save_behavior": "best",
        "side": "R",
        "out_voxel_size": 2.0,
        "region_name": "S.Or.",
        "brain_regions_json": str(regions_json),
        "parallel": False,
        "nb_subjects": -1,
        "input_type": "skeleton",
        "labeled_subjects_dir": "labeled",
        "path_to_graph_supervised": "pgs",
        "supervised_output_dir": str(tmp_path / "supervised"),
        "nb_subjects_mask": -1,
        "graphs_dir": "graphs",
        "path_to_graph": "pg",
        "path_to_skeleton_with_hull": "ph",
        "skel_qc_path": "",
        "output_dir": str(tmp_path / "derivatives"),
        "junction": "thin",
        "bids": False,
        "new_sulcus": None,
        "resampled_skel": False,
        "cropping_type": "mask",
        "combine_type": False,
        "no_mask": False,
        "threshold": 0,
        "dilation": 5,
        "njobs": 1,
        "masks_version": "canonical_25",
    }
    params.update(overrides)
    return params


def _resample_files_kwargs(step_calls):
    called = [kwargs for name, kwargs in step_calls if name == "resample_files"]
    assert len(called) == 1
    return called[0]


def test_default_skel_seed_is_42():
    """DEFAULT_SKEL_SEED is 42, the seed verified deterministic in the TASK-126 evidence."""
    assert getattr(gosr, "DEFAULT_SKEL_SEED", None) == 42


@pytest.mark.parametrize("seed", [7, 0])
def test_run_with_params_passes_config_skel_seed_to_resample_files(tmp_path, step_calls, seed):
    """input_type skeleton: the skel_seed configuration value is handed to resample_files(skel_seed=)."""
    gosr.run_with_params(_region_params(tmp_path, skel_seed=seed))

    assert _resample_files_kwargs(step_calls).get("skel_seed") == seed


def test_run_with_params_without_skel_seed_passes_default_seed(tmp_path, step_calls):
    """A configuration without skel_seed (pre-change pipeline_loop_2mm.json copies) still seeds with 42."""
    params = _region_params(tmp_path)
    assert "skel_seed" not in params
    gosr.run_with_params(params)

    assert _resample_files_kwargs(step_calls).get("skel_seed") == 42


@pytest.mark.parametrize("seed", [7, 0])
def test_resample_files_skeleton_passes_skel_seed_to_resample_as_srand(tmp_path, monkeypatch, seed):
    """resample_files(input_type='skeleton', skel_seed=N) reaches resample(srand=N) for each subject."""
    raw = tmp_path / "raw" / "R"
    raw.mkdir(parents=True)
    (tmp_path / "transforms" / "R").mkdir(parents=True)
    for subject in ("s1", "s2"):
        aims.write(_skeleton_volume(), str(raw / f"Rskeleton_generated_{subject}.nii.gz"))

    received = []

    def fake_resample(*args, **kwargs):
        received.append(kwargs)
        return aims.Volume(np.zeros((4, 4, 4, 1), dtype=np.int16))

    monkeypatch.setattr(rf, "resample", fake_resample)

    rf.resample_files(
        src_dir=str(tmp_path / "raw"),
        input_type="skeleton",
        resampled_dir=str(tmp_path / "out"),
        transform_dir=str(tmp_path / "transforms"),
        side="R",
        out_voxel_size=2.0,
        src_filename="skeleton_generated_",
        output_filename="resampled_skeleton_",
        skel_seed=seed,
    )

    assert len(received) == 2
    assert [kwargs.get("srand") for kwargs in received] == [seed, seed]
    assert all(kwargs.get("do_skel") is True for kwargs in received)


# --- REQ-SKELSEED-BDRABCZUK-860C364DCA0B: --skel_seed flag ------------------------------------------


@pytest.fixture
def gsr(monkeypatch):
    """generate_sulcal_regions imports its sibling as a top-level module: brainvisa dir on sys.path."""
    monkeypatch.syspath_prepend(BRAINVISA_DIR)
    import generate_sulcal_regions

    return generate_sulcal_regions


@pytest.fixture
def configs(gsr, monkeypatch):
    """Replaces run_with_params by a recorder of (a copy of) each configuration it receives."""
    recorded = []
    monkeypatch.setattr(gsr, "run_with_params", lambda cfg: recorded.append(dict(cfg)))
    return recorded


BASE_CONFIG = {
    "masks_version": "canonical_25",
    "path_to_graph": "cfg/graph",
    "path_to_skeleton_with_hull": "cfg/hull",
    "skel_qc_path": "cfg/qc.tsv",
    "output_dir": "/cfg/out",
}

REQUIRED_FLAGS = ["-d", "D", "--path_to_graph", "G", "--path_sk_with_hull", "H"]


def _generate(gsr, tmp_path, config, **overrides):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "pipeline_loop_2mm.json").write_text(json.dumps(config))
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


def test_parse_args_skel_seed_is_int(gsr):
    """--skel_seed N is parsed as the integer N under the skel_seed key."""
    params = gsr.parse_args(REQUIRED_FLAGS + ["--skel_seed", "7"])
    assert params.get("skel_seed") == 7
    assert isinstance(params.get("skel_seed"), int)


def test_parse_args_skel_seed_defaults_to_none(gsr):
    """Without --skel_seed, parse_args gives skel_seed None (no override)."""
    params = gsr.parse_args(REQUIRED_FLAGS)
    assert "skel_seed" in params
    assert params["skel_seed"] is None


@pytest.mark.parametrize("seed", [7, 0])
def test_skel_seed_argument_overrides_config(gsr, configs, tmp_path, seed):
    """An integer skel_seed argument replaces the skel_seed of pipeline_loop_2mm.json (0 included)."""
    _generate(gsr, tmp_path, dict(BASE_CONFIG, skel_seed=42), skel_seed=seed)
    assert configs[0]["skel_seed"] == seed


@pytest.mark.parametrize("config, expected", [(dict(BASE_CONFIG, skel_seed=42), 42), (BASE_CONFIG, "absent")])
def test_absent_skel_seed_argument_keeps_config(gsr, configs, tmp_path, config, expected):
    """skel_seed=None leaves the configuration untouched: its value, or no key (DEFAULT_SKEL_SEED applies)."""
    _generate(gsr, tmp_path, config, skel_seed=None)
    assert configs[0].get("skel_seed", "absent") == expected
