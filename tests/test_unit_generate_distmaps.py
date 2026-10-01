"""Unit tests for cortical_tiles.brainvisa.generate_distmaps (REQ-CTILESTEST-162..166).

The distance-map computation itself (utils/distmap) is covered by test_unit_utils_distmap.py
and test_generate_distmaps.py; these tests pin the orchestration: which skeleton file is
handed to which distmap function, which subjects are processed, and the CLI. Recorders
replace the two distmap functions (they write an empty output file), so no volume is
processed; everything lives in a pytest tmp dir. The parallel path runs in a fresh pathos
pool so the forked workers see the recorders.
"""

import os

import pytest
from cortical_tiles.brainvisa import generate_distmaps as gd
from pathos import multiprocessing as pathos_mp


def _clear_pathos_pools():
    state = pathos_mp._ProcessPool__STATE
    for pool in list(state.values()):
        pool.terminate()
        pool.join()
    state.clear()


@pytest.fixture
def fresh_process_pool():
    _clear_pathos_pools()
    yield
    _clear_pathos_pools()


@pytest.fixture
def recorders(monkeypatch):
    calls = []

    def _make(kind):
        def _fake(skeleton_file, distmap_file):
            calls.append((kind, skeleton_file, distmap_file))
            open(distmap_file, "w").close()

        return _fake

    monkeypatch.setattr(gd, "generate_distmap_from_skeleton_file", _make("skeleton"))
    monkeypatch.setattr(gd, "generate_distmap_from_resampled_skeleton", _make("resampled"))
    return calls


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return str(path)


# --- REQ-CTILESTEST-162 / 163: generate_one_distmap ------------------------------------------------


def test_raw_skeleton_mode_reads_side_subdir(tmp_path, recorders):
    """REQ-CTILESTEST-162: resampled_skel False -> skeleton-file distmap of <src>/<side>/*<subject>*.nii.gz."""
    skel = _touch(tmp_path / "src" / "R" / "Rskeleton_generated_s01.nii.gz")
    _touch(tmp_path / "src" / "Rresampled_s01.nii.gz")  # must be ignored in this mode
    conv = gd.SkelConvert2DistMap(
        str(tmp_path / "src"), str(tmp_path / "dm"), "R", parallel=False, resampled_skel=False
    )

    conv.generate_one_distmap("s01")

    assert recorders == [("skeleton", skel, f"{tmp_path}/dm/R/Rdistmap_generated_s01.nii.gz")]


def test_resampled_mode_reads_src_dir_root(tmp_path, recorders):
    """REQ-CTILESTEST-163: resampled_skel True -> resampled-skeleton distmap of <src>/*<subject>*.nii.gz."""
    skel = _touch(tmp_path / "src" / "Rresampled_s01.nii.gz")
    _touch(tmp_path / "src" / "R" / "Rskeleton_generated_s01.nii.gz")  # must be ignored in this mode
    conv = gd.SkelConvert2DistMap(str(tmp_path / "src"), str(tmp_path / "dm"), "R", parallel=False, resampled_skel=True)

    conv.generate_one_distmap("s01")

    assert recorders == [("resampled", skel, f"{tmp_path}/dm/R/Rdistmap_generated_s01.nii.gz")]


# --- REQ-CTILESTEST-164: compute ----------------------------------------------------------------


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_compute_generates_distmaps_only_for_unprocessed_subjects(tmp_path, recorders, fresh_process_pool, parallel):
    """REQ-CTILESTEST-164: compute writes one distmap per <side>skeleton_generated_<subject> lacking one."""
    for s in ("s01", "s02", "s03"):
        _touch(tmp_path / "src" / "L" / f"Lskeleton_generated_{s}.nii.gz")
    done = _touch(tmp_path / "dm" / "L" / "Ldistmap_generated_s02.nii.gz")
    os.utime(done, (1, 1))
    conv = gd.SkelConvert2DistMap(
        str(tmp_path / "src"), str(tmp_path / "dm"), "L", parallel=parallel, resampled_skel=False
    )

    conv.compute(number_subjects="all")

    assert sorted(os.listdir(tmp_path / "dm" / "L")) == [
        f"Ldistmap_generated_{s}.nii.gz" for s in ("s01", "s02", "s03")
    ]
    assert os.stat(done).st_mtime == 1  # already-processed subject not regenerated
    if not parallel:
        assert sorted(c[2] for c in recorders) == [
            f"{tmp_path}/dm/L/Ldistmap_generated_{s}.nii.gz" for s in ("s01", "s03")
        ]


# --- REQ-CTILESTEST-165 / 166: main / parse_args -------------------------------------------------


@pytest.fixture
def captured_main(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(gd, "generate_distmaps", lambda **kwargs: received.append(kwargs))
    monkeypatch.chdir(tmp_path)
    return received


def test_main_maps_arguments_to_generate_distmaps(tmp_path, captured_main):
    """REQ-CTILESTEST-165: main(argv) calls generate_distmaps once; dirs absolute, side/parallel kept, nb int."""
    gd.main(["-s", "skel", "-o", "dm", "-i", "L", "-a", "-n", "4"])
    assert captured_main == [
        dict(
            src_dir=str(tmp_path / "skel"),
            distmaps_dir=str(tmp_path / "dm"),
            side="L",
            parallel=True,
            resampled_skel=False,
            number_subjects=4,
        )
    ]


@pytest.mark.parametrize("value", ["False", "0", "no"])
def test_main_resampled_flag_keeps_raw_string(captured_main, value):
    """REQ-CTILESTEST-166 (DEFECT): '-r <value>' passes the string itself, so '-r False' is truthy."""
    gd.main(["-o", "dm", "-r", value])
    assert captured_main[0]["resampled_skel"] == value
    assert bool(captured_main[0]["resampled_skel"]) is True
