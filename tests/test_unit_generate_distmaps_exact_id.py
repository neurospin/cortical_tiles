"""generate_one_distmap picks the skeleton file by exact subject id (DEF-CTFAIL generate_distmaps glob).

A subject id that is a prefix of another (sub-1, sub-10) must never take the skeleton of the
other subject, whatever order the directory listing returns. No file or several files for the
requested id is an error that names the id and the count. Recorders replace the two distmap
functions, so no volume is processed.
"""

import glob
import os

import pytest
from cortical_tiles.brainvisa import generate_distmaps as gd


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


@pytest.fixture
def reverse_listing(monkeypatch):
    """Make glob.glob return its matches in reverse sorted order: the wrong file would come first."""
    real = glob.glob
    monkeypatch.setattr(gd.glob, "glob", lambda pattern, *a, **k: sorted(real(pattern, *a, **k), reverse=True))


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return str(path)


def _conv(tmp_path, resampled):
    return gd.SkelConvert2DistMap(
        str(tmp_path / "src"), str(tmp_path / "dm"), "R", parallel=False, resampled_skel=resampled
    )


@pytest.mark.parametrize("listing", ["plain", "reverse"])
def test_raw_mode_prefix_id_takes_own_skeleton(tmp_path, recorders, request, listing):
    """sub-1 takes the skeleton of sub-1, not that of sub-10, in either listing order."""
    if listing == "reverse":
        request.getfixturevalue("reverse_listing")
    other = _touch(tmp_path / "src" / "R" / "Rskeleton_generated_sub-10.nii.gz")
    own = _touch(tmp_path / "src" / "R" / "Rskeleton_generated_sub-1.nii.gz")
    _conv(tmp_path, False).generate_one_distmap("sub-1")
    assert recorders == [("skeleton", own, f"{tmp_path}/dm/R/Rdistmap_generated_sub-1.nii.gz")]
    assert other not in [c[1] for c in recorders]


@pytest.mark.parametrize("listing", ["plain", "reverse"])
def test_resampled_mode_prefix_id_takes_own_skeleton(tmp_path, recorders, request, listing):
    """Resampled mode: sub-1 takes Rresampled_sub-1, not Rresampled_sub-10, in either listing order."""
    if listing == "reverse":
        request.getfixturevalue("reverse_listing")
    other = _touch(tmp_path / "src" / "Rresampled_sub-10.nii.gz")
    own = _touch(tmp_path / "src" / "Rresampled_sub-1.nii.gz")
    _conv(tmp_path, True).generate_one_distmap("sub-1")
    assert recorders == [("resampled", own, f"{tmp_path}/dm/R/Rdistmap_generated_sub-1.nii.gz")]
    assert other not in [c[1] for c in recorders]


@pytest.mark.parametrize("resampled", [False, True])
def test_longer_id_takes_own_skeleton(tmp_path, recorders, resampled):
    """sub-10 takes its own skeleton when sub-1 also exists."""
    if resampled:
        _touch(tmp_path / "src" / "Rresampled_sub-1.nii.gz")
        own = _touch(tmp_path / "src" / "Rresampled_sub-10.nii.gz")
    else:
        _touch(tmp_path / "src" / "R" / "Rskeleton_generated_sub-1.nii.gz")
        own = _touch(tmp_path / "src" / "R" / "Rskeleton_generated_sub-10.nii.gz")
    _conv(tmp_path, resampled).generate_one_distmap("sub-10")
    assert [c[1] for c in recorders] == [own]


@pytest.mark.parametrize("resampled", [False, True])
def test_only_prefixed_neighbour_present_is_an_error(tmp_path, recorders, resampled):
    """Only sub-10 exists: sub-1 has no file, so it fails instead of using sub-10's file."""
    if resampled:
        _touch(tmp_path / "src" / "Rresampled_sub-10.nii.gz")
    else:
        _touch(tmp_path / "src" / "R" / "Rskeleton_generated_sub-10.nii.gz")
    with pytest.raises(ValueError, match=r"sub-1\b.*\b0\b"):
        _conv(tmp_path, resampled).generate_one_distmap("sub-1")
    assert recorders == []


def test_raw_mode_no_file_is_an_error_naming_id_and_count(tmp_path, recorders):
    os.makedirs(tmp_path / "src" / "R")
    with pytest.raises(ValueError, match=r"sub-7.*\b0\b"):
        _conv(tmp_path, False).generate_one_distmap("sub-7")
    assert recorders == []


def test_resampled_mode_several_files_is_an_error_naming_id_and_count(tmp_path, recorders):
    """Two files carry the id as their subject token: no silent first pick."""
    _touch(tmp_path / "src" / "Rresampled_skeleton_sub-1.nii.gz")
    _touch(tmp_path / "src" / "Rresampled_extremities_sub-1.nii.gz")
    with pytest.raises(ValueError, match=r"sub-1.*\b2\b"):
        _conv(tmp_path, True).generate_one_distmap("sub-1")
    assert recorders == []
