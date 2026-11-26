"""Several labelling sets without a graph label (TASK-ARGDISC-BDRABCZUK-817BCB001B96).

User answer 2026-11-25: when graph discovery meets a subject folder holding labelled .arg files of two or
more labelling sets and no graph label is given, the command line stops with exit status 2 and lists the
sets, after a pre-scan of every subject folder and before any region run, mask increment or box read, before
compute_mask creates the side folder of its output directory (REQ-ARGDISC-BDRABCZUK-A7FDADCC7F0E) and before
compute_bounding_box writes its general information to the bounding box JSON file (REQ-ARGDISC-BDRABCZUK-E0A131863927).
The resolver raises a distinct error type, SeveralLabellingSetsError (a ValueError), so that the command
lines tell this case from the graphs-map-beside-path_to_graph ValueError.

Synthetic tmp_path trees of empty .arg files (tests/graph_trees.py). The readers are replaced, on the
imported modules, by recorders, so no .arg file is opened. The subject folder with one labelling set
sorts first: a run without a pre-scan would read it before meeting the other one.
"""

import json
import logging

import numpy as np
import pytest

from cortical_tiles.brainvisa import compute_bounding_box as cbb
from cortical_tiles.brainvisa import compute_mask as cm
from cortical_tiles.brainvisa import subject_graphs
from tests.graph_trees import (
    LABEL,
    OTHER_LABEL,
    SIDE,
    classic_subject,
    import_generate_sulcal_regions,
    record_graph_files,
)

ONE_SET = "subj01"
TWO_SETS = "subj02"
SULCUS = "S.C."


def _tree(src):
    """Subject folder ONE_SET holds one labelling set, TWO_SETS holds LABEL and OTHER_LABEL."""
    classic_subject(src, subject=ONE_SET)
    classic_subject(src, subject=TWO_SETS, labels=(LABEL, OTHER_LABEL))


def _error_text(caplog, capsys):
    """The ERROR (and higher) log messages and the standard error output of the run, joined."""
    records = [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]
    return " ".join(records) + capsys.readouterr().err


def test_resolver_raises_several_labelling_sets_error(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-03ABA004D9B9: two labelling sets, no label -> SeveralLabellingSetsError (a ValueError).

    The error type is read from the module at run time, so the file imports before the type exists.
    """
    _tree(tmp_path)

    with pytest.raises(ValueError) as raised:
        subject_graphs.resolve_subject_graphs(tmp_path / TWO_SETS, subject_graphs.GraphSpec())

    assert isinstance(raised.value, subject_graphs.SeveralLabellingSetsError)
    assert LABEL in str(raised.value) and OTHER_LABEL in str(raised.value)


@pytest.fixture
def gsr(monkeypatch):
    return import_generate_sulcal_regions(monkeypatch)


@pytest.fixture
def increments(monkeypatch):
    """Graph files given to increment_one_mask (replaced by a recorder returning the mask unchanged)."""
    return record_graph_files(monkeypatch, cm, "increment_one_mask", lambda _graph, mask, *_rest: mask)


@pytest.fixture
def box_reads(monkeypatch):
    """Graph files given to get_one_bounding_box (replaced by a recorder returning a unit box)."""
    unit_box = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    return record_graph_files(monkeypatch, cbb, "get_one_bounding_box", lambda *_arguments: unit_box)


def _stop_code(main, argv, caplog):
    """Runs a command line main expected to stop; returns its exit status."""
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as stopped:
        main(argv)
    return stopped.value.code


def test_generate_sulcal_regions_exits_2_before_any_region_run(gsr, monkeypatch, tmp_path, caplog, capsys):
    """REQ-ARGDISC-BDRABCZUK-66C2B47AC856: no label, a subject folder with two sets -> exit 2 listing them, no region run."""
    region_runs = []
    monkeypatch.setattr(gsr, "run_with_params", lambda cfg: region_runs.append(cfg["region_name"]))
    _tree(tmp_path / "graphs")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    config = {
        "graphs_dir": str(tmp_path / "graphs"),
        "output_dir": str(tmp_path / "out"),
        "path_to_graph": None,
        "path_to_skeleton_with_hull": "hull",
        "skel_qc_path": "",
    }
    (dataset / "pipeline_loop_2mm.json").write_text(json.dumps(config))

    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as stopped:
        gsr.main(["-d", str(dataset), "--path_sk_with_hull", "H", "-i", SIDE, "-r", "S.Or.", "-y", "skeleton"])

    assert stopped.value.code == 2
    assert region_runs == []
    assert not (tmp_path / "out").exists()  # region runs are the only writers of the output folder
    errors = _error_text(caplog, capsys)
    assert LABEL in errors and OTHER_LABEL in errors


def test_compute_mask_exits_2_before_any_mask_increment(increments, tmp_path, caplog, capsys):
    """REQ-ARGDISC-BDRABCZUK-5ACE42CB7BF4: compute_mask command line, no label, a subject folder with two sets -> exit 2 listing them, no increment."""
    _tree(tmp_path / "src")

    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as stopped:
        cm.main(["-s", str(tmp_path / "src"), "-o", str(tmp_path / "masks"), "-u", SULCUS, "-i", SIDE])

    assert stopped.value.code == 2
    assert increments == []
    errors = _error_text(caplog, capsys)
    assert LABEL in errors and OTHER_LABEL in errors


def test_compute_bounding_box_exits_2_before_any_box_read(box_reads, tmp_path, caplog, capsys):
    """REQ-ARGDISC-BDRABCZUK-DEEE898F7317: compute_bounding_box command line, no label, a subject folder with two sets -> exit 2 listing them."""
    # "-n 2": the default "-n all" fails today in compute (number_subjects converted twice), a separate defect.
    _tree(tmp_path / "src")

    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as stopped:
        cbb.main(["-s", str(tmp_path / "src"), "-o", str(tmp_path / "bbox"), "-u", SULCUS, "-i", SIDE, "-n", "2"])

    assert stopped.value.code == 2
    assert box_reads == []
    errors = _error_text(caplog, capsys)
    assert LABEL in errors and OTHER_LABEL in errors


def test_compute_mask_exits_2_before_creating_the_side_folder(increments, tmp_path, caplog):
    """REQ-ARGDISC-BDRABCZUK-A7FDADCC7F0E: exit 2 and no <output>/<side> folder (the output folder holds the logs)."""
    _tree(tmp_path / "src")
    argv = ["-s", str(tmp_path / "src"), "-o", str(tmp_path / "masks"), "-u", SULCUS, "-i", SIDE]

    assert _stop_code(cm.main, argv, caplog) == 2
    assert not (tmp_path / "masks" / SIDE).exists()


def test_compute_bounding_box_exits_2_before_writing_general_info(box_reads, tmp_path, caplog):
    """REQ-ARGDISC-BDRABCZUK-E0A131863927: exit 2 and the bounding box JSON file, created empty with the box object, stays empty."""
    _tree(tmp_path / "src")
    argv = ["-s", str(tmp_path / "src"), "-o", str(tmp_path / "bbox"), "-u", SULCUS, "-i", SIDE, "-n", "2"]

    assert _stop_code(cbb.main, argv, caplog) == 2
    json_files = sorted((tmp_path / "bbox").rglob("*.json"))
    assert [json.loads(path.read_text()) for path in json_files] in ([], [{}])
