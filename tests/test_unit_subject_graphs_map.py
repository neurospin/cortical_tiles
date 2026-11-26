"""Graphs map file of the cortical_tiles graph resolver (REQ-ARGDISC-BDRABCZUK-BE031223681D).

A graphs map is a JSON file {subject folder name: {"L": [.arg paths], "R": [.arg paths]}}. It is read
once by ``load_graphs_map`` into ``GraphSpec.graphs``. Synthetic tmp_path trees only. ``load_graphs_map`` is read
from the module at run time, so the file imports before the function exists.
"""

import json

from cortical_tiles.brainvisa import subject_graphs
from cortical_tiles.brainvisa.subject_graphs import GraphSpec, resolve_subject_graphs


def _write_map(tmp_path, graphs):
    path = tmp_path / "graphs.json"
    path.write_text(json.dumps(graphs))
    return path


def test_load_graphs_map_reads_the_json_map(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-BE031223681D: the map read equals the JSON content, per subject and side."""
    graphs = {
        "sub-01": {"L": ["/data/a/Lsub-01.arg"], "R": ["/data/a/Rsub-01.arg"]},
        "sub-02": {"L": [], "R": ["/data/b/sub-02_hemi-R.arg"]},
    }

    assert subject_graphs.load_graphs_map(str(_write_map(tmp_path, graphs))) == graphs


def test_loaded_map_feeds_resolve_subject_graphs(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-BE031223681D: the loaded map gives the resolver the listed .arg files."""
    left = str(tmp_path / "elsewhere" / "Lsub-01_x.arg")
    right = str(tmp_path / "elsewhere" / "Rsub-01_x.arg")
    path = _write_map(tmp_path, {"sub-01": {"L": [left], "R": [right]}})

    spec = GraphSpec(graphs=subject_graphs.load_graphs_map(path))

    assert resolve_subject_graphs(tmp_path / "subjects" / "sub-01", spec) == {"L": [left], "R": [right]}
