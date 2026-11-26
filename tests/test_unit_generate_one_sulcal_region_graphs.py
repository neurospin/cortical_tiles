"""Graph inputs a region run hands to its graph-reading steps (TASK-ARGDISC-BDRABCZUK-817BCB001B96).

A region run receives in its configuration the keys ``graphs`` (graphs map) and ``graph_label`` from
generate_sulcal_regions. run_with_params passes them, as the keywords ``graphs`` and ``graph_label``,
to the graph-reading steps it calls: generate_skeletons, generate_foldlabels and
generate_ICBM2009c_transforms (keyword names of COMP-ARGDISC-BDRABCZUK-C28FF0F14F70, user answer
2026-11-25). Every step function is replaced, on the imported module, by a recorder: the ``calls`` fixture of
test_unit_generate_one_sulcal_region.py, reused here; the real signatures must accept what run_with_params passes.
"""

import inspect
import json

import pytest

from cortical_tiles.brainvisa import generate_one_sulcal_region as gosr
from tests import test_unit_generate_one_sulcal_region as region_tests
from tests.graph_trees import LABEL

REGION = "S.Or."

GRAPH_STEPS = ("generate_skeletons", "generate_foldlabels", "generate_ICBM2009c_transforms")

# The real graph-reading steps, kept before the `calls` fixture replaces them, for their signatures.
REAL_STEPS = {name: getattr(gosr, name) for name in GRAPH_STEPS}

# (input_type, graph-reading step a 2 mm region run of that input type calls)
RUNS = [
    ("skeleton", "generate_skeletons"),
    ("skeleton", "generate_ICBM2009c_transforms"),
    ("foldlabel", "generate_foldlabels"),
    ("foldlabel", "generate_ICBM2009c_transforms"),
]

GRAPHS = {"sub-01": {"L": ["/data/a/Lsub-01.arg"], "R": ["/data/a/Rsub-01.arg"]}}


@pytest.fixture
def regions_json(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text(json.dumps({"brain": {"S.Or._right": {"S.Or._right": 1}, "S.Or._left": {"S.Or._left": 1}}}))
    return str(path)


# The `calls` fixture of the region run tests (every step function replaced by a recorder of (name, kwargs)).
calls = region_tests.calls


def _params(tmp_path, regions_json, input_type):
    return {
        "save_behavior": "best",
        "side": "R",
        "out_voxel_size": 2.0,
        "region_name": REGION,
        "brain_regions_json": regions_json,
        "parallel": False,
        "nb_subjects": -1,
        "input_type": input_type,
        "labeled_subjects_dir": "labeled",
        "path_to_graph_supervised": "pgs",
        "supervised_output_dir": str(tmp_path / "supervised"),
        "nb_subjects_mask": -1,
        "graphs_dir": "graphs",
        "path_to_graph": None,
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
        "graphs": GRAPHS,
        "graph_label": LABEL,
    }


def _step_kwargs(calls, step):
    (kwargs,) = [kwargs for called, kwargs in calls if called == step]
    return kwargs


@pytest.mark.parametrize(("input_type", "step"), RUNS)
def test_region_run_passes_graph_label_to_step(tmp_path, regions_json, calls, input_type, step):
    """REQ-ARGDISC-BDRABCZUK-37BD85482FB8: the region configuration's graph_label reaches the step as its graph_label keyword."""
    gosr.run_with_params(_params(tmp_path, regions_json, input_type))

    kwargs = _step_kwargs(calls, step)
    assert kwargs.get("graph_label") == LABEL
    inspect.signature(REAL_STEPS[step]).bind(**kwargs)


@pytest.mark.parametrize(("input_type", "step"), RUNS)
def test_region_run_passes_graphs_map_to_step(tmp_path, regions_json, calls, input_type, step):
    """REQ-ARGDISC-BDRABCZUK-60C94B82EB05: the region configuration's graphs map reaches the step as its graphs keyword."""
    gosr.run_with_params(_params(tmp_path, regions_json, input_type))

    kwargs = _step_kwargs(calls, step)
    assert kwargs.get("graphs") == GRAPHS
    inspect.signature(REAL_STEPS[step]).bind(**kwargs)
