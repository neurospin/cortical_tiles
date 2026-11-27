"""Scan key of a morphologist-cli scan: subject plus ses, acq, run and ana (TASK-ARGDISC-BDRABCZUK-09582A3290C3).

User decision 2026-11-27: every sub/ses/run/ana folder of a morphologist-cli output is its own scan, with its own
crops. So the session key of a graph file ends with its ana- entity, the subject form of a key strips that part, and
the graphs map accepts one entry per scan, keyed by the scan key: one subject folder can then hold several graphs per
side. Synthetic tmp_path trees and literal paths only; no .arg file is ever opened.
"""

from pathlib import Path

from cortical_tiles.brainvisa import generate_skeletons as gs
from cortical_tiles.brainvisa.subject_graphs import GraphSpec, resolve_subject_graphs
from cortical_tiles.brainvisa.utils.skip_list import covered_by, session_key, subject_form
from tests.graph_trees import CLI_LABEL, CLI_SUBJECT, cli_subject

SIDE = "L"
SESSION = "ses-1/run-0"
ANALYSES = ("ana-0", "ana-1")
SCAN_KEYS = tuple(f"{CLI_SUBJECT}_ses-1_run-0_{analysis}" for analysis in ANALYSES)
CLI_GRAPH = f"/data/{CLI_SUBJECT}/{SESSION}/ana-0/folds/3.1/{CLI_LABEL}/{CLI_SUBJECT}_run-0_sul-0_hemi-L_auto.arg"


# --- REQ-ARGDISC-BDRABCZUK-7F41241203DD: the session key ends with the ana- entity ---------------------------------


def test_session_key_ends_with_the_ana_entity():
    """REQ-ARGDISC-BDRABCZUK-7F41241203DD: a path holding an ana- folder gives a key ending with it, prefix kept."""
    assert session_key("sub-01", "/data/sub-01/ana-default/folds/Lsub-01.arg", True) == "sub-01_ana-default"


def test_session_key_puts_the_ana_entity_after_ses_acq_and_run():
    """REQ-ARGDISC-BDRABCZUK-7F41241203DD: the ana- part comes last, after the ses-, acq- and run- parts."""
    graph = "/data/sub-1/ses-1_acq-a/run-2/ana-b/folds/Lsub-1.arg"

    assert session_key("sub-1", graph, True) == "sub-1_ses-1_acq-a_run-2_ana-b"


def test_session_keys_of_two_analyses_of_one_session_differ():
    """REQ-ARGDISC-BDRABCZUK-7F41241203DD: the morphologist-cli graphs of ana-0 and ana-1 of one session get two keys."""
    other = CLI_GRAPH.replace("/ana-0/", "/ana-1/")

    keys = (session_key(CLI_SUBJECT, CLI_GRAPH, True), session_key(CLI_SUBJECT, other, True))

    assert keys == SCAN_KEYS


# --- REQ-ARGDISC-BDRABCZUK-319245776240: the subject form strips a trailing ana- part -------------------------------


def test_subject_form_strips_a_trailing_ana_part():
    """REQ-ARGDISC-BDRABCZUK-319245776240: the ana- part goes with the ses-, acq- and run- parts."""
    assert subject_form("sub-01_ses-1_acq-a_run-0_ana-0") == "sub-01"
    assert subject_form("sub-01_ana-default") == "sub-01"


def test_subject_form_covers_a_scan_key_with_an_ana_part():
    """REQ-ARGDISC-BDRABCZUK-319245776240: an entry naming the subject covers the scan keys of its analyses."""
    assert covered_by(SCAN_KEYS[1], {CLI_SUBJECT})


# --- REQ-ARGDISC-BDRABCZUK-2B4CDEFB0D38, REQ-ARGDISC-BDRABCZUK-D176D869F4DA: graphs map entries keyed by scan ---------


def test_resolver_returns_the_graphs_of_a_scan_entry(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-2B4CDEFB0D38: an entry keyed by a scan key of the subject feeds that subject folder."""
    graphs = {SCAN_KEYS[0]: {"L": ["/data/a/L.arg"], "R": ["/data/a/R.arg"]}}

    result = resolve_subject_graphs(tmp_path / CLI_SUBJECT, GraphSpec(graphs=graphs))

    assert result == {"L": ["/data/a/L.arg"], "R": ["/data/a/R.arg"]}


def test_resolver_leaves_out_the_scan_entries_of_another_subject(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-2B4CDEFB0D38: sub-010 shares the first characters of sub-01, not its subject form."""
    graphs = {
        "sub-010_ses-1_run-0_ana-0": {"L": ["/data/other/L.arg"]},
        SCAN_KEYS[0]: {"L": ["/data/a/L.arg"]},
    }

    result = resolve_subject_graphs(tmp_path / CLI_SUBJECT, GraphSpec(graphs=graphs))

    assert result == {"L": ["/data/a/L.arg"], "R": []}


def test_resolver_returns_several_graphs_per_side_in_sorted_key_order(tmp_path):
    """REQ-ARGDISC-BDRABCZUK-D176D869F4DA: the entries of one subject join per side in sorted key order, not map order."""
    graphs = {
        SCAN_KEYS[1]: {"L": ["/data/ana-1/L.arg"]},
        CLI_SUBJECT: {"L": ["/data/plain/L.arg"]},
        SCAN_KEYS[0]: {"L": ["/data/ana-0/L.arg"], "R": ["/data/ana-0/R.arg"]},
    }

    result = resolve_subject_graphs(tmp_path / CLI_SUBJECT, GraphSpec(graphs=graphs))

    assert result == {
        "L": ["/data/plain/L.arg", "/data/ana-0/L.arg", "/data/ana-1/L.arg"],
        "R": ["/data/ana-0/R.arg"],
    }


# --- REQ-ARGDISC-BDRABCZUK-AAD9ED88106C: each analysis of one subject gets its own outputs ------------------------


def test_skeletons_of_two_analyses_of_one_session_are_written_apart(tmp_path, monkeypatch):
    """REQ-ARGDISC-BDRABCZUK-AAD9ED88106C: a graphs map of two scans of sub-01 gives two skeleton files, one per scan.

    The crop steps take their keys from the skeleton file names, so two skeleton keys are two crops.
    """
    src = tmp_path / "graphs"
    graphs = {}
    for key, analysis in zip(SCAN_KEYS, ANALYSES):
        graphs[key] = {SIDE: [cli_subject(src, folds=f"{SESSION}/{analysis}/folds/3.1")[CLI_LABEL]]}
    out = tmp_path / "skeletons"
    monkeypatch.setattr(gs, "generate_skeleton_from_graph_file", lambda graph, output, _: Path(output).write_text(""))
    step = gs.GraphConvert2Skeleton(
        src_dir=str(src),
        skeleton_dir=str(out),
        side=SIDE,
        junction="thin",
        parallel=False,
        path_to_graph=None,
        bids=True,
        qc_path="",
        njobs=1,
        skip_dir=str(tmp_path / "skip_lists"),
        graphs=graphs,
    )

    step.compute(nb_subjects=-1)

    written = sorted(path.name for path in (out / SIDE).iterdir())
    assert written == [f"{SIDE}skeleton_generated_{key}.nii.gz" for key in SCAN_KEYS]
