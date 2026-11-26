"""Synthetic subject trees and shared helpers for the graph consumers.

Shared by the tests of TASK-ARGDISC-BDRABCZUK-7E631CF8D649 (generate_foldlabels, generate_ICBM2009c_transforms,
generate_skeletons) and TASK-ARGDISC-BDRABCZUK-817BCB001B96 (generate_sulcal_regions, compute_mask,
compute_bounding_box), which read each subject's graph through the cortical_tiles graph resolver. The trees hold
empty .arg files with synthetic names only:

* classic BrainVISA: <subject>/t1mri/BL/default_analysis/folds/3.1/<side><subject>.arg (unlabelled) and
  <same folder>/<labelling set>/<side><subject>_<labelling set>.arg (labelled);
* morphologist-cli: <folder>/<folds>/<stem>_hemi-<side>.arg (unlabelled) and
  <folds>/sul-0_auto/<stem>_sul-0_hemi-<side>_auto.arg (labelled).

The helpers below the trees replace a script's graph reader or region run by a recorder, so no .arg file is
ever opened.
"""

import os
import re

SIDE = "L"
SUBJECT = "subj01"
CLASSIC_FOLDS = "t1mri/BL/default_analysis/folds/3.1"
LABEL = "default_session_auto"
OTHER_LABEL = "long_session_auto"  # noqa: V107 - used by the consumer test files
CLI_SUBJECT = "sub-01"
CLI_FOLDS = "ses-1/run-0/ana-0/folds/3.1"
CLI_LABEL = "sul-0_auto"
UNLABELLED = "unlabelled"

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAINVISA_DIR = os.path.join(REPO_DIR, "cortical_tiles", "brainvisa")


def touch(root, relative):
    """Creates the empty file root/relative and its folders; returns its path as a string."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return str(path)


def classic_subject(src, subject=SUBJECT, labels=(LABEL,), side=SIDE):  # noqa: V103 - used by the consumer test files
    """Creates the classic subject folder src/<subject>.

    It holds the unlabelled graph of side in CLASSIC_FOLDS and one labelled graph per labelling set of
    labels. Returns {UNLABELLED: path, <labelling set>: path, ...}.
    """
    folder = src / subject
    graphs = {UNLABELLED: touch(folder, f"{CLASSIC_FOLDS}/{side}{subject}.arg")}
    for label in labels:
        graphs[label] = touch(folder, f"{CLASSIC_FOLDS}/{label}/{side}{subject}_{label}.arg")
    return graphs


def cli_subject(src, folder=CLI_SUBJECT, folds=CLI_FOLDS, side=SIDE):  # noqa: V103 - used by the consumer test files
    """Creates the morphologist-cli subject folder src/<folder>.

    It holds the unlabelled hemi-<side> graph in folds and the labelled one in folds/sul-0_auto.
    Returns {UNLABELLED: path, CLI_LABEL: path}.
    """
    root = src / folder
    stem = f"{folder}_run-0"
    return {
        UNLABELLED: touch(root, f"{folds}/{stem}_hemi-{side}.arg"),
        CLI_LABEL: touch(root, f"{folds}/{CLI_LABEL}/{stem}_sul-0_hemi-{side}_auto.arg"),
    }


# --- layouts of one left graph read in graph discovery or through a path_to_graph --------------------------------


def classic_session_graph(src):  # noqa: V103 - used by the compute_mask and compute_bounding_box test files
    """Classic t1mri layout of src/sujet01: unlabelled L, labelled L and R; returns the labelled left graph."""
    folds = "sujet01/t1mri/default_acquisition/default_analysis/folds/3.1"
    touch(src, f"{folds}/Lsujet01.arg")
    touch(src, f"{folds}/{LABEL}/Rsujet01_{LABEL}.arg")
    return touch(src, f"{folds}/{LABEL}/Lsujet01_{LABEL}.arg")


def cli_session_graph(src):  # noqa: V103 - used by the compute_mask and compute_bounding_box test files
    """Morphologist-cli layout of src/sub-01: unlabelled L, labelled L and R; returns the labelled left graph."""
    folds = "sub-01/ses-01/morphologist/folds/3.1"
    touch(src, f"{folds}/sub-01_ses-01_hemi-L.arg")
    touch(src, f"{folds}/{LABEL}/sub-01_ses-01_hemi-R_{LABEL}.arg")
    return touch(src, f"{folds}/{LABEL}/sub-01_ses-01_hemi-L_{LABEL}.arg")


def subject_named_glob_graph(src):  # noqa: V103 - used by the compute_mask and compute_bounding_box test files
    """src/sujet01/graphs holds La_other.arg and Lsujet01_x.arg; returns the second, the one named after the subject."""
    touch(src, "sujet01/graphs/La_other.arg")
    return touch(src, "sujet01/graphs/Lsujet01_x.arg")


# --- recorders ---------------------------------------------------------------------------------------------------


def record_graph_files(monkeypatch, module, reader, answer):  # noqa: V103 - used by the consumer test files
    """Replaces module.<reader> by a recorder of its first argument (the graph file); returns the record list.

    The recorder returns answer(*arguments), so the caller keeps the reader's return shape.
    """
    recorded = []

    def _record(graph_filename, *arguments):
        recorded.append(graph_filename)
        return answer(graph_filename, *arguments)

    monkeypatch.setattr(module, reader, _record)
    return recorded


def import_generate_sulcal_regions(monkeypatch):  # noqa: V103 - used by the generate_sulcal_regions test files
    """generate_sulcal_regions imports its siblings as top-level modules: brainvisa folder on sys.path first."""
    monkeypatch.syspath_prepend(BRAINVISA_DIR)
    import generate_sulcal_regions

    return generate_sulcal_regions


def record_region_runs(module, monkeypatch):  # noqa: V103 - used by the generate_sulcal_regions test files
    """Replaces module.run_with_params by a recorder of a copy of each region run configuration; returns the list."""
    configurations = []
    monkeypatch.setattr(module, "run_with_params", lambda cfg: configurations.append(dict(cfg)))
    return configurations


def option_help(module, option, capsys, monkeypatch):  # noqa: V103 - used by the consumer test files
    """The help entry of one option (for example -p) of module's command line, whitespace collapsed."""
    monkeypatch.setenv("COLUMNS", "1000")  # argparse breaks long words at the terminal width
    capsys.readouterr()
    try:
        module.main(["--help"])
    except SystemExit:
        pass
    text = " ".join(capsys.readouterr().out.split())
    entries = re.split(r" (?=-\w(?: [A-Z_]+)?, --)", text)  # an entry starts at "-x NAME, --" or "-x, --"
    return next(entry for entry in entries if entry.startswith(f"{option} "))
