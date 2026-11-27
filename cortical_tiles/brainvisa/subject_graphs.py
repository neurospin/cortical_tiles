"""Per-subject graph resolver: the left and right .arg files of one subject folder.

``resolve_subject_graphs`` is the one place that turns a subject folder and a
graph specification into the subject's .arg files, per side. Its pure core,
``select_graphs``, chooses among file names only: no file is opened.
"""

import glob
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import NamedTuple

from cortical_tiles.brainvisa.utils.skip_list import subject_form
from cortical_tiles.config.logs import set_file_logger

__all__ = [
    "EXIT_SEVERAL_LABELLING_SETS",
    "GraphSelection",
    "GraphSpec",
    "SeveralLabellingSetsError",
    "check_spec_modes",
    "labelling_sets",
    "load_graphs_map",
    "prescan_labelling_sets",
    "resolve_subject_graphs",
    "select_graphs",
    "stop_on_several_labelling_sets",
]

log = set_file_logger(__file__)

SIDES = ("L", "R")
LABELLED_SUFFIXES = ("_auto", "_manual")
GRAPH_EXTENSION = ".arg"
HEMI_ENTITY = re.compile(r"(?:^|_)hemi-([LR])(?:_|\.arg$)")
DIGIT_RUNS = re.compile(r"(\d+)")
# Process exit status of a command line stopped by several labelling sets
EXIT_SEVERAL_LABELLING_SETS = 2


class SeveralLabellingSetsError(ValueError):
    """Graph discovery without a label met subject folders holding several labelling sets.

    sets_by_folder: folder path -> sorted labelling set names, folders in the order given.
    """

    def __init__(self, sets_by_folder):
        self.sets_by_folder = {folder: tuple(sets) for folder, sets in sets_by_folder.items()}
        found = "; ".join(
            f"{folder}: several labelling sets {list(sets)}" for folder, sets in self.sets_by_folder.items()
        )
        super().__init__(f"{found}; name one with a graph label (--graph_label)")


@dataclass(frozen=True)
class GraphSpec:
    """How the graphs of a subject are found.

    graphs: per subject folder name or scan key, per side, the .arg files the caller lists.
    path_to_graph: the legacy relative path of today's glob.
    label: the labelling set chosen when graph discovery finds several.
    path_filter: graph discovery keeps a candidate only when it returns True for the
        candidate's path relative to the subject folder; None keeps every candidate.
    """

    graphs: Mapping[str, Mapping[str, Sequence[str]]] | None = None
    path_to_graph: str | None = None
    label: str | None = None
    path_filter: Callable[[str], bool] | None = None


class GraphSelection(NamedTuple):
    """What graph discovery chose for one subject folder.

    graphs: per side ("L", "R"), the kept .arg file, or an empty list.
    left_out: (side, kept path, left-out path) per candidate not kept.
    unlabelled_sides: the sides whose kept .arg file is unlabelled.
    """

    graphs: dict[str, list[str]]
    left_out: tuple[tuple[str, str, str], ...]
    unlabelled_sides: tuple[str, ...]


def check_spec_modes(graphs, path_to_graph):
    """Raises ValueError when a graphs map and a path_to_graph are both given."""
    if not (graphs is None or path_to_graph is None):
        raise ValueError("graphs map and path_to_graph are exclusive")


def _graphs_map_problem(graphs):
    """Returns what keeps graphs from being {subject: {side: [str, ...]}}, else None."""
    if not isinstance(graphs, dict):
        return "the top level is not a JSON object"
    for subject, sides in graphs.items():
        if not isinstance(sides, dict):
            return f"the value of {subject} is not a JSON object"
        for side, paths in sides.items():
            if not (isinstance(paths, list) and all(isinstance(p, str) for p in paths)):
                return f"{subject} side {side} is not a list of paths"
    return None


def load_graphs_map(path):
    """Returns the graphs map of a JSON file: subject folder name -> side -> .arg paths.

    Raises OSError when the file cannot be read, ValueError when it is not such a map.
    """
    with open(path, encoding="utf-8") as stream:
        try:
            graphs = json.load(stream)
        except json.JSONDecodeError as error:
            raise ValueError(f"graphs map {path}: {error}") from error
    problem = _graphs_map_problem(graphs)
    if problem is not None:
        raise ValueError(f"graphs map {path}: {problem}")
    assert isinstance(graphs, dict), "load_graphs_map post: isinstance(result, dict)"
    return graphs


def stop_on_several_labelling_sets(main):
    """Decorator of a command-line main: several labelling sets end it with exit status 2.

    The ERROR record lists each subject folder and its labelling sets; any other
    exception passes through unchanged.
    """

    @wraps(main)
    def wrapper(*args, **kwargs):
        try:
            return main(*args, **kwargs)
        except SeveralLabellingSetsError as error:
            log.error("%s", error)
            sys.exit(EXIT_SEVERAL_LABELLING_SETS)

    return wrapper


def _side(path):
    """Returns the side of a .arg file: its hemi- entity, else its first letter."""
    name = os.path.basename(path)
    match = HEMI_ENTITY.search(name)
    if match:
        return match.group(1)
    return name[0] if name[:1] in SIDES else None


def _labelling_set(path):
    """Returns the folder of a .arg file when its name marks it labelled, else None."""
    folder = os.path.basename(os.path.dirname(path))
    return folder if folder.endswith(LABELLED_SUFFIXES) else None


def _natural_key(path):
    """Sort key: digit runs compare as numbers; the plain path breaks ties."""
    parts = DIGIT_RUNS.split(path)
    return tuple(int(p) if i % 2 else p for i, p in enumerate(parts)), path


def labelling_sets(arg_paths):
    """Returns the sorted distinct labelling sets of the .arg files that have a side."""
    found = {_labelling_set(path) for path in arg_paths if _side(path) is not None}
    result = sorted(labelling for labelling in found if labelling is not None)
    assert result == sorted(result), "labelling_sets post: result == sorted(result)"
    return result


def _select_side(side, entries, chosen):
    """Returns ([kept] or [], left-out entries, kept file unlabelled) for one side.

    entries: (path, labelling set) of the side's candidates.
    """
    pool = [path for path, labelling in entries if labelling and labelling == chosen]
    unlabelled = not pool
    if unlabelled:
        pool = [path for path, labelling in entries if labelling is None]
    if not pool:
        return [], (), False
    kept, *others = sorted(pool, key=_natural_key)
    return [kept], tuple((side, kept, path) for path in others), unlabelled


def select_graphs(subject_dir, arg_paths, label):
    """Chooses the .arg file of each side among arg_paths (no I/O, no logging).

    Returns a GraphSelection. Raises SeveralLabellingSetsError (a ValueError)
    when label is None and the candidates hold two or more labelling sets.
    """
    entries = {side: [] for side in SIDES}
    for path in arg_paths:
        side = _side(path)
        if side is not None:
            entries[side].append((path, _labelling_set(path)))
    sets = labelling_sets(arg_paths)
    if label is None and len(sets) >= 2:
        raise SeveralLabellingSetsError({subject_dir: sets})
    chosen = label if label is not None else next(iter(sets), None)
    graphs, left_out, unlabelled_sides = {}, [], []
    for side in SIDES:
        graphs[side], dropped, unlabelled = _select_side(side, entries[side], chosen)
        left_out.extend(dropped)
        if unlabelled:
            unlabelled_sides.append(side)
    return GraphSelection(graphs, tuple(left_out), tuple(unlabelled_sides))


def _walk_arg_files(subject_dir):
    """Returns every .arg file below subject_dir, walked depth-first in sorted order.

    A folder's own .arg files come before those of its subfolders; folder
    symlinks are not followed; an unreadable folder is skipped.
    """
    try:
        names = sorted(os.listdir(subject_dir))
    except OSError:
        return []
    found, subdirs = [], []
    for name in names:
        path = os.path.join(subject_dir, name)
        if os.path.isdir(path):
            if not os.path.islink(path):
                subdirs.append(path)
        elif name.endswith(GRAPH_EXTENSION):
            found.append(path)
    for path in subdirs:
        found.extend(_walk_arg_files(path))
    return found


def _discovery_candidates(root, path_filter):
    """Returns the .arg files below root that path_filter accepts (relative path), walk order kept."""
    candidates = _walk_arg_files(root)
    if path_filter is None:
        return candidates
    return [path for path in candidates if path_filter(os.path.relpath(path, root))]


def prescan_labelling_sets(subject_dirs, spec):
    """Raises SeveralLabellingSetsError naming every folder of subject_dirs with several sets.

    Graph discovery without a label only: a graphs map, a path_to_graph or a
    label makes it a no-op. Every folder is walked before the error is raised.
    """
    if spec.graphs is not None or spec.path_to_graph is not None or spec.label is not None:
        return
    collected = {}
    for folder in subject_dirs:
        root = os.fspath(folder)
        sets = labelling_sets(_discovery_candidates(root, spec.path_filter))
        if len(sets) >= 2:
            collected[root] = sets
    if collected:
        raise SeveralLabellingSetsError(collected)


def _keys_of_folder(graphs, folder):
    """Returns, sorted, the graphs map keys of folder: the folder name or a scan key of that subject."""
    return sorted(key for key in graphs if key == folder or subject_form(key) == folder)


def _resolve_listed(subject_dir, graphs):
    """Returns per side the .arg files the graphs map lists for subject_dir, entries in sorted key order."""
    keys = _keys_of_folder(graphs, os.path.basename(os.path.normpath(subject_dir)))
    return {side: [os.fspath(path) for key in keys for path in graphs[key].get(side, [])] for side in SIDES}


def _resolve_glob(subject_dir, path_to_graph):
    """Returns per side the sorted matches of today's glob <side>*.arg."""
    folder = os.path.join(os.fspath(subject_dir), path_to_graph)
    result = {}
    for side in SIDES:
        result[side] = sorted(glob.glob(f"{folder}/{side}*{GRAPH_EXTENSION}"))
    return result


def _log_selection(subject_dir, selection):
    """Logs a WARNING per side resolved to an unlabelled graph, an INFO per left-out."""
    for side in selection.unlabelled_sides:
        log.warning("%s: side %s resolved to an unlabelled graph", subject_dir, side)
    for side, kept, other in selection.left_out:
        log.info("%s: side %s keeps %s, leaves out %s", subject_dir, side, kept, other)


def resolve_subject_graphs(subject_dir, spec):
    """Returns {"L": [...], "R": [...]}: the .arg files of subject_dir per side.

    Mode order: the graphs map, else the legacy path_to_graph glob, else graph
    discovery below subject_dir. Raises ValueError for a graphs map beside a
    path_to_graph, or for several labelling sets found without a label.
    """
    check_spec_modes(spec.graphs, spec.path_to_graph)
    if spec.graphs is not None:
        result = _resolve_listed(subject_dir, spec.graphs)
    elif spec.path_to_graph is not None:
        result = _resolve_glob(subject_dir, spec.path_to_graph)
    else:
        root = os.fspath(subject_dir)
        selection = select_graphs(root, _discovery_candidates(root, spec.path_filter), spec.label)
        _log_selection(root, selection)
        result = selection.graphs
    assert len(result) == 2, "resolve_subject_graphs post: len(result) == 2"
    assert "L" in result, 'resolve_subject_graphs post: "L" in result'
    assert "R" in result, 'resolve_subject_graphs post: "R" in result'
    return result
