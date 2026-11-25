"""Per-subject graph resolver: the left and right .arg files of one subject folder.

``resolve_subject_graphs`` is the one place that turns a subject folder and a
graph specification into the subject's .arg files, per side. Its pure core,
``select_graphs``, chooses among file names only: no file is opened.
"""

import glob
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from cortical_tiles.config.logs import set_file_logger

__all__ = [
    "GraphSelection",
    "GraphSpec",
    "check_spec_modes",
    "resolve_subject_graphs",
    "select_graphs",
]

log = set_file_logger(__file__)

SIDES = ("L", "R")
LABELLED_SUFFIXES = ("_auto", "_manual")
GRAPH_EXTENSION = ".arg"
HEMI_ENTITY = re.compile(r"(?:^|_)hemi-([LR])(?:_|\.arg$)")
DIGIT_RUNS = re.compile(r"(\d+)")


@dataclass(frozen=True)
class GraphSpec:
    """How the graphs of a subject are found.

    graphs: per subject folder name, per side, the .arg files the caller lists.
    path_to_graph: the legacy relative path of today's glob.
    label: the labelling set chosen when graph discovery finds several.
    """

    graphs: Mapping[str, Mapping[str, Sequence[str]]] | None = None
    path_to_graph: str | None = None
    label: str | None = None


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

    Returns a GraphSelection. Raises ValueError when label is None and the
    candidates hold two or more labelling sets.
    """
    entries = {side: [] for side in SIDES}
    for path in arg_paths:
        side = _side(path)
        if side is not None:
            entries[side].append((path, _labelling_set(path)))
    found = {labelling for pairs in entries.values() for _, labelling in pairs}
    sets = sorted(labelling for labelling in found if labelling is not None)
    if label is None and len(sets) >= 2:
        raise ValueError(f"{subject_dir}: several labelling sets {sets}; name one")
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


def _resolve_listed(subject_dir, graphs):
    """Returns per side the .arg files the graphs map lists for subject_dir."""
    listed = graphs.get(os.path.basename(os.path.normpath(subject_dir)), {})
    return {side: [os.fspath(path) for path in listed.get(side, [])] for side in SIDES}


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
        selection = select_graphs(root, _walk_arg_files(root), spec.label)
        _log_selection(root, selection)
        result = selection.graphs
    assert len(result) == 2, "resolve_subject_graphs post: len(result) == 2"
    assert "L" in result, 'resolve_subject_graphs post: "L" in result'
    assert "R" in result, 'resolve_subject_graphs post: "R" in result'
    return result
