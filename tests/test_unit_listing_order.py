"""Listing order of the graph and subject listings of cortical_tiles (TASK-READORDER-BDRABCZUK-D018CA8BBF02).

User rule: files must always be read in the same order, for reproducibility. A real directory
gives one arbitrary order, so these tests replace what os.scandir and os.listdir yield, below
glob and pathlib, and stay valid whichever listing call the code uses. Every test first checks,
inside the controlled listing, that the order it set really reaches the listing call: without
that check a broken control would pass on any code.

* REQ-READORDER-BDRABCZUK-BB45F549A2F7 - increment_mask reads the sorted-first graph.
* REQ-READORDER-BDRABCZUK-1EB646CB0E8D - get_all_subjects_as_dictionary lists subjects in sorted order.
* REQ-READORDER-BDRABCZUK-56B9F6AFBCDE - get_bounding_boxes reads the sorted-first graph.
* REQ-READORDER-BDRABCZUK-E0E242574B81 - BoundingBoxMax.compute accepts two or more subjects.
* REQ-READORDER-BDRABCZUK-E57EA5AA3DBC - generate_one_skeleton reads the sorted-first graph.
* REQ-READORDER-BDRABCZUK-8AE4FE5A2FB6 - generate_one_foldlabel reads the sorted-first graph.
* REQ-READORDER-BDRABCZUK-09F50ED77134 - generate_one_transform reads the sorted-first graph.
* REQ-READORDER-BDRABCZUK-4956612FDCF0 - quality_checks refuses an ambiguous source folder.
"""

import contextlib
import glob
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from cortical_tiles.brainvisa import compute_bounding_box as cbb
from cortical_tiles.brainvisa import compute_mask as cm
from cortical_tiles.brainvisa import generate_foldlabels as gf
from cortical_tiles.brainvisa import generate_ICBM2009c_transforms as gt
from cortical_tiles.brainvisa import generate_skeletons as gs
from cortical_tiles.brainvisa.utils import quality_checks as qc
from cortical_tiles.brainvisa.utils.bbox import compute_max
from cortical_tiles.brainvisa.utils.subjects import get_all_subjects_as_dictionary
from soma import aims

SIDE = "R"
SUBJECT = "sub01"
SULCUS = "S.C."
SULCUS_ON_SIDE = "S.C._right"
PATH_TO_GRAPH = "t1mri/default/folds"
GRAPH_FILE = f"%(subject)s/{PATH_TO_GRAPH}/%(side)s%(subject)s*.arg"
OUT_VS = (2.0, 2.0, 2.0)

# One voxel per graph of SUBJECT, far enough apart to fall in three different mask voxels.
GRAPH_VOXELS = {
    f"{SIDE}{SUBJECT}_a.arg": (10, 10, 10),
    f"{SIDE}{SUBJECT}_b.arg": (20, 20, 20),
    f"{SIDE}{SUBJECT}_c.arg": (30, 30, 30),
}
# "sub-10" before "sub-2": Python sorted() order, not numeric order.
SUBJECTS = ["sub-1", "sub-10", "sub-2"]


def _descending(names):
    return sorted(names, reverse=True)


def _rotated(names):
    """Sorted names with the first one moved to the end: neither ascending nor descending."""
    ordered = sorted(names)
    return ordered[1:] + ordered[:1]


unsorted_orders = pytest.mark.parametrize("order", [_descending, _rotated], ids=["descending", "rotated"])


@contextlib.contextmanager
def _listing_in(order):
    """Make os.scandir and os.listdir yield their entries in the given order of names."""
    real_scandir = os.scandir
    real_listdir = os.listdir

    @contextlib.contextmanager
    def scandir(path="."):
        with real_scandir(path) as entries:
            by_name = {entry.name: entry for entry in entries}
        yield iter([by_name[name] for name in order(list(by_name))])

    def listdir(path="."):
        return order(real_listdir(path))

    with patch("os.scandir", scandir), patch("os.listdir", listdir):
        yield


def _graph(voxel):
    """In-memory CorticalFoldArg graph holding the sulcus on one voxel."""
    bucket = aims.BucketMap_VOID()
    bucket.setSizeXYZT(1.0, 1.0, 1.0, 1.0)
    bucket[0][aims.Point3d(*voxel)] = 0
    graph = aims.Graph("CorticalFoldArg")
    graph["voxel_size"] = [1.0, 1.0, 1.0, 1.0]
    graph["boundingbox_min"] = [0, 0, 0]
    graph["boundingbox_max"] = [40, 40, 40]
    sulcus = graph.addVertex("fold")
    sulcus["name"] = SULCUS_ON_SIDE
    sulcus["aims_ss"] = aims.rc_ptr_BucketMap_VOID(bucket)
    return graph


def _mask_index(graph, voxel):
    """Mask index of a 1 mm graph voxel: ICBM2009c position / output voxel size, rounded."""
    transform = aims.GraphManip.getICBM2009cTemplateTransform(graph)
    position = np.asarray(transform.transform(np.asarray(voxel, dtype=float)))
    return tuple(int(i) for i in np.round(position / np.asarray(OUT_VS)).astype(int))


class _AimsProxy:
    """soma.aims with read answering the in-memory graphs of the placeholder .arg files."""

    def __init__(self, graphs):
        self._graphs = graphs

    def read(self, filename, *args, **kwargs):
        if filename in self._graphs:
            return self._graphs[filename]
        return aims.read(filename, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(aims, name)


@pytest.fixture
def graph_dir(tmp_path):
    """Folder of SUBJECT holding one empty .arg placeholder per name of GRAPH_VOXELS."""
    folder = tmp_path / "src" / SUBJECT / PATH_TO_GRAPH
    folder.mkdir(parents=True)
    for name in GRAPH_VOXELS:
        (folder / name).write_text("")
    return folder


def _assert_control_reaches_glob(folder, order):
    """The controlled order must show in glob, or a pass below proves nothing."""
    listed = [os.path.basename(path) for path in glob.glob(str(folder / "*.arg"))]
    assert listed == order(list(GRAPH_VOXELS))


def _subject_dict(tmp_path):
    """SUBJECT as get_all_subjects_as_dictionary gives it."""
    return {
        "subject": SUBJECT,
        "side": SIDE,
        "dir": str(tmp_path / "src"),
        "graph_file": GRAPH_FILE % {"side": SIDE, "subject": SUBJECT},
    }


class TestIncrementMaskOrder:
    """compute_mask.py increment_mask: list_graph_file[0] of a glob picks the graph behind the mask."""

    @unsorted_orders
    def test_sorted_first_graph_is_read(self, tmp_path, graph_dir, monkeypatch, order):
        read = []

        def increment_one_mask(graph_filename, *_):
            read.append(graph_filename)

        monkeypatch.setattr(cm, "increment_one_mask", increment_one_mask)
        monkeypatch.setattr(cm, "aims", SimpleNamespace(write=lambda *args: None))

        with _listing_in(order):
            _assert_control_reaches_glob(graph_dir, order)
            cm.increment_mask([_subject_dict(tmp_path)], None, SULCUS_ON_SIDE, OUT_VS, str(tmp_path))

        assert read == [str(graph_dir / min(GRAPH_VOXELS))]

    @unsorted_orders
    def test_mask_holds_the_sorted_first_graph(self, tmp_path, graph_dir, monkeypatch, order):
        graphs = {str(graph_dir / name): _graph(voxel) for name, voxel in GRAPH_VOXELS.items()}
        monkeypatch.setattr(cm, "aims", _AimsProxy(graphs))
        mask_dir = tmp_path / "mask"

        with _listing_in(order):
            _assert_control_reaches_glob(graph_dir, order)
            cm.compute_mask(
                src_dir=str(tmp_path / "src"),
                mask_dir=str(mask_dir),
                path_to_graph=PATH_TO_GRAPH,
                sulcus=SULCUS,
                side=SIDE,
                out_voxel_size=OUT_VS[0],
            )

        first = min(GRAPH_VOXELS)
        expected = _mask_index(graphs[str(graph_dir / first)], GRAPH_VOXELS[first])
        final = np.asarray(aims.read(str(mask_dir / SIDE / f"{SULCUS_ON_SIDE}.nii.gz")))
        assert [tuple(int(c) for c in index[:3]) for index in np.argwhere(final)] == [expected]


def _subject_dirs(src: Path):
    for subject in SUBJECTS:
        (src / subject).mkdir(parents=True)
    return str(src)


class TestSubjectsDictionaryOrder:
    """subjects.py get_all_subjects_as_dictionary: the subject list follows os.listdir."""

    @unsorted_orders
    def test_subjects_are_sorted(self, tmp_path, order):
        src = _subject_dirs(tmp_path / "src")

        with _listing_in(order):
            assert os.listdir(src) == order(SUBJECTS)
            subjects = get_all_subjects_as_dictionary([src], [GRAPH_FILE], SIDE)

        assert [sub["subject"] for sub in subjects] == sorted(SUBJECTS)

    @unsorted_orders
    def test_each_source_directory_is_sorted_in_turn(self, tmp_path, order):
        # src_dir_list order stays the caller's: "z_src" sorts after "a_src" but is listed first.
        sources = [_subject_dirs(tmp_path / "z_src"), _subject_dirs(tmp_path / "a_src")]

        with _listing_in(order):
            assert [os.listdir(src) for src in sources] == [order(SUBJECTS)] * 2
            subjects = get_all_subjects_as_dictionary(sources, [GRAPH_FILE, GRAPH_FILE], SIDE)

        expected = [(src, subject) for src in sources for subject in sorted(SUBJECTS)]
        assert [(sub["dir"], sub["subject"]) for sub in subjects] == expected


def _box_voxel(graph_files):
    """Voxel box enclosing the sulcus of the graph files, by the module's own functions."""
    boxes = [cbb.get_one_bounding_box(graph_file, SULCUS_ON_SIDE) for graph_file in graph_files]
    bbmin, bbmax = compute_max([box[0].tolist() for box in boxes], [box[1].tolist() for box in boxes])
    return [array.tolist() for array in cbb.compute_box_voxel(bbmin, bbmax, OUT_VS)]


def _compute_box(tmp_path, number_subjects):
    box = cbb.compute_bounding_box(
        src_dir=[str(tmp_path / "src")],
        bbox_dir=str(tmp_path / "bbox"),
        path_to_graph=[PATH_TO_GRAPH],
        sulcus=SULCUS,
        side=SIDE,
        number_subjects=number_subjects,
        out_voxel_size=OUT_VS[0],
    )
    return [array.tolist() for array in box]


class TestBoundingBoxGraphOrder:
    """compute_bounding_box.py get_bounding_boxes: list_graph_file[0] of a glob picks the graph behind the box."""

    @unsorted_orders
    def test_sorted_first_graph_is_read(self, tmp_path, graph_dir, monkeypatch, order):
        read = []

        def get_one_bounding_box(graph_filename, _sulcus):
            read.append(graph_filename)
            return [0, 0, 0], [1, 1, 1]

        monkeypatch.setattr(cbb, "get_one_bounding_box", get_one_bounding_box)

        with _listing_in(order):
            _assert_control_reaches_glob(graph_dir, order)
            cbb.get_bounding_boxes([_subject_dict(tmp_path)], SULCUS_ON_SIDE)

        assert read == [str(graph_dir / min(GRAPH_VOXELS))]

    @unsorted_orders
    def test_box_holds_the_sorted_first_graph(self, tmp_path, graph_dir, monkeypatch, order):
        graphs = {str(graph_dir / name): _graph(voxel) for name, voxel in GRAPH_VOXELS.items()}
        monkeypatch.setattr(cbb, "aims", _AimsProxy(graphs))

        with _listing_in(order):
            _assert_control_reaches_glob(graph_dir, order)
            box = _compute_box(tmp_path, 1)

        assert box == _box_voxel([str(graph_dir / min(GRAPH_VOXELS))])


@pytest.fixture
def three_subjects(tmp_path, monkeypatch):
    """One graph per name of SUBJECTS, with one voxel each; returns the graph files in sorted subject order."""
    graphs = {}
    for subject, voxel in zip(sorted(SUBJECTS), GRAPH_VOXELS.values(), strict=True):
        folder = tmp_path / "src" / subject / PATH_TO_GRAPH
        folder.mkdir(parents=True)
        graph_file = folder / f"{SIDE}{subject}_auto.arg"
        graph_file.write_text("")
        graphs[str(graph_file)] = _graph(voxel)
    monkeypatch.setattr(cbb, "aims", _AimsProxy(graphs))
    return list(graphs)


class TestBoundingBoxSubjects:
    """compute_bounding_box.py compute: select_subjects sorts the subject dictionaries themselves."""

    @pytest.mark.parametrize("number_subjects", [1, 2, 3])
    def test_box_encloses_the_sorted_first_subjects(self, tmp_path, three_subjects, number_subjects):
        box = _compute_box(tmp_path, number_subjects)

        assert box == _box_voxel(three_subjects[:number_subjects])

    @unsorted_orders
    def test_kept_subjects_do_not_follow_the_listing(self, tmp_path, three_subjects, order):
        with _listing_in(order):
            assert os.listdir(tmp_path / "src") == order(SUBJECTS)
            box = _compute_box(tmp_path, 2)

        assert box == _box_voxel(three_subjects[:2])


class TestFirstGraphOutsideBids:
    """Outside BIDS mode each step stops after the first graph of a glob: that graph feeds the crops."""

    @unsorted_orders
    def test_skeleton_reads_the_sorted_first_graph(self, tmp_path, graph_dir, monkeypatch, order):
        read = []
        monkeypatch.setattr(gs, "generate_skeleton_from_graph_file", lambda graph_file, *_: read.append(graph_file))
        step = gs.GraphConvert2Skeleton(
            src_dir=str(tmp_path / "src"),
            skeleton_dir=str(tmp_path / "skeletons"),
            side=SIDE,
            junction="wide",
            parallel=False,
            path_to_graph=PATH_TO_GRAPH,
            bids=False,
            qc_path="",
            njobs=1,
            skip_dir=str(tmp_path / "skip"),
        )

        with _listing_in(order):
            _assert_control_reaches_glob(graph_dir, order)
            step.generate_one_skeleton(SUBJECT)

        assert read == [str(graph_dir / min(GRAPH_VOXELS))]

    @unsorted_orders
    def test_foldlabel_reads_the_sorted_first_graph(self, tmp_path, graph_dir, monkeypatch, order):
        read = []
        monkeypatch.setattr(gf, "generate_foldlabel_from_graph_file", lambda graph_file, *_: read.append(graph_file))
        step = gf.GraphConvert2FoldLabel(
            src_dir=str(tmp_path / "src"),
            foldlabel_dir=str(tmp_path / "foldlabels"),
            side=SIDE,
            junction="wide",
            parallel=False,
            path_to_graph=PATH_TO_GRAPH,
            bids=False,
            qc_path="",
            skip_dir=str(tmp_path / "skip"),
        )

        with _listing_in(order):
            _assert_control_reaches_glob(graph_dir, order)
            step.generate_one_foldlabel(SUBJECT)

        assert read == [str(graph_dir / min(GRAPH_VOXELS))]

    @unsorted_orders
    def test_transform_reads_the_sorted_first_graph(self, tmp_path, graph_dir, monkeypatch, order):
        read = []
        fake_aims = SimpleNamespace(
            read=lambda graph_file: read.append(graph_file),
            write=lambda *args: None,
            GraphManip=SimpleNamespace(getICBM2009cTemplateTransform=lambda graph: graph),
        )
        monkeypatch.setattr(gt, "aims", fake_aims)
        step = gt.GraphGenerateTransform(
            src_dir=str(tmp_path / "src"),
            transform_dir=str(tmp_path / "transforms"),
            side=SIDE,
            parallel=False,
            path_to_graph=PATH_TO_GRAPH,
            bids=False,
            qc_path="",
            skip_dir=str(tmp_path / "skip"),
        )

        with _listing_in(order):
            _assert_control_reaches_glob(graph_dir, order)
            step.generate_one_transform(SUBJECT)

        assert read == [str(graph_dir / min(GRAPH_VOXELS))]


SRC_FILENAME = "skeleton_"


def _not_processed_files(src, tgt):
    return qc.get_not_processed_files(src, tgt, SRC_FILENAME)


def _not_processed_files_general(src, tgt):
    return qc.get_not_processed_files_general(src, tgt, SRC_FILENAME, "skeleton_no_ventricle_")


def _not_processed_cropped_files(src, tgt):
    return qc.get_not_processed_cropped_files(src, tgt)


# Per function: two files giving one subject key, two files with different text before the
# subject key, and a clean folder of two subjects.
AMBIGUOUS_FOLDERS = {
    _not_processed_files: {
        "same_key": ["Rskeleton_sub01.nii.gz", "Rskeleton_sub01.v2.nii.gz"],
        "mixed_prefix": ["Lskeleton_sub02.nii.gz", "Rskeleton_sub01.nii.gz"],
        "clean": ["Rskeleton_sub01.nii.gz", "Rskeleton_sub02.nii.gz"],
    },
    _not_processed_files_general: {
        "same_key": ["Rskeleton_sub01.nii.gz", "Rskeleton_sub01.v2.nii.gz"],
        "mixed_prefix": ["Rskeleton_sub01.nii.gz", "sub02.nii.gz"],
        "clean": ["Rskeleton_sub01.nii.gz", "Rskeleton_sub02.nii.gz"],
    },
    _not_processed_cropped_files: {
        "same_key": ["Rresampled_skeleton_sub01.nii.gz", "Rresampled_skeleton_sub01.v2.nii.gz"],
        "mixed_prefix": ["Rresampled_foldlabel_sub02.nii.gz", "Rresampled_skeleton_sub01.nii.gz"],
        "clean": ["Rresampled_skeleton_sub01.nii.gz", "Rresampled_skeleton_sub02.nii.gz"],
    },
}
FUNCTION_IDS = ["get_not_processed_files", "get_not_processed_files_general", "get_not_processed_cropped_files"]
each_function = pytest.mark.parametrize("function", list(AMBIGUOUS_FOLDERS), ids=FUNCTION_IDS)
both_orders = pytest.mark.parametrize("order", [sorted, _descending], ids=["ascending", "descending"])


def _source_folder(tmp_path, names):
    """Source folder holding one empty file per name, and an empty target folder."""
    src = tmp_path / "src"
    src.mkdir()
    for name in names:
        (src / name).write_text("")
    tgt = tmp_path / "tgt"
    tgt.mkdir()
    return src, tgt


class TestAmbiguousSourceFolder:
    """quality_checks.py :108, :157, :239: a folder with two candidate files is refused, never resolved by order."""

    @each_function
    @both_orders
    @pytest.mark.parametrize("case", ["same_key", "mixed_prefix"])
    def test_two_candidate_files_raise_value_error_naming_both(self, tmp_path, function, order, case):
        names = AMBIGUOUS_FOLDERS[function][case]
        src, tgt = _source_folder(tmp_path, names)

        with _listing_in(order):
            assert [os.path.basename(path) for path in glob.glob(f"{src}/*.nii.gz")] == order(names)
            with pytest.raises(ValueError) as refused:
                function(str(src), str(tgt))

        for name in names:
            assert str(src / name) in str(refused.value)

    @each_function
    @both_orders
    def test_clean_folder_is_not_refused(self, tmp_path, function, order):
        names = AMBIGUOUS_FOLDERS[function]["clean"]
        src, tgt = _source_folder(tmp_path, names)

        with _listing_in(order):
            result = function(str(src), str(tgt))

        assert sorted(result) == [str(src / name) for name in names]
