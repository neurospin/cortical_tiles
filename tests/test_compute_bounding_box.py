import glob
import json
from pathlib import Path

from cortical_tiles.brainvisa import compute_bounding_box

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Gets the source directory (list)
SRC_DIR = [str(DATA_DIR / "source" / "supervised")]

# Gets the reference directory
REF_DIR = DATA_DIR / "reference" / "bbox"

# Gets relative path to graph file name
path_to_graph = ["t1mri/t1/default_analysis/folds/3.3/base2018_manual"]

# Gets sulcus name
sulcus = "S.T.s.ter.asc.ant."
side = "L"
out_voxel_size = 1


def test_resample_files_help():
    args = "--help"
    argv = args.split(" ")
    compute_bounding_box.main(argv)


def test_bounding_box_n_0(tmp_path):
    """Tests the function when number of subjects is 0"""
    bbox_dir = str(tmp_path / "bbox")

    compute_bounding_box.compute_bounding_box(
        src_dir=SRC_DIR, bbox_dir=bbox_dir, sulcus=sulcus, side=side, out_voxel_size=out_voxel_size, number_subjects=0
    )


def test_bounding_box_n_1(tmp_path):
    """Tests if the bounding box one one subject gives the expected result.

    The source and the reference are in the data subdirectory
    """
    bbox_dir = str(tmp_path / "bbox")

    # Determines the bounding box around the sulcus
    compute_bounding_box.compute_bounding_box(
        src_dir=SRC_DIR,
        bbox_dir=bbox_dir,
        path_to_graph=path_to_graph,
        sulcus=sulcus,
        side=side,
        number_subjects=1,
        out_voxel_size=out_voxel_size,
    )

    # Selected keys to test
    selected_keys = ["bbmin_voxel", "bbmax_voxel", "bbmin_AIMS_Talairach", "bbmin_AIMS_Talairach"]

    # Gets and reads the first reference json file
    ref_dir_side = str(REF_DIR / side)
    ref_file = glob.glob(ref_dir_side + "/*.json")[0]
    print("ref_file = ", ref_file, "\n")
    with open(ref_file, "r") as f:
        data_ref = json.load(f)
        print(json.dumps(data_ref, sort_keys=True, indent=4))
        box_ref = {k: data_ref[k] for k in selected_keys}

    # Gets and reads the first target json file
    tgt_dir_side = bbox_dir + "/" + side
    tgt_file = glob.glob(tgt_dir_side + "/*.json")[0]
    print("tgt_file = ", tgt_file, "\n")
    with open(tgt_file, "r") as f:
        data_target = json.load(f)
        print(json.dumps(data_target, sort_keys=True, indent=4))
        box_target = {k: data_ref[k] for k in selected_keys}

    assert box_target == box_ref
