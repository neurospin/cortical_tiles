from cortical_tiles.brainvisa import generate_crops


def test_generate_crops_help():
    args = "--help"
    argv = args.split(" ")
    generate_crops.main(argv)


def test_generate_crops_n_0(tmp_path):
    """Tests the function when number of subjects is 0"""
    generate_crops.generate_crops(crop_dir=str(tmp_path), nb_subjects=0)
