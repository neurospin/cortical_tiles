from cortical_tiles.brainvisa import generate_skeletons


def test_generate_skeletons_help():
    args = "--help"
    argv = args.split(" ")
    generate_skeletons.main(argv)


def test_generate_skeletons_n_0(tmp_path):
    """Tests the function when number of subjects is 0"""
    generate_skeletons.generate_skeletons(skeleton_dir=str(tmp_path), nb_subjects=0)
