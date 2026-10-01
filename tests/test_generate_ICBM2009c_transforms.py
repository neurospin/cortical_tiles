from cortical_tiles.brainvisa import generate_ICBM2009c_transforms


def test_generate_ICBM2009c_transforms_help():
    args = "--help"
    argv = args.split(" ")
    generate_ICBM2009c_transforms.main(argv)


def test_generate_ICBM2009c_transforms_n_0(tmp_path):
    """Tests the function when number of subjects is 0"""

    generate_ICBM2009c_transforms.generate_ICBM2009c_transforms(transform_dir=str(tmp_path), nb_subjects=0)
