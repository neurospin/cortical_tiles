from cortical_tiles.brainvisa import compute_mask


def test_compute_mask_help():
    args = "--help"
    argv = args.split(" ")
    compute_mask.main(argv)


def test_compute_mask_n_0(tmp_path):
    """Tests the function when number of subjects is 0"""
    compute_mask.compute_mask(mask_dir=str(tmp_path), number_subjects=0)
