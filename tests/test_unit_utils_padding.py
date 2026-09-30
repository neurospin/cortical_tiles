"""Unit tests for cortical_tiles.brainvisa.utils.padding.padd (REQ-CTILESTEST-21).

``padd`` is used by ``utils/distmap.py`` to bring volumes to a target shape. Pure numpy.
"""

import numpy as np

from cortical_tiles.brainvisa.utils.padding import padd


def test_padd_centres_array_with_extra_value_after():
    """REQ-CTILESTEST-21: floor(diff/2) before, the rest after, filled with fill_value."""
    arr = np.arange(1, 1 + 2 * 3 * 4).reshape(2, 3, 4)

    out = padd(arr, [5, 3, 8], fill_value=-7)

    assert out.shape == (5, 3, 8)
    # axis 0: diff 3 -> 1 before, 2 after; axis 1: diff 0; axis 2: diff 4 -> 2 before, 2 after
    np.testing.assert_array_equal(out[1:3, :, 2:6], arr)
    mask = np.ones(out.shape, dtype=bool)
    mask[1:3, :, 2:6] = False
    assert (out[mask] == -7).all()


def test_padd_leaves_unlisted_trailing_axes_unpadded():
    """REQ-CTILESTEST-21: axes beyond len(shape) are not padded."""
    arr = np.ones((2, 2, 2, 3), dtype=np.int16)

    out = padd(arr, [4, 3, 2])

    assert out.shape == (4, 3, 2, 3)
    np.testing.assert_array_equal(out[1:3, 0:2, :, :], arr)
    assert out.sum() == arr.sum()


def test_padd_default_fill_is_zero():
    """REQ-CTILESTEST-21: default fill_value is 0."""
    out = padd(np.full((1,), 9), [4])

    np.testing.assert_array_equal(out, [0, 9, 0, 0])
