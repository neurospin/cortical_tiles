#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guards two upstream regression tests against vacuous assertions.

``test_bounding_box_n_1`` must fail when the generated bounding box differs
from the reference on any compared key, and
``test_generate_foldlabels_value_correspondance`` must fail when the
generated skeleton bucket differs from the reference bucket. Each guard
injects a known mismatch and expects the guarded test to raise
``AssertionError``; a test that still passes is checking nothing.

Covers REQ-UPSTREAM-05 and REQ-UPSTREAM-06 (tracked in
champollion_pipeline/elm/REQUIREMENTS.md).
"""

import glob
import json
import os

import pytest

from tests import test_compute_bounding_box as bbox_test
from tests import test_generate_foldlabels as foldlabels_test

COMPARED_BBOX_KEYS = [
    "bbmin_voxel",
    "bbmax_voxel",
    "bbmin_AIMS_Talairach",
    "bbmax_AIMS_Talairach",
]


def _reference_bbox():
    ref_file = glob.glob(str(bbox_test.REF_DIR / bbox_test.side / "*.json"))[0]
    with open(ref_file, "r") as f:
        return json.load(f)


@pytest.mark.parametrize("key", COMPARED_BBOX_KEYS)
def test_bounding_box_n_1_fails_on_mismatched_key(key, tmp_path, monkeypatch):
    """REQ-UPSTREAM-05: a generated box differing on ``key`` must fail."""
    generated = _reference_bbox()
    generated[key] = [value + 1 for value in generated[key]]

    def fake_compute_bounding_box(*, bbox_dir, side, **_kwargs):
        os.makedirs(os.path.join(bbox_dir, side), exist_ok=True)
        with open(os.path.join(bbox_dir, side, "generated.json"), "w") as f:
            json.dump(generated, f)

    monkeypatch.setattr(bbox_test.compute_bounding_box, "compute_bounding_box", fake_compute_bounding_box)

    with pytest.raises(AssertionError):
        bbox_test.test_bounding_box_n_1(tmp_path)


def test_generate_foldlabels_value_correspondance_fails_on_unequal_buckets(tmp_path, monkeypatch):
    """REQ-UPSTREAM-06: unequal generated and reference buckets must fail."""
    monkeypatch.setattr(foldlabels_test, "equal_buckets", lambda _bck1, _bck2: False)

    with pytest.raises(AssertionError):
        foldlabels_test.test_generate_foldlabels_value_correspondance(tmp_path)
