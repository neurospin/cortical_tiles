"""Unit tests for select_good_qc id handling (DEF-CORTILES-BDRABCZUK-777B67E5881A).

A participant_id of the QC file is read as text, so an all-numeric id with leading zeros (012) keeps its zeros and
matches the subject folder of the same name instead of being read as the number 12 and dropped.
"""

import pytest
from cortical_tiles.brainvisa.utils.subjects import select_good_qc


def _write_qc(tmp_path, suffix, rows):
    sep = "\t" if suffix == ".tsv" else ","
    path = tmp_path / f"qc{suffix}"
    path.write_text("participant_id" + sep + "qc\n" + "".join(f"{pid}{sep}{qc}\n" for pid, qc in rows))
    return str(path)


@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_leading_zero_numeric_ids_are_kept(tmp_path, suffix):
    qc_path = _write_qc(tmp_path, suffix, [("012", 1), ("007", 1), ("100", 1)])
    assert select_good_qc(["012", "007", "100"], qc_path) == ["007", "012", "100"]


@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_leading_zero_id_with_qc_zero_is_the_only_one_dropped(tmp_path, suffix):
    qc_path = _write_qc(tmp_path, suffix, [("012", 0), ("007", 1)])
    assert select_good_qc(["012", "007"], qc_path) == ["007"]


def test_unpadded_name_does_not_match_padded_id(tmp_path):
    qc_path = _write_qc(tmp_path, ".csv", [("012", 1)])
    assert select_good_qc(["12", "012"], qc_path) == ["012"]
