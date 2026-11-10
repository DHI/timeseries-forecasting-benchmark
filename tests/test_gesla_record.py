"""Parsing a GESLA record: values GESLA marks not for use must not reach the target."""

from __future__ import annotations

from tsforecastbench.data.cases.surge_cuxhaven.download import _read_gesla_record

RECORD = """\
# SITE NAME Cuxhaven_Steubenhoft
# NUMBER OF HEADER LINES 2
2003/06/14 10:00:00 0.1100 1 1
2003/06/14 11:00:00 -4.5700 4 0
2003/06/14 12:00:00 -99.9999 5 0
2003/06/14 13:00:00 2.0000 3 0
2003/06/14 14:00:00 0.2300 0 1
"""


def test_values_flagged_not_for_use_become_nan(tmp_path):
    path = tmp_path / "cuxhaven"
    path.write_text(RECORD)
    df = _read_gesla_record(path)
    assert list(df.columns) == ["ds", "sea_level"]
    assert len(df) == 5
    assert df["sea_level"].isna().tolist() == [False, True, True, True, False]
    assert df["sea_level"].dropna().tolist() == [0.11, 0.23]
