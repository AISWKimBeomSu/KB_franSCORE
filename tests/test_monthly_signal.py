"""월별 인허가 신호의 화면 연결 — 신호표가 있을 때만, 범위와 함께 보인다."""
from __future__ import annotations

import pandas as pd

from src.views import common as C


def _write(tmp_path, rows):
    pd.DataFrame(rows).to_csv(tmp_path / "localdata_signal.csv", index=False, encoding="utf-8-sig")


def test_signal_line_shows_trend_scope_and_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "out_dir", lambda: tmp_path)
    _write(tmp_path, [{"brand_id": "B1", "close_3m": 7, "open_3m": 2, "net_3m": -5, "close_rate_3m": 0.05,
                       "trend": "악화", "brand_name": "가상브랜드", "as_of_month": "2026-08",
                       "n_active_end": 140, "reason": "폐업 7건(전년 동기 1건) — 전년보다 유의하게 많음.",
                       "scope": "전국"}])
    html = C.localdata_html("B1")
    assert "월별 인허가 신호" in html and "2026-08" in html and "전국" in html
    assert "폐업 7건" in html and "개업 2건" in html and "관찰 점포 140곳" in html
    assert "g-High" in html                                   # 악화는 경고색 배지


def test_no_signal_file_or_brand_draws_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "out_dir", lambda: tmp_path)
    assert C.localdata_html("B1") == ""                        # 파일 없음
    _write(tmp_path, [{"brand_id": "B2", "close_3m": 0, "open_3m": 1, "trend": "판단보류",
                       "as_of_month": "2026-08", "reason": "관측 점포 30곳 미만"}])
    assert C.localdata_html("B1") == ""                        # 다른 브랜드뿐
    assert "지역 표본" in C.localdata_html("B2")                # 범위가 없으면 지역 표본으로 밝힌다
