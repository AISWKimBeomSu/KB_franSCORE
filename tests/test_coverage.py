"""평가 범위 보정(src/coverage.py) — 빈 해를 무엇으로 메우고, 무엇은 메우지 않는가.

보정이 헐거우면 근거 없는 브랜드가 등급을 받고, 빡빡하면 BBQ 같은 대형 브랜드가 계속
빠진다. 판정 경로를 하나씩 고정한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import coverage as CV
from src.score import _place_on_steps

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CFG = {"sample": {"min_stores": 30, "min_consecutive_years": 3}}
T = 2024


def _row(bid, year, n, *, name=None, mnno=None, corp="본부", elig=False):
    return {"brand_id": bid, "brand_name": name or bid, "company_name": corp, "brand_mnno": mnno,
            "year": year, "n_stores": n, "eligible_t": elig}


def _frames(rows):
    pf = pd.DataFrame(rows)
    return pf, pf.copy()                      # 합성 자료에서는 전체 패널 = 업종 범위 패널


def _one(table, bid):
    return table.set_index("brand_id").loc[bid]


def test_regular_eligible_brand_is_scored_without_basis_text():
    pf, panel = _frames([_row("A", y, 100, elig=(y == T)) for y in (2022, 2023, T)])
    r = _one(CV.assess(CFG, pf, panel, T, {"region": {}, "registry": {}}), "A")
    assert r["status"] == CV.STATUS_SCORED and r["basis"] == CV.BASIS_REGULAR


def test_one_year_stats_hole_confirmed_by_region_data_is_bridged():
    """BBQ 형 — 통계 API 에 2022년만 없고, 지역·직영 통계에는 그해 가맹점이 있다."""
    rows = [_row("B", y, 2000, mnno="M1") for y in (2019, 2020, 2021, 2023, T)]
    pf, panel = _frames(rows)
    r = _one(CV.assess(CFG, pf, panel, T, {"region": {"M1": {2022}}, "registry": {}}), "B")
    assert r["status"] == CV.STATUS_BRIDGED
    assert r["gap_years"] == "2022" and "지역·직영 통계" in r["basis"]


def test_same_brand_split_across_ids_is_bridged_by_link():
    """고봉민김밥인 형 — 같은 관리번호의 옛 ID 에 그해 행이 있다."""
    rows = ([_row("OLD", y, 500, mnno="M2", name="고봉민김밥人") for y in (2019, 2020, 2021, 2022)]
            + [_row("NEW", y, 450, mnno="M2", name="고봉민김밥인") for y in (2023, T)])
    pf, panel = _frames(rows)
    r = _one(CV.assess(CFG, pf, panel, T, {"region": {}, "registry": {}}), "NEW")
    assert r["status"] == CV.STATUS_BRIDGED and "다른 ID" in r["basis"] and "500개" in r["basis"]


def test_registry_only_confirmation_is_bridged_and_labelled():
    rows = [_row("C", y, 1200, mnno="M3") for y in (2020, 2021, 2023, T)]
    pf, panel = _frames(rows)
    r = _one(CV.assess(CFG, pf, panel, T, {"region": {}, "registry": {"M3": {2022}}}), "C")
    assert r["status"] == CV.STATUS_BRIDGED and "정보공개서 등록" in r["basis"]


def test_missing_previous_year_is_not_scored_even_if_confirmed_elsewhere():
    """파리바게뜨 형 — 직전 연도가 비면 변화 지표가 결측이라 점수가 낙관적으로 나온다."""
    rows = [_row("D", y, 3300, mnno="M4") for y in (2020, 2021, 2022, T)]
    pf, panel = _frames(rows)
    r = _one(CV.assess(CFG, pf, panel, T, {"region": {"M4": {2023}}, "registry": {}}), "D")
    assert r["status"] == CV.STATUS_UNSCORED
    assert "2023년 가맹점 통계" in r["reason"] and "지역·직영 통계" in r["reason"]


def test_unconfirmed_gap_short_history_and_small_brands_get_their_own_reasons():
    rows = ([_row("E", y, 300, mnno="M5") for y in (2019, 2020, 2021, 2023, T)]      # 확인 안 된 공백
            + [_row("F", y, 80) for y in (2023, T)]                                  # 신규
            + [_row("G", y, 12) for y in (2020, 2021, 2022, 2023, T)]                # 소형
            + [_row("H", T, 0)])                                                     # 0개
    pf, panel = _frames(rows)
    t = CV.assess(CFG, pf, panel, T, {"region": {}, "registry": {}}).set_index("brand_id")
    assert "확인되지 않아" in t.loc["E", "reason"]
    assert "이력이 2년" in t.loc["F", "reason"]
    assert "30개에 이른 적이 없어" in t.loc["G", "reason"]
    assert "0이거나 비어" in t.loc["H", "reason"]
    assert (t["status"] == CV.STATUS_UNSCORED).all()


def test_bridged_display_value_sits_between_regular_neighbours_on_the_same_step():
    raw_r = np.array([0.10, 0.20, 0.30, 0.40])
    step_r = np.array([0.05, 0.05, 0.05, 0.12])
    smooth_r = np.array([0.041, 0.050, 0.059, 0.12])
    got = _place_on_steps(np.array([0.25, 0.45]), np.array([0.05, 0.12]), raw_r, step_r, smooth_r)
    assert 0.050 < got[0] < 0.059            # 같은 계단의 정규 두 브랜드 사이
    assert got[1] == pytest.approx(0.12)     # 같은 계단 정규 브랜드가 하나뿐 → 계단값


@pytest.mark.skipif(not (OUT / "coverage_report.csv").exists(), reason="coverage_report 없음")
def test_outputs_agree_scores_report_and_meta():
    rep = pd.read_csv(OUT / "coverage_report.csv", encoding="utf-8-sig")
    sc = pd.read_csv(OUT / "scores_latest.csv", encoding="utf-8-sig")
    meta = json.loads((OUT / "scores_latest_meta.json").read_text(encoding="utf-8"))
    scored = set(rep.loc[rep["status"] != CV.STATUS_UNSCORED, "brand_id"].astype(str))
    assert scored == set(sc["brand_id"].astype(str))
    n_b = int((rep["status"] == CV.STATUS_BRIDGED).sum())
    assert meta["n_bridged"] == n_b == int((sc["eligibility_basis"] != "정규").sum())
    assert meta["n_regular"] + meta["n_bridged"] == meta["n_scored"] == len(sc)
    assert rep.loc[rep["status"] == CV.STATUS_UNSCORED, "reason"].astype(str).str.len().gt(0).all()
