"""은행 연체 자료 검증(src/delinquency.py) — 절차가 스스로를 속이지 않는지.

검증 도구가 틀리면 '등급이 연체를 설명한다'는 잘못된 결론이 결재에 올라간다. 그래서
결과가 좋게 나오는 쪽으로 새는 길 — 이후 정보 사용(look-ahead), 사라진 브랜드 누락
(생존 편향), 빈칸을 정상으로 세기, 대출을 독립으로 본 좁은 구간 — 을 하나씩 막는다.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import delinquency as D

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def _hist(rows: list[tuple]) -> pd.DataFrame:
    """(brand_id, year, brand_name, grade, step, state) → 이력 표."""
    return pd.DataFrame([{"brand_id": b, "year": y, "brand_name": n, "n_stores": 50, "grade": g,
                          "risk_grade": {"FS1": "Low", "FS2": "Medium", "FS3": "High"}[g],
                          "deterioration_step": s, "deterioration_1y": s, "brand_state": st,
                          "n_events_at_t": 0 if st == "건전" else 1, "basis": "테스트"}
                         for b, y, n, g, s, st in rows])


def _loans(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).astype(str)


# ── 산출물 계약 ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not (OUT / "grade_history.csv").exists(), reason="grade_history 없음")
def test_grade_history_reproduces_published_grade_record():
    """이력의 검증 연도 등급이 공표 등급 실적(grade_bands.json pooled)과 한 건도 다르지 않다."""
    hist = pd.read_csv(OUT / "grade_history.csv", encoding="utf-8-sig")
    wf = pd.read_parquet(OUT / "walkforward_predictions.parquet")[["brand_id", "year", "y_true"]]
    m = hist.merge(wf, on=["brand_id", "year"], how="inner")
    got = m.groupby("grade").agg(n=("y_true", "size"), events=("y_true", "sum"))
    pooled = json.loads((OUT / "grade_bands.json").read_text(encoding="utf-8"))["pooled"]
    for row in pooled:
        assert int(got.loc[row["grade"], "n"]) == row["n"]
        assert int(got.loc[row["grade"], "events"]) == row["events"]


@pytest.mark.skipif(not (OUT / "grade_history.csv").exists(), reason="grade_history 없음")
def test_grade_history_latest_equals_operational_scores_and_skips_training_years():
    hist = pd.read_csv(OUT / "grade_history.csv", encoding="utf-8-sig")
    train = set(json.loads((OUT / "split_years.json").read_text(encoding="utf-8"))["train_years"])
    assert not (set(hist["year"]) & train), "학습 연도 등급(표본 내)이 이력에 섞였다"
    sl = pd.read_csv(OUT / "scores_latest.csv", encoding="utf-8-sig")
    latest = hist[hist["year"] == int(sl["year"].max())].set_index("brand_id")["grade"]
    assert latest.sort_index().equals(sl.set_index("brand_id")["grade"].sort_index())


# ── 값 해석 ──────────────────────────────────────────────────────────────────

def test_parse_outcome_reads_flags_and_never_counts_blank_as_good():
    s = pd.Series(["Y", "n", "1", "0", "예", "아니오", "", None, "연체", "모름"])
    got = D.parse_outcome(s).tolist()
    assert got[:6] == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert np.isnan(got[6]) and np.isnan(got[7]), "빈칸을 정상으로 세면 연체율이 낮게 나온다"
    assert got[8] == 1.0 and np.isnan(got[9])


def test_parse_outcome_days_past_due_threshold():
    s = pd.Series(["0", "29", "30", "91", "1,200", ""])
    assert D.parse_outcome(s, "dpd", 30).tolist()[:5] == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert D.parse_outcome(s, "dpd", 90).tolist()[:4] == [0.0, 0.0, 0.0, 1.0]
    assert np.isnan(D.parse_outcome(s, "dpd", 90).iloc[5])


def test_parse_year_handles_bank_date_formats():
    s = pd.Series(["2023-05-01", "2023.05.01", "2023/05/01", "20230501", "2023", "2023년",
                   "45047", "2023-05-01 00:00:00", "abc", ""])
    got = D.parse_year(s).tolist()
    assert got[:8] == [2023.0] * 8
    assert np.isnan(got[8]) and np.isnan(got[9])


def test_detect_columns_on_template():
    df = pd.read_excel(io.BytesIO(D.template_bytes()), dtype=str)
    c = D.detect_columns(df)
    assert (c.brand, c.outcome, c.outcome_kind, c.date, c.internal) == (
        "브랜드명", "연체여부", "flag", "취급일", "내부등급")
    assert c.amount == "대출금액(백만원)"


def test_detect_columns_prefers_flag_but_reads_numeric_days_as_dpd():
    df = pd.DataFrame({"가맹 브랜드": ["a"], "최장연체일수": ["35"], "실행일자": ["2023-01-01"]})
    c = D.detect_columns(df)
    assert (c.brand, c.outcome, c.outcome_kind, c.date) == ("가맹 브랜드", "최장연체일수", "dpd", "실행일자")


# ── 시점 정합 · 생존 편향 ────────────────────────────────────────────────────

HIST = _hist([
    ("A", 2022, "알파치킨", "FS1", 0.02, "건전"),
    ("A", 2023, "알파치킨", "FS3", 0.30, "요주의"),
    ("A", 2024, "알파치킨", "FS3", 0.35, "요주의"),
    ("B", 2023, "베타커피", "FS2", 0.08, "건전"),
    ("B", 2024, "베타커피", "FS2", 0.09, "건전"),
    ("C", 2022, "감마분식", "FS3", 0.40, "요주의"),      # 2023년 이후 사라진 브랜드
])


def test_point_in_time_uses_prior_year_disclosure_grade():
    df = _loans([
        {"브랜드명": "알파치킨", "취급일": "2023-06-01", "연체여부": "N"},   # → 2022 FS1
        {"브랜드명": "알파치킨", "취급일": "2024-02-01", "연체여부": "Y"},   # → 2023 FS3
        {"브랜드명": "알파치킨", "취급일": "2022-03-01", "연체여부": "N"},   # → 2021: 이력 이전
        {"브랜드명": "베타커피", "취급일": "2023-09-09", "연체여부": "N"},   # → 2022: 평가 전
        {"브랜드명": "알파치킨", "취급일": "2026-01-01", "연체여부": "N"},   # → 2025: 2024 사용(1년 이내)
    ])
    prep = D.prepare(df, D.detect_columns(df), HIST)
    got = dict(zip(prep.loans["row"], prep.loans["grade"], strict=True))
    assert got == {2: "FS1", 3: "FS3", 6: "FS3"}
    ex = prep.excluded.set_index("row")
    assert ex.loc[4, "reason"] == D.EXCL_PIT and "학습 연도" in ex.loc[4, "detail"]
    assert ex.loc[5, "reason"] == D.EXCL_PIT and "평가 대상이 아니었" in ex.loc[5, "detail"]
    assert prep.pit and not prep.notes


def test_same_year_option_and_no_date_fallback_is_flagged():
    df = _loans([{"브랜드명": "알파치킨", "취급일": "2023-06-01", "연체여부": "N"}])
    same = D.prepare(df, D.detect_columns(df), HIST, lag_years=0)
    assert same.loans["grade"].tolist() == ["FS3"]                 # 2023 공시
    nodate = df.drop(columns="취급일")
    p = D.prepare(nodate, D.detect_columns(nodate), HIST)
    assert not p.pit and p.loans["grade"].tolist() == ["FS3"]      # 최신(2024)
    assert p.notes and "좋게 나올 수" in p.notes[0]


def test_disappeared_brand_is_still_matched_no_survivorship():
    """2024년에 없는 브랜드도 과거 이력으로 찾는다 — 빠지면 연체 많은 곳이 사라진다."""
    df = _loans([{"브랜드명": "감마분식", "취급일": "2023-04-01", "연체여부": "Y"}])
    prep = D.prepare(df, D.detect_columns(df), HIST)
    assert prep.loans["brand_id"].tolist() == ["C"] and prep.loans["grade"].tolist() == ["FS3"]


def test_unknown_and_unreadable_rows_are_listed_not_dropped_silently():
    df = _loans([
        {"브랜드명": "없는브랜드", "취급일": "2023-06-01", "연체여부": "N"},
        {"브랜드명": "알파치킨", "취급일": "모름", "연체여부": "N"},
        {"브랜드명": "알파치킨", "취급일": "2023-06-01", "연체여부": ""},
    ])
    prep = D.prepare(df, D.detect_columns(df), HIST)
    assert prep.loans.empty
    assert prep.excluded["reason"].tolist() == [D.EXCL_NOT_FOUND, D.EXCL_DATE, D.EXCL_OUTCOME]


# ── 통계 ─────────────────────────────────────────────────────────────────────

def test_auc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = np.round(rng.random(400) + 0.3 * y, 1)                      # 동점 다수
    assert D.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert np.isnan(D.auc(np.zeros(5), np.arange(5)))


def test_clopper_pearson_known_values():
    assert D.clopper_pearson(0, 10) == pytest.approx((0.0, 0.30850), abs=1e-4)
    lo, hi = D.clopper_pearson(5, 10)
    assert (lo, hi) == pytest.approx((0.18709, 0.81291), abs=1e-4)


def _clustered(n_brands: int, per: int, brand_sd: float, beta: float, seed: int,
               internal_effect: float = 0.3) -> D.Prepared:
    """브랜드 공통 충격이 있는 가상 대출 — 등급은 브랜드 위험으로 매긴다."""
    rng = np.random.default_rng(seed)
    risk = rng.uniform(0.01, 0.4, n_brands)
    grade = np.where(risk < 0.045, "FS1", np.where(risk < 0.16, "FS2", "FS3"))
    shock = rng.normal(0, brand_sd, n_brands)
    b = np.repeat(np.arange(n_brands), per)
    internal = rng.integers(1, 11, len(b))
    z = np.log(risk / (1 - risk))
    lp = -3.6 + internal_effect * (internal - 5) + beta * (z[b] - z.mean()) + shock[b]
    y = (rng.random(len(b)) < 1 / (1 + np.exp(-lp))).astype(int)
    loans = pd.DataFrame({"row": np.arange(len(b)) + 2, "brand_id": [f"B{i}" for i in b],
                          "brand_name": [f"브랜드{i}" for i in b], "default": y,
                          "grade_year": 2023, "grade": grade[b], "risk": risk[b],
                          "brand_state": "건전", "internal": internal.astype(float), "amount": np.nan})
    return D.Prepared(loans=loans, excluded=pd.DataFrame(columns=["row", "brand_input", "reason", "detail"]),
                      pit=True, lag_years=1, synthetic=False)


def test_cluster_bootstrap_interval_is_wider_than_naive_independent_interval():
    """같은 브랜드 연체가 함께 움직이면 대출 독립 가정 구간은 너무 좁다 — 브랜드를 재표집한다."""
    prep = _clustered(n_brands=120, per=60, brand_sd=0.9, beta=0.5, seed=3)
    res = D.evaluate(prep, n_boot=300, seed=1)
    L = prep.loans
    p1, n1 = L.loc[L.grade == "FS1", "default"].mean(), (L.grade == "FS1").sum()
    p3, n3 = L.loc[L.grade == "FS3", "default"].mean(), (L.grade == "FS3").sum()
    naive_width = 2 * 1.96 * np.sqrt(p1 * (1 - p1) / n1 + p3 * (1 - p3) / n3)
    cluster_width = res["rank"]["diff_hi"] - res["rank"]["diff_lo"]
    assert cluster_width > 1.3 * naive_width


def test_incremental_value_detects_planted_signal_and_rejects_null():
    planted = D.evaluate(_clustered(300, 40, 0.3, 0.9, seed=5), n_boot=200, seed=2)
    assert planted["incremental"]["or_lo"] > 1
    assert planted["verdict"]["incremental"][0] == "확인"
    null = D.evaluate(_clustered(300, 40, 0.3, 0.0, seed=6), n_boot=200, seed=2)
    assert null["verdict"]["incremental"][0] != "확인"


def test_incremental_test_false_positive_rate_is_controlled_under_null():
    """브랜드 효과가 없는 자료 60벌에서 '확인'(오즈비 하한 > 1)이 나오는 비율 — 명목 2.5% 근처.

    예전 판정(교차검증 예측을 모아 잰 AUC 증가분)은 귀무에서도 증가분이 72% 양수였다.
    """
    hits = 0
    for seed in range(60):
        L = _clustered(200, 30, 0.6, 0.0, seed=100 + seed).loans
        w = D.cluster_wald(L["default"].to_numpy(), L["internal"].to_numpy(float),
                           L["risk"].to_numpy(float), L["brand_id"].to_numpy())
        hits += w["or_lo"] > 1
    assert hits / 60 <= 0.10


def test_cluster_robust_se_is_wider_than_naive_when_brands_share_shocks():
    L = _clustered(150, 60, 1.0, 0.5, seed=4).loans
    y, ii, ri = L["default"].to_numpy(), L["internal"].to_numpy(float), L["risk"].to_numpy(float)
    robust = D.cluster_wald(y, ii, ri, L["brand_id"].to_numpy())
    naive = D.cluster_wald(y, ii, ri, np.arange(len(y)))          # 대출 하나 = 군집 하나
    assert robust["se"] > 1.3 * naive["se"]


def test_small_sample_defers_overall_verdict():
    res = D.evaluate(_clustered(20, 10, 0.3, 0.9, seed=9), n_boot=100, seed=0)
    assert res["verdict"]["sample"][0] == "부족"
    assert "보류" in res["verdict"]["overall"]


# ── 시연 자료 ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not (OUT / "grade_history.csv").exists(), reason="grade_history 없음")
def test_synthetic_sample_is_marked_end_to_end_and_deterministic():
    hist = D.load_history(OUT)
    pd.testing.assert_frame_equal(D.sample_frame(hist, 2400, 11), D.sample_frame(hist, 2400, 11))
    df = pd.read_excel(io.BytesIO(D.sample_bytes(hist, n_loans=2400, seed=11)), dtype=str)
    prep = D.prepare(df, D.detect_columns(df), hist)
    assert prep.synthetic, "다시 올린 가상 자료도 가상으로 표시돼야 한다"
    res = D.evaluate(prep, n_boot=100)
    assert res["synthetic"] and set(res["verdict"]) == {"rank", "incremental", "sample", "overall"}
    xls = pd.read_excel(io.BytesIO(D.to_excel(res, prep, {"generated": "t"})), sheet_name=None)
    assert "가상" in str(xls["요약"].iloc[0, 1])
    assert {"등급별", "제외 목록", "안내"} <= set(xls)
