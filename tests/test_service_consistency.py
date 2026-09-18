"""화면·상담이 **같은 숫자를 같은 규칙으로** 말하는지 — 실무자가 제일 먼저 대조하는 곳.

실측으로 났던 결함을 그대로 고정한다.
  · 자기 이름으로 검색해도 그 브랜드가 안 나오던 47개 (한글만 떼어 비교 + 정렬 전 절단)
  · "커피" 한 단어가 별칭을 거쳐 빽다방 한 건으로 좁혀지던 문제
  · 경계 바로 아래 점수가 반올림으로 "16.0% · 관찰"이 되던 51개
  · "가장 위험한 브랜드"에 안정 등급이 섞이던 상담 답변
  · 키 없이 예시 질문을 누르면 "브랜드를 찾지 못했습니다"로 끝나던 상담
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import chat, grading
from src.brand_search import search
from src.common import load_config

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


@pytest.fixture(scope="module")
def scores() -> pd.DataFrame:
    return pd.read_csv(OUT / "scores_latest.csv", encoding="utf-8-sig")


def test_every_brand_finds_itself(scores):
    misses = [nm for nm in scores["brand_name"].astype(str).unique()
              if nm not in set(search(scores, nm)[0]["brand_name"].astype(str))]
    assert misses == []


def test_exact_name_returns_only_that_brand(scores):
    hit, _ = search(scores, "DDC치킨")
    assert hit["brand_name"].tolist() == ["DDC치킨"]


def test_generic_word_ranks_by_store_count_not_alias(scores):
    hit, _ = search(scores, "커피")
    assert len(hit) > 10                                   # 별칭 때문에 한 건으로 좁혀지지 않는다
    n = pd.to_numeric(hit["n_stores"], errors="coerce").fillna(0).tolist()
    assert n == sorted(n, reverse=True)                    # 같은 일치 등급 안에서 가맹점 수 순


def test_displayed_percent_never_crosses_its_grade_cut(scores):
    cuts = grading.cuts(OUT)
    assert cuts is not None
    lo, hi = (c * 100 for c in cuts)
    shown = scores["deterioration_1y"].map(lambda p: grading.display_pct(p, cuts))
    bad_mid = scores[(scores["risk_grade"] == "Medium") & (shown >= hi)]
    bad_low = scores[(scores["risk_grade"] == "Low") & (shown >= lo)]
    assert bad_mid.empty and bad_low.empty


def test_grade_matches_fixed_cuts(scores):
    """화면의 등급 색·문구가 쓰는 경계와 점수표의 등급이 같은 체계인지."""
    lo, hi = grading.cuts(OUT)
    p = scores["deterioration_1y"]
    expect = pd.Series("Low", index=scores.index).mask(p >= lo, "Medium").mask(p >= hi, "High")
    mismatch = (expect != scores["risk_grade"]).mean()
    assert mismatch < 0.001


def test_priority_by_grade_puts_high_first(scores):
    rates = grading.watch_rates(OUT)
    top = grading.prioritize(scores, rates, by_grade=True).head(50)
    order = top["risk_grade"].map(grading.GRADE_ORDER).tolist()
    assert order == sorted(order)


def test_watch_brands_use_realized_rate(scores):
    rates = grading.watch_rates(OUT)
    row = scores[scores["brand_state"] == "요주의"].iloc[0]
    k = int(row["n_events_at_t"])
    assert grading.priority_risk(row, rates) == pytest.approx(rates[k]["rate"])


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


def test_no_key_industry_answer_lists_high_grade_first(cfg, monkeypatch):
    monkeypatch.setattr(chat.llm, "is_enabled", lambda _cfg: False)
    res = chat.answer(cfg, "치킨 업종에서 지금 가장 위험한 브랜드는?", [])
    assert res["intent"] == "industry"
    assert "찾지 못했습니다" not in res["text"]
    table = res["text"].split("| 브랜드 | 등급 |", 1)[1].splitlines()[2:]   # 머리글·구분선 다음 행들
    first_grades = [ln.split("|")[2].strip() for ln in table if ln.startswith("| ")][:3]
    assert first_grades == ["주의"] * 3


def test_no_key_greeting_explains_capabilities(cfg, monkeypatch):
    monkeypatch.setattr(chat.llm, "is_enabled", lambda _cfg: False)
    res = chat.answer(cfg, "안녕하세요", [])
    assert res["text"] == chat.CAPABILITY_TEXT


def test_no_key_trend_answer_has_table(cfg, monkeypatch):
    monkeypatch.setattr(chat.llm, "is_enabled", lambda _cfg: False)
    res = chat.answer(cfg, "달콤왕가탕후루 가맹점 수 추이를 가져와줘", [])
    assert "| 연도 |" in res["text"]
    assert "2,0" not in res["text"].split("| 연도 |")[1].split("\n\n")[0]   # 연도에 쉼표 없음


def test_chat_uses_short_interactive_timeout(cfg):
    lc = chat._interactive(cfg)["llm"]
    assert lc["timeout_sec"] <= 30 and lc["max_retries"] <= 2
