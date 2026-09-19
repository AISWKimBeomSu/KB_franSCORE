"""등급·표시·점검순서 규칙 — 화면과 상담이 **같은 규칙**을 쓰게 하는 한 곳.

왜 따로 뺐나 (실측)
    등급 경계·확률 표기·점검 순서가 화면(src/views)과 상담(src/chat.py)에 따로 구현돼
    있었다. 화면만 고치면 상담이 옛 규칙으로 답했다 — 화면은 "15.9% · 관찰"로 고쳤는데
    상담 표는 같은 브랜드를 "16.0% · 관찰"로 적었고, "가장 위험한 브랜드"를 물으면
    안정 등급 브랜드가 2·3위에 올랐다. 규칙을 한 곳에 두고 양쪽이 불러 쓴다.
    streamlit 을 import 하지 않는다 — 배치·테스트에서도 그대로 쓴다.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

GRADE_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def year_label(year, *, short: bool = False) -> str:
    """패널 연도 표기 — '2024년 실적 · 2025년 정보공개서'.

    패널 연도는 **실적연도**다(정보공개서 기준연도 − 1, src/panel.py 상단 규칙). 예전 화면은
    이 값을 '2024년 공시'로 적었다. 실무자가 그 말대로 2024년 정보공개서(2023년 실적)를 열면
    가맹점 수가 1년씩 어긋난다(메가MGC커피: 패널 2024 = 3,325개 = 2024년 말).
    """
    try:
        y = int(float(year))
    except (TypeError, ValueError):
        return "-"
    return f"{y}년 실적" if short else f"{y}년 실적 · {y + 1}년 정보공개서"
GRADE_KR = {"High": "주의", "Medium": "관찰", "Low": "안정"}

_CACHE: dict[tuple[str, float], object] = {}


def _cached_json(path: Path) -> dict:
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return {}
    if key not in _CACHE:
        try:
            _CACHE[key] = json.loads(path.read_text(encoding="utf-8-sig")) or {}
        except (OSError, ValueError):
            _CACHE[key] = {}
    obj = _CACHE[key]
    return obj if isinstance(obj, dict) else {}


def cuts(outputs: Path) -> tuple[float, float] | None:
    """등급 경계 (관찰 시작, 주의 시작) — 0~1 비율. 공표 밴드가 없으면 None."""
    c = _cached_json(Path(outputs) / "grade_bands.json").get("cuts") or []
    return (float(c[0]), float(c[1])) if len(c) == 2 else None


def display_pct(p, grade_cuts: tuple[float, float] | None) -> float | None:
    """화면·답변에 적을 확률(%) — 반올림이 등급 경계를 넘지 않게 한다.

    경계 바로 아래(예: 15.97%)로 잘린 점수를 소수 첫째 자리로 반올림하면 "16.0% · 관찰"
    이 돼 숫자와 등급이 모순돼 보인다(실측 51개 브랜드). 그런 값만 내림으로 적는다.
    """
    v = pd.to_numeric(pd.Series([p]), errors="coerce").iloc[0]
    if pd.isna(v):
        return None
    v = float(v) * 100
    shown = round(v, 1)
    for cut in (grade_cuts or ()):
        c = cut * 100
        if v < c <= shown:
            return math.floor(v * 10) / 10
    return shown


def watch_rates(outputs: Path) -> dict[int, dict]:
    """악화 사건수별 다음 해 재발동 실현율 (`tools/watch_base_rates.py` 산출)."""
    obj = _cached_json(Path(outputs) / "watch_base_rates.json")
    out = {}
    for r in obj.get("table") or []:
        out[int(r["n_events_at_t"])] = {
            "rate": float(r["rate"]), "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"]), "n": int(r["n"]), "state": str(r["state"])}
    return out


def watch_hit(row, rates: dict[int, dict]) -> dict | None:
    """악화 발생('요주의') 브랜드면 같은 사건수 과거 브랜드의 재발동 실현율. 아니면 None."""
    if str(row.get("brand_state")) != "요주의":
        return None
    k = pd.to_numeric(pd.Series([row.get("n_events_at_t")]), errors="coerce").iloc[0]
    return None if pd.isna(k) else rates.get(int(k))


def priority_risk(row, rates: dict[int, dict]) -> float:
    """점검 순서용 1년 내 악화 위험 — 건전은 모형 확률, 악화 발생은 재발동 실현율.

    악화 발생 구간은 학습 표본 밖이라 `MODEL_USE_SPEC` 이 확률로 줄세우기를 금지한다.
    둘 다 '1년 내 악화' 비율이라 같은 축에서 비교할 수 있다.
    """
    hit = watch_hit(row, rates)
    if hit:
        return float(hit["rate"])
    v = pd.to_numeric(pd.Series([row.get("deterioration_1y")]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(v) else float(v)


def prioritize(df: pd.DataFrame, rates: dict[int, dict], *, by_grade: bool = False) -> pd.DataFrame:
    """점검 우선순위 = 위험(priority_risk) × 가맹점 수.

    by_grade=True 면 등급(주의→관찰→안정)을 먼저 본다 — "가장 위험한 브랜드"를 물었는데
    가맹점이 많다는 이유로 안정 등급이 앞에 오면 질문에 답하지 않은 것이다.
    """
    v = df.copy()
    if v.empty:
        return v.assign(_risk=pd.Series(dtype=float), _pri=pd.Series(dtype=float))
    v["_risk"] = v.apply(lambda r: priority_risk(r, rates), axis=1)
    v["_pri"] = v["_risk"] * pd.to_numeric(v["n_stores"], errors="coerce").fillna(0)
    keys, asc = ["_pri"], [False]
    if by_grade and "risk_grade" in v.columns:
        v["_g"] = v["risk_grade"].map(GRADE_ORDER).fillna(9)
        keys, asc = ["_g", "_pri"], [True, False]
    out = v.sort_values(keys, ascending=asc, kind="stable")
    return out.drop(columns=["_g"], errors="ignore")
