"""참고의견서(src/report.py)와 권고 문구(src/guidance.py) 검증.

검사 항목
  1. 규칙 엔진이 낼 수 있는 소견 코드 전부, 그리고 실제 산출물에 나온 코드 전부가
     확인·징구 항목을 갖는가 (새 규칙이 체크리스트에서 조용히 빠지지 않게)
  2. 권고·확인 문구에 금지 용도(승인·거절·한도·금리)가 섞이지 않았는가 (MODEL_USE_SPEC §3)
  3. 중대 신호 추출이 입력 형태(DataFrame·dict·Finding·코드)와 무관하게 같은 답을 내는가
  4. FS3·중대 신호 브랜드 문서 — 등급 구간이 grade_bands.json 컷과 같은 문자열이고,
     고지문이 있고, 외부 자원이 없는가
  5. 본부 재무 미매칭 브랜드 — '확인되지 않음'으로만 쓰고 원인(외부감사 비대상)을 단정하지 않는가
  6. 따옴표가 든 브랜드명이 이스케이프되는가, 원천에 엔티티가 든 이름이 이중 이스케이프되지 않는가
  7. 경계 반올림 방지, 최소 ctx 로도 문서가 만들어지는가

네트워크·API 키 없이 저장소의 산출물만 읽는다.
실행: python -m pytest -q tests/test_report.py
"""
from __future__ import annotations

import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import guidance as G
from src import report as R

OUT = ROOT / "outputs"
_ARTIFACTS = ("scores_latest.csv", "brand_diagnosis.parquet", "grade_bands.json")
needs_artifacts = pytest.mark.skipif(
    not all((OUT / f).exists() for f in _ARTIFACTS), reason="산출물(outputs/) 없음 — 파이프라인 실행 후 검사")

FORBIDDEN_USE = ("승인", "거절", "한도", "금리")
# 감사의견 유형 '의견거절'(disclaimer of opinion)은 회계 용어다 — 여신 거절과 무관하므로 검사에서 뺀다.
AUDIT_TERMS = ("의견거절",)
# 매칭 실패를 '외부감사 대상이 아니다'로 단정하던 문장(diagnosis HQ_NO_DATA 원문·화면 문구)
NOT_AUDITED_CLAIM = "외부감사 대상이 아"


# ---------------------------------------------------------------------------
# 공용
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scores() -> pd.DataFrame:
    return pd.read_csv(OUT / "scores_latest.csv", encoding="utf-8-sig")


@pytest.fixture(scope="module")
def findings() -> pd.DataFrame:
    return pd.read_parquet(OUT / "brand_diagnosis.parquet")


@pytest.fixture(scope="module")
def bands() -> dict:
    return json.loads((OUT / "grade_bands.json").read_text(encoding="utf-8-sig"))


def _esc(s: str) -> str:
    """문서가 쓰는 표기 — 원천 엔티티를 푼 뒤 이스케이프."""
    return html.escape(html.unescape(str(s)), quote=True)


def _band_strings(cuts: list[float]) -> dict[str, str]:
    lo, hi = cuts[0] * 100, cuts[1] * 100
    return {"FS1": f"{lo:.1f}% 미만", "FS2": f"{lo:.1f}~{hi:.1f}%", "FS3": f"{hi:.1f}% 이상"}


def _body(doc: str) -> str:
    """<style> 을 뺀 본문."""
    return re.sub(r"<style>.*?</style>", "", doc, flags=re.S)


def _scrub_audit_terms(text: str) -> str:
    for term in AUDIT_TERMS:
        text = text.replace(term, "")
    return text


def _render(bid: str) -> tuple[dict, str]:
    ctx = R.load_brand_context(bid)
    return ctx, R.build_brand_report_html(ctx)


def _assert_document_contract(doc: str, bands: dict) -> None:
    """모든 참고의견서가 지켜야 하는 것."""
    assert doc.startswith("<!DOCTYPE html>")
    assert R.DOC_TITLE in doc
    assert R.SCOPE_LINE in doc, "머리말 사용 범위 문구가 없다"
    assert R.DISCLAIMER in doc, "사용 범위 고지문이 없다"
    assert NOT_AUDITED_CLAIM not in doc, "확인하지 않은 원인(외부감사 비대상)을 단정했다"
    # 등급은 grade_bands.json 의 고정 컷으로만 설명한다 — 순위 백분위 표기가 섞이면 안 된다.
    for band in _band_strings(bands["cuts"]).values():
        assert band in doc, f"등급 구간 '{band}' 가 문서에 없다"
    assert "상위 10%가 주의" not in doc
    # 자기완결 문서 — 외부 스크립트·스타일·이미지·폰트를 부르지 않는다.
    low = doc.lower()
    for bad in ("<script", "<link", "@import", "url("):
        assert bad not in low, f"외부 자원 참조 '{bad}'"
    assert not re.search(r"\ssrc\s*=", low), "이미지 등 외부 src 참조"
    assert "@page" in doc and "@media print" in doc, "A4 인쇄 규칙이 없다"
    assert "/*" not in doc, "CSS 주석이 문서에 실려 나간다 — 본문 문구 검사를 속인다"


# ---------------------------------------------------------------------------
# 1~3. 권고 문구 데이터
# ---------------------------------------------------------------------------

def test_every_rule_code_has_check() -> None:
    """규칙 엔진 소스에 있는 모든 코드 + 뉴스 유형별 코드가 확인 항목을 갖는다."""
    src = (ROOT / "src" / "diagnosis.py").read_text(encoding="utf-8")
    rule_codes = set(re.findall(r'code="([A-Z_]+)"', src))
    assert len(rule_codes) >= 33, f"규칙 코드 추출이 이상하다: {sorted(rule_codes)}"
    from src.news_llm import EVENT_TYPES
    news_codes = {f"NEWS_{t}" for t in EVENT_TYPES if t != "무관"}
    missing = sorted((rule_codes | news_codes) - set(G.FINDING_CHECKS))
    assert not missing, f"확인 항목이 없는 소견 코드: {missing}"
    for code, entry in G.FINDING_CHECKS.items():
        assert str(entry.get("check") or "").strip(), f"{code}: check 비어 있음"
        assert isinstance(entry.get("docs"), list), f"{code}: docs 는 목록이어야 한다"
        assert all(isinstance(d, str) and d.strip() for d in entry["docs"]), f"{code}: 빈 서류명"
    # 새 뉴스 유형이 생겨도 체크리스트에서 빠지지 않는다
    assert G.check_for("NEWS_새로운유형") == G.FINDING_CHECKS["NEWS_기타"]
    assert G.check_for("NO_SUCH_CODE") is None


@needs_artifacts
def test_every_emitted_code_has_check(findings: pd.DataFrame) -> None:
    """실제 산출물(brand_diagnosis.parquet)에 나온 코드 전부가 확인 항목을 갖는다."""
    emitted = set(findings["code"].dropna().astype(str))
    missing = sorted(emitted - set(G.FINDING_CHECKS))
    assert not missing, f"확인 항목이 없는 소견 코드: {missing}"
    assert set(G.FINDING_CHECKS) >= G.CRITICAL_CODES


def test_guidance_wording_avoids_forbidden_uses() -> None:
    """권고·확인 문구가 한도·금리·승인·거절을 말하지 않는다 (MODEL_USE_SPEC §3)."""
    texts = {f"GRADE_ACTIONS[{k}]": v for k, v in G.GRADE_ACTIONS.items()}
    for code, entry in G.FINDING_CHECKS.items():
        texts[f"{code}.check"] = entry["check"]
        for i, d in enumerate(entry["docs"]):
            texts[f"{code}.docs[{i}]"] = d
    texts.update({f"CRITICAL_RATIONALE[{k}]": v for k, v in G.CRITICAL_RATIONALE.items()})
    bad = [(where, w) for where, t in texts.items() for w in FORBIDDEN_USE if w in _scrub_audit_terms(t)]
    assert not bad, f"금지 용도 표현: {bad}"
    assert set(G.GRADE_ACTIONS) == {"FS1", "FS2", "FS3"}
    for risk, fs in G.GRADE_FROM_RISK.items():
        assert G.grade_action(risk) == G.grade_action(fs) == G.GRADE_ACTIONS[fs]
    assert G.grade_action(None) == "" and G.grade_action("??") == ""


def test_critical_findings_accepts_every_shape() -> None:
    """DataFrame·dict 목록·Finding 객체·코드 문자열 어느 쪽이든 같은 코드를 같은 순서로."""
    codes = ["STORE_DECLINE", "HQ_CAPITAL_IMPAIRED", "HQ_AUDIT_OPINION", "HQ_NO_DATA", "HQ_GOING_CONCERN",
             "NEWS_재무이슈"]
    expected = ["HQ_GOING_CONCERN", "HQ_CAPITAL_IMPAIRED"]   # 무거운 순서. 보도·자동판독 감사의견은 제외

    @dataclass
    class _F:
        code: str

    df = pd.DataFrame({"code": codes, "title": codes})
    assert [r["code"] for r in G.critical_findings(df)] == expected
    assert [r["code"] for r in G.critical_findings(df.to_dict("records"))] == expected
    assert [f.code for f in G.critical_findings([_F(c) for c in codes])] == expected
    assert G.critical_findings(codes) == expected
    assert G.critical_findings(None) == [] and G.critical_findings(pd.DataFrame()) == []
    assert frozenset(G.CRITICAL_RATIONALE) == G.CRITICAL_CODES


def test_checklist_dedupes_and_keeps_order() -> None:
    items = G.checklist(["HQ_GOING_CONCERN", "NEWS_새유형A", "NEWS_기타", "HQ_GOING_CONCERN", "NOPE"])
    assert [it["check"] for it in items] == [G.FINDING_CHECKS["HQ_GOING_CONCERN"]["check"],
                                             G.FINDING_CHECKS["NEWS_기타"]["check"]]
    assert items[1]["codes"] == ["NEWS_새유형A", "NEWS_기타"]


def test_modules_do_not_import_streamlit() -> None:
    """배치·테스트에서 같은 문서를 만들려면 화면 프레임워크에 기대면 안 된다."""
    for rel in ("src/report.py", "src/guidance.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+streamlit", text, re.M), f"{rel} 가 streamlit 을 import"


# ---------------------------------------------------------------------------
# 4~6. 실제 브랜드 문서
# ---------------------------------------------------------------------------

@needs_artifacts
def test_report_fs3_brand_with_findings(scores: pd.DataFrame, findings: pd.DataFrame,
                                        bands: dict) -> None:
    fs3 = set(scores.loc[scores["grade"] == "FS3", "brand_id"].astype(str))
    crit = set(findings.loc[findings["code"].isin(G.CRITICAL_CODES), "brand_id"].astype(str))
    cand = sorted(fs3 & crit) or sorted(fs3 & set(findings["brand_id"].astype(str)))
    if not cand:
        pytest.skip("FS3 이면서 소견이 있는 브랜드가 없다")
    bid = cand[0]
    ctx, doc = _render(bid)
    _assert_document_contract(doc, bands)

    assert "FS3 주의" in doc
    band = _band_strings(bands["cuts"])["FS3"]
    assert f"업종 하위구간 악화 전환 확률 {band}" in doc, "결론 요약의 등급 구간이 컷과 다르다"
    assert _esc(G.GRADE_ACTIONS["FS3"]) in doc, "권고 처리가 guidance 와 다르다"

    mine = findings[findings["brand_id"].astype(str) == bid]
    for title in mine["title"]:
        assert _esc(title) in doc, f"소견 제목 누락: {title}"
    for code in mine["code"]:
        assert _esc(G.check_for(code)["check"]) in doc, f"{code} 의 확인 항목이 체크리스트에 없다"
    assert "☐" in doc

    # 문구 검사는 <style> 밖에서만 한다 — CSS 속 글자가 본문 검사를 통과시키지 않게.
    body = _body(doc)
    n_crit = int(mine["code"].isin(G.CRITICAL_CODES).sum())
    if n_crit:
        assert "<div class='critical'>" in body and f"중대 신호 {n_crit}건" in body
        assert body.index("<div class='critical'>") < body.index("핵심 소견"), "중대 신호는 소견 목록보다 앞"
    # 중대 신호가 없는 브랜드에는 상자가 없다
    calm = sorted(set(findings["brand_id"].astype(str))
                  - set(findings.loc[findings["code"].isin(G.CRITICAL_CODES), "brand_id"].astype(str)))
    if calm:
        assert "<div class='critical'>" not in _body(_render(calm[0])[1])

    # 검증 노트는 시점 밖(out_of_time) 실현율을 싣는다
    for r in bands["out_of_time"]["by_grade"]:
        assert f"{float(r['rate']) * 100:.1f}%" in doc
    # 요주의면 사건 수 실현율(watch_base_rates.json)을 싣는다
    b = ctx["brand"]
    if b.get("brand_state") == "요주의":
        rates = {int(r["n_events_at_t"]): r for r in ctx["watch"]["table"]}
        k = int(b["n_events_at_t"])
        if k in rates:
            assert f"{float(rates[k]['rate']) * 100:.1f}%" in doc


@needs_artifacts
def test_report_brand_without_hq_financials(findings: pd.DataFrame, bands: dict) -> None:
    raw_detail = str(findings.loc[findings["code"] == "HQ_NO_DATA", "detail"].iloc[0]) \
        if (findings["code"] == "HQ_NO_DATA").any() else ""
    for bid in sorted(set(findings.loc[findings["code"] == "HQ_NO_DATA", "brand_id"].astype(str)))[:40]:
        ctx, doc = _render(bid)
        if ctx["hq"]["matched"]:
            continue            # 소견은 직전연도까지만 본다 — 최신 결산이 매칭된 브랜드는 건너뛴다
        _assert_document_contract(doc, bands)
        assert "확인되지 않음 (공시 매칭 불가)" in doc
        assert _esc(G.FINDING_CHECKS["HQ_NO_DATA"]["check"]) in doc
        if raw_detail:
            assert _esc(raw_detail) not in doc, "규칙 원문(원인 단정)이 그대로 실렸다"
        return
    pytest.skip("본부 재무가 매칭되지 않은 브랜드가 없다")


@needs_artifacts
def test_report_escapes_quoted_brand_name(scores: pd.DataFrame, bands: dict) -> None:
    names = scores[scores["brand_name"].astype(str).str.contains("['\"]", regex=True)]
    if names.empty:
        pytest.skip("따옴표가 든 브랜드명이 없다")
    pref = names[names["brand_name"].astype(str).str.contains("GGUL'S", regex=False)]
    row = (pref if not pref.empty else names).iloc[0]
    name = str(row["brand_name"])
    _, doc = _render(str(row["brand_id"]))
    _assert_document_contract(doc, bands)
    assert html.escape(name, quote=True) in doc
    assert name not in doc, "따옴표가 이스케이프되지 않은 채 나갔다"
    assert f"<title>{R.DOC_TITLE} — {html.escape(name, quote=True)}</title>" in doc


@needs_artifacts
def test_report_does_not_double_escape_source_entities(scores: pd.DataFrame) -> None:
    """공시 원천에 이미 엔티티가 든 이름("Han&#39;s", "bread&amp;co")이 인쇄물에 글자로 찍히지 않는다."""
    ent = scores[scores["brand_name"].astype(str).str.contains(r"&(?:#\d+|amp|quot|#x[0-9a-f]+);", regex=True)]
    if ent.empty:
        pytest.skip("엔티티가 든 브랜드명이 없다")
    row = ent.iloc[0]
    _, doc = _render(str(row["brand_id"]))
    assert "&amp;#" not in doc and "&amp;amp;" not in doc
    assert _esc(row["brand_name"]) in doc


@needs_artifacts
def test_load_brand_context_unknown_brand() -> None:
    with pytest.raises(KeyError):
        R.load_brand_context("NO_SUCH_BRAND_ID")


# ---------------------------------------------------------------------------
# 7. 합성 ctx
# ---------------------------------------------------------------------------

_CUTS = {"cuts": [0.045, 0.16]}


def test_risk_pct_never_rounds_across_grade_cut() -> None:
    """15.97% 관찰(FS2)이 '16.0%'로 찍혀 주의 경계와 모순돼 보이면 안 된다."""
    doc = R.build_brand_report_html({"brand": {"brand_id": "T1", "brand_name": "경계", "grade": "FS2",
                                               "deterioration_1y": 0.15996}, "bands": _CUTS})
    assert "15.9<small>%</small>" in doc and "16.0<small>%</small>" not in doc
    doc = R.build_brand_report_html({"brand": {"brand_id": "T2", "brand_name": "보통", "grade": "FS3",
                                               "deterioration_1y": 0.2046}, "bands": _CUTS})
    assert "20.5<small>%</small>" in doc


def test_minimal_ctx_renders_and_escapes() -> None:
    """화면이 일부 키만 넘겨도 문서가 나오고, 동적 문자열은 모두 이스케이프된다."""
    evil = "<script>alert('x')</script>\"&"
    doc = R.build_brand_report_html({
        "brand": {"brand_id": "T3", "brand_name": evil, "risk_grade": "High"},
        "findings": pd.DataFrame([{"code": "HQ_NO_DATA", "category": "재무", "severity": "Low",
                                   "direction": "info", "title": evil, "detail": evil,
                                   "source": evil},
                                  {"code": "STORE_DECLINE", "category": evil, "severity": evil,
                                   "direction": "risk", "title": "가맹점이 줄었습니다", "detail": evil,
                                   "source": evil}]),
        "bands": _CUTS})
    assert "<script" not in doc.lower()
    assert html.escape(evil, quote=True) in doc
    assert "FS3 주의" in doc                          # risk_grade(High) → FS3
    assert "확인되지 않음 (공시 매칭 불가)" in doc      # 본부 재무 키가 없으면 미확인으로
    assert R.DISCLAIMER in doc and R.SCOPE_LINE in doc
    assert NOT_AUDITED_CLAIM not in doc
