"""실제 여신 장부 올리기 — 읽기·매칭·LGD·꼬리손실이 약속한 대로 동작하는지.

무엇을 고정하나
    · 단위: 열 이름에 적힌 단위가 이기고, 없으면 값 크기로 추정하되 그 근거를 남긴다.
    · 매칭: 화면 검색(src/brand_search.py)과 같은 규칙 — 브랜드ID·정확·통칭 일치만 확정하고,
      나머지는 '확인 필요', 못 찾으면 '미매칭'으로 따로 모은다.
    · LGD: 행에 적힌 값 > 담보유형 가정 > 기본값. 브랜드로 합칠 때는 금액 가중.
    · 꼬리손실: 같은 입력이면 같은 결과(시드 고정), 행 순서와 무관하고, **파이프라인 산출물
      (outputs/correlation_impact.json)을 같은 입력·같은 시드로 재현**한다.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import loanbook as lb
from src.brand_search import search
from src.common import load_config

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
LEVELS = ("p95", "p99", "p999")


@pytest.fixture(scope="module")
def scores() -> pd.DataFrame:
    return pd.read_csv(OUT / "scores_latest.csv", encoding="utf-8-sig")


@pytest.fixture(scope="module")
def all_brands() -> pd.DataFrame | None:
    p = ROOT / "data" / "processed" / "panel_full.parquet"
    return pd.read_parquet(p, columns=["brand_name"]).drop_duplicates() if p.exists() else None


def _csv(text: str, enc: str = "utf-8-sig") -> bytes:
    return text.encode(enc)


def _meta(book: pd.DataFrame) -> dict:
    return book.attrs["loanbook"]


# ---------------------------------------------------------------------------
# 읽기 · 단위
# ---------------------------------------------------------------------------

def test_header_unit_wins_and_is_converted_to_mkrw():
    book = lb.read_book("b.csv", _csv("브랜드명,여신잔액(억원)\n빽다방,45\n이디야커피,12.5\n"))
    m = _meta(book)
    assert (m["brand_col"], m["exposure_col"]) == ("브랜드명", "여신잔액(억원)")
    assert (m["unit"], m["unit_source"]) == ("억원", "열 이름")
    assert book["exposure_mkrw"].tolist() == [4500.0, 1250.0]           # 억원 → 백만원


def test_cp949_export_with_title_row_thousands_separators_and_total_row():
    """은행 시스템 내보내기의 흔한 모양 — 제목 줄, CP949, 천 단위 쉼표, 맨 아래 합계 줄."""
    raw = ("여신 잔액 현황 (2026.08월말)\n\n"
           "브랜드명,대출잔액(원),담보구분\n"
           '빽다방,"4,500,000,000",보증서\n'
           "교촌치킨,2200000000,부동산담보\n"
           "합계,6700000000,\n")
    book = lb.read_book("book.csv", raw.encode("cp949"))
    m = _meta(book)
    assert m["header_row"] == 3 and m["unit"] == "원"
    assert book["brand_input"].tolist() == ["빽다방", "교촌치킨"]      # 합계 줄은 빠진다
    assert book["exposure_mkrw"].tolist() == pytest.approx([4500.0, 2200.0])
    assert m["dropped"]["합계·소계 줄"] == 1
    assert book["collateral"].tolist() == ["보증서", "담보"]
    assert book["src_row"].tolist() == [4, 5]                          # 파일에서 찾아갈 행 번호


@pytest.mark.parametrize(("body", "unit"), [
    ("빽다방,4500000000\n교촌치킨,1200000000\n", "원"),        # 한 줄 1조 원 이상이 될 수 없다
    ("빽다방,4500\n교촌치킨,1200\n굽네치킨,3000\n", "백만원"),  # 억원이면 한 줄 1,000억 원 이상
    ("빽다방,45.5\n교촌치킨,12\n굽네치킨,3.2\n", "억원"),      # 소수점 금액
    ("빽다방,45\n교촌치킨,12\n", "억원"),                      # 백만원이면 브랜드당 1억 원 미만
])
def test_unit_by_magnitude_is_estimated_and_explained(body, unit):
    book = lb.read_book("b.csv", _csv("브랜드명,여신잔액\n" + body))
    m = _meta(book)
    assert m["unit"] == unit
    assert m["unit_source"] == "값 크기 추정" and m["unit_reason"]    # 화면이 가정을 밝힐 근거


def test_borrower_count_decides_ambiguous_unit():
    # 브랜드당 300·120 이면 백만원(3억·1.2억)도 억원(300억·120억)도 말이 된다. 차주 1명당으로
    # 나누면 백만원이면 150만·100만 원이라 소상공인 대출로 말이 안 되고, 억원이면 1.5억·1억이다.
    book = lb.read_book("b.csv", _csv("브랜드명,잔액,차주 수\n빽다방,300,200\n교촌치킨,120,120\n"))
    assert _meta(book)["unit"] == "억원"
    assert book["n_borrowers"].tolist() == [200.0, 120.0]


def test_column_and_unit_can_be_overridden():
    raw = _csv("브랜드명,신청금액,여신잔액\n빽다방,10,45\n교촌치킨,20,12\n")
    assert _meta(lb.read_book("b.csv", raw))["exposure_col"] == "여신잔액"   # 잔액이 신청금액을 이긴다
    book = lb.read_book("b.csv", raw, exposure_col="신청금액", unit="백만원")
    m = _meta(book)
    assert m["exposure_col"] == "신청금액"
    assert (m["unit"], m["unit_source"]) == ("백만원", "사용자 지정")
    assert book["exposure_mkrw"].tolist() == [10.0, 20.0]


def test_unreadable_books_raise_with_columns_for_manual_mapping():
    with pytest.raises(lb.BookReadError) as e:
        lb.read_book("b.csv", _csv("지점,금액(백만원)\n강남,100\n"))
    assert "지점" in e.value.columns                     # 화면이 열을 직접 고르게 한다
    with pytest.raises(lb.BookReadError):
        lb.read_book("b.csv", b"")


def test_template_is_real_brands_with_example_amounts(scores):
    data = lb.template_bytes(scores)
    book = lb.read_book(lb.TEMPLATE_NAME, data)
    m = _meta(book)
    assert m["sheet"] == "여신장부" and m["unit"] == "억원" and m["is_example"]
    assert (m["collateral_col"], m["lgd_col"], m["borrower_col"]) == ("담보유형", "LGD(%, 선택)", "차주 수")
    guide = pd.read_excel(io.BytesIO(data), sheet_name="작성 안내")
    assert "예시(가상)" in guide.iloc[0, 0]
    notes = pd.read_excel(io.BytesIO(data), sheet_name="여신장부")["비고"].astype(str)
    assert notes.str.startswith("예시(가상 금액)").all()
    matched = lb.match_book(book, scores)
    # 이름은 전부 평가 대상 브랜드로 풀린다 — 오타 예시 한 줄만 미매칭(비슷한 이름 제안 포함)
    none = matched[matched["match_status"] == lb.MATCH_NONE]
    assert none["brand_input"].tolist() == ["빽다빵"] and len(none.iloc[0]["candidates"]) > 0
    ok = matched[matched["match_status"] != lb.MATCH_NONE]
    assert ok["brand_id"].isin(scores["brand_id"].astype(str)).all()
    assert float(book.loc[book["brand_input"] == "투다리", "lgd_input"].iloc[0]) == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# 매칭
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matched(scores, all_brands) -> pd.DataFrame:
    raw = ("brand_id,브랜드명,잔액(백만원)\n"
           "BRD_20090100502,,100\n,빽다방,100\n,메가커피,100\n,국수나무,100\n"
           ",컴포즈커피,100\n,빽다빵,100\n,크린토피아,100\n,국수나무,50\n")
    return lb.match_book(lb.read_book("b.csv", _csv(raw)), scores, all_brands=all_brands)


def test_match_statuses_follow_search_rules(matched, all_brands):
    got = {(r.brand_input if isinstance(r.brand_input, str) else r.brand_id_input):
           (r.match_status, r.match_rule) for r in matched.itertuples()}
    assert got["BRD_20090100502"] == (lb.MATCH_OK, "브랜드ID")
    assert got["빽다방"] == (lb.MATCH_OK, "정확 일치")
    assert got["메가커피"] == (lb.MATCH_OK, "통칭 일치")
    assert got["국수나무"] == (lb.MATCH_CHECK, "동명 브랜드")
    assert got["컴포즈커피"] == (lb.MATCH_CHECK, "유사 일치")        # 후보가 하나여도 확정하지 않는다
    assert got["빽다빵"] == (lb.MATCH_NONE, "공시에 없음")
    if all_brands is not None:
        assert got["크린토피아"] == (lb.MATCH_NONE, "평가 대상 아님")
    mega = matched[matched["brand_input"] == "메가커피"].iloc[0]
    assert mega["brand_name"].startswith("메가엠지씨커피")
    dup = matched[matched["brand_input"] == "국수나무"].iloc[0]
    assert len(dup["candidates"]) >= 2
    typo = matched[matched["brand_input"] == "빽다빵"].iloc[0]
    assert "BRD_20090100502" in typo["candidates"]                   # 비슷한 이름을 제안한다


def test_matched_brand_is_what_screen_search_puts_first(matched, scores):
    """'같은 규칙'의 뜻 — 이름으로 찾은 행은 화면 검색의 첫 결과와 같은 브랜드에 붙는다."""
    rows = matched[matched["brand_input"].notna() & matched["brand_id"].notna()]
    for r in rows.itertuples():
        hit, _ = search(scores, r.brand_input)
        if r.match_rule == "통칭 일치":
            assert r.brand_id in set(hit["brand_id"].astype(str))
        else:
            assert r.brand_id == str(hit.iloc[0]["brand_id"])


def test_summary_counts_unmatched_separately(matched):
    s = lb.summarize(matched)
    assert (s["n_ok"], s["n_check"], s["n_none"]) == (3, 3, 2)
    assert s["expo_none_mkrw"] == pytest.approx(200.0)
    assert s["expo_used_mkrw"] == pytest.approx(550.0)
    assert s["n_brands"] == 4                         # 빽다방(ID·이름 같은 브랜드) · 메가 · 국수나무 · 컴포즈


def test_user_fixes_confirm_change_or_drop(matched, scores):
    dup = matched[matched["brand_input"] == "국수나무"].iloc[0]
    other = next(c for c in dup["candidates"] if c != dup["brand_id"])
    fixes = {dup["match_key"]: {"brand_id": other, "reviewed": True},
             matched[matched["brand_input"] == "컴포즈커피"].iloc[0]["match_key"]: {"brand_id": ""},
             matched[matched["brand_input"] == "빽다빵"].iloc[0]["match_key"]:
                 {"brand_id": "BRD_20090100502", "reviewed": True}}
    fixed = lb.apply_fixes(matched, scores, fixes)
    kuk = fixed[fixed["brand_input"] == "국수나무"]
    assert (kuk["brand_id"] == other).all() and (kuk["match_status"] == lb.MATCH_OK).all()
    assert (kuk["match_rule"] == "사용자 확인").all()                 # 사람이 본 것은 따로 남는다
    s = scores.assign(brand_id=scores["brand_id"].astype(str)).set_index("brand_id")
    assert float(kuk["n_stores"].iloc[0]) == float(s.loc[other, "n_stores"])   # 점수도 새 브랜드 것
    assert fixed.loc[fixed["brand_input"] == "컴포즈커피", "match_status"].item() == lb.MATCH_DROP
    assert fixed.loc[fixed["brand_input"] == "빽다빵", "match_status"].item() == lb.MATCH_OK
    bb = lb.brand_book(fixed)
    assert "BRD_20141250" not in set(bb["brand_id"])                  # 제외한 컴포즈커피는 빠진다
    assert bb.loc[bb["brand_id"] == "BRD_20090100502", "exposure_mkrw"].item() == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# 담보유형 · LGD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "cat"), [
    ("보증서", "보증서"), ("보증서담보대출", "보증서"), ("신용보증재단 보증", "보증서"), ("지역신보", "보증서"),
    ("임차보증금 질권", "담보"), ("부동산 근저당", "담보"), ("담보", "담보"),
    ("신용", "신용"), ("무담보", "신용"), ("무보증 신용", "신용"),
    ("", "미기재"), (None, "미기재"), ("기타", "기타"),
])
def test_collateral_keywords(text, cat):
    assert lb.classify_collateral(text) == cat


def test_lgd_by_collateral_row_override_and_brand_weighting(scores):
    raw = ("브랜드명,여신잔액(백만원),담보유형,LGD(%)\n"
           "빽다방,100,보증서,\n빽다방,100,신용,\n교촌치킨,100,담보,30\n굽네치킨,100,,\n")
    m = lb.match_book(lb.read_book("b.csv", _csv(raw)), scores)
    lgd, src = lb.row_lgd(m)
    assert lgd.tolist() == pytest.approx([0.10, 0.45, 0.30, 0.45])
    assert src.tolist() == ["담보유형 가정", "담보유형 가정", "행 입력", "기본값"]
    bb = lb.brand_book(m).set_index("brand_name")
    assert bb.loc["빽다방", "lgd"] == pytest.approx((100 * 0.10 + 100 * 0.45) / 200)
    assert bb.loc["빽다방", "exposure_mkrw"] == pytest.approx(200.0)
    assert bb.loc["빽다방", "collateral_mix"] == "보증서 50% · 신용 50%"
    assert bb.loc["교촌치킨", "lgd"] == pytest.approx(0.30)
    assert bb.loc["굽네치킨", "lgd"] == pytest.approx(0.45)             # 모르면 없다고 본다
    # 차주 수가 없으면 가맹점 수 가정 — 그 사실이 표에 남는다
    assert (bb["n_borrowers_basis"] == "가맹점 수 가정").all()
    assert bb.loc["빽다방", "n_borrowers"] == float(bb.loc["빽다방", "n_stores"])
    custom = lb.brand_book(m, {"보증서": 0.20}, 0.60).set_index("brand_name")
    assert custom.loc["빽다방", "lgd"] == pytest.approx((0.20 + 0.45) / 2)
    assert custom.loc["굽네치킨", "lgd"] == pytest.approx(0.60)
    assert custom.loc["교촌치킨", "lgd"] == pytest.approx(0.30)          # 행 입력은 가정보다 우선


def test_recovery_rate_column_becomes_lgd():
    book = lb.read_book("b.csv", _csv("브랜드명,잔액(억원),회수율(%)\n빽다방,10,70\n"))
    assert book["lgd_input"].tolist() == pytest.approx([0.30])


def test_lgd_assumptions_come_from_config():
    lmap, ldef = lb.lgd_assumptions(load_config())
    assert lmap == pytest.approx({"신용": 0.45, "담보": 0.25, "보증서": 0.10}) and ldef == pytest.approx(0.45)
    lmap2, ldef2 = lb.lgd_assumptions({"loanbook": {"lgd_by_collateral": {"담보": 0.3}, "lgd_default": 0.5}})
    assert lmap2["담보"] == pytest.approx(0.3) and lmap2["신용"] == pytest.approx(0.45)
    assert ldef2 == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 꼬리손실
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def template_brands(scores) -> pd.DataFrame:
    book = lb.read_book(lb.TEMPLATE_NAME, lb.template_bytes(scores))
    return lb.brand_book(lb.match_book(book, scores))


def _quantiles(r: dict) -> list[float]:
    return [r[f"{lab}_{lvl}_mkrw"] for lab in ("independent", "brand_correlated") for lvl in LEVELS]


def test_tail_risk_is_deterministic_and_row_order_free(template_brands, scores):
    a = lb.tail_risk(template_brands, scores, 0.40, 0.005, n_sims=5000, seed=7)
    b = lb.tail_risk(template_brands, scores, 0.40, 0.005, n_sims=5000, seed=7)
    assert _quantiles(a) == _quantiles(b)
    pd.testing.assert_frame_equal(a["contrib"], b["contrib"])
    shuffled = template_brands.sample(frac=1.0, random_state=3)
    c = lb.tail_risk(shuffled, scores, 0.40, 0.005, n_sims=5000, seed=7)
    assert _quantiles(c) == _quantiles(a)                      # 같은 장부면 행 순서와 무관
    d = lb.tail_risk(template_brands, scores, 0.40, 0.005, n_sims=5000, seed=8)
    assert _quantiles(d) != _quantiles(a)
    assert a["euler"]["component_addup_rel_err"] < 1e-9        # 성분 ES 의 합 = 전체 ES
    assert a["contrib"]["ul_share"].sum() == pytest.approx(1.0)
    assert not a["lgd_uniform"]                                # 담보별 LGD 경로를 탔다
    # 상관을 넣으면 평균은 그대로, 꼬리만 두꺼워진다
    assert a["brand_correlated_p99_mkrw"] > a["independent_p99_mkrw"]
    assert a["brand_correlated_mean_mkrw"] == pytest.approx(a["independent_mean_mkrw"], rel=0.05)


def test_tail_risk_fills_missing_inputs_from_scores(scores):
    book = pd.DataFrame({"brand_id": ["BRD_20090100502", "BRD_20080600002", "NOPE_1"],
                         "exposure_mkrw": [5000.0, 2000.0, 100.0]})
    r = lb.tail_risk(book, scores, 0.40, 0.005, n_sims=2000, seed=1)
    assert r["n_brands"] == 2 and r["excluded_brands"] == ["NOPE_1"]   # 위험도 없는 브랜드는 빠진다
    assert r["lgd"] == pytest.approx(lb.DEFAULT_LGD)
    s = scores.assign(brand_id=scores["brand_id"].astype(str)).set_index("brand_id")
    assert r["n_franchisees"] == int(s.loc["BRD_20090100502", "n_stores"] + s.loc["BRD_20080600002", "n_stores"])


def test_book_hash_tracks_inputs_and_params(template_brands):
    h = lb.book_hash(template_brands, 0.4, 0.005, 20000, 42)
    assert h == lb.book_hash(template_brands.copy(), 0.4, 0.005, 20000, 42)
    moved = template_brands.copy()
    moved.loc[0, "exposure_mkrw"] += 1.0
    assert lb.book_hash(moved, 0.4, 0.005, 20000, 42) != h
    assert lb.book_hash(template_brands, 0.4, 0.005, 20000, 43) != h


@pytest.fixture(scope="module")
def pipeline() -> tuple[pd.DataFrame, dict]:
    ci = json.loads((OUT / "correlation_impact.json").read_text(encoding="utf-8"))
    port = pd.read_csv(OUT / "portfolio.csv")
    fp = ci["input_fingerprint"]
    if (round(float(port["deterioration_1y"].sum()), 9) != fp["det_sum"]
            or round(float(port["exposure_mkrw"].sum()), 6) != fp["exposure_sum_mkrw"]):
        pytest.skip("portfolio.csv 가 correlation_impact.json 이후 다시 만들어졌다 — 대조 대상이 다르다")
    return port, ci


def test_reproduces_pipeline_correlation_impact_bit_for_bit(pipeline):
    """파이프라인과 **같은 입력·같은 난수 배정 순서·같은 시드(20만 회)** 면 같은 수가 나와야 한다.

    tail_risk 는 수식을 다시 쓰지 않고 src/correlation 의 함수를 그대로 부른다. 그 주장이 참이면
    산출물이 부동소수점 반올림 수준까지 재현된다 — 수식이 한 군데라도 갈라지면 여기서 드러난다.
    """
    port, ci = pipeline
    r = lb.tail_risk(port, None, ci["rho_within_brand"], ci["rho_between_brand"],
                     n_sims=int(ci["n_sims"]), seed=int(ci["input_fingerprint"]["seed"]),
                     lgd=float(ci["lgd"]), keep_order=True)
    for lab in ("independent", "brand_correlated"):
        for key in (*LEVELS, "mean"):
            k = f"{lab}_{key}_mkrw"
            assert r[k] == pytest.approx(ci[k], rel=1e-9), k
    assert r["ul99_multiple"] == pytest.approx(ci["ul99_multiple"], rel=1e-9)
    eu = ci["euler_allocation"]
    assert r["euler"]["n_tail_scenarios"] == eu["n_tail_scenarios"]
    assert r["euler"]["es99_total_mkrw"] == pytest.approx(eu["es99_total_mkrw"], rel=1e-9)
    old = pd.read_csv(OUT / "brand_ul_contribution.csv", encoding="utf-8-sig")
    both = r["contrib"].merge(old, on="brand_id", suffixes=("", "_pipe"))
    assert len(both) == len(old)
    assert np.allclose(both["ul_contrib_mkrw"], both["ul_contrib_mkrw_pipe"], rtol=1e-9, atol=1e-6)


def test_matches_pipeline_within_monte_carlo_error_in_default_order(pipeline):
    """화면이 쓰는 기본 순서(brand_id 정렬)는 난수 배정이 달라 비트 단위로는 다르다.

    허용오차 — 두 추정이 서로 다른 난수로 같은 분위수를 잰 것이므로 차이의 표준오차는 각자
    표준오차의 √2 배다. 파이프라인이 스스로 보고한 몬테카를로 표준오차(mc_se_*)로 4σ 를 준다
    (정규 근사에서 우연히 넘을 확률 약 0.006%). 상관 99% 분위수 기준 약 51백만원(±1.9%)이라,
    LGD·상관·차주 분할이 틀리면(수십 % 차이) 반드시 걸리고 몬테카를로 잡음으로는 걸리지 않는다.
    """
    port, ci = pipeline
    r = lb.tail_risk(port, None, ci["rho_within_brand"], ci["rho_between_brand"],
                     n_sims=int(ci["n_sims"]), seed=int(ci["input_fingerprint"]["seed"]),
                     lgd=float(ci["lgd"]))
    for lab, se_lab in (("independent", "independent"), ("brand_correlated", "correlated")):
        for lvl in LEVELS:
            k = f"{lab}_{lvl}_mkrw"
            tol = 4.0 * np.sqrt(2.0) * float(ci[f"mc_se_{se_lab}_{lvl}_mkrw"])
            assert abs(r[k] - ci[k]) <= tol, (k, r[k], ci[k], tol)


def test_interactive_speed_60_brands_20k_sims(pipeline):
    """화면은 실행·회수 한 번마다 다시 계산한다 — 60개 브랜드 · 20,000회가 3초 안이어야 한다."""
    port, ci = pipeline
    best = min(lb.tail_risk(port, None, ci["rho_within_brand"], ci["rho_between_brand"],
                            n_sims=20_000, seed=42, lgd=0.45)["elapsed_sec"] for _ in range(2))
    assert best < 3.0, best


# ---------------------------------------------------------------------------
# 화면 — 업로드 경로가 예외 없이 끝까지 도는지 (파일 올리기 위젯 대신 세션 상태로 주입)
# ---------------------------------------------------------------------------

def test_portfolio_screen_switches_to_uploaded_book(monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FRANSCORE_QUEUE_STORE", "session")
    data = lb.template_bytes()
    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=180)
    at.session_state["nav_view"] = "여신 포트폴리오"
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("공시 창업비용 기반 추정" in m.value for m in at.markdown)

    at.session_state["pf_upload"] = {"name": "장부.xlsx", "data": data,
                                     "sha": hashlib.sha256(data).hexdigest(),
                                     "at": "2026-09-19 09:00", "example": False}
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    labels = [b.label for b in at.button]
    assert "이 장부로 화면 전환" in labels
    at.button(key="pf_confirm_all").click().run()               # 확인 필요 후보를 모두 확정
    assert not at.exception, [e.message for e in at.exception]
    assert not any(b.key == "pf_confirm_all" for b in at.button)

    next(b for b in at.button if b.label == "이 장부로 화면 전환").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("은행 제공 장부" in m.value and "장부.xlsx" in m.value for m in at.markdown)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["총 여신"] == "240 억"                        # 미매칭(빽다빵 5억)은 빠진다
    assert "대리 예상손실 (담보별 LGD)" in metrics
    assert "99% 손실 · 브랜드 상관 반영" in metrics             # 꼬리손실도 올린 장부로 다시 계산


def test_unreadable_upload_offers_manual_column_choice(monkeypatch):
    """브랜드 열을 못 찾은 파일은 화면이 죽지 않고, 발견한 열을 보여 주며 직접 고르게 한다."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FRANSCORE_QUEUE_STORE", "session")
    bad = _csv("지점,금액(백만원)\n빽다방,100\n")
    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=180)
    at.session_state["nav_view"] = "여신 포트폴리오"
    at.session_state["pf_upload"] = {"name": "bad.csv", "data": bad,
                                     "sha": hashlib.sha256(bad).hexdigest(),
                                     "at": "2026-09-19 09:00", "example": False}
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("브랜드 열을 찾지 못했습니다" in e.value for e in at.error)
    at.selectbox(key="pf_opt_brand").set_value("지점").run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error                                           # 고른 열로 다시 읽었다
    assert "이 장부로 화면 전환" in [b.label for b in at.button]
