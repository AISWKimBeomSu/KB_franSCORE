"""브랜드 탐색·비교 — 조건으로 거르고, 여러 브랜드를 나란히 놓고 본다.

왜 필요한가
    목록 화면은 '지금 봐야 할 8개'를 보여 준다. 그런데 실무 질문은 대개 조건에서 시작한다 —
    "치킨 업종에서 가맹점 100개 이상인데 관찰 이상인 곳", "중대 신호가 있는 브랜드 전부".
    그리고 신규 협약·신규 취급 검토는 **후보 브랜드 몇 개를 나란히** 놓고 한다.
    검색창에서 하나씩 열어 메모장에 옮겨 적게 하면 도구가 일을 늘린다.
"""
from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import grading, theme
from src.views import common as C

_MAX_COMPARE = 4


def render() -> None:
    df, meta = C.load_scores()
    if df is None:
        st.warning("아직 평가 결과가 없습니다.")
        return
    theme.page_header(
        "브랜드 탐색·비교",
        f"{meta.get('scored_year', '-')}년 공시 기준 {len(df):,}개 브랜드를 조건으로 거르고, "
        "후보 브랜드를 나란히 비교합니다. 신규 협약·신규 취급 검토에 쓰십시오.",
        eyebrow="심사")
    base = _frame(df)
    t1, t2 = st.tabs(["전체 브랜드 탐색", "브랜드 비교"])
    with t1:
        _explore(base)
    with t2:
        _compare(base)


# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _frame_cached(df: pd.DataFrame, crit_titles: dict, m_panel: float) -> pd.DataFrame:
    out = df.copy()
    out["brand_id"] = out["brand_id"].astype(str)
    panel = C.load_panel()
    if panel is not None and not panel.empty:
        year = int(panel["year"].max())
        sec = C._section_frame(panel, year)
        sec["brand_id"] = sec["brand_id"].astype(str)
        keep = ["brand_id", "_growth", "_new_rate", "_end_rate", "_cancel_rate", "_direct_ratio",
                "_age", "n_regions", "avg_sales", "company_name"]
        out = out.merge(sec[[c for c in keep if c in sec.columns]].drop_duplicates("brand_id"),
                        on="brand_id", how="left")
    out["중대 신호"] = out["brand_id"].map(lambda b: crit_titles.get(b, ""))
    return out


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    crit = {b: " · ".join(x["title"] for x in items) for b, items in C.critical_map().items()}
    out = _frame_cached(df, crit, C._mtime(C.proc_dir() / "panel.parquet"))
    rates = grading.watch_rates(C.out_dir())
    out = out.assign(
        _shown=out["deterioration_1y"].map(C.risk_pct),
        _risk=out.apply(lambda r: grading.priority_risk(r, rates), axis=1) * 100,
        _state=[C.state_label(s, k) for s, k in zip(out["brand_state"], out["n_events_at_t"], strict=False)],
    )
    diag = C.load_diagnosis_summary()
    if diag is not None and not diag.empty:
        d = diag[["brand_id", "headline_detail", "n_risk"]].copy()
        d["brand_id"] = d["brand_id"].astype(str)
        out = out.merge(d, on="brand_id", how="left")
    return out


def _explore(base: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns([1.4, 1.1, 1.1, 1])
    inds = sorted(base["industry_mid"].dropna().astype(str).unique())
    pick_ind = c1.multiselect("세부 업종", inds, placeholder="전체")
    pick_grade = c2.multiselect("등급", ["High", "Medium", "Low"], format_func=lambda g: C.GRADE_KR.get(g, g),
                                placeholder="전체")
    states = [s for s in ("건전", "요주의", "평가불가") if s in set(base["brand_state"])]
    pick_state = c3.multiselect("브랜드 상태", states, format_func=lambda s: C.STATE_LABEL.get(s, s),
                                placeholder="전체")
    min_n = c4.number_input("최소 가맹점 수", min_value=0, value=0, step=10)
    d1, d2 = st.columns([2.2, 1])
    q = d1.text_input("브랜드명 포함", placeholder="예: 치킨, 커피, 메가")
    only_crit = d2.checkbox("중대 신호만", value=False)

    v = base
    if pick_ind:
        v = v[v["industry_mid"].astype(str).isin(pick_ind)]
    if pick_grade:
        v = v[v["risk_grade"].isin(pick_grade)]
    if pick_state:
        v = v[v["brand_state"].isin(pick_state)]
    if min_n:
        v = v[pd.to_numeric(v["n_stores"], errors="coerce").fillna(0) >= min_n]
    if q.strip():
        from src.brand_search import normalize
        key = normalize(q)
        v = v[v["brand_name"].astype(str).map(normalize).str.contains(key, regex=False, na=False)]
    if only_crit:
        v = v[v["중대 신호"].astype(str).str.len() > 0]
    v = v.sort_values("_risk", ascending=False)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("조건에 맞는 브랜드", f"{len(v):,}")
    k2.metric("주의 등급", f"{int((v['risk_grade'] == 'High').sum()):,}")
    k3.metric("가맹점 합계", f"{int(pd.to_numeric(v['n_stores'], errors='coerce').fillna(0).sum()):,}")
    k4.metric("중대 신호", f"{int((v['중대 신호'].astype(str).str.len() > 0).sum()):,}")

    table = _table(v)
    st.dataframe(
        table, hide_index=True, width="stretch", height=min(560, 44 + 35 * max(len(table), 1)),
        column_config={
            "상세": st.column_config.LinkColumn("상세", display_text="열기", width="small"),
            "브랜드": st.column_config.TextColumn(width="medium"),
            "가맹점 수": st.column_config.NumberColumn(format="%d"),
            "브랜드 리스크(%)": st.column_config.NumberColumn(format="%.1f%%"),
            "1년 내 악화 위험(%)": st.column_config.NumberColumn(
                format="%.1f%%", help="건전 브랜드는 모형 확률, 악화 발생 브랜드는 같은 사건수 과거 브랜드의 "
                                      "다음 해 재발동률 — 점검 큐와 같은 기준입니다."),
            "전년 대비 가맹점(%)": st.column_config.NumberColumn(format="%+.1f%%"),
            "계약종료율(%)": st.column_config.NumberColumn(format="%.1f%%"),
            "대표 소견": st.column_config.TextColumn(width="large"),
        })
    st.caption("표 머리글을 누르면 정렬됩니다. '열기'는 브랜드 상세를 새 탭으로 엽니다. "
               "1년 내 악화 위험 순으로 정렬해 두었습니다.")
    st.download_button("이 목록 내려받기 (Excel)", _excel(table.drop(columns=["상세"]), "브랜드 탐색"),
                       file_name="FranSCORE_브랜드탐색.xlsx", on_click="ignore",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _table(v: pd.DataFrame) -> pd.DataFrame:
    num = lambda c: pd.to_numeric(v.get(c), errors="coerce")  # noqa: E731
    return pd.DataFrame({
        "상세": "/?brand=" + v["brand_id"].astype(str),
        "브랜드": v["brand_name"].astype(str),
        "세부 업종": v["industry_mid"].astype(str),
        "가맹점 수": num("n_stores"),
        "등급": v["risk_grade"].map(C.GRADE_KR),
        "브랜드 상태": v["_state"],
        "브랜드 리스크(%)": v["_shown"],
        "1년 내 악화 위험(%)": v["_risk"].round(1),
        "전년 대비 가맹점(%)": num("_growth").round(1),
        "계약종료율(%)": num("_end_rate").round(1),
        "중대 신호": v["중대 신호"],
        "대표 소견": v.get("headline_detail", pd.Series("", index=v.index)).fillna(""),
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------

def _compare(base: pd.DataFrame) -> None:
    sized = base.assign(n_sort=pd.to_numeric(base["n_stores"], errors="coerce").fillna(0)) \
                .sort_values("n_sort", ascending=False)
    # itertuples 는 '_' 로 시작하는 컬럼 이름을 바꿔 버리므로 zip 으로 읽는다
    label = {b: f"{n} ({i} · {int(s):,}개)" for b, n, i, s in
             zip(sized["brand_id"], sized["brand_name"], sized["industry_mid"], sized["n_sort"], strict=True)}
    defaults = list(sized["brand_id"].head(2))
    picks = st.multiselect(f"비교할 브랜드 (최대 {_MAX_COMPARE}개)", list(label), default=defaults,
                           format_func=lambda b: label.get(b, b), max_selections=_MAX_COMPARE)
    if len(picks) < 2:
        st.info("브랜드를 두 개 이상 고르십시오. 같은 업종의 후보를 나란히 두면 차이가 가장 잘 보입니다.")
        return
    sel = base.set_index("brand_id").loc[picks]

    st.markdown("##### 한눈에 비교")
    comp = _comparison_table(sel)
    st.dataframe(comp, width="stretch", height=min(640, 44 + 35 * len(comp)))
    st.caption("본부 재무는 금융감독원 감사보고서·공정위 정보공개서에서 확인된 경우만 표시합니다. "
               "'확인 못 함'은 비대상이거나 매칭에 실패한 경우입니다.")

    panel = C.load_panel()
    if panel is not None and not panel.empty:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**가맹점 수 추이**")
            theme.plot(_trend_fig(panel, picks, sel, "n_stores", "개"), key="cmp_stores")
        with g2:
            st.markdown("**계약종료·해지율 추이** (연초 점포 대비)")
            theme.plot(_trend_fig(panel, picks, sel, "_churn", "%"), key="cmp_churn")

    links = " · ".join(f"[{C.esc(sel.loc[b, 'brand_name'])} 상세](/?brand={b})" for b in picks)
    st.markdown(f"상세 보기 — {links}")
    st.download_button("비교표 내려받기 (Excel)", _excel(comp.reset_index(), "브랜드 비교"),
                       file_name="FranSCORE_브랜드비교.xlsx", on_click="ignore",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _hq_latest(company: str) -> dict:
    fin_all = C.load_hq_financials()
    if fin_all is None or not company or company == "nan":
        return {}
    from src.dart import norm_corp
    fin = fin_all[fin_all["key"] == norm_corp(company)].sort_values("fiscal_year")
    if fin.empty:
        return {}
    last = fin.iloc[-1]
    eq, li = pd.to_numeric(last.get("equity"), errors="coerce"), pd.to_numeric(last.get("liabilities"), errors="coerce")
    return {"fy": int(last["fiscal_year"]), "equity": eq, "op": pd.to_numeric(last.get("operating_income"),
            errors="coerce"), "debt_ratio": (li / eq * 100) if pd.notna(eq) and eq > 0 and pd.notna(li) else None}


def _comparison_table(sel: pd.DataFrame) -> pd.DataFrame:
    def f(v, fmt):
        v = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        return "-" if pd.isna(v) else fmt.format(v)

    cols = {}
    for bid, r in sel.iterrows():
        hq = _hq_latest(str(r.get("company_name") or ""))
        eok = lambda x: "-" if x is None or pd.isna(x) else f"{x / 1e8:,.1f}억"  # noqa: E731
        cols[str(r["brand_name"])] = {
            "등급": f"{C.GRADE_KR.get(str(r['risk_grade']), '-')} ({r.get('grade', '-')})",
            "브랜드 리스크": f(r["_shown"], "{:.1f}%"),
            "1년 내 악화 위험": f(r["_risk"], "{:.1f}%"),
            "브랜드 상태": r["_state"],
            "중대 신호": r["중대 신호"] or "없음",
            "세부 업종": str(r.get("industry_mid") or "-"),
            "가맹점 수": f(r.get("n_stores"), "{:,.0f}개"),
            "전년 대비 가맹점": f(r.get("_growth"), "{:+.1f}%"),
            "신규 개점률": f(r.get("_new_rate"), "{:.1f}%"),
            "계약종료율": f(r.get("_end_rate"), "{:.1f}%"),
            "중도해지율": f(r.get("_cancel_rate"), "{:.1f}%"),
            "직영점 비중": f(r.get("_direct_ratio"), "{:.1f}%"),
            "업력": f(r.get("_age"), "{:.0f}년"),
            "진출 시·도": f(r.get("n_regions"), "{:.0f}곳"),
            "가맹점 평균매출": f(pd.to_numeric(r.get("avg_sales"), errors="coerce") / 1e5, "{:.2f}억"),
            "가맹본부": str(r.get("company_name") or "-"),
            "본부 재무 (결산)": f"{hq['fy']}년" if hq else "확인 못 함",
            "본부 자본총계": eok(hq.get("equity")) if hq else "-",
            "본부 영업이익": eok(hq.get("op")) if hq else "-",
            "본부 부채비율": f(hq.get("debt_ratio"), "{:.0f}%") if hq else "-",
            "위험 소견 수": f(r.get("n_risk"), "{:.0f}건"),
            "대표 소견": str(r.get("headline_detail") or "-"),
        }
        del bid
    return pd.DataFrame(cols)


def _trend_fig(panel: pd.DataFrame, picks: list[str], sel: pd.DataFrame, col: str, unit: str) -> go.Figure:
    p = panel[panel["brand_id"].astype(str).isin(picks)].copy().sort_values("year")
    if col == "_churn":
        prev = p.groupby("brand_id")["n_stores"].shift(1)
        out = (pd.to_numeric(p["n_contract_end"], errors="coerce").fillna(0)
               + pd.to_numeric(p["n_contract_cancel"], errors="coerce").fillna(0))
        p["_churn"] = (out / prev * 100).where(prev > 0)
    fig = go.Figure()
    palette = ["#2F6FB5", "#CC6536", "#1F8A5B", "#8E3A1C"]
    for i, b in enumerate(picks):
        d = p[p["brand_id"].astype(str) == b]
        fig.add_trace(go.Scatter(
            x=d["year"].astype(int), y=d[col], mode="lines+markers", name=str(sel.loc[b, "brand_name"]),
            line={"width": 2.2, "color": palette[i % len(palette)]}, marker={"size": 7},
            hovertemplate=f"%{{x}}년<br><b>%{{y:,.1f}}</b>{unit}<extra>%{{fullData.name}}</extra>"))
    fig.update_layout(height=260, margin={"l": 4, "r": 4, "t": 8, "b": 4},
                      legend={"orientation": "h", "y": -0.18, "x": 0})
    if unit == "%":
        fig.update_yaxes(ticksuffix="%")
    return fig


def _excel(df: pd.DataFrame, sheet: str) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name=sheet)
        ws = w.sheets[sheet]
        for i, col in enumerate(df.columns, start=1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = (
                60 if col in ("대표 소견", "index") else max(10, min(28, len(str(col)) * 2 + 4)))
    return buf.getvalue()
