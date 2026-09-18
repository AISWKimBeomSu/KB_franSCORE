"""여신 포트폴리오 — 여신이 움직이면 집중도와 예상손실이 그 자리에서 다시 계산된다.

왜 정적인 그림이 아니라 계산 화면인가
    여신 포트폴리오는 매일 바뀐다. 어느 브랜드에 신규로 나가면 그 브랜드 비중이 오르고,
    상환되면 내린다. 고정된 그림 한 장을 붙여두면 '분석 결과 보고서'이지 업무 도구가
    아니다. 이 화면에서는 실행·회수를 입력하면 총 익스포저·HHI·상위 집중도·
    예상손실(EL)·스트레스 EL이 즉시 다시 계산된다.

계산식은 전부 화면에서 다시 푼다 (portfolio.py 와 같은 규칙)
    EL        = 익스포저 × 악화확률 × 손실률(LGD)
    스트레스   = 위험 상위 stress_top_pct 브랜드의 악화확률 × 배수 (1.0 상한) 로 다시 계산
    HHI       = Σ(브랜드 여신 비중)²
    조정분을 반영해 **다시 계산**하지 않으면 화면의 숫자가 조작 결과와 어긋난다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import theme
from src.views import common as C

_ADJ = "pf_adjust"          # 브랜드ID → 조정액(백만원)
EXPOSURE_NOTE = {
    "actual_loan_book": "은행이 제공한 실제 여신 잔액입니다.",
    "disclosed_startup_cost":
        "공정거래위원회 공시 창업비용 실측값에 대출조달비율·은행점유율 가정을 곱해 "
        "추정한 값입니다. 실제 여신 잔액은 은행 내부 자료이므로 공개 데이터에 없습니다.",
    "synthetic_lognormal": "방법론 실증을 위한 합성 예시입니다.",
}


def _adj() -> dict:
    if _ADJ not in st.session_state:
        st.session_state[_ADJ] = {}
    return st.session_state[_ADJ]


def render() -> None:
    pf, summary = C.load_portfolio()
    if pf is None:
        st.warning("포트폴리오 산출물이 없습니다.")
        return
    cfg = C.cfg()
    pcfg = cfg["portfolio"]
    basis = ((summary.get("assumptions") or {}).get("exposure") or {}).get("basis", "")

    theme.page_header(
        "여신 포트폴리오",
        "브랜드별 여신 쏠림과 예상손실입니다. 아래에서 실행·회수를 입력하면 "
        "모든 지표가 즉시 다시 계산됩니다.",
        eyebrow="여신관리")

    adj = _adj()
    base = pf.copy()
    base["brand_id"] = base["brand_id"].astype(str)
    base["_adj"] = base["brand_id"].map(lambda b: float(adj.get(b, 0.0)))
    base["exposure_mkrw"] = (pd.to_numeric(base["exposure_mkrw"], errors="coerce").fillna(0)
                             + base["_adj"]).clip(lower=0.0)

    # 목록에 없던 브랜드에 신규 실행한 경우 행을 만들어 붙인다
    extra = [b for b in adj if b not in set(base["brand_id"])]
    if extra:
        base = pd.concat([base, _new_rows(extra, adj)], ignore_index=True)

    # ⚠️ 고지를 화면 맨 아래 캡션으로만 두면 아무도 안 읽는다. 여기 숫자는 **실제 여신이
    #    아니라 공시 창업비용에서 유도한 추정치**이고, 그 사실을 모르고 HHI·예상손실을
    #    인용하면 근거 없는 금액이 결재 문서로 넘어간다. 표 위에 먼저 밝힌다.
    if basis != "actual_loan_book":
        a = ((summary.get("assumptions") or {}).get("exposure") or {})
        st.warning(
            f"**여기 익스포저는 추정치입니다.** 실제 여신 잔액은 은행 내부 자료라 공개 "
            f"데이터에 없습니다. 공정거래위원회 공시 창업비용에 대출조달비율 "
            f"{float(a.get('loan_to_startup_cost', 0)) * 100:.0f}% · 은행점유율 "
            f"{float(a.get('bank_share', 0)) * 100:.0f}% 가정을 곱해 만든 값이며, "
            f"창업비용이 공시된 **{int(a.get('n_brands', 0))}개 브랜드**만 담겨 있습니다.\n\n"
            f"→ **금액 자체가 아니라 '구조'를 보십시오** — 어느 브랜드에 쏠렸는지, "
            f"신규 실행이 집중도를 얼마나 움직이는지가 이 화면의 쓸모입니다. "
            f"행내 여신 잔액을 연결하면 같은 화면이 그대로 실측 기준으로 바뀝니다.")

    port = _recompute(base, pcfg)
    _kpis(port, summary, adj)
    st.write("")

    tabs = st.tabs(["여신 조정", "집중도", "예상손실", "브랜드별 명세"])
    with tabs[0]:
        _adjust_panel(port, adj)
    with tabs[1]:
        _concentration(port)
    with tabs[2]:
        _expected_loss(port, pcfg)
    with tabs[3]:
        _detail_table(port)

    st.caption("익스포저 산출 근거 — " + EXPOSURE_NOTE.get(basis, EXPOSURE_NOTE[
        "synthetic_lognormal"]) + "  이 지표는 2선 리스크 관리 참고용이며 "
        "자동 여신 결정에 사용되지 않습니다.")


# ---------------------------------------------------------------------------
# 재계산
# ---------------------------------------------------------------------------

def _new_rows(brand_ids: list[str], adj: dict) -> pd.DataFrame:
    """포트폴리오에 없던 브랜드를 점수표에서 끌어와 행으로 만든다."""
    scores, _ = C.load_scores()
    rows = []
    for b in brand_ids:
        s = scores[scores["brand_id"].astype(str) == b] if scores is not None else None
        r = s.iloc[0] if s is not None and not s.empty else None
        rows.append({
            "brand_id": b,
            "brand_name": str(r["brand_name"]) if r is not None else b,
            "industry_major": str(r.get("industry_major", "")) if r is not None else "",
            "industry_mid": str(r.get("industry_mid", "")) if r is not None else "",
            "n_stores": float(r["n_stores"]) if r is not None else np.nan,
            "deterioration_1y": float(r["deterioration_1y"]) if r is not None else 0.0,
            "risk_grade": str(r["risk_grade"]) if r is not None else "Low",
            "exposure_mkrw": max(float(adj.get(b, 0.0)), 0.0),
            "_adj": float(adj.get(b, 0.0)),
        })
    return pd.DataFrame(rows)


def _recompute(port: pd.DataFrame, pcfg: dict) -> pd.DataFrame:
    """조정 후 익스포저로 비중·스트레스 악화확률·EL 을 전부 다시 계산한다."""
    p = port[port["exposure_mkrw"] > 0].copy().reset_index(drop=True)
    if p.empty:
        return p
    total = float(p["exposure_mkrw"].sum())
    p["exposure_share"] = p["exposure_mkrw"] / total if total > 0 else 0.0
    p["deterioration_1y"] = pd.to_numeric(p["deterioration_1y"], errors="coerce").fillna(0.0)

    top_pct = float(pcfg["stress_top_pct"])
    mult = float(pcfg["stress_pd_multiplier"])
    n_stress = max(1, math.ceil(len(p) * top_pct))
    idx = p["deterioration_1y"].nlargest(n_stress).index
    p["is_stressed"] = p.index.isin(idx)
    p["det_stressed"] = np.where(p["is_stressed"],
                                np.minimum(p["deterioration_1y"] * mult, 1.0), p["deterioration_1y"])

    for lgd in sorted(float(x) for x in pcfg["lgd_scenarios"]):
        k = f"lgd{round(lgd * 100):02d}"
        p[f"el_{k}_mkrw"] = p["exposure_mkrw"] * p["deterioration_1y"] * lgd
        p[f"stress_el_{k}_mkrw"] = p["exposure_mkrw"] * p["det_stressed"] * lgd
    return p


def _mid_lgd(pcfg: dict) -> str:
    lgds = sorted(float(x) for x in pcfg["lgd_scenarios"])
    return f"lgd{round(lgds[len(lgds) // 2] * 100):02d}"


def _eok(mkrw) -> float:
    return float(mkrw) / 100.0


# ---------------------------------------------------------------------------
# 화면 조각
# ---------------------------------------------------------------------------

def _kpis(port: pd.DataFrame, summary: dict, adj: dict) -> None:
    if port.empty:
        st.info("여신 잔액이 0입니다. '여신 조정'에서 실행을 입력해 보십시오.")
        return
    cfg = C.cfg()
    mid = _mid_lgd(cfg["portfolio"])
    total = float(port["exposure_mkrw"].sum())
    shares = port["exposure_share"].sort_values(ascending=False)
    hhi = float((shares ** 2).sum())
    top10 = float(shares.head(10).sum())
    el = float(port[f"el_{mid}_mkrw"].sum())
    stress = float(port[f"stress_el_{mid}_mkrw"].sum())

    base_total = ((summary.get("exposure") or {}).get("total_mkrw")
                  if isinstance(summary.get("exposure"), dict) else None)
    delta_total = None
    if adj:
        moved = sum(float(v) for v in adj.values())
        if abs(moved) > 1e-9:
            delta_total = f"{_eok(moved):+,.1f} 억"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("총 여신", f"{_eok(total):,.0f} 억", delta=delta_total)
    k2.metric("쏠림 정도 (HHI)", f"{hhi:.4f}",
              help="Σ(브랜드 비중)². 0에 가까울수록 고르게 분산, 1에 가까울수록 한 곳에 집중.")
    k3.metric("상위 10 집중도", f"{top10 * 100:.1f}%")
    k4.metric(f"예상손실 (LGD {int(mid[3:])}%)", f"{_eok(el):,.1f} 억")
    k5.metric("스트레스 시", f"{_eok(stress):,.1f} 억",
              delta=f"+{_eok(stress - el):,.1f} 억", delta_color="inverse")
    del base_total


def _adjust_panel(port: pd.DataFrame, adj: dict) -> None:
    scores, _ = C.load_scores()
    st.markdown("##### 여신 실행 · 회수")
    st.caption("브랜드를 고르고 금액을 입력하면 위쪽 지표가 즉시 다시 계산됩니다. "
               "회수는 음수로 입력합니다.")

    names = (scores[["brand_id", "brand_name", "n_stores", "deterioration_1y", "risk_grade"]]
             if scores is not None else port[["brand_id", "brand_name"]])
    names = names.copy()
    names["brand_id"] = names["brand_id"].astype(str)
    label = {str(r["brand_id"]): f"{r['brand_name']}" for _, r in names.iterrows()}

    c1, c2, c3 = st.columns([2.4, 1.2, 1])
    pick = c1.selectbox("브랜드", options=list(label), format_func=lambda b: label[b],
                        key="pf_pick")
    amount = c2.number_input("금액 (억원)", value=0.0, step=1.0, format="%.1f",
                             key="pf_amt")
    c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c3.button("반영", type="primary", use_container_width=True) and abs(amount) > 1e-9:
        adj[pick] = adj.get(pick, 0.0) + amount * 100.0         # 억원 → 백만원
        st.rerun()

    if adj:
        st.markdown("##### 반영된 조정")
        rows = [{"브랜드": label.get(b, b), "조정액(억원)": round(_eok(v), 1)}
                for b, v in adj.items() if abs(v) > 1e-9]
        if rows:
            d1, d2 = st.columns([3, 1])
            d1.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            d2.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if d2.button("전체 되돌리기", use_container_width=True):
                st.session_state[_ADJ] = {}
                st.rerun()
        else:
            st.caption("조정 내역이 없습니다.")
    else:
        st.caption("아직 조정한 여신이 없습니다. 현재 지표는 산출된 기준 포트폴리오입니다.")

    _limit_check(port)


def _limit_check(port: pd.DataFrame) -> None:
    """한도 점검 — 어디가 위험한 쏠림인지 화면에서 바로 알려준다."""
    if port.empty:
        return
    st.markdown("##### 한도 점검")
    c1, c2 = st.columns(2)
    brand_cap = c1.slider("브랜드 1개 한도 (총 여신 대비 %)", 1.0, 20.0, 5.0, 0.5)
    ind_cap = c2.slider("업종 1개 한도 (총 여신 대비 %)", 5.0, 60.0, 30.0, 1.0)

    over_b = port[port["exposure_share"] * 100 > brand_cap].sort_values(
        "exposure_share", ascending=False)
    key = "industry_mid" if "industry_mid" in port.columns else "industry_major"
    ind = port.groupby(key)["exposure_share"].sum().sort_values(ascending=False)
    over_i = ind[ind * 100 > ind_cap]

    if over_b.empty and over_i.empty:
        st.success(f"설정한 한도(브랜드 {brand_cap:.1f}% · 업종 {ind_cap:.0f}%)를 "
                   "넘는 쏠림이 없습니다.")
        return
    for _, r in over_b.head(6).iterrows():
        risky = " · 주의 등급" if str(r.get("risk_grade")) == "High" else ""
        st.warning(f"**{r['brand_name']}** 한 브랜드에 총 여신의 "
                   f"{r['exposure_share'] * 100:.1f}% ({_eok(r['exposure_mkrw']):,.1f}억원)가 "
                   f"나가 있습니다 — 한도 {brand_cap:.1f}% 초과{risky}.")
    for name, share in over_i.head(4).items():
        st.warning(f"**{name}** 업종에 총 여신의 {share * 100:.1f}%가 몰려 있습니다 "
                   f"— 한도 {ind_cap:.0f}% 초과. 업종 전체가 동시에 나빠지면 "
                   "분산 효과가 없습니다.")


def _concentration(port: pd.DataFrame) -> None:
    if port.empty:
        return
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("##### 여신 상위 브랜드")
        top = port.nlargest(12, "exposure_mkrw").sort_values("exposure_mkrw")
        colors = [theme.GRADE_FILL.get(str(g), theme.YELLOW_DEEP)
                  for g in top["risk_grade"]]
        fig = C.bar_chart(top["brand_name"].astype(str).tolist(),
                          (top["exposure_mkrw"] / 100).round(1).tolist(),
                          colors=colors, unit="억원")
        fig.update_layout(height=max(240, 26 * len(top)))
        theme.plot(fig, key="pf_top")
        st.caption("막대 색 = 위험등급 (빨강 주의 · 주황 관찰 · 초록 양호)")
    with c2:
        st.markdown("##### 위험 × 여신")
        fig = go.Figure()
        for g in ("Low", "Medium", "High"):
            m = port["risk_grade"].astype(str) == g
            if not m.any():
                continue
            fig.add_trace(go.Scatter(
                x=port.loc[m, "deterioration_1y"] * 100,
                y=port.loc[m, "exposure_mkrw"] / 100,
                mode="markers", name=C.GRADE_KR.get(g, g),
                marker={"size": 11, "color": theme.GRADE_FILL[g], "opacity": .78,
                        "line": {"width": 1, "color": "#FFFFFF"}},
                text=port.loc[m, "brand_name"],
                hovertemplate="<b>%{text}</b><br>브랜드 리스크 %{x:.1f}%<br>"
                              "여신 %{y:,.1f}억원<extra></extra>"))
        fig.update_layout(height=330, xaxis_title="브랜드 리스크",
                          yaxis_title="여신 (억원)",
                          margin={"l": 4, "r": 4, "t": 30, "b": 30})
        fig.update_xaxes(ticksuffix="%")
        theme.plot(fig, key="pf_scatter")
        st.caption("오른쪽 위에 있을수록 위험하면서 금액도 큰 브랜드입니다.")


def _expected_loss(port: pd.DataFrame, pcfg: dict) -> None:
    if port.empty:
        return
    lgds = sorted(float(x) for x in pcfg["lgd_scenarios"])
    labels = [f"LGD {int(v * 100)}%" for v in lgds]
    base = [_eok(port[f"el_lgd{round(v * 100):02d}_mkrw"].sum()) for v in lgds]
    strs = [_eok(port[f"stress_el_lgd{round(v * 100):02d}_mkrw"].sum()) for v in lgds]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=base, name="기본",
                         marker_color=theme.YELLOW_DEEP,
                         hovertemplate="%{x}<br>기본 <b>%{y:,.1f}</b>억원<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=strs, name="스트레스",
                         marker_color=theme.DANGER_FILL,
                         hovertemplate="%{x}<br>스트레스 <b>%{y:,.1f}</b>억원<extra></extra>"))
    fig.update_layout(barmode="group", height=300, yaxis_title="예상손실 (억원)",
                      margin={"l": 4, "r": 4, "t": 30, "b": 20})
    theme.plot(fig, key="pf_el")
    st.caption(
        f"예상손실 = 여신 × 브랜드 리스크 × 손실률(LGD). "
        f"스트레스는 위험 상위 {float(pcfg['stress_top_pct']) * 100:.0f}% 브랜드의 "
        f"브랜드 리스크에 {pcfg['stress_pd_multiplier']}배를 적용해 동시 악화를 가정한 값입니다.")

    imp = C.out_dir() / "correlation_impact.json"
    if imp.exists():
        _correlation_note(imp)
    _tail_contribution()


def _tail_contribution() -> None:
    """어느 브랜드가 99% 꼬리를 만드는가 — 측정한 ρ 가 여신 판단으로 이어지는 자리.

    왜 이 표가 접힌 참고표가 아니라 본문인가 (심사 지적)
        상관을 측정해 놓고 화면에는 포트폴리오 총량(UL 배수)만 있었다. 심사역의 실제
        질문은 "그래서 어느 브랜드의 한도를 봐야 하는가"다. 성분 ES(Euler 배분)는
        합이 전체 ES 와 정확히 일치하는 유일한 가법 배분이라, 꼬리손실을 브랜드별로
        조작 없이 나눠 그 질문에 답한다.
    """
    p = C.out_dir() / "brand_ul_contribution.csv"
    if not p.exists():
        return
    df = pd.read_csv(p, encoding="utf-8-sig").head(10)
    st.markdown(f"<div style='font-weight:700;font-size:{theme.FS_LG};color:{theme.INK};"
                f"margin-top:18px'>꼬리손실 기여 상위 — 어느 브랜드가 99% 손실을 만드는가"
                f"</div>", unsafe_allow_html=True)
    view = pd.DataFrame({
        "브랜드": df["brand_name"],
        "여신 비중": df["exposure_share"],
        "UL 기여 비중": df["ul_share"],
        "쏠림 배수": df["concentration_ratio"],
        "꼬리손실 기여(억)": df["ul_contrib_mkrw"] / 100,
    })
    st.dataframe(
        view, hide_index=True, use_container_width=True,
        column_config={
            "여신 비중": st.column_config.NumberColumn(format="percent"),
            "UL 기여 비중": st.column_config.ProgressColumn(
                format="percent", min_value=0.0,
                max_value=float(view["UL 기여 비중"].max() or 1.0)),
            "쏠림 배수": st.column_config.NumberColumn(
                format="%.2f배",
                help="UL 기여 비중 ÷ 여신 비중. 1보다 크면 여신 규모에 비해 꼬리손실 "
                     "기여가 큰 브랜드 — 브랜드 내부 상관(ρ=0.416)과 집중이 만드는 "
                     "위험 쏠림입니다."),
            "꼬리손실 기여(억)": st.column_config.NumberColumn(format="%.2f"),
        })
    st.caption(
        "총손실이 99% 분위수를 넘는 시나리오에서 각 브랜드가 평균적으로 만든 손실"
        "(성분 ES, Euler 배분 — 브랜드별 기여의 합이 전체 꼬리손실과 정확히 일치)에서 "
        "그 브랜드의 기대손실을 뺀 값입니다. 한도 재검토 우선순위는 여신 잔액이 아니라 "
        "이 표의 순서를 따르는 것이 이 도구의 처방입니다.")


def _correlation_note(path) -> None:
    import json
    try:
        ci = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    with st.expander("차주를 서로 독립으로 보면 손실이 얼마나 과소평가되는가"):
        rows = []
        for lvl, nm in (("p95", "95%"), ("p99", "99%"), ("p999", "99.9%")):
            if f"independent_{lvl}_mkrw" not in ci:
                continue
            rows.append({
                "신뢰수준": nm,
                "차주 독립 가정": round(ci[f"independent_{lvl}_mkrw"] / 100, 1),
                "브랜드 상관 반영": round(ci[f"brand_correlated_{lvl}_mkrw"] / 100, 1),
                "과소평가": round(ci[f"understatement_{lvl}_mkrw"] / 100, 1)})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(c, format="%.1f 억")
                                        for c in ("차주 독립 가정", "브랜드 상관 반영", "과소평가")})
        st.caption("같은 브랜드 가맹점은 본부라는 공통 원인을 공유합니다. 서로 독립이라고 "
                   "보면 동시에 무너질 확률을 과소평가하게 됩니다.")


def _detail_table(port: pd.DataFrame) -> None:
    if port.empty:
        return
    mid = _mid_lgd(C.cfg()["portfolio"])
    view = port[["brand_name", "industry_mid", "n_stores", "deterioration_1y", "risk_grade",
                 "exposure_mkrw", "exposure_share", f"el_{mid}_mkrw",
                 f"stress_el_{mid}_mkrw"]].copy()
    view["exposure_mkrw"] = view["exposure_mkrw"] / 100
    view[f"el_{mid}_mkrw"] = view[f"el_{mid}_mkrw"] / 100
    view[f"stress_el_{mid}_mkrw"] = view[f"stress_el_{mid}_mkrw"] / 100
    view["risk_grade"] = view["risk_grade"].map(C.GRADE_KR).fillna(view["risk_grade"])
    # 비중은 비율(0~1)로 저장돼 있다 — %.2f%% 서식은 값을 그대로 찍으므로
    # 5%가 "0.05%"로 나온다. 표시 직전에 100을 곱한다.
    view["exposure_share"] = view["exposure_share"] * 100
    view["deterioration_1y"] = pd.to_numeric(view["deterioration_1y"], errors="coerce") * 100
    view = view.sort_values("exposure_mkrw", ascending=False)
    st.dataframe(
        view, hide_index=True, use_container_width=True, height=430,
        column_config={
            "brand_name": st.column_config.TextColumn("브랜드", width="medium"),
            "industry_mid": st.column_config.TextColumn("업종"),
            "n_stores": st.column_config.NumberColumn("가맹점", format="%d"),
            "deterioration_1y": st.column_config.NumberColumn("브랜드 리스크", format="%.1f%%"),
            "risk_grade": st.column_config.TextColumn("등급"),
            "exposure_mkrw": st.column_config.NumberColumn("여신", format="%.1f 억"),
            "exposure_share": st.column_config.ProgressColumn(
                "비중", format="%.2f%%", min_value=0.0,
                max_value=float(view["exposure_share"].max() or 1)),
            f"el_{mid}_mkrw": st.column_config.NumberColumn("예상손실", format="%.2f 억"),
            f"stress_el_{mid}_mkrw": st.column_config.NumberColumn(
                "스트레스", format="%.2f 억"),
        })
    st.download_button("포트폴리오 내려받기 (CSV)",
                       port.to_csv(index=False).encode("utf-8-sig"),
                       file_name="franscore_portfolio.csv", mime="text/csv")
