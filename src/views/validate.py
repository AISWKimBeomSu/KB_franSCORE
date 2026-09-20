"""등급 검증 — 은행 연체 자료로 FranSCORE 등급이 실제 가맹점주 연체를 가려내는지 본다.

왜 이 화면이 있는가
    등급의 라벨은 공시 지표 악화이지 연체가 아니다. 심사 보조지표로 올리려면 은행이 가진
    연체 자료로 "등급이 연체를 서열화하는가, 내부등급에 없는 정보를 더하는가"를 먼저
    통과해야 한다. 그 검증을 은행 담당자가 파일 하나 올려 바로 돌릴 수 있게 한다.
    계산은 src/delinquency.py 가 하고, 이 화면은 입력·열 확인·판정 표시·반출만 맡는다.
    업로드 파일은 디스크에 쓰지 않는다(세션 메모리에서만 처리).
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import batch, theme
from src import delinquency as D
from src.views import common as C

_STATE = "val_input"
_NONE = "(없음)"
_STATUS_KIND = {"확인": "Low", "충분": "Low", "부분 확인": "Medium", "방향만 일치": "Medium",
                "부족": "Medium", "확인 안 됨": "High", "판단 불가": "Neutral", "평가 생략": "Neutral"}
_GRADE_FILL = {"FS1": theme.SAFE_FILL, "FS2": theme.WARN_FILL, "FS3": theme.DANGER_FILL}


@st.cache_data(show_spinner=False)
def _history(m: float) -> pd.DataFrame:
    return D.load_history(C.out_dir())


@st.cache_data(show_spinner=False)
def _all_brands(m: float) -> pd.DataFrame | None:
    p = C.proc_dir() / "panel_full.parquet"
    return pd.read_parquet(p, columns=["brand_name"]).drop_duplicates() if p.exists() else None


@st.cache_data(show_spinner=False)
def _sample(m: float) -> bytes:
    return D.sample_bytes(_history(m))


@st.cache_data(show_spinner=False, max_entries=6)
def _run(df: pd.DataFrame, cols: tuple, lag: int, dpd: int, m: float) -> tuple[D.Prepared, dict]:
    brand, outcome, kind, date, internal, amount = cols
    spec = D.Columns(brand=brand, outcome=outcome, outcome_kind=kind, date=date,
                     internal=internal, amount=amount)
    prep = D.prepare(df, spec, _history(m), lag_years=lag, dpd_threshold=dpd,
                     all_brands=_all_brands(C._mtime(C.proc_dir() / "panel_full.parquet")),
                     outputs=C.out_dir())
    return prep, D.evaluate(prep)


def render() -> None:
    theme.page_header(
        "등급 검증",
        "은행의 가맹점주 대출 연체 자료를 올리면 FranSCORE 등급이 실제 연체를 가려내는지, "
        "기존 내부등급에 없는 정보를 더하는지 바로 검증합니다.",
        eyebrow="모형 검증")
    st.markdown(
        f"<div style='font-size:{theme.FS_SM};color:{theme.TEXT_SUB};line-height:1.7;margin:-4px 0 12px'>"
        "FranSCORE 등급은 브랜드의 <b>공시 지표 악화</b> 확률이지 연체 확률이 아닙니다. 심사 보조지표로 "
        "쓰려면 이 검증을 먼저 통과해야 합니다. 대출은 <b>취급 당시 볼 수 있던 등급</b>에 맞추고, "
        "판정 기준은 코드에 고정돼 있어 결과를 보고 바뀌지 않습니다.</div>", unsafe_allow_html=True)

    m = C._mtime(C.out_dir() / "grade_history.csv")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        up = st.file_uploader("대출·연체 자료 (엑셀 또는 CSV)", type=["xlsx", "xls", "csv"],
                              help="브랜드명과 연체 여부(또는 연체일수) 열이 있으면 됩니다. 취급일·"
                                   "내부등급·대출금액이 있으면 더 많은 검증을 합니다.")
    with c2:
        st.markdown(f"<div style='font-size:{theme.FS_SM};color:{theme.TEXT_SUB};margin:6px 0 8px'>"
                    "은행 자료가 없으면 <b>가상 자료</b>로 화면을 먼저 보십시오. 결과는 시연용이며 "
                    "성능 근거가 아닙니다.</div>", unsafe_allow_html=True)
        names = None
        if C.masked():                          # 가명 모드는 실명 예시를 쓰지 않는다 (src/public.py)
            from src import public
            scores, _ = C.load_scores()
            names = public.example_names(scores, 3) if scores is not None else None
        st.download_button("업로드 양식 내려받기 (.xlsx)", D.template_bytes(names),
                           file_name="FranSCORE_등급검증_양식.xlsx", width="stretch",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           on_click="ignore")
        demo = st.button("가상 자료로 시연", width="stretch")

    if up is not None:
        try:
            st.session_state[_STATE] = {"df": batch.read_upload(up.name, up.getvalue()),
                                        "name": up.name}
        except Exception as exc:                     # 파일 형식 문제로 화면이 죽으면 안 된다
            st.error(f"파일을 읽지 못했습니다 — {exc}")
            return
    elif demo:
        st.session_state[_STATE] = {"df": pd.read_excel(io.BytesIO(_sample(m)), dtype=str),
                                    "name": "가상 시연 자료"}

    state = st.session_state.get(_STATE)
    if not state:
        _empty_hint()
        return
    df: pd.DataFrame = state["df"]
    if df.empty:
        st.warning("자료가 비어 있습니다.")
        return
    if len(df) > 200_000:
        st.warning("한 번에 200,000건까지 검증합니다 — 앞 200,000건만 썼습니다.")
        df = df.head(200_000)

    spec, lag, dpd = _column_picker(df)
    if spec is None:
        return
    try:
        prep, res = _run(df, spec, lag, dpd, m)
    except ValueError as exc:
        st.error(str(exc))
        return

    if res["synthetic"]:
        st.warning("**가상 자료입니다.** 연체 확률에 브랜드 위험을 일부러 반영해 만든 시연 자료라, "
                   "아래 판정은 FranSCORE 성능의 근거가 아닙니다. 절차가 어떻게 판정하는지만 "
                   "보여 줍니다.", icon="⚠️")
    for note in res["notes"]:
        st.info(note)

    _verdict(res)
    _kpis(res)
    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("##### 등급별 연체율")
        theme.plot(_grade_chart(res), height=300, key="val_grade_chart")
    with right:
        st.markdown("##### 등급별 표")
        _grade_table(res)
    _incremental(res)
    _state_table(res)

    with st.expander(f"실적연도별 · 브랜드별 · 제외 목록 ({res['n_excluded']:,}행 제외)"):
        t1, t2, t3 = st.tabs(["실적연도별", "브랜드별", "제외 목록"])
        with t1:
            st.caption("등급 기준 실적연도마다 서열이 유지되는지 봅니다. 최근 취급분은 관찰 기간이 "
                       "짧아 연체율이 낮게 나올 수 있습니다.")
            st.dataframe(res["by_year"], hide_index=True, width="stretch")
        with t2:
            st.caption("연체가 많은 브랜드부터 — 사후관리 대상 선정에 씁니다.")
            st.dataframe(res["brands"], hide_index=True, width="stretch", height=380)
        with t3:
            st.dataframe(prep.excluded.rename(columns={"row": "원본 행", "brand_input": "입력 브랜드",
                                                       "reason": "제외 사유", "detail": "상세"}),
                         hide_index=True, width="stretch", height=320)

    with st.expander("검증 설계 — 모형검증 담당자용"):
        st.markdown(
            "- **시점 정합**: 대출을 `취급연도 − 2년` 실적의 등급에 맞춥니다. t년 실적은 t+1년 "
            "정보공개서로 공개되므로 취급 시점에 확실히 볼 수 있던 것은 2년 전 실적입니다. 모형 학습 "
            "연도(2018~2021년 실적) 등급은 표본 내 점수라 쓰지 않습니다.\n"
            "- **생존 편향 방지**: 브랜드를 최신 목록이 아니라 과거에 평가된 브랜드 전체에서 찾습니다. "
            "그사이 사라진 브랜드가 빠지면 결과가 좋게 나오기 때문입니다.\n"
            "- **군집 부트스트랩**: 같은 브랜드 가맹점주의 연체는 함께 움직입니다(브랜드 내 상관 "
            f"ρ_W 0.416). 구간은 브랜드를 통째로 재표집해 {res['n_boot']}회 계산합니다.\n"
            "- **추가 정보**: `연체 ~ 내부등급 + 브랜드 위험` 로지스틱 회귀에서 브랜드 위험 계수를 "
            "**브랜드 군집 강건 표준오차**로 검정합니다(오즈비 하한 > 1이면 확인). 효과 크기로는 "
            "**브랜드 단위 5겹 교차검증**의 폴드별 표본 밖 AUC를 비교합니다 — 같은 브랜드가 학습·평가에 "
            "함께 들어가면 브랜드 효과를 외워서 맞히기 때문입니다. 폴드 예측을 한데 모아 AUC를 재면 "
            "효과가 없어도 증가분이 양수로 치우쳐(귀무 모의실험 72%) 판정에 쓰지 않습니다.\n"
            f"- **표본 기준**: 등급별 연체 {D.MIN_DEFAULTS_PER_GRADE}건·전체 {D.MIN_DEFAULTS_TOTAL}건 "
            "미만이면 판정을 보류합니다.\n"
            "- **빈칸**: 연체 여부가 빈 행은 정상으로 세지 않고 제외합니다(추출 누락일 수 있음).")

    _, meta = C.load_scores()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "검증 결과 내려받기 (Excel · 요약/등급별/상태별/연도별/브랜드별/제외 목록)",
        D.to_excel(res, prep, {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                               "scored_year": meta.get("scored_year", "-")}),
        file_name=f"FranSCORE_등급검증_{stamp}.xlsx", type="primary",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", on_click="ignore")
    st.caption("업로드한 파일은 서버에 저장하지 않고 이 세션 안에서만 처리합니다. 반출 파일에는 원본 "
               "행 번호와 브랜드명만 남고 차주 정보는 담지 않습니다.")


# ---------------------------------------------------------------------------

def _empty_hint() -> None:
    st.write("")
    st.markdown(
        f"<div style='padding:18px 20px;border-radius:{theme.RADIUS_LG};border:1px dashed "
        f"{theme.BORDER_STRONG};color:{theme.TEXT_SUB};font-size:{theme.FS_BASE};line-height:1.7'>"
        f"<b style='color:{theme.INK}'>이렇게 쓰십시오</b><br>"
        "① 가맹점주 대출 목록에 <b>브랜드명·연체 여부</b>를 담아 올립니다. 취급일·내부등급·"
        "대출금액이 있으면 함께 넣습니다.<br>"
        "② 대출마다 <b>취급 당시의 FranSCORE 등급</b>을 붙여 등급별 연체율을 계산합니다.<br>"
        "③ 서열성 · 내부등급 대비 추가 정보 · 표본 충분성을 판정하고, 결과를 엑셀로 내려받아 "
        "모형검증 보고에 붙입니다.</div>", unsafe_allow_html=True)


def _column_picker(df: pd.DataFrame) -> tuple[tuple | None, int, int]:
    guess = D.detect_columns(df)
    cols = [str(c) for c in df.columns]
    opt = [_NONE, *cols]

    def _idx(name: str | None, options: list[str]) -> int:
        return options.index(name) if name in options else 0

    st.markdown("##### 열 확인")
    a, b, c = st.columns(3)
    brand = a.selectbox("브랜드명 열", cols, index=_idx(guess.brand, cols))
    outcome = b.selectbox("연체 여부 / 연체일수 열", opt, index=_idx(guess.outcome, opt))
    kind = c.radio("연체 열의 형식", ["연체 여부 (Y/N·1/0)", "연체일수 (숫자)"],
                   index=1 if guess.outcome_kind == "dpd" else 0, horizontal=True)
    d, e, f, g = st.columns(4)
    date = d.selectbox("취급일 열 (권장)", opt, index=_idx(guess.date, opt))
    internal = e.selectbox("내부등급·CB점수 열 (권장)", opt, index=_idx(guess.internal, opt))
    amount = f.selectbox("대출금액 열 (선택)", opt, index=_idx(guess.amount, opt))
    lag = g.radio("취급 당시 볼 수 있던 등급", ["취급연도 − 2년 실적", "취급연도 − 1년 실적"], index=0,
                  help="t년 실적은 t+1년 정보공개서로, 빨라야 그해 중반에 공개됩니다. 취급 시점에 확실히 "
                       "볼 수 있던 것은 2년 전 실적입니다(권장). 하반기 취급이 대부분이고 공개 시점을 "
                       "확인했다면 − 1년을 고르십시오.")
    dpd = 90
    if kind.startswith("연체일수"):
        dpd = int(st.number_input("연체로 볼 연체일수 기준 (일 이상)", min_value=1, max_value=365,
                                  value=90, step=30))
    if outcome == _NONE:
        st.error("연체 여부(또는 연체일수) 열을 골라 주십시오.")
        return None, 2, dpd
    pick = lambda v: None if v == _NONE else v            # noqa: E731
    spec = (brand, outcome, "dpd" if kind.startswith("연체일수") else "flag", pick(date),
            pick(internal), pick(amount))
    return spec, (2 if lag.endswith("2년 실적") else 1), dpd


def _card_html(title: str, status: tuple[str, str]) -> str:
    return (f"<div style='flex:1 1 240px;background:{theme.SURFACE};border:1px solid {theme.BORDER};"
            f"border-radius:{theme.RADIUS_LG};padding:14px 16px'>"
            f"<div style='font-size:{theme.FS_SM};color:{theme.TEXT_SUB};margin-bottom:8px'>{title}</div>"
            f"{theme.chip(status[0], _STATUS_KIND.get(status[0], 'Neutral'))}"
            f"<div style='font-size:{theme.FS_SM};color:{theme.TEXT};line-height:1.6;margin-top:8px'>"
            f"{C.esc(status[1])}</div></div>")


def _verdict(res: dict) -> None:
    v = res["verdict"]
    st.markdown("##### 판정")
    # 세 카드를 한 줄의 flex 로 그린다 — st.columns 안의 카드는 글 길이에 따라 높이가 제각각이다
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:12px;align-items:stretch;margin-bottom:10px'>"
        + _card_html("① 등급 서열성 — 나쁜 등급일수록 연체가 많은가", v["rank"])
        + _card_html("② 추가 정보 — 내부등급에 없는 정보를 더하는가", v["incremental"])
        + _card_html("③ 표본 — 판정할 만큼 연체가 관측됐는가", v["sample"])
        + "</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='padding:12px 16px;border-radius:{theme.RADIUS_MD};background:{theme.YELLOW_SOFT};"
        f"border:1px solid {theme.YELLOW_LINE};color:{theme.INK};font-size:{theme.FS_MD};margin:4px 0 8px'>"
        f"<b>종합</b> · {'<b>[가상 시연]</b> ' if res['synthetic'] else ''}{C.esc(v['overall'])}</div>",
        unsafe_allow_html=True)


def _kpis(res: dict) -> None:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("검증 대출 수", f"{res['n']:,}")
    k2.metric("연체 수 (연체율)", f"{res['defaults']:,} ({res['rate'] * 100:.2f}%)")
    k3.metric("브랜드 수", f"{res['n_brands']:,}")
    r = res["rank"]
    k4.metric("주의 ÷ 안정 연체율", f"{r['ratio']:.1f}배" if np.isfinite(r["ratio"]) else "-",
              help=(f"주의 − 안정 차이 {r['diff'] * 100:.2f}%p, 95% 구간 "
                    f"[{r['diff_lo'] * 100:.2f}, {r['diff_hi'] * 100:.2f}]%p (브랜드 군집 부트스트랩)"
                    if np.isfinite(r["diff"]) else None))


def _grade_chart(res: dict) -> go.Figure:
    t = res["by_grade"]
    grades = list(D.GRADES)
    rate, lo, hi = t["연체율(%)"], t["95% 구간 하한(%)"], t["95% 구간 상한(%)"]
    fig = go.Figure(go.Bar(
        x=t["구분"], y=rate, width=0.52,
        marker={"color": [_GRADE_FILL[g] for g in grades], "line": {"width": 0},
                "cornerradius": 4},
        error_y={"type": "data", "symmetric": False, "array": hi - rate, "arrayminus": rate - lo,
                 "color": theme.TEXT_SUB, "thickness": 1.5, "width": 7},
        customdata=np.column_stack([t["대출 수"], t["연체 수"], lo, hi]),
        hovertemplate="%{x}<br>연체율 <b>%{y:.2f}%</b><br>95% 구간 %{customdata[2]:.2f}~"
                      "%{customdata[3]:.2f}%<br>연체 %{customdata[1]:,} / 대출 %{customdata[0]:,}"
                      "<extra></extra>"))
    # 값 표시는 막대가 아니라 **구간 위**에 — 막대 끝에 두면 오차 막대와 겹친다
    for x, v, h in zip(t["구분"], rate, hi, strict=True):
        if pd.notna(v):
            fig.add_annotation(x=x, y=h, text=f"<b>{v:.2f}%</b>", showarrow=False, yshift=12,
                               font={"color": theme.TEXT, "size": 13})
    overall = res["rate"] * 100
    fig.add_hline(y=overall, line={"color": theme.TEXT_MUTED, "width": 1, "dash": "dot"},
                  annotation_text=f"전체 {overall:.2f}%", annotation_position="top left",
                  annotation_font={"color": theme.TEXT_SUB, "size": 12})
    top = float(np.nanmax(hi)) if len(hi) else 1.0
    fig.update_layout(showlegend=False, margin={"l": 4, "r": 4, "t": 16, "b": 4},
                      yaxis={"title": None, "ticksuffix": "%", "range": [0, top * 1.22]},
                      xaxis={"title": None})
    return fig


_COMPACT_CFG = {"연체율": st.column_config.NumberColumn(format="%.2f%%"),
                "금액가중": st.column_config.NumberColumn(
                    format="%.2f%%", help="대출금액으로 가중한 연체율 — 큰 대출의 연체가 더 무겁다"),
                "대출": st.column_config.NumberColumn(format="%,.0f"),
                "연체": st.column_config.NumberColumn(format="%,.0f"),
                "브랜드": st.column_config.NumberColumn(format="%,.0f")}


def _compact(t: pd.DataFrame) -> pd.DataFrame:
    """화면용 좁은 표 — 엑셀에는 원래 열을 그대로 둔다."""
    out = pd.DataFrame({
        "구분": t["구분"], "대출": t["대출 수"], "연체": t["연체 수"], "연체율": t["연체율(%)"],
        "95% 구간": [f"{a:.2f}~{b:.2f}%" if pd.notna(a) else "-"
                     for a, b in zip(t["95% 구간 하한(%)"], t["95% 구간 상한(%)"], strict=True)],
        "브랜드": t["브랜드 수"]})
    if "금액 가중 연체율(%)" in t.columns:
        out["금액가중"] = t["금액 가중 연체율(%)"]
    return out


def _grade_table(res: dict) -> None:
    st.dataframe(_compact(res["by_grade"]), hide_index=True, width="stretch",
                 column_config=_COMPACT_CFG)
    a = res["auc"]
    st.caption(f"브랜드 위험만으로 대출 연체를 가려내는 힘(AUC) {a['est']:.3f} "
               f"[95% {a['lo']:.3f}~{a['hi']:.3f}] · 0.5면 무작위, 1이면 완벽. "
               f"등급 기준: {'취급연도 − ' + str(res['lag_years']) + '년 실적' if res['pit'] else '최신 실적'}.")


def _incremental(res: dict) -> None:
    inc = res["incremental"]
    st.markdown("##### 내부등급 대비 추가 정보")
    if inc is None:
        st.caption("내부등급(또는 CB점수) 열이 없어 이 검증은 건너뛰었습니다. 이 검증이 있어야 "
                   "'기존 심사에 없는 정보인가'에 답할 수 있습니다.")
        return
    k1, k2, k3 = st.columns(3)
    k1.metric("브랜드 위험 오즈비 (내부등급 통제)", f"{inc['or']:.2f}배",
              help=f"내부등급이 같을 때 브랜드 위험이 1표준편차 높으면 연체 오즈가 몇 배인지. "
                   f"95% 구간 {inc['or_lo']:.2f}~{inc['or_hi']:.2f} · p={inc['p']:.4f} · "
                   f"브랜드 {inc['n_clusters']:,}개 군집 강건 표준오차. 구간 하한이 1보다 크면 "
                   "추가 정보가 있다고 판정합니다.")
    k2.metric("내부등급만 · 표본 밖 AUC", f"{inc['base']:.3f}")
    k3.metric("+ 브랜드 위험 · 표본 밖 AUC", f"{inc['aug']:.3f}",
              delta=f"{inc['delta']:+.3f}", delta_color="normal",
              help="브랜드 단위 5겹 교차검증 — 폴드마다 표본 밖 AUC 를 재고 평균했습니다.")
    st.caption(f"오즈비 95% 구간 {inc['or_lo']:.2f}~{inc['or_hi']:.2f} · 대출 {inc['n']:,}건")
    if res["strata"] is not None:
        st.caption("같은 내부등급 층 안에서도 FranSCORE 등급에 따라 연체율이 갈리는지 — "
                   "갈린다면 브랜드 위험이 내부등급에 없는 정보입니다.")
        s = res["strata"]
        show = ["내부등급 층", "대출 수"] + [f"{g} {D.GRADE_KR[g]} 연체율(%)" for g in D.GRADES]
        st.dataframe(s[show], hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f%%")
                                    for c in show[2:]})


def _state_table(res: dict) -> None:
    t = res["by_state"]
    if t.empty:
        return
    st.markdown("##### 브랜드 상태별")
    st.caption("'악화 발생'은 취급 당시 공시에 이미 악화 사건이 나타난 브랜드입니다.")
    st.dataframe(_compact(t), hide_index=True, width="stretch", column_config=_COMPACT_CFG)
