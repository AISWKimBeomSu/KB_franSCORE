"""일괄 조회 — 신청 목록을 올리면 브랜드별 등급·상태·중대 신호·확인 서류를 한 번에.

심사역의 하루는 브랜드 하나가 아니라 **신청 목록**으로 시작한다. 목록을 올리면 행 순서를
그대로 둔 채 진단 열을 붙여 돌려준다. 계산은 src/batch.py 가 하고, 이 화면은 입력·확인·반출만
맡는다. 업로드 파일은 디스크에 쓰지 않는다(세션 메모리에서만 처리).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src import batch, theme
from src.views import common as C

_RES = "batch_result"


@st.cache_data(ttl=1800, show_spinner=False)
def _biz_status(values: tuple) -> pd.DataFrame:
    """국세청 원천이 30분 주기로 갱신되므로 같은 목록은 30분 동안 다시 묻지 않는다."""
    from src import nts
    return nts.lookup_status(list(values))


def _template() -> bytes:
    """예시 양식 — 공개 배포는 가명 예시(실명 등급을 공표하지 않는다 · src/public.py)."""
    if not C.is_public():
        return batch.template_bytes()
    from src import public
    scores, _ = C.load_scores()
    rows = [(n, amt, "") for n, amt in zip(public.example_names(scores, 5), (150, 80, 120, 200, 60),
                                           strict=False)]
    panel = C.load_panel(full=True)
    other = public.unscored_example(panel) if panel is not None else None
    if other:
        rows.append((other, 180, "공시에는 있으나 외식 업종이 아니어서 평가 대상이 아닌 예"))
    return batch.template_bytes(rows)


@st.cache_resource(show_spinner=False)
def _context(m_scores: float, m_diag: float) -> batch.Context:
    """산출물 묶음은 프로세스당 한 번만 읽는다 (산출물이 바뀌면 키가 바뀐다)."""
    return batch.Context.from_files(C.out_dir(), C.proc_dir())


def render() -> None:
    theme.page_header(
        "일괄 조회",
        "가맹점주 대출 신청 목록을 올리면 브랜드를 찾아 등급·상태·중대 신호·확인 서류를 한 번에 "
        "붙여 드립니다. 결과는 받은 목록 순서 그대로 엑셀로 내려받을 수 있습니다.",
        eyebrow="심사")

    c1, c2 = st.columns([1.6, 1])
    with c1:
        up = st.file_uploader("신청 목록 (엑셀 또는 CSV)", type=["xlsx", "xls", "csv"],
                              help="브랜드명 열이 있으면 됩니다. 신청번호·신청금액 등 다른 열은 "
                                   "그대로 되돌려 드립니다.")
    with c2:
        st.markdown(f"<div style='font-size:{theme.FS_SM};color:{theme.TEXT_SUB};margin:6px 0 8px'>"
                    "양식이 없으면 예시 파일로 시작하십시오. 통칭·동명 브랜드·평가 대상 아님이 "
                    "어떻게 처리되는지 함께 보입니다.</div>", unsafe_allow_html=True)
        st.download_button("예시 양식 내려받기 (.xlsx)", _template(),
                           file_name="FranSCORE_일괄조회_양식.xlsx", width="stretch",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           on_click="ignore")
        use_sample = st.button("예시로 바로 실행", width="stretch")

    with st.expander("파일 없이 붙여넣기 — 한 줄에 브랜드 하나 (쉼표 뒤 금액은 선택)"):
        text = st.text_area("브랜드 목록", height=140, label_visibility="collapsed",
                            placeholder="메가커피, 150\n인생냉면, 80\n달리는커피")
        pasted = st.button("붙여넣은 목록 조회")

    df = None
    if up is not None:
        try:
            df = batch.read_upload(up.name, up.getvalue())
        except Exception as exc:                       # 파일 형식 문제로 화면이 죽으면 안 된다
            st.error(f"파일을 읽지 못했습니다 — {exc}")
            return
    elif pasted and text.strip():
        df = batch.from_text(text)
    elif use_sample:
        import io
        df = pd.read_excel(io.BytesIO(_template()), dtype=str)

    if df is not None:
        if df.empty:
            st.warning("목록이 비어 있습니다.")
            return
        if len(df) > 2000:
            st.warning("한 번에 2,000건까지 조회합니다 — 앞 2,000건만 처리했습니다.")
            df = df.head(2000)
        st.session_state[_RES] = {"input": df}

    state = st.session_state.get(_RES)
    if not state:
        _empty_hint()
        return
    df = state["input"]

    cols = [str(c) for c in df.columns]
    guess_b = batch.detect_brand_column(df)
    guess_a = batch.detect_amount_column(df)
    guess_n = batch.detect_bno_column(df)
    s1, s2, s3 = st.columns(3)
    brand_col = s1.selectbox("브랜드명 열", cols, index=cols.index(guess_b) if guess_b in cols else 0)
    amt_opts = ["(없음)", *cols]
    amount_col = s2.selectbox("신청금액 열 (선택)", amt_opts,
                              index=amt_opts.index(guess_a) if guess_a in amt_opts else 0)
    amount_col = None if amount_col == "(없음)" else amount_col
    bno_col = s3.selectbox("사업자번호 열 (선택)", amt_opts,
                           index=amt_opts.index(guess_n) if guess_n in amt_opts else 0,
                           help="있으면 국세청 사업자등록 상태(계속·휴업·폐업)를 함께 붙입니다. "
                                "공시는 1~2년 늦지만 국세청은 30분 주기로 갱신됩니다.")
    bno_col = None if bno_col == "(없음)" else bno_col

    ctx = _context(C._mtime(C.out_dir() / "scores_latest.csv"),
                   C._mtime(C.out_dir() / "brand_diagnosis.parquet"))
    with st.spinner("브랜드를 찾아 진단을 붙이는 중…"):
        res = batch.screen(df, brand_col, ctx, amount_col=amount_col)
    summ = batch.summarize(res, amount_col)
    if bno_col:
        with st.spinner("국세청 사업자 상태를 확인하는 중…"):
            res, biz = batch.attach_business_status(res, bno_col,
                                                    lookup=lambda v: _biz_status(tuple(v)))
        summ["biz"] = biz
        if biz["no_key"]:
            st.info("사업자 휴·폐업 확인에는 국세청 상태조회 키가 필요합니다 — 공공데이터포털 "
                    "「국세청_사업자등록정보 진위확인 및 상태조회 서비스」(15081808)를 활용신청한 뒤 "
                    "`DATA_GO_KR_KEY` 로 설정하십시오. 지금은 '확인불가'로 표시합니다.")
        elif biz["closed"] or biz["suspended"]:
            st.error(f"**국세청 기준 폐업 {biz['closed']:,}건 · 휴업 {biz['suspended']:,}건** — 브랜드 "
                     "등급과 무관하게 먼저 확인하십시오. '사업자 상태' 열을 보십시오.")
    _kpis(summ, amount_col)
    _table(res, amount_col)

    _, meta = C.load_scores()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "조회 결과 내려받기 (Excel · 결과/요약/안내 3개 시트)",
        batch.to_excel(res, summ, {"scored_year": meta.get("scored_year", "-"),
                                   "generated": datetime.now().strftime("%Y-%m-%d %H:%M")}),
        file_name=f"FranSCORE_일괄조회_{stamp}.xlsx", type="primary",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", on_click="ignore")
    st.caption("업로드한 파일은 서버에 저장하지 않고 이 세션 안에서만 처리합니다. "
               "등급은 브랜드의 구조악화 확률로 매긴 참고 지표이며 차주의 부도확률이 아닙니다 — "
               "여신 승인·거절, 한도·금리 결정에 사용하지 않습니다.")


def _empty_hint() -> None:
    st.write("")
    st.markdown(
        f"<div style='padding:18px 20px;border-radius:{theme.RADIUS_LG};border:1px dashed "
        f"{theme.BORDER_STRONG};color:{theme.TEXT_SUB};font-size:{theme.FS_BASE};line-height:1.7'>"
        "<b style='color:{ink}'>이렇게 쓰십시오</b><br>"
        "① 신청 목록 엑셀을 올리거나 브랜드명을 붙여넣습니다.<br>"
        "② 브랜드를 공시 등록명으로 찾아 붙입니다 — 통칭(메가커피)도 찾고, 이름이 정확히 같지 "
        "않으면 <b>확인 필요</b>로 표시합니다.<br>"
        "③ 등급·브랜드 상태·중대 신호·확인 사항·권고 서류가 붙은 결과를 엑셀로 내려받아 "
        "품의 자료에 붙입니다.<br>"
        "④ 목록에 <b>사업자번호</b> 열이 있으면 국세청 사업자 상태(계속·휴업·폐업)도 함께 붙입니다 "
        "— 공시보다 최신이라 이미 문을 닫은 신청인을 먼저 걸러 냅니다.</div>".replace("{ink}", theme.INK),
        unsafe_allow_html=True)


def _kpis(s: dict, amount_col: str | None) -> None:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("조회 건수", f"{s['n']:,}")
    k2.metric("매칭 성공", f"{s['matched']:,}",
              help="정확·통칭 일치에 더해, 이름이 달라 가장 가까운 브랜드를 붙인 '확인 필요'를 포함합니다.")
    k3.metric("확인 필요", f"{s['need_check']:,}", help="동명 브랜드 또는 이름이 정확히 같지 않은 건입니다.")
    k4.metric("주의 등급", f"{s['fs3']:,}",
              help=(f"신청금액 {s['amount_fs3']:,.0f}" if amount_col and "amount_fs3" in s else None))
    k5.metric("중대 신호", f"{s['critical']:,}",
              help="본부 계속기업 불확실성·자본잠식·정보공개서 등록취소 — 등급과 무관하게 먼저 확인합니다.")


def _table(res: pd.DataFrame, amount_col: str | None) -> None:
    show = res.copy()
    order = {batch.MATCH_SAME_NAME: 0, batch.MATCH_FUZZY: 0, batch.MATCH_NOT_SCORED: 2,
             batch.MATCH_NOT_FOUND: 2}
    need = show["매칭 상태"].map(order).fillna(1)
    if (need == 0).any():
        st.warning(f"**확인 필요 {int((need == 0).sum())}건** — 이름이 정확히 같지 않거나 같은 이름의 "
                   "브랜드가 여럿이라 가장 가까운(가맹점이 많은) 브랜드를 붙였습니다. "
                   "'매칭 브랜드'와 '다른 후보'를 확인하십시오.")
    st.dataframe(
        show, hide_index=True, width="stretch", height=min(620, 42 + 36 * len(show)),
        column_config={
            "매칭 상태": st.column_config.TextColumn(width="small"),
            "매칭 브랜드": st.column_config.TextColumn(width="medium"),
            "가맹점 수": st.column_config.NumberColumn(format="%,.0f"),
            "브랜드 리스크(%)": st.column_config.NumberColumn(format="%.1f%%"),
            "1년 내 악화 위험(%)": st.column_config.NumberColumn(
                format="%.1f%%", help="건전 브랜드는 모형 확률, 악화 발생 브랜드는 같은 사건수 "
                                      "과거 브랜드의 다음 해 재발동률"),
            "확인 사항": st.column_config.TextColumn(width="large"),
            "대표 소견": st.column_config.TextColumn(width="large"),
            **({amount_col: st.column_config.NumberColumn(format="%,.0f")} if amount_col else {}),
        })
