"""점검 큐 — 이번 분기에 실제로 처리해야 할 브랜드 목록과 그 처리 상태.

이 화면이 하는 일은 하나다: **누가 무엇을 언제까지 확인할지 정하고, 결과를 남기는 것.**
'점수 목록을 필터링해서 CSV로 내려받는 화면'이 아니라 업무 흐름 화면이다.
배정 → 검토 → 처리결과 기록 → 반출 순서로 화면을 배치한다.
"""
from __future__ import annotations

import io
import json
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from src import theme
from src.views import common as C

STATUS = ["미착수", "검토 중", "조치 완료", "이상 없음"]
STATUS_KIND = {"미착수": "High", "검토 중": "Medium",
               "조치 완료": "Low", "이상 없음": "Neutral"}
_KEY = "queue_state"


_STATE_FILE = "queue_state.json"


def _state_path():
    return C.out_dir() / _STATE_FILE


def _state() -> dict:
    """처리 상태 저장소 (브랜드ID → {status, owner, note}).

    ⚠️ 예전에는 세션에만 담았다. 새로고침 한 번에 배정과 메모가 전부 사라졌다 —
       **기록이 사라지는 큐는 큐가 아니다.** 파일로 남겨 새로고침·재기동을 견디게 한다.

       다만 이것이 은행 업무 시스템을 대신하지는 못한다. 사용자 인증도, 변경 이력도,
       결재 연동도 없다. 그 사실을 화면에 그대로 적고, 실제 운영은 반출(엑셀)로
       기존 결재 흐름에 넘기는 것을 전제로 한다.
    """
    if _KEY not in st.session_state:
        p = _state_path()
        try:
            st.session_state[_KEY] = (json.loads(p.read_text(encoding="utf-8"))
                                      if p.exists() else {})
        except (OSError, ValueError):
            st.session_state[_KEY] = {}
    return st.session_state[_KEY]


def _save_state() -> None:
    try:
        p = _state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(st.session_state.get(_KEY, {}), ensure_ascii=False,
                                indent=1), encoding="utf-8")
    except OSError:
        pass          # 쓰기 불가 환경(읽기전용 배포)에서도 화면은 계속 동작해야 한다


def render() -> None:
    df, meta = C.load_scores()
    if df is None:
        st.warning("아직 평가 결과가 없습니다.")
        return
    diag = C.load_diagnosis_summary()
    yr = meta.get("scored_year", "-")

    theme.page_header(
        "점검 큐",
        f"{yr}년 공시 기준으로 우선 확인이 필요한 브랜드입니다. "
        "담당자를 지정하고 확인 결과를 기록하면 목록에서 정리됩니다.",
        eyebrow="업무")

    work = df[df["risk_grade"].isin(["High", "Medium"])].copy()
    if diag is not None and not diag.empty:
        work = work.merge(
            diag[["brand_id", "headline_detail", "n_risk", "n_high", "categories",
                  "watch_score"]],
            on="brand_id", how="left")
    else:
        work["headline_detail"] = ""
        work["watch_score"] = pd.to_numeric(work["deterioration_rank_pct"], errors="coerce") * 100

    state = _state()
    work["처리상태"] = work["brand_id"].astype(str).map(
        lambda b: state.get(b, {}).get("status", "미착수"))
    work["담당"] = work["brand_id"].astype(str).map(
        lambda b: state.get(b, {}).get("owner", ""))

    done = work[work["처리상태"].isin(["조치 완료", "이상 없음"])]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("점검 대상", f"{len(work):,}")
    k2.metric("미착수", f"{int((work['처리상태'] == '미착수').sum()):,}")
    k3.metric("검토 중", f"{int((work['처리상태'] == '검토 중').sum()):,}")
    k4.metric("처리 완료", f"{len(done):,}",
              delta=f"{len(done) / max(len(work), 1) * 100:.0f}%", delta_color="off")

    st.write("")
    tab_work, tab_all = st.tabs(["오늘 처리할 건", "전체 목록 · 반출"])
    with tab_work:
        _worklist(work)
    with tab_all:
        _fulltable(work, yr)


# ---------------------------------------------------------------------------

def _worklist(work: pd.DataFrame) -> None:
    f1, f2, f3 = st.columns([1.4, 1.4, 1.2])
    grades = f1.multiselect("등급", ["High", "Medium"], default=["High"],
                            format_func=lambda g: C.GRADE_KR.get(g, g))
    stats = f2.multiselect("처리상태", STATUS, default=["미착수", "검토 중"])
    min_stores = f3.number_input("최소 가맹점 수", min_value=0, value=0, step=10)

    view = work.copy()
    if grades:
        view = view[view["risk_grade"].isin(grades)]
    if stats:
        view = view[view["처리상태"].isin(stats)]
    if min_stores > 0:
        view = view[pd.to_numeric(view["n_stores"], errors="coerce").fillna(0)
                    >= min_stores]
    view = _prioritize(view)

    n_watch = int((view["brand_state"] == "요주의").sum()) if "brand_state" in view else 0
    st.caption(
        f"조건에 맞는 **{len(view):,}건** · 상위 20건을 펼쳐 둡니다. "
        "순서는 **위험도 × 가맹점 수**입니다 — 같은 위험도라도 점포가 많으면 "
        "은행 익스포저가 크기 때문입니다.")
    if n_watch:
        st.caption(
            f"이 중 **{n_watch:,}건({n_watch / max(len(view), 1) * 100:.0f}%)이 요주의** — "
            "올해 공시에 이미 악화 사건이 발동한 브랜드입니다. 이들은 모델 학습 표본 밖이라 "
            "**확률값 대신 같은 사건수 브랜드의 실제 재발동률**로 순서를 잡았습니다.")

    if view.empty:
        st.success("조건에 해당하는 미처리 건이 없습니다.")
        return

    state = _state()
    for _, r in view.head(20).iterrows():
        bid = str(r["brand_id"])
        cur = state.get(bid, {})
        with st.container(border=True):
            a, b = st.columns([3, 1.5])
            with a:
                st.markdown(
                    f"<div style='display:flex;gap:11px;align-items:center'>"
                    f"{C.brand_mark_html(str(r['brand_name']), 56)}"
                    f"<div><div style='font-weight:700;font-size:{theme.FS_LG};color:{theme.INK}'>"
                    f"{r['brand_name']} {theme.grade_chip(str(r['risk_grade']))} "
                    f"{theme.chip(cur.get('status', '미착수'), STATUS_KIND[cur.get('status', '미착수')])}"
                    f"</div>"
                    f"<div style='font-size:{theme.FS_SM};color:{theme.TEXT_SUB}'>"
                    f"{r.get('industry_mid', '-')} · 가맹점 {int(r['n_stores']):,}개 · "
                    f"{_risk_label(r)}</div></div></div>",
                    unsafe_allow_html=True)
                detail = str(r.get("headline_detail") or "")
                if detail and detail != "nan":
                    st.markdown(
                        f"<div style='font-size:{theme.FS_BASE};color:{theme.TEXT};margin-top:9px;"
                        f"line-height:1.6'>{detail}</div>", unsafe_allow_html=True)
                # 이 브랜드의 숫자가 어느 근거에서 나왔는지 — 큐에서도 숨기지 않는다
                note = C.population_note(r)
                if note:
                    st.markdown(note, unsafe_allow_html=True)
            with b:
                owner = st.text_input("담당자", value=cur.get("owner", ""),
                                      key=f"own_{bid}", placeholder="이름 입력")
                status = st.selectbox("처리상태", STATUS,
                                      index=STATUS.index(cur.get("status", "미착수")),
                                      key=f"st_{bid}")
            note = st.text_input("확인 결과 메모", value=cur.get("note", ""),
                                 key=f"nt_{bid}",
                                 placeholder="예: 본부 재무자료 징구 완료, 자본잠식 아님")
            if (status != cur.get("status", "미착수") or owner != cur.get("owner", "")
                    or note != cur.get("note", "")):
                state[bid] = {"status": status, "owner": owner, "note": note,
                              "brand_name": str(r["brand_name"]),
                              "updated": datetime.now(UTC)
                                                 .astimezone().strftime("%Y-%m-%d %H:%M")}
                _save_state()


def _prioritize(view: pd.DataFrame) -> pd.DataFrame:
    """상태별로 다른 위험도를 쓴다 — 한 척도로 줄세우면 명세를 어긴다.

    왜 이렇게 바꿨나 (실측)
        예전에는 `deterioration_1y × n_stores` 하나로 전부 줄세웠다. 그런데 큐 상위
        20건 중 **18건이 요주의**였다. 요주의 구간은 `MODEL_USE_SPEC` 이 "확률값에
        성능 근거가 없으니 순위를 매기지 말라"고 못박은 바로 그 구간이다.
        즉 심사역이 가장 먼저 보는 화면이 **우리 명세가 금지한 줄세우기**를 하고 있었고,
        화면 어디에도 그 사실이 없었다(queue.py 에 brand_state 참조 0건).

        그렇다고 요주의를 큐에서 빼면 안 된다 — 실제로 다음 해 재발동률이 사건 1건
        24.1% / 2건 45.9% / 3건 64.8% 로 건전(9.4%)보다 훨씬 높다. 봐야 할 브랜드가
        맞다. 잘못된 것은 **근거 없는 숫자로 줄세운 것**이지 목록에 넣은 것이 아니다.

        그래서 건전은 모델 확률로, 요주의는 `watch_base_rates.csv` 의 **실현율**로
        위험도를 잡는다. 둘 다 '1년 내 악화' 확률이라 같은 축에서 비교 가능하고,
        요주의 쪽은 모델이 아니라 실적이라 명세를 어기지 않는다.
    """
    rates = C.watch_rates(0.0)

    def risk(r) -> float:
        if str(r.get("brand_state")) == "요주의":
            k = r.get("n_events_at_t")
            try:
                hit = rates.get(int(k)) if k is not None and str(k) != "nan" else None
            except (TypeError, ValueError):
                hit = None
            if hit:
                return float(hit["rate"])
        return float(pd.to_numeric(pd.Series([r.get("deterioration_1y")]),
                                   errors="coerce").fillna(0).iloc[0])

    v = view.copy()
    v["_risk"] = v.apply(risk, axis=1) if len(v) else []
    v["_pri"] = v["_risk"] * pd.to_numeric(v["n_stores"], errors="coerce").fillna(0)
    return v.sort_values("_pri", ascending=False)


def _risk_label(r) -> str:
    """카드에 띄울 위험도 문구 — 근거가 다르면 이름도 달라야 한다."""
    if str(r.get("brand_state")) == "요주의":
        rates = C.watch_rates(0.0)
        k = r.get("n_events_at_t")
        try:
            hit = rates.get(int(k)) if k is not None and str(k) != "nan" else None
        except (TypeError, ValueError):
            hit = None
        if hit:
            return (f"악화 사건 {int(k)}건 · <b>같은 조건 브랜드의 다음 해 재발동률 "
                    f"{hit['rate'] * 100:.1f}%</b>")
    return f"브랜드 리스크 {float(r['deterioration_1y']) * 100:.1f}%"


def _fulltable(work: pd.DataFrame, yr) -> None:
    state = _state()
    cols = [c for c in ("brand_name", "industry_major", "industry_mid", "n_stores",
                        "brand_state", "deterioration_1y", "risk_grade", "watch_score",
                        "n_risk", "n_high", "categories", "처리상태", "담당",
                        "headline_detail")
            if c in work.columns]
    view = work[cols].copy()
    view["risk_grade"] = view["risk_grade"].map(C.GRADE_KR).fillna(view["risk_grade"])
    # 상태를 컬럼으로 내보낸다 — 반출한 엑셀에서도 '이 확률에 근거가 있는가'가 보여야 한다
    if "brand_state" in view:
        ev = work.get("n_events_at_t")
        if ev is not None:
            view["brand_state"] = [
                f"요주의 {int(k)}건" if s == "요주의" and pd.notna(k) else s
                for s, k in zip(view["brand_state"], ev, strict=False)]
    view = view.sort_values("deterioration_1y", ascending=False)
    # ⚠️ deterioration_1y 는 0~1 비율이다. "%.1f%%" 서식은 값을 그대로 찍으므로 45.3%가
    #    "0.5%" 로 나온다(구버전 화면의 실제 결함). 표시 직전에 100을 곱한다.
    view["deterioration_1y"] = pd.to_numeric(view["deterioration_1y"], errors="coerce") * 100
    st.dataframe(
        view, hide_index=True,
        use_container_width=True, height=460,
        column_config={
            "brand_name": st.column_config.TextColumn("브랜드", width="medium"),
            "industry_major": st.column_config.TextColumn("업종"),
            "industry_mid": st.column_config.TextColumn("세부 업종"),
            "n_stores": st.column_config.NumberColumn("가맹점", format="%d"),
            "brand_state": st.column_config.TextColumn(
                "상태", help="요주의는 올해 공시에 이미 악화 사건이 발동한 브랜드입니다. "
                            "모델 학습 표본 밖이라 옆의 확률값에는 성능 근거가 없습니다 — "
                            "사건수별 실제 재발동률로 판단하십시오."),
            "deterioration_1y": st.column_config.NumberColumn("브랜드 리스크", format="%.1f%%"),
            "risk_grade": st.column_config.TextColumn("등급"),
            "watch_score": st.column_config.ProgressColumn(
                "감시 우선순위", format="%.0f", min_value=0, max_value=100),
            "n_risk": st.column_config.NumberColumn("소견", format="%d"),
            "n_high": st.column_config.NumberColumn("중대", format="%d"),
            "categories": st.column_config.TextColumn("위험 영역"),
            "headline_detail": st.column_config.TextColumn("대표 소견", width="large"),
        })

    st.caption(
        f"처리 상태는 `{_state_path().name}` 에 저장돼 새로고침·앱 재기동 후에도 남습니다. "
        "다만 이 데모는 클라우드 컨테이너 위에서 돌기 때문에 **저장소가 다시 배포되면 "
        "초기화**됩니다 — 기록을 보존하려면 아래 엑셀로 반출하십시오. "
        "은행 내부 도입 시에는 이 파일을 업무 DB 테이블로 대체합니다. "
        "다만 이 화면에는 사용자 인증도, 변경 이력도, 결재 연동도 없습니다 — "
        "**공식 기록은 아래에서 내려받아 은행 결재 흐름에 넘기십시오.**")
    c1, c2 = st.columns(2)
    c1.download_button(
        "점검 목록 내려받기 (Excel)", _excel(view),
        file_name=f"franscore_점검큐_{yr}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
    log = pd.DataFrame([{"brand_id": k, **v} for k, v in state.items()])
    c2.download_button(
        f"처리 기록 내려받기 ({len(log)}건)",
        (log if not log.empty else pd.DataFrame(
            columns=["brand_id", "brand_name", "status", "owner", "note", "updated"])
         ).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"franscore_처리기록_{yr}.csv", mime="text/csv",
        disabled=log.empty, use_container_width=True)


def _excel(df: pd.DataFrame) -> bytes:
    """엑셀로 반출. openpyxl 이 없으면 CSV 바이트로 물러선다."""
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="점검큐")
        return buf.getvalue()
    except Exception:
        return df.to_csv(index=False).encode("utf-8-sig")
