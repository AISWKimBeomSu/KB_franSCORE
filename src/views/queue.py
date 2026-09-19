"""점검 큐 — 이번 분기에 실제로 처리해야 할 브랜드 목록과 그 처리 상태.

이 화면이 하는 일은 하나다: **누가 무엇을 언제까지 확인할지 정하고, 결과를 남기는 것.**
'점수 목록을 필터링해서 CSV로 내려받는 화면'이 아니라 업무 흐름 화면이다.
배정 → 검토 → 처리결과 기록 → 반출 순서로 화면을 배치한다.
"""
from __future__ import annotations

import io
import json
import os
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


def store_mode() -> str:
    """처리 기록을 어디에 둘지 — 'file'(공유 파일) 또는 'session'(방문자별).

    ⚠️ 한때 모든 방문자가 `outputs/queue_state.json` 한 파일을 같이 썼다. 각 세션은
       시작할 때 읽은 내용 **전체로** 파일을 덮어써서, 공개 데모에서는 동시에 쓰는
       사람끼리 기록을 지웠고 남의 메모가 그대로 보였다. 사용자 인증이 없는 공개
       배포에서 공유 저장소는 쓰면 안 된다.
       → 공개 클라우드(Streamlit Community Cloud 는 `/mount/src` 에서 앱을 띄운다)에서는
         방문자별 세션에만 둔다. 사내·로컬 설치는 파일에 남겨 재기동을 견디게 한다.
         환경변수 `FRANSCORE_QUEUE_STORE=file|session` 이 있으면 그것이 우선한다.
    """
    env = os.getenv("FRANSCORE_QUEUE_STORE", "").strip().lower()
    if env in ("file", "session"):
        return env
    root = str(C.cfg().get("_root", ""))
    return "session" if root.startswith("/mount/src") else "file"


def _state() -> dict:
    """처리 상태 저장소 (브랜드ID → {status, owner, note, updated})."""
    if _KEY not in st.session_state:
        loaded: dict = {}
        if store_mode() == "file":
            p = _state_path()
            try:
                loaded = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            except (OSError, ValueError):
                loaded = {}
        st.session_state[_KEY] = loaded
    return st.session_state[_KEY]


def _save_state() -> None:
    if store_mode() != "file":
        return
    try:
        p = _state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(st.session_state.get(_KEY, {}), ensure_ascii=False,
                                indent=1), encoding="utf-8")
    except OSError:
        pass          # 쓰기 불가 환경(읽기전용 배포)에서도 화면은 계속 동작해야 한다


_LOG_KEY = "queue_log"
_LOG_FILE = "queue_log.jsonl"
_FIELD_KR = {"status": "처리상태", "owner": "담당자", "note": "확인 결과 메모"}


def _log() -> list[dict]:
    """변경 이력 — 덮어쓰지 않고 **덧붙이기만** 한다 (감사 추적).

    처리상태·담당·메모의 '현재 값'만 남기면, 누가 언제 '이상 없음'으로 바꿨는지, 그때 이 브랜드의
    등급이 무엇이었는지를 나중에 설명할 수 없다. 여신 사후관리 기록은 결과보다 **경위**가 중요하다.
    """
    if _LOG_KEY not in st.session_state:
        rows: list[dict] = []
        if store_mode() == "file":
            p = C.out_dir() / _LOG_FILE
            try:
                rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except (OSError, ValueError):
                rows = []
        st.session_state[_LOG_KEY] = rows
    return st.session_state[_LOG_KEY]


def _append_log(entries: list[dict]) -> None:
    _log().extend(entries)
    if store_mode() != "file":
        return
    try:
        p = C.out_dir() / _LOG_FILE
        with p.open("a", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _on_edit(bid: str, brand_name: str, snapshot: dict | None = None) -> None:
    """카드 입력이 바뀌는 **즉시** 기록한다 (위젯 콜백은 화면을 다시 그리기 전에 돈다).

    ⚠️ 예전에는 카드를 그리는 도중에 이전 값과 비교해 저장했다. 그런데 상단 KPI 는 그보다
       먼저 계산되므로 상태를 바꿔도 숫자가 한 박자 늦게 따라왔다. 콜백으로 옮기면 KPI 가
       방금 바꾼 상태를 바로 반영한다.
    snapshot: 바꾸는 시점의 등급·위험·상태 — 변경 이력에 함께 남긴다.
    """
    state = _state()
    before = state.get(bid, {})
    now = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    after = {
        "status": st.session_state.get(f"st_{bid}", "미착수"),
        "owner": st.session_state.get(f"own_{bid}", ""),
        "note": st.session_state.get(f"nt_{bid}", ""),
    }
    changes = []
    for k, v in after.items():
        old = before.get(k, "미착수" if k == "status" else "")
        if v != old:
            changes.append({"시각": now, "브랜드ID": bid, "브랜드": brand_name, "항목": _FIELD_KR[k],
                            "이전": old, "변경": v, "변경자": after["owner"] or "(미지정)",
                            **(snapshot or {})})
    state[bid] = {**after, "brand_name": brand_name, "updated": now[:16]}
    _save_state()
    if changes:
        _append_log(changes)


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

    # 주의·관찰 + **중대 신호가 있는 브랜드는 등급과 무관하게** 큐에 올린다
    # (계속기업 불확실성·자본잠식·등록취소가 있는데 '안정'이라 큐 밖에 있던 브랜드가 있었다).
    crit = C.critical_map()
    work = df[df["risk_grade"].isin(["High", "Medium"])
              | df["brand_id"].astype(str).isin(crit)].copy()
    work["중대 신호"] = work["brand_id"].astype(str).map(
        lambda b: " · ".join(x["title"] for x in crit.get(b, [])))
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
    # 중대 신호 브랜드를 먼저, 그 안팎은 같은 우선순위 규칙(위험 × 가맹점 수)으로
    work = C.prioritize(work)
    work = (work.assign(_crit=work["중대 신호"].astype(str).str.len() > 0)
                .sort_values("_crit", ascending=False, kind="stable"))

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
    grades = f1.multiselect("등급", ["High", "Medium", "Low"], default=["High"],
                            format_func=lambda g: C.GRADE_KR.get(g, g))
    stats = f2.multiselect("처리상태", STATUS, default=["미착수", "검토 중"])
    min_stores = f3.number_input("최소 가맹점 수", min_value=0, value=0, step=10)
    with_crit = st.checkbox("중대 신호 브랜드는 등급과 무관하게 포함", value=True,
                            help="본부 계속기업 불확실성·자본잠식·정보공개서 등록취소처럼 사건 자체가 "
                                 "위험 신호인 브랜드입니다. 모형 등급이 낮아도 먼저 확인합니다.")

    view = work
    is_crit = view["중대 신호"].astype(str).str.len() > 0
    if grades:
        view = view[view["risk_grade"].isin(grades) | (is_crit if with_crit else False)]
    if stats:
        view = view[view["처리상태"].isin(stats)]
    if min_stores > 0:
        view = view[pd.to_numeric(view["n_stores"], errors="coerce").fillna(0)
                    >= min_stores]

    n_watch = int((view["brand_state"] == "요주의").sum()) if "brand_state" in view else 0
    watch = C.STATE_LABEL["요주의"]
    st.caption(
        f"조건에 맞는 **{len(view):,}건** · 상위 20건을 펼쳐 둡니다. "
        "순서는 **중대 신호 우선 → 1년 내 악화 위험 × 가맹점 수**입니다 — 같은 위험이라도 "
        "점포가 많으면 은행 익스포저가 크기 때문입니다.")
    if n_watch:
        st.caption(
            f"이 중 **{n_watch:,}건({n_watch / max(len(view), 1) * 100:.0f}%)이 {watch}** — "
            "올해 공시에 이미 악화 사건이 나타난 브랜드입니다. 이들은 모델 학습 표본 밖이라 "
            "**확률값 대신 같은 사건수 브랜드의 실제 재발동률**로 순서를 잡았습니다.")

    if view.empty:
        st.success("조건에 해당하는 미처리 건이 없습니다.")
        return

    state = _state()
    for _, r in view.head(20).iterrows():
        bid = str(r["brand_id"])
        name = str(r["brand_name"])
        cur = state.get(bid, {})
        status_now = cur.get("status", "미착수")
        # 바꾸는 시점의 판단 근거 — 변경 이력에 함께 남긴다 (나중에 "왜 그때 이상 없음이었나"에 답하려고)
        snap = {"당시 등급": C.GRADE_KR.get(str(r["risk_grade"]), str(r["risk_grade"])),
                "당시 브랜드 상태": C.state_label(r.get("brand_state"), r.get("n_events_at_t")),
                "당시 1년 내 악화 위험(%)": round(float(r.get("_risk", 0.0)) * 100, 1),
                "당시 중대 신호": str(r.get("중대 신호") or ""),
                "기준 공시연도": str(C.scored_year())}
        with st.container(border=True):
            a, b = st.columns([3, 1.5])
            with a:
                st.markdown(
                    f"<div style='display:flex;gap:11px;align-items:center'>"
                    f"{C.brand_mark_html(name, 56)}"
                    f"<div><div style='font-weight:700;font-size:{theme.FS_LG};color:{theme.INK}'>"
                    f"{C.esc(name)} {theme.grade_chip(str(r['risk_grade']))} "
                    f"{theme.chip(status_now, STATUS_KIND[status_now])}"
                    f"</div>"
                    f"<div style='font-size:{theme.FS_SM};color:{theme.TEXT_SUB}'>"
                    f"{C.esc(r.get('industry_mid', '-'))} · 가맹점 {int(r['n_stores']):,}개 · "
                    f"{C.risk_basis_label(r)}</div></div></div>",
                    unsafe_allow_html=True)
                detail = str(r.get("headline_detail") or "")
                if detail and detail != "nan":
                    st.markdown(
                        f"<div style='font-size:{theme.FS_BASE};color:{theme.TEXT};margin-top:9px;"
                        f"line-height:1.6'>{C.esc(detail)}</div>", unsafe_allow_html=True)
                # 이 브랜드의 숫자가 어느 근거에서 나왔는지 — 큐에서도 숨기지 않는다
                note_html = C.population_note(r)
                if note_html:
                    st.markdown(note_html, unsafe_allow_html=True)
                crit_items = C.critical_map().get(bid)
                if crit_items:
                    st.markdown(C.critical_banner_html(crit_items), unsafe_allow_html=True)
            with b:
                st.text_input("담당자", value=cur.get("owner", ""), key=f"own_{bid}",
                              placeholder="이름 입력", on_change=_on_edit, args=(bid, name, snap))
                st.selectbox("처리상태", STATUS, index=STATUS.index(status_now),
                             key=f"st_{bid}", on_change=_on_edit, args=(bid, name, snap))
            st.text_input("확인 결과 메모", value=cur.get("note", ""), key=f"nt_{bid}",
                          placeholder="예: 본부 재무자료 징구 완료, 자본잠식 아님",
                          on_change=_on_edit, args=(bid, name, snap))


def _export_frame(work: pd.DataFrame, state: dict) -> pd.DataFrame:
    """반출용 표 — **화면과 같은 순서·같은 이름**으로.

    ⚠️ 예전 엑셀은 헤더가 brand_name·deterioration_1y 같은 내부 컬럼명 그대로였고,
       브랜드ID·메모·수정시각이 빠졌으며 정렬도 화면 우선순위와 달랐다. 결재에 첨부할
       파일은 받은 사람이 설명 없이 읽을 수 있어야 한다.
    """
    rows = []
    for rank, (_, r) in enumerate(work.iterrows(), start=1):
        bid = str(r["brand_id"])
        s = state.get(bid, {})
        shown = C.risk_pct(r.get("deterioration_1y"))
        rows.append({
            "우선순위": rank,
            "브랜드ID": bid,
            "브랜드": r.get("brand_name"),
            "업종": r.get("industry_major"),
            "세부 업종": r.get("industry_mid"),
            "가맹점 수": pd.to_numeric(r.get("n_stores"), errors="coerce"),
            "등급": C.GRADE_KR.get(str(r.get("risk_grade")), r.get("risk_grade")),
            "브랜드 상태": C.state_label(r.get("brand_state"), r.get("n_events_at_t")),
            "브랜드 리스크(%)": shown,
            "점검 기준 위험(%)": round(float(r.get("_risk", 0.0)) * 100, 1),
            "위험 소견 수": pd.to_numeric(r.get("n_risk"), errors="coerce"),
            "중대 소견 수": pd.to_numeric(r.get("n_high"), errors="coerce"),
            "위험 영역": r.get("categories"),
            "중대 신호": r.get("중대 신호") or "",
            "대표 소견": r.get("headline_detail"),
            "처리상태": s.get("status", "미착수"),
            "담당자": s.get("owner", ""),
            "확인 결과 메모": s.get("note", ""),
            "최종 수정": s.get("updated", ""),
        })
    return pd.DataFrame(rows)


def _fulltable(work: pd.DataFrame, yr) -> None:
    state = _state()
    view = _export_frame(work, state)
    st.dataframe(
        view.drop(columns=["브랜드ID"]), hide_index=True,
        use_container_width=True, height=460,
        column_config={
            "브랜드": st.column_config.TextColumn(width="medium"),
            "가맹점 수": st.column_config.NumberColumn(format="%d"),
            "브랜드 상태": st.column_config.TextColumn(
                help=f"{C.STATE_LABEL['요주의']}은 올해 공시에 이미 악화 사건이 나타난 브랜드입니다. "
                     "모델 학습 표본 밖이라 브랜드 리스크 확률에는 성능 근거가 없습니다 — "
                     "사건수별 실제 재발동률(점검 기준 위험)로 판단하십시오."),
            "브랜드 리스크(%)": st.column_config.NumberColumn(format="%.1f%%"),
            "점검 기준 위험(%)": st.column_config.NumberColumn(
                format="%.1f%%", help="건전 브랜드는 모형 확률, 악화 발생 브랜드는 재발동 실현율"),
            "대표 소견": st.column_config.TextColumn(width="large"),
        })

    where = (f"`{_state_path().name}` 파일에 저장돼 새로고침·재기동 후에도 남습니다"
             if store_mode() == "file" else
             "**이 브라우저 세션에만** 저장됩니다 — 공개 데모라 방문자끼리 기록을 공유하지 않습니다")
    st.caption(
        f"처리 상태는 {where}. 이 화면에는 사용자 인증도, 변경 이력도, 결재 연동도 없습니다 — "
        "**공식 기록은 아래에서 내려받아 은행 결재 흐름에 넘기십시오.** "
        "은행 내부 도입 시에는 이 저장소를 업무 DB 테이블로 대체합니다.")
    c1, c2 = st.columns(2)
    data, ext, mime = _excel(view)
    c1.download_button(
        f"점검 목록 내려받기 ({'Excel' if ext == 'xlsx' else 'CSV'})", data,
        file_name=f"franscore_점검큐_{yr}.{ext}", mime=mime, use_container_width=True)
    log = pd.DataFrame([{"brand_id": k, **v} for k, v in state.items()])
    c2.download_button(
        f"처리 기록 내려받기 ({len(log)}건)",
        (log if not log.empty else pd.DataFrame(
            columns=["brand_id", "brand_name", "status", "owner", "note", "updated"])
         ).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"franscore_처리기록_{yr}.csv", mime="text/csv",
        disabled=log.empty, use_container_width=True)

    hist = pd.DataFrame(_log())
    with st.expander(f"변경 이력 {len(hist):,}건 — 누가·언제·무엇을·당시 등급", expanded=False):
        if hist.empty:
            st.caption("아직 변경 이력이 없습니다. 담당자·처리상태·메모를 바꾸면 여기에 쌓입니다.")
        else:
            st.dataframe(hist.iloc[::-1].head(200), hide_index=True, use_container_width=True)
            st.download_button(
                "변경 이력 내려받기 (CSV)", hist.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"franscore_변경이력_{yr}.csv", mime="text/csv", on_click="ignore")
        st.caption("이력은 덧붙이기만 하고 고치지 않습니다. 은행 내부 도입 시에는 이 기록을 "
                   "사용자 인증과 묶어 업무 DB 에 남깁니다.")


def _excel(df: pd.DataFrame) -> tuple[bytes, str, str]:
    """엑셀로 반출. openpyxl 이 없으면 **CSV 로, 확장자도 CSV 로** 물러선다.

    ⚠️ 예전에는 실패하면 CSV 내용을 .xlsx 이름으로 내보내 엑셀이 파일을 열지 못했다.
    """
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="점검큐")
            ws = w.sheets["점검큐"]
            ws.freeze_panes = "C2"
            widths = {"브랜드": 22, "대표 소견": 60, "확인 결과 메모": 36, "위험 영역": 18}
            for i, col in enumerate(df.columns, start=1):
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = widths.get(
                    col, max(10, min(18, len(str(col)) * 2 + 2)))
        return (buf.getvalue(), "xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        return df.to_csv(index=False).encode("utf-8-sig"), "csv", "text/csv"
