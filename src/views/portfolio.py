"""여신 포트폴리오 — 여신이 움직이면 집중도·예상손실·꼬리손실이 그 자리에서 다시 계산된다.

왜 정적인 그림이 아니라 계산 화면인가
    여신 포트폴리오는 매일 바뀐다. 어느 브랜드에 신규로 나가면 그 브랜드 비중이 오르고,
    상환되면 내린다. 고정된 그림 한 장을 붙여두면 '분석 결과 보고서'이지 업무 도구가
    아니다. 이 화면에서는 실행·회수를 입력하면 총 익스포저·HHI·상위 집중도·
    예상손실(EL)·스트레스 EL·꼬리손실이 즉시 다시 계산된다.

장부는 둘 중 하나다 — 어느 쪽으로 계산했는지 화면 맨 위 '기준 장부' 띠가 항상 밝힌다
    · 공시 창업비용 기반 추정 (기본) — 파이프라인이 만든 60개 브랜드 예시 장부. 금액은 가정이다.
    · 은행 제공 장부 (업로드) — 은행이 가진 실제 잔액. 양식을 받아 채워 올리고, 브랜드 매칭을
      확인해 적용하면 이 화면 전체(지표·한도 점검·차트·표·꼬리손실·내려받기)가 그 장부로
      바뀐다. 파일은 이 세션의 메모리에만 있다(디스크에 쓰지 않는다).
    읽기·매칭·LGD·꼬리손실 계산은 src/loanbook.py 가 하고, 이 파일은 입력·확인·표시만 맡는다.

계산식은 전부 화면에서 다시 푼다 (portfolio.py 와 같은 규칙)
    EL        = 익스포저 × 악화확률 × 손실률(LGD)   (올린 장부는 담보유형별 LGD)
    스트레스   = 위험 상위 stress_top_pct 브랜드의 악화확률 × 배수 (1.0 상한) 로 다시 계산
    HHI       = Σ(브랜드 여신 비중)²
    꼬리손실   = 파이프라인(src/correlation.py)과 같은 2단계 요인 모형을 **지금 화면의 장부**로
                다시 돌린다. 같은 장부·같은 조정이면 캐시에서 바로 나온다.
    조정분을 반영해 **다시 계산**하지 않으면 화면의 숫자가 조작 결과와 어긋난다.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import loanbook, theme
from src.views import common as C

_ADJ = "pf_adjust"          # 추정 장부 위의 실행·회수: 브랜드ID → 조정액(백만원)
_ADJ_UP = "pf_adjust_up"    # 올린 장부 위의 실행·회수 (새 파일을 올리면 비운다)
_UP = "pf_upload"           # 올린 파일 원본 {name, data, sha, at, example} — 세션 메모리에만
_USE = "pf_use_upload"      # 참이면 화면 전체가 올린 장부 기준
_PARSED = "pf_parsed"       # 읽기·매칭 결과 (파일·읽는 방식이 같으면 다시 계산하지 않는다)
_OPT = "pf_read_opt"        # 읽는 방식 수동 지정 {brand_col, exposure_col, unit}
_FIX = "pf_match_fix"       # 매칭 키 → {"brand_id", "reviewed"} — 사용자가 확인·수정한 매칭
_LGD = "pf_lgd"             # 담보유형 → LGD (이 세션에서만 바꾼 가정)
_GEN = "pf_file_gen"        # 파일 올리기 위젯을 비울 때 키를 바꾸는 세대 번호
_APPLIED = "pf_applied"     # 이 파일을 한 번이라도 적용했는가 (되돌린 뒤에는 미리보기를 접어 둔다)

EXPOSURE_NOTE = {
    "actual_loan_book": "은행이 제공한 실제 여신 잔액입니다.",
    "disclosed_startup_cost":
        "공정거래위원회 공시 창업비용 실측값에 대출조달비율·은행점유율 가정을 곱해 "
        "추정한 값입니다. 실제 여신 잔액은 은행 내부 자료이므로 공개 데이터에 없습니다.",
    "synthetic_lognormal": "방법론 실증을 위한 합성 예시입니다.",
}
# 기본 장부의 이름은 파이프라인이 무엇으로 만들었는지(basis)에 따른다 — config 의
# portfolio.exposure_source 로 실제 장부를 물린 배포라면 기본 장부가 곧 실제 잔액이다.
_BASIS_LABEL = {"actual_loan_book": "은행 제공 장부 (파이프라인 설정)",
                "disclosed_startup_cost": "공시 창업비용 기반 추정",
                "synthetic_lognormal": "합성 예시 장부"}


def _book_label(kind: str, basis: str) -> str:
    return "은행 제공 장부" if kind == "upload" else _BASIS_LABEL.get(basis, "공시 창업비용 기반 추정")


def _adj(key: str = _ADJ) -> dict:
    if key not in st.session_state:
        st.session_state[key] = {}
    return st.session_state[key]


def _align_latest(pf: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """포트폴리오의 위험도·등급을 **FRANSCORE 와 같은 최신 점수표**로 맞춘다.

    ⚠️ 파이프라인의 포트폴리오 산출물은 백테스트 코호트(test 연도 예측)와 60개 안에서의
       순위 등급으로 만들어진다 — 방법론 검증용으로는 맞지만 업무 화면으로는 어긋났다.
       60개 중 23개 브랜드가 FRANSCORE 와 등급이 달랐고(예: 한 브랜드가 목록에서는 27.4%
       주의, 여기서는 4.9% 안정), 새로 추가한 행만 최신 점수를 써서 한 표에 두 기준이
       섞였다. 익스포저(창업비용 기반 추정)는 그대로 두고 **위험도·등급만** 최신 점수로
       바꾼다. 최신 점수표에 없는 브랜드는 산출물 값을 유지한다.
    반환: (정렬된 표, 최신 점수로 바뀐 행 수)
    """
    scores, _ = C.load_scores()
    out = pf.copy()
    out["brand_id"] = out["brand_id"].astype(str)
    if scores is None or scores.empty:
        return out, 0
    s = scores[["brand_id", "deterioration_1y", "risk_grade"]].copy()
    s["brand_id"] = s["brand_id"].astype(str)
    s = s.drop_duplicates("brand_id").set_index("brand_id")
    hit = out["brand_id"].isin(s.index)
    out.loc[hit, "deterioration_1y"] = out.loc[hit, "brand_id"].map(s["deterioration_1y"])
    out.loc[hit, "risk_grade"] = out.loc[hit, "brand_id"].map(s["risk_grade"])
    return out, int(hit.sum())


def render() -> None:
    cfg = C.cfg()
    pcfg = cfg["portfolio"]
    theme.page_header(
        "여신 포트폴리오",
        "브랜드별 여신 쏠림과 대리 예상손실·꼬리손실입니다. 은행 여신 장부를 올리면 화면 전체가 "
        "그 장부로 바뀌고, 실행·회수를 입력하면 모든 지표가 즉시 다시 계산됩니다.",
        eyebrow="여신관리")

    scores, _ = C.load_scores()
    est, summary = C.load_portfolio()
    state = _upload_state(scores) if scores is not None else None
    use_up = bool(st.session_state.get(_USE)) and _usable(state)
    if st.session_state.get(_USE) and not use_up:
        st.session_state[_USE] = False          # 올린 장부를 더 쓸 수 없으면 추정 장부로 돌아간다

    basis = ((summary.get("assumptions") or {}).get("exposure") or {}).get("basis", "")
    n_aligned = 0
    if use_up:
        kind, pf = "upload", state["brands"].copy()
    elif est is not None:
        kind = "estimate"
        pf, n_aligned = _align_latest(est)
        # 추정 장부의 가정 — 파이프라인 꼬리손실(correlation_impact)과 같다:
        # LGD 는 중간 시나리오, 여신은 그 브랜드 가맹점 전체에 고르게 나뉜다.
        pf["lgd"] = _mid_lgd_value(pcfg)
        pf["n_borrowers"] = pd.to_numeric(pf["n_stores"], errors="coerce").fillna(1.0).clip(lower=1.0)
    else:
        kind, pf = "none", None

    label = _book_label(kind, basis)
    _basis_banner(kind, pf, state, label, basis)
    _upload_card(state, use_up, scores)

    if pf is None:
        st.warning("추정 포트폴리오 산출물이 없습니다. 위에서 은행 여신 장부를 올리면 그 장부로 계산합니다.")
        return

    if kind == "estimate":
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
                f"위에서 행내 여신 장부를 올리면 같은 화면이 그대로 실측 기준으로 바뀝니다.")
        st.caption(f"위험도·등급은 FRANSCORE 와 같은 **{C.scored_year()}년 실적 점수**입니다"
                   f"(포트폴리오 {n_aligned}/{len(pf)}개 브랜드). 대리 예상손실 = 여신 × 브랜드 리스크 "
                   "× 손실률(LGD) — 브랜드 리스크는 차주의 부도확률(PD)이 아니라 브랜드 공시 지표의 "
                   "구조악화 확률이라, 이 금액은 충당금·규제자본 산출에 쓰지 않습니다.")
    else:
        _upload_basis_note(state)

    adj_key = _ADJ_UP if kind == "upload" else _ADJ
    adj = _adj(adj_key)
    base = pf.copy()
    base["brand_id"] = base["brand_id"].astype(str)
    base["_adj"] = base["brand_id"].map(lambda b: float(adj.get(b, 0.0)))
    base["exposure_mkrw"] = (pd.to_numeric(base["exposure_mkrw"], errors="coerce").fillna(0)
                             + base["_adj"]).clip(lower=0.0)

    # 목록에 없던 브랜드에 신규 실행한 경우 행을 만들어 붙인다
    extra = [b for b in adj if b not in set(base["brand_id"])]
    if extra:
        base = pd.concat([base, _new_rows(extra, adj, _lgd_current()[1])], ignore_index=True)

    port = _recompute(base, pcfg)
    _kpis(port, adj)
    st.write("")

    tabs = st.tabs(["여신 조정", "집중도", "예상손실", "꼬리손실", "브랜드별 명세"])
    with tabs[0]:
        _adjust_panel(port, adj, adj_key, kind)
    with tabs[1]:
        _concentration(port)
    with tabs[2]:
        _expected_loss(port, pcfg)
    with tabs[3]:
        _tail_section(port, kind, label)
    with tabs[4]:
        _detail_table(port, kind, label)

    note = (EXPOSURE_NOTE["actual_loan_book"] if kind == "upload"
            else EXPOSURE_NOTE.get(basis, EXPOSURE_NOTE["synthetic_lognormal"]))
    st.caption("익스포저 산출 근거 — " + note + "  이 지표는 2선 리스크 관리 참고용이며 "
               "자동 여신 결정에 사용되지 않습니다.")


# ---------------------------------------------------------------------------
# 올린 장부 — 상태
# ---------------------------------------------------------------------------

def _lgd_current() -> tuple[dict[str, float], float]:
    """지금 쓰는 담보유형별 LGD. 화면에서 바꾼 값이 있으면 그것, 없으면 config 기본값."""
    lmap, ldef = loanbook.lgd_assumptions(C.cfg())
    cur = st.session_state.get(_LGD) or {}
    lmap = {k: float(cur.get(k, v)) for k, v in lmap.items()}
    return lmap, float(cur.get("_default", ldef))


@st.cache_data(show_spinner=False)
def _all_brand_names_cached(path: str, m: float) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["brand_name"]).drop_duplicates().reset_index(drop=True)


def _all_brand_names() -> pd.DataFrame | None:
    """공시 전체 브랜드명 — 미매칭을 '평가 대상 아님'과 '공시에 없음'으로 가르는 데만 쓴다."""
    p = C.proc_dir() / "panel_full.parquet"
    if not p.exists():
        return None
    try:
        return _all_brand_names_cached(str(p), C._mtime(p))
    except (OSError, ValueError, KeyError):
        return None


@st.cache_data(show_spinner=False)
def _template_cached(m: float) -> bytes:
    scores, _ = C.load_scores()
    return loanbook.template_bytes(scores, C.cfg())


def _template() -> bytes:
    return _template_cached(C._mtime(C.out_dir() / "scores_latest.csv"))


def _upload_state(scores: pd.DataFrame) -> dict | None:
    """세션에 올린 장부를 읽고 매칭한 결과. 같은 파일·같은 읽는 방식이면 다시 계산하지 않는다.

    ⚠️ 이 결과를 st.cache_data 에 두지 않는다. 그 캐시는 **프로세스 전체가 공유**하고 세션이
       끝나도 남는다. 은행 장부 원본은 이 세션의 session_state 에만 두어, 창을 닫으면 사라진다.
    """
    up = st.session_state.get(_UP)
    if not up:
        return None
    opt = st.session_state.get(_OPT) or {}
    key = (up["sha"], opt.get("brand_col"), opt.get("exposure_col"), opt.get("unit"),
           C._mtime(C.out_dir() / "scores_latest.csv"))
    cache = st.session_state.get(_PARSED)
    if not cache or cache.get("key") != key:
        cache = {"key": key}
        try:
            raw = loanbook.read_book(up["name"], up["data"], brand_col=opt.get("brand_col"),
                                     exposure_col=opt.get("exposure_col"), unit=opt.get("unit"))
            cache["meta"] = dict(raw.attrs.get("loanbook") or {})
            cache["matched"] = loanbook.match_book(raw, scores, all_brands=_all_brand_names())
        except loanbook.BookReadError as exc:
            cache.update(error=str(exc), columns=exc.columns)
        except Exception as exc:                  # 예상 못 한 파일 모양으로 화면이 죽으면 안 된다
            cache.update(error=f"장부를 해석하지 못했습니다 — {exc}", columns=[])
        st.session_state[_PARSED] = cache
    out = {"upload": up, "opt": opt}
    if cache.get("error"):
        return {**out, "error": cache["error"], "meta": {"columns": cache.get("columns", [])}}
    lmap, ldef = _lgd_current()
    fixed = loanbook.apply_fixes(cache["matched"], scores, st.session_state.get(_FIX))
    return {**out, "error": None, "meta": cache["meta"], "matched_raw": cache["matched"],
            "matched": fixed, "brands": loanbook.brand_book(fixed, lmap, ldef),
            "summary": loanbook.summarize(fixed), "lgd": (lmap, ldef)}


def _usable(state: dict | None) -> bool:
    return bool(state) and not state.get("error") and not state["brands"].empty


def _reset_widgets() -> None:
    for k in list(st.session_state.keys()):
        if str(k).startswith(("pf_fix_", "pf_ok_", "pf_opt_")):
            del st.session_state[k]


def _set_upload(name: str, data: bytes, example: bool) -> None:
    _reset_widgets()
    st.session_state[_UP] = {"name": name, "data": data,
                             "sha": hashlib.sha256(data).hexdigest(),
                             "at": datetime.now().strftime("%Y-%m-%d %H:%M"), "example": example}
    st.session_state[_USE] = False          # 새 파일은 매칭을 확인한 뒤 적용한다
    for k in (_PARSED, _FIX, _OPT, _ADJ_UP, _APPLIED):
        st.session_state.pop(k, None)


def _on_file(wkey: str) -> None:
    f = st.session_state.get(wkey)
    if f is None:                            # 사용자가 올린 파일을 지웠다
        _clear_upload()
        return
    _set_upload(f.name, f.getvalue(), example=False)


def _load_example() -> None:
    st.session_state[_GEN] = int(st.session_state.get(_GEN, 0)) + 1   # 올리기 칸을 비운다
    _set_upload("예시_" + loanbook.TEMPLATE_NAME, _template(), example=True)


def _clear_upload() -> None:
    _reset_widgets()
    for k in (_UP, _PARSED, _FIX, _OPT, _ADJ_UP, _APPLIED):
        st.session_state.pop(k, None)
    st.session_state[_USE] = False
    st.session_state[_GEN] = int(st.session_state.get(_GEN, 0)) + 1


def _activate() -> None:
    st.session_state[_USE] = True
    st.session_state[_APPLIED] = True


def _deactivate() -> None:
    st.session_state[_USE] = False


def _default_kw(wkey: str, **kw) -> dict:
    """위젯 기본값은 **처음 그릴 때만** 넘긴다 — 이미 상태가 있으면 Streamlit 이 경고를 띄운다."""
    return {} if wkey in st.session_state else kw


# ---------------------------------------------------------------------------
# 올린 장부 — 화면
# ---------------------------------------------------------------------------

def _basis_banner(kind: str, pf: pd.DataFrame | None, state: dict | None, label: str,
                  basis: str) -> None:
    """기준 장부 띠 — 이 화면의 모든 숫자가 **어느 장부**에서 나왔는지 맨 위에서 밝힌다.

    두 장부는 같은 화면·같은 표로 보인다. 띠가 없으면 추정 금액으로 만든 HHI 가 실제 장부의
    숫자로 인용되거나, 그 반대가 일어난다.
    """
    if kind == "upload":
        up, s = state["upload"], state["summary"]
        tone, soft = theme.SAFE, theme.SAFE_SOFT
        tag = label + (" · 양식 예시(가상 금액)" if up.get("example")
                       or state["meta"].get("is_example") else "")
        bits = [f"<b>{C.esc(up['name'])}</b>", f"업로드 {up['at']}",
                f"<b>{s['n_brands']}개 브랜드</b>", f"총 {s['expo_used_mkrw'] / 100:,.1f}억원"]
        if s["n_check"]:
            bits.append(f"확인 필요 {s['n_check']}행 포함")
        if s["n_none"]:
            bits.append(f"미매칭 {s['n_none']}행 · {s['expo_none_mkrw'] / 100:,.1f}억원 제외")
        sub = ("이 장부는 이 세션의 메모리에만 있습니다 — 서버에 저장하지 않으며, 새로고침하거나 "
               "창을 닫으면 사라집니다.")
    elif kind == "estimate":
        real = basis == "actual_loan_book"
        tone, soft, tag = (theme.SAFE, theme.SAFE_SOFT, label) if real else (theme.WARN, theme.WARN_SOFT, label)
        total = float(pd.to_numeric(pf["exposure_mkrw"], errors="coerce").sum())
        bits = [f"<b>{len(pf)}개 브랜드</b>", f"총 {total / 100:,.0f}억원" + ("" if real else "(추정)")]
        sub = ("파이프라인 설정(portfolio.exposure_source)으로 물린 장부입니다. 다른 장부를 올리면 이 세션에서만 "
               "그 장부로 바뀝니다." if real else
               "실제 여신 잔액이 아닙니다. 아래에서 은행 여신 장부를 올리면 이 화면 전체가 그 장부로 다시 계산됩니다.")
    else:
        return
    st.markdown(
        f"<div style='padding:12px 16px;border-radius:{theme.RADIUS_LG};background:{soft};"
        f"border:1px solid {theme.BORDER};border-left:5px solid {tone};margin:0 0 12px'>"
        f"<div style='font-size:{theme.FS_XS};font-weight:700;letter-spacing:.04em;color:{tone}'>"
        f"기준 장부</div>"
        f"<div style='font-weight:800;color:{theme.INK};font-size:{theme.FS_LG};margin-top:1px'>{tag}</div>"
        f"<div style='color:{theme.TEXT};font-size:{theme.FS_MD};margin-top:3px'>{' · '.join(bits)}</div>"
        f"<div style='color:{theme.TEXT_SUB};font-size:{theme.FS_SM};margin-top:4px'>{sub}</div></div>",
        unsafe_allow_html=True)


def _step(n: str, label: str) -> None:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0 6px'>"
        f"<span style='width:22px;height:22px;border-radius:50%;background:{theme.YELLOW};"
        f"color:{theme.INK};font-weight:800;font-size:{theme.FS_XS};display:inline-flex;"
        f"align-items:center;justify-content:center;flex:0 0 22px'>{n}</span>"
        f"<span style='font-weight:700;color:{theme.INK};font-size:{theme.FS_MD}'>{label}</span></div>",
        unsafe_allow_html=True)


def _upload_card(state: dict | None, active: bool, scores: pd.DataFrame | None) -> None:
    """양식 내려받기 → 올리기 → 매칭 확인 → 적용. 한 카드 안에서 순서대로 보이게 한다."""
    with st.container(border=True):
        st.markdown(
            f"<div style='display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:4px'>"
            f"<span style='font-weight:800;font-size:{theme.FS_LG};color:{theme.INK}'>실제 여신 장부 올리기</span>"
            f"<span style='color:{theme.TEXT_SUB};font-size:{theme.FS_SM}'>브랜드별 잔액만 있으면 됩니다 — "
            f"올리고 매칭을 확인해 적용하면 이 화면 전체가 그 장부로 다시 계산됩니다.</span></div>",
            unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1.45, 1.15], gap="medium")
        with c1:
            _step("1", "양식 내려받기")
            st.download_button("양식 내려받기 (.xlsx)", _template(), file_name=loanbook.TEMPLATE_NAME,
                               mime=loanbook.XLSX_MIME, on_click="ignore", width="stretch")
            st.button("예시 장부로 해 보기", on_click=_load_example, width="stretch")
        with c2:
            _step("2", "장부 파일 올리기")
            wkey = f"pf_file_{int(st.session_state.get(_GEN, 0))}"
            st.file_uploader("은행 여신 장부 (CSV · 엑셀)", type=["csv", "xlsx", "xls"], key=wkey,
                             on_change=_on_file, args=(wkey,), label_visibility="collapsed",
                             help="브랜드명(또는 brand_id)과 잔액 열이 있으면 됩니다. 담보유형·LGD·"
                                  "차주 수 열은 선택입니다.")
        with c3:
            _step("3", "매칭 확인 후 적용")
            _apply_box(state, active)
        st.caption("올린 파일은 서버 디스크에 저장하지 않고 이 세션의 메모리에서만 처리합니다 — "
                   "새로고침하거나 창을 닫으면 사라집니다. 차주 개인정보(성명·주민등록번호·계좌번호)는 "
                   "넣지 마십시오. 브랜드별 금액이면 충분합니다.")
        if not state:
            return
        if state.get("error"):
            st.error(f"**{C.esc(state['upload']['name'])}** — {state['error']}")
            if state["meta"].get("columns"):
                _read_options(state["meta"], expanded=True)
            return
        # 처음 올린 파일은 매칭을 확인해야 하므로 펼쳐 두고, 한 번 적용한 뒤에는 접어 둔다
        # (되돌린 뒤에도 긴 미리보기가 지표를 화면 아래로 밀어내지 않게).
        if active or st.session_state.get(_APPLIED):
            with st.expander("올린 장부 내용과 매칭 결과 보기", expanded=False):
                _review(state, scores)
        else:
            _review(state, scores)
        _lgd_editor(state)


def _apply_box(state: dict | None, active: bool) -> None:
    if not state:
        st.markdown(f"<div style='font-size:{theme.FS_SM};color:{theme.TEXT_SUB};line-height:1.6'>"
                    "파일을 올리면 브랜드 매칭 결과가 아래에 나옵니다. <b>확인 필요</b> 행을 살핀 뒤 "
                    "적용하면 화면 전체가 그 장부로 바뀝니다.</div>", unsafe_allow_html=True)
        return
    if state.get("error"):
        st.markdown(f"<div style='font-size:{theme.FS_SM};color:{theme.DANGER}'>파일을 읽지 못했습니다. "
                    "아래 안내를 보십시오.</div>", unsafe_allow_html=True)
        st.button("파일 지우기", on_click=_clear_upload, width="stretch", key="pf_clear_err")
        return
    s = state["summary"]
    chips = [theme.chip(f"확정 {s['n_ok']}", "Low")]
    if s["n_check"]:
        chips.append(theme.chip(f"확인 필요 {s['n_check']}", "Medium"))
    if s["n_none"]:
        chips.append(theme.chip(f"미매칭 {s['n_none']}", "High"))
    if s["n_drop"]:
        chips.append(theme.chip(f"제외 {s['n_drop']}", "Neutral"))
    st.markdown(f"<div style='margin:0 0 8px;line-height:2'>{' '.join(chips)}</div>",
                unsafe_allow_html=True)
    if active:
        st.markdown(f"<div style='font-size:{theme.FS_SM};color:{theme.SAFE};font-weight:700;"
                    f"margin-bottom:6px'>이 장부로 화면을 계산하고 있습니다</div>", unsafe_allow_html=True)
        st.button("추정 장부로 되돌리기", on_click=_deactivate, width="stretch")
    else:
        ok = not state["brands"].empty
        again = bool(st.session_state.get(_APPLIED))
        st.button("이 장부로 다시 전환" if again else "이 장부로 화면 전환", type="primary",
                  on_click=_activate, width="stretch", disabled=not ok)
        if not ok:
            st.caption("평가 대상 브랜드와 연결된 행이 없어 적용할 수 없습니다.")
    st.button("파일 지우기", on_click=_clear_upload, width="stretch", key="pf_clear")


def _review(state: dict, scores: pd.DataFrame | None) -> None:
    meta = state["meta"]
    _read_summary(meta)
    _read_options(meta, expanded=meta.get("unit_source") == "값 크기 추정")
    _preview(state)
    if scores is not None:
        _resolve_panel(state, scores)
    _unmatched(state)


def _read_summary(meta: dict) -> None:
    """'이렇게 읽었습니다' — 고른 열과 단위, 그 근거를 먼저 밝힌다(틀렸으면 바로 고칠 수 있게)."""
    cols = [("브랜드명", meta.get("brand_col")), ("브랜드ID", meta.get("brand_id_col")),
            ("잔액", meta.get("exposure_col")), ("담보유형", meta.get("collateral_col")),
            ("LGD", meta.get("lgd_col")), ("차주 수", meta.get("borrower_col"))]
    found = " · ".join(f"{k} ← <b>{C.esc(v)}</b>" for k, v in cols if v)
    sheet = f" · 시트 '{C.esc(meta['sheet'])}'" if meta.get("sheet") else ""
    st.markdown(
        f"<div style='font-size:{theme.FS_MD};color:{theme.TEXT};line-height:1.7;margin-top:6px'>"
        f"<b>읽은 방식</b> — {C.esc(meta.get('file_name', ''))}{sheet} · 머리글 {meta.get('header_row', 1)}행 · "
        f"장부 {meta.get('n_rows', 0):,}행<br>"
        f"<span style='color:{theme.TEXT_SUB}'>열 연결</span> {found}<br>"
        f"<span style='color:{theme.TEXT_SUB}'>금액 단위</span> <b>{meta.get('unit')}</b> "
        f"<span style='color:{theme.TEXT_SUB}'>— {C.esc(meta.get('unit_source', ''))}: "
        f"{C.esc(meta.get('unit_reason', ''))}</span></div>",
        unsafe_allow_html=True)
    if meta.get("unit_source") == "값 크기 추정":
        st.warning(f"**금액 단위를 값 크기로 추정했습니다** — {meta.get('unit_reason')}. 열 이름에 단위가 "
                   "없어서입니다. 틀렸으면 아래 '열·단위 직접 고르기'에서 단위를 고르십시오.")
    if meta.get("is_example"):
        st.info("이 파일에는 **'예시' 표기**가 남아 있습니다 — 양식의 가상 금액일 수 있습니다. "
                "실제 장부라면 비고 열의 예시 표기를 지우고 올리십시오.")
    for n in meta.get("notes") or []:
        st.caption(n)


def _on_opt() -> None:
    def pick(k: str) -> str | None:
        v = st.session_state.get(k)
        return None if v in (None, "(자동)") else str(v)
    st.session_state[_OPT] = {"brand_col": pick("pf_opt_brand"), "exposure_col": pick("pf_opt_expo"),
                              "unit": pick("pf_opt_unit")}


def _read_options(meta: dict, expanded: bool = False) -> None:
    cols = [str(c) for c in meta.get("columns") or []]
    if not cols:
        return
    opt = st.session_state.get(_OPT) or {}
    with st.expander("열·단위 직접 고르기", expanded=expanded):
        a, b, c = st.columns(3)
        o1 = ["(자동)", *cols]
        a.selectbox("브랜드 열", o1, key="pf_opt_brand", on_change=_on_opt,
                    **_default_kw("pf_opt_brand", index=o1.index(opt["brand_col"])
                                  if opt.get("brand_col") in o1 else 0))
        b.selectbox("잔액 열", o1, key="pf_opt_expo", on_change=_on_opt,
                    **_default_kw("pf_opt_expo", index=o1.index(opt["exposure_col"])
                                  if opt.get("exposure_col") in o1 else 0))
        o3 = ["(자동)", *loanbook.UNITS]
        c.selectbox("금액 단위", o3, key="pf_opt_unit", on_change=_on_opt,
                    **_default_kw("pf_opt_unit", index=o3.index(opt["unit"])
                                  if opt.get("unit") in o3 else 0))
        st.caption("'(자동)'은 열 이름으로 찾고, 단위는 열 이름 → 값 크기 순서로 정합니다.")


_STATUS_BG = {loanbook.MATCH_CHECK: theme.WARN_SOFT, loanbook.MATCH_DROP: "#F2F0EC"}


def _preview(state: dict) -> None:
    """연결된 행 미리보기 — '확인 필요'를 맨 위로 올리고 색으로 표시한다. 미매칭은 따로 모은다."""
    m = state["matched"]
    v = m[m["match_status"] != loanbook.MATCH_NONE].copy()
    if v.empty:
        return
    lmap, ldef = state["lgd"]
    lgd, _ = loanbook.row_lgd(v, lmap, ldef)
    v["_lgd"] = lgd
    v["_o"] = v["match_status"].map(loanbook.MATCH_ORDER).fillna(9)
    v = v.sort_values(["_o", "src_row"], kind="stable")
    raw = v["collateral_raw"].fillna("").astype(str)
    view = pd.DataFrame({
        "파일 행": v["src_row"].astype(int),
        "입력 브랜드": v["brand_input"].where(v["brand_input"].notna(), v["brand_id_input"]),
        "상태": v["match_status"],
        "매칭 브랜드": v["brand_name"].fillna("—"),
        "근거": v["match_rule"],
        "여신(억원)": pd.to_numeric(v["exposure_mkrw"], errors="coerce") / 100,
        "담보유형": np.where((raw != v["collateral"]) & (raw != ""), raw + " → " + v["collateral"],
                         v["collateral"]),
        "LGD": v["_lgd"] * 100,
        "차주 수": pd.to_numeric(v["n_borrowers"], errors="coerce"),
        "등급": v["risk_grade"].map(C.GRADE_KR).fillna("—"),
    })

    def paint(row: pd.Series) -> list[str]:
        bg = _STATUS_BG.get(row["상태"])
        if not bg:
            return [""] * len(row)
        return [f"background-color: {bg}" + (f"; color: {theme.WARN}; font-weight: 700"
                                             if c == "상태" and row["상태"] == loanbook.MATCH_CHECK else "")
                for c in row.index]

    styled = (view.style.apply(paint, axis=1)
              .format({"여신(억원)": "{:,.1f}", "LGD": "{:.0f}%", "차주 수": "{:,.0f}"}, na_rep="—"))
    st.markdown("##### 매칭 결과")
    st.dataframe(styled, hide_index=True, width="stretch", height=min(460, 40 + 35 * len(view)),
                 column_config={"입력 브랜드": st.column_config.TextColumn(width="medium"),
                                "매칭 브랜드": st.column_config.TextColumn(width="medium"),
                                "담보유형": st.column_config.TextColumn(width="medium")})
    st.caption("노란 행은 **확인 필요** — 등록명과 정확히 같지 않거나 같은 이름의 브랜드가 여럿이라 "
               "가장 가까운 후보를 임시로 붙였습니다(분석에는 포함). 아래에서 맞는 브랜드를 고르거나 "
               "'확인'을 누르면 확정됩니다. LGD 는 행에 적힌 값이 있으면 그 값, 없으면 담보유형 가정입니다.")


def _cand_label(bid: str, idx: pd.DataFrame, dup_names: set[str]) -> str:
    if not bid:
        return "반영하지 않음 (분석에서 뺌)"
    if bid not in idx.index:
        return bid
    r = idx.loc[bid]
    n = pd.to_numeric(pd.Series([r.get("n_stores")]), errors="coerce").iloc[0]
    name = str(r.get("brand_name"))
    bits = [name, str(r.get("industry_mid") or "-"), f"가맹점 {int(n):,}개" if pd.notna(n) else ""]
    if name in dup_names:
        bits.append(bid)
    return " · ".join(b for b in bits if b)


def _on_pick(key: str, wkey: str, okkey: str) -> None:
    fixes = st.session_state.setdefault(_FIX, {})
    fixes[key] = {"brand_id": str(st.session_state.get(wkey) or ""), "reviewed": True}
    st.session_state[okkey] = True           # 고른 것이 곧 확인한 것이다


def _on_ok(key: str, wkey: str, okkey: str, default: str) -> None:
    fixes = st.session_state.setdefault(_FIX, {})
    bid = st.session_state.get(wkey, default)
    fixes[key] = {"brand_id": str(bid or ""), "reviewed": bool(st.session_state.get(okkey))}


def _on_confirm_all(items: list[tuple[str, str, str, str]]) -> None:
    fixes = st.session_state.setdefault(_FIX, {})
    for key, wkey, okkey, default in items:
        fixes[key] = {"brand_id": str(st.session_state.get(wkey, default) or ""), "reviewed": True}
        st.session_state[okkey] = True


def _resolve_panel(state: dict, scores: pd.DataFrame) -> None:
    """확인 필요(와 비슷한 이름이 있는 미매칭) 이름마다 후보를 고르게 한다.

    같은 이름은 한 번만 묻는다 — 장부에 '국수나무'가 30줄 있어도 한 번 고르면 30줄에 적용된다.
    """
    raw = state["matched_raw"]
    need = raw[(raw["match_status"] == loanbook.MATCH_CHECK)
               | ((raw["match_status"] == loanbook.MATCH_NONE) & (raw["candidates"].map(len) > 0))]
    if need.empty:
        return
    fixes = st.session_state.get(_FIX) or {}
    idx = scores.assign(brand_id=scores["brand_id"].astype(str)).drop_duplicates("brand_id") \
        .set_index("brand_id")
    names = idx["brand_name"].astype(str)
    dup = set(names[names.duplicated(keep=False)])
    st.markdown("##### 확인할 이름")
    st.caption("후보가 맞으면 '확인'을, 다르면 맞는 브랜드를 고르십시오. 장부에 같은 이름이 여러 줄이면 "
               "한 번에 적용됩니다. '반영하지 않음'을 고르면 그 이름의 행을 분석에서 뺍니다.")
    pending: list[tuple[str, str, str, str]] = []
    for key, g in need.groupby("match_key", sort=False):
        r = g.iloc[0]
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        wkey, okkey = f"pf_fix_{h}", f"pf_ok_{h}"
        unmatched = r["match_status"] == loanbook.MATCH_NONE
        default = "" if unmatched else str(r["brand_id"] or "")
        fx = fixes.get(key) or {}
        current = str(fx.get("brand_id", default) or "")
        opts = list(dict.fromkeys([*[str(c) for c in r["candidates"]], current, ""]))
        rows = ", ".join(str(x) for x in g["src_row"].head(4)) + (" …" if len(g) > 4 else "")
        expo = float(pd.to_numeric(g["exposure_mkrw"], errors="coerce").sum()) / 100
        why = "미매칭 · 비슷한 이름" if unmatched else str(r["match_rule"])
        c1, c2, c3 = st.columns([1.35, 2.3, 0.7], vertical_alignment="center")
        c1.markdown(f"<div style='line-height:1.35'><b>{C.esc(r['brand_input'] or r['brand_id_input'])}</b>"
                    f"<div style='font-size:{theme.FS_XS};color:{theme.TEXT_SUB}'>{why} · 파일 {rows}행 · "
                    f"{expo:,.1f}억원</div></div>", unsafe_allow_html=True)
        c2.selectbox("매칭 브랜드", opts, key=wkey, label_visibility="collapsed",
                     format_func=lambda b: _cand_label(b, idx, dup),
                     on_change=_on_pick, args=(key, wkey, okkey),
                     **_default_kw(wkey, index=opts.index(current)))
        if unmatched:
            c3.markdown(f"<div style='font-size:{theme.FS_XS};color:{theme.TEXT_SUB}'>고르면 반영</div>",
                        unsafe_allow_html=True)
            continue
        c3.checkbox("확인", key=okkey, on_change=_on_ok, args=(key, wkey, okkey, default),
                    **_default_kw(okkey, value=bool(fx.get("reviewed"))))
        if not fx.get("reviewed"):
            pending.append((key, wkey, okkey, default))
    if pending:
        st.button(f"남은 {len(pending)}건 후보 그대로 모두 확인", on_click=_on_confirm_all,
                  args=(pending,), key="pf_confirm_all")


def _unmatched(state: dict) -> None:
    """찾지 못한 행은 **따로** 모아 보여 준다 — 분석에서 빠진 금액이 조용히 사라지면 안 된다."""
    m = state["matched"]
    um = m[m["match_status"] == loanbook.MATCH_NONE]
    if um.empty:
        return
    idx_names = {}
    scores, _ = C.load_scores()
    if scores is not None:
        idx_names = dict(zip(scores["brand_id"].astype(str), scores["brand_name"].astype(str),
                             strict=False))
    total = float(pd.to_numeric(um["exposure_mkrw"], errors="coerce").sum()) / 100
    st.markdown(f"##### 미매칭 {len(um)}행 · {total:,.1f}억원 — 분석에서 빠집니다")
    view = pd.DataFrame({
        "파일 행": um["src_row"].astype(int),
        "입력 브랜드": um["brand_input"].where(um["brand_input"].notna(), um["brand_id_input"]),
        "여신(억원)": pd.to_numeric(um["exposure_mkrw"], errors="coerce") / 100,
        "사유": um["match_rule"],
        "비슷한 이름": um["candidates"].map(lambda c: " · ".join(idx_names.get(str(b), str(b))
                                                            for b in list(c)[:3])),
    })
    st.dataframe(view, hide_index=True, width="stretch",
                 column_config={"여신(억원)": st.column_config.NumberColumn(format="%.1f")})
    st.caption("공시에 없음: 공정위 가맹사업 공시에서 찾지 못한 이름(오타·통칭일 수 있음). "
               "평가 대상 아님: 공시에는 있으나 외식업·가맹점 30개 이상 브랜드가 아니라 브랜드 리스크가 "
               "없습니다. 비슷한 이름이 있으면 위 '확인할 이름'에서 골라 넣을 수 있습니다.")


def _on_lgd() -> None:
    lmap, ldef = loanbook.lgd_assumptions(C.cfg())
    cur = {k: float(st.session_state.get(f"pf_lgd_{k}", v * 100)) / 100 for k, v in lmap.items()}
    cur["_default"] = float(st.session_state.get("pf_lgd__default", ldef * 100)) / 100
    st.session_state[_LGD] = cur


def _reset_lgd() -> None:
    lmap, ldef = loanbook.lgd_assumptions(C.cfg())
    for k, v in lmap.items():
        st.session_state[f"pf_lgd_{k}"] = v * 100
    st.session_state["pf_lgd__default"] = ldef * 100
    st.session_state.pop(_LGD, None)


def _lgd_editor(state: dict) -> None:
    """담보유형별 LGD 가정 — 은행 LGD 모형이 없을 때의 출발값. 바꾸면 화면 전체가 다시 계산된다."""
    lmap, ldef = _lgd_current()
    with st.expander("담보·손실률(LGD) 가정 바꾸기", expanded=False):
        cols = st.columns(4)
        for col, cat in zip(cols, loanbook.COLLATERAL_TYPES, strict=False):
            col.number_input(f"{cat} (%)", min_value=0.0, max_value=100.0, step=5.0, format="%.0f",
                             key=f"pf_lgd_{cat}", on_change=_on_lgd,
                             **_default_kw(f"pf_lgd_{cat}", value=float(lmap.get(cat, ldef)) * 100))
        cols[3].number_input("미기재·기타 (%)", min_value=0.0, max_value=100.0, step=5.0, format="%.0f",
                             key="pf_lgd__default", on_change=_on_lgd,
                             **_default_kw("pf_lgd__default", value=ldef * 100))
        n_over = int(state["summary"].get("n_lgd_override", 0))
        st.caption(
            "기본값의 근거 — 신용 45%: 바젤 기초 IRB 무담보 선순위 감독 LGD 와 같은 값(추정 장부의 중간 "
            "시나리오와도 같다). 담보 25%: 부동산 담보 감독 LGD 20%에 임차보증금 회수 시 미납 차임·"
            "원상복구비가 먼저 빠지는 점을 더했다. 보증서 10%: 신보·지역신보 보증비율 85% 이상 → 비보증분 "
            "15% × 45% ≈ 7%에 보증 면책 위험을 더했다. 담보유형을 모르면 없다고 본다(신용과 같게)."
            + (f" 장부에 LGD 가 적힌 {n_over}행은 이 가정 대신 그 값을 씁니다." if n_over else ""))
        st.button("기본값으로 되돌리기", on_click=_reset_lgd, key="pf_lgd_reset")


def _upload_basis_note(state: dict) -> None:
    """올린 장부로 계산할 때 쓴 가정을 표 위에 모아 둔다 — 단위·LGD·차주 수·빠진 행."""
    meta, s, bb = state["meta"], state["summary"], state["brands"]
    lmap, ldef = state["lgd"]
    n_book = int((bb["n_borrowers_basis"] == "장부 차주 수").sum())
    n_assume = len(bb) - n_book
    if not n_assume:
        who = f"장부에 적힌 값 (브랜드 {n_book}개 전부)"
    else:
        who = ((f"장부 값 {n_book}개 브랜드 · " if n_book else "")
               + f"<b>{n_assume}개 브랜드는 장부에 없어 가맹점 수로 가정</b> — 실제 차주가 더 적으면 "
               "여신이 덜 쪼개져 꼬리손실이 더 큽니다")
    lines = [
        f"금액 단위 <b>{meta.get('unit')}</b> ({C.esc(meta.get('unit_source', ''))})",
        "손실률(LGD) 담보유형별 가정 — "
        + " · ".join(f"{k} {lmap[k] * 100:.0f}%" for k in loanbook.COLLATERAL_TYPES if k in lmap)
        + f" · 미기재 {ldef * 100:.0f}%"
        + (f" (LGD 가 적힌 {s['n_lgd_override']}행은 그 값)" if s["n_lgd_override"] else ""),
        f"차주 수 — {who}",
    ]
    gone = [f"{nm} {n}행({e / 100:,.1f}억원)" for nm, n, e in (
        ("미매칭", s["n_none"], s["expo_none_mkrw"]), ("제외", s["n_drop"], s["expo_drop_mkrw"])) if n]
    if gone:
        lines.append("빠진 행 — " + " · ".join(gone) + ". 브랜드 리스크가 없어 계산할 수 없습니다.")
    st.markdown(
        f"<div style='padding:10px 14px;border-radius:{theme.RADIUS_MD};background:{theme.INFO_SOFT};"
        f"border:1px solid #CBDDF0;font-size:{theme.FS_SM};color:{theme.TEXT};line-height:1.7;"
        f"margin-bottom:8px'><b style='color:{theme.INFO}'>이 장부로 계산한 방식</b><br>"
        + "<br>".join(f"· {x}" for x in lines) + "</div>", unsafe_allow_html=True)
    st.caption(f"위험도·등급은 FRANSCORE 와 같은 **{C.scored_year()}년 실적 점수**입니다. 대리 예상손실 = "
               "여신 × 브랜드 리스크 × 담보유형별 LGD — 브랜드 리스크는 차주의 부도확률(PD)이 아니라 "
               "브랜드 공시 지표의 구조악화 확률이라, 이 금액은 충당금·규제자본 산출에 쓰지 않습니다.")


# ---------------------------------------------------------------------------
# 재계산
# ---------------------------------------------------------------------------

def _new_rows(brand_ids: list[str], adj: dict, lgd_default: float) -> pd.DataFrame:
    """포트폴리오에 없던 브랜드를 점수표에서 끌어와 행으로 만든다.

    새로 실행한 여신은 담보유형을 모르므로 기본 LGD(모름 = 신용과 같게)로 계산한다.
    """
    scores, _ = C.load_scores()
    rows = []
    for b in brand_ids:
        s = scores[scores["brand_id"].astype(str) == b] if scores is not None else None
        r = s.iloc[0] if s is not None and not s.empty else None
        n = float(r["n_stores"]) if r is not None else np.nan
        rows.append({
            "brand_id": b,
            "brand_name": str(r["brand_name"]) if r is not None else b,
            "industry_major": str(r.get("industry_major", "")) if r is not None else "",
            "industry_mid": str(r.get("industry_mid", "")) if r is not None else "",
            "n_stores": n,
            "deterioration_1y": float(r["deterioration_1y"]) if r is not None else 0.0,
            "risk_grade": str(r["risk_grade"]) if r is not None else "Low",
            "exposure_mkrw": max(float(adj.get(b, 0.0)), 0.0),
            "lgd": float(lgd_default),
            "n_borrowers": max(n, 1.0) if np.isfinite(n) else 1.0,
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

    # 장부 LGD — 올린 장부는 담보유형별(브랜드마다 다름), 추정 장부는 중간 시나리오 하나
    mid = _mid_lgd_value(pcfg)
    p["lgd"] = (pd.to_numeric(p["lgd"], errors="coerce").fillna(mid) if "lgd" in p.columns
                else mid)
    p["el_mkrw"] = p["exposure_mkrw"] * p["deterioration_1y"] * p["lgd"]
    p["stress_el_mkrw"] = p["exposure_mkrw"] * p["det_stressed"] * p["lgd"]

    for lgd in sorted(float(x) for x in pcfg["lgd_scenarios"]):
        k = f"lgd{round(lgd * 100):02d}"
        p[f"el_{k}_mkrw"] = p["exposure_mkrw"] * p["deterioration_1y"] * lgd
        p[f"stress_el_{k}_mkrw"] = p["exposure_mkrw"] * p["det_stressed"] * lgd
    return p


def _mid_lgd(pcfg: dict) -> str:
    lgds = sorted(float(x) for x in pcfg["lgd_scenarios"])
    return f"lgd{round(lgds[len(lgds) // 2] * 100):02d}"


def _mid_lgd_value(pcfg: dict) -> float:
    lgds = sorted(float(x) for x in pcfg["lgd_scenarios"])
    return lgds[len(lgds) // 2]


def _lgd_uniform(port: pd.DataFrame) -> bool:
    lg = pd.to_numeric(port["lgd"], errors="coerce")
    return bool(lg.notna().all() and float(lg.max() - lg.min()) < 1e-9)


def _lgd_label(port: pd.DataFrame) -> str:
    if _lgd_uniform(port):
        return f"LGD {float(port['lgd'].iloc[0]) * 100:.0f}%"
    return "담보별 LGD"


def _eok(mkrw) -> float:
    return float(mkrw) / 100.0


# ---------------------------------------------------------------------------
# 화면 조각
# ---------------------------------------------------------------------------

def _kpis(port: pd.DataFrame, adj: dict) -> None:
    if port.empty:
        st.info("여신 잔액이 0입니다. '여신 조정'에서 실행을 입력해 보십시오.")
        return
    total = float(port["exposure_mkrw"].sum())
    shares = port["exposure_share"].sort_values(ascending=False)
    hhi = float((shares ** 2).sum())
    top10 = float(shares.head(10).sum())
    el = float(port["el_mkrw"].sum())
    stress = float(port["stress_el_mkrw"].sum())

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
    k4.metric(f"대리 예상손실 ({_lgd_label(port)})", f"{_eok(el):,.1f} 억",
              help="여신 × 브랜드 리스크 × LGD. 브랜드 리스크는 부도확률(PD)이 아니라 브랜드 "
                   "구조악화 확률이므로, 이 값은 쏠림의 상대 크기를 보는 대리 지표입니다.")
    k5.metric("스트레스 시", f"{_eok(stress):,.1f} 억",
              delta=f"+{_eok(stress - el):,.1f} 억", delta_color="inverse")


def _adjust_panel(port: pd.DataFrame, adj: dict, adj_key: str, kind: str) -> None:
    scores, _ = C.load_scores()
    st.markdown("##### 여신 실행 · 회수")
    st.caption("브랜드를 고르고 금액을 입력하면 위쪽 지표와 꼬리손실이 즉시 다시 계산됩니다. "
               "회수는 음수로 입력합니다."
               + (" 장부에 없던 브랜드에 새로 실행하면 담보유형을 모르므로 기본 LGD 로 계산합니다."
                  if kind == "upload" else ""))

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
    if c3.button("반영", type="primary", width="stretch") and abs(amount) > 1e-9:
        adj[pick] = adj.get(pick, 0.0) + amount * 100.0         # 억원 → 백만원
        st.rerun()

    if adj:
        st.markdown("##### 반영된 조정")
        rows = [{"브랜드": label.get(b, b), "조정액(억원)": round(_eok(v), 1)}
                for b, v in adj.items() if abs(v) > 1e-9]
        if rows:
            d1, d2 = st.columns([3, 1])
            d1.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            d2.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if d2.button("전체 되돌리기", width="stretch"):
                st.session_state[adj_key] = {}
                st.rerun()
        else:
            st.caption("조정 내역이 없습니다.")
    else:
        st.caption("아직 조정한 여신이 없습니다. 현재 지표는 "
                   + ("올린 장부 그대로입니다." if kind == "upload" else "산출된 기준 포트폴리오입니다."))

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
        st.caption("막대 색 = 위험등급 (빨강 주의 · 주황 관찰 · 초록 안정)")
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
    by_book = not _lgd_uniform(port)
    if by_book:
        # 담보유형별 LGD 를 쓰는 장부는 그 값이 본 숫자다 — 균일 LGD 막대는 민감도로 뒤에 둔다
        labels = ["장부 LGD (담보별)", *labels]
        base = [_eok(port["el_mkrw"].sum()), *base]
        strs = [_eok(port["stress_el_mkrw"].sum()), *strs]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=base, name="기본",
                         marker_color=theme.YELLOW_DEEP,
                         hovertemplate="%{x}<br>기본 <b>%{y:,.1f}</b>억원<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=strs, name="스트레스",
                         marker_color=theme.DANGER_FILL,
                         hovertemplate="%{x}<br>스트레스 <b>%{y:,.1f}</b>억원<extra></extra>"))
    fig.update_layout(barmode="group", height=300, yaxis_title="대리 예상손실 (억원)",
                      margin={"l": 4, "r": 4, "t": 30, "b": 20})
    theme.plot(fig, key="pf_el")
    st.caption(
        "대리 예상손실 = 여신 × 브랜드 리스크 × 손실률(LGD). "
        + ("'장부 LGD'는 담보유형별 가정(행에 LGD 가 있으면 그 값)을 브랜드별 금액 가중으로 쓴 값이고, "
           "나머지 막대는 모든 여신에 같은 LGD 를 가정한 민감도입니다. " if by_book else "")
        + f"스트레스는 위험 상위 {float(pcfg['stress_top_pct']) * 100:.0f}% 브랜드의 "
        f"브랜드 리스크에 {pcfg['stress_pd_multiplier']}배를 적용해 동시 악화를 가정한 값입니다. "
        "여러 브랜드 차주가 함께 무너지는 해의 손실은 '꼬리손실' 탭에서 봅니다.")


# ---------------------------------------------------------------------------
# 꼬리손실 — 지금 화면의 장부로 다시 계산
# ---------------------------------------------------------------------------

_TAIL_INPUT = ["brand_id", "brand_name", "exposure_mkrw", "deterioration_1y", "n_borrowers", "lgd"]


@st.cache_data(show_spinner=False, max_entries=48, scope="session")
def _tail_cached(key: str, _book: pd.DataFrame, rho_w: float, rho_b: float,
                 n_sims: int, seed: int) -> dict:
    """꼬리손실 계산 캐시 — 키는 장부(계산에 쓰는 열)·ρ·횟수·시드의 지문이다.

    같은 장부로 돌아오면(조정 되돌리기, 탭 이동) 다시 계산하지 않는다.
    ⚠️ scope="session" 이 핵심이다. 기본(global) 캐시는 프로세스 전체가 공유하고 세션이 끝나도
       남아서, 올린 장부의 브랜드별 금액이 창을 닫은 뒤에도 서버 메모리에 머문다. 화면이
       "이 세션의 메모리에만 있다"고 말하는 것이 참이 되도록 세션이 끊기면 비운다.
    """
    del key
    return loanbook.tail_risk(_book, None, rho_w, rho_b, n_sims=n_sims, seed=seed,
                              return_samples=True)


def _tail_section(port: pd.DataFrame, kind: str, label: str) -> None:
    """어느 정도의 손실이 100년에 한 번 오는가, 그리고 **어느 브랜드가** 그 꼬리를 만드는가.

    왜 고정 산출물이 아니라 여기서 계산하는가
        예전 화면은 파이프라인이 기준 포트폴리오로 미리 계산한 표를 붙여 두었다. 그러면 실행·회수를
        넣거나 실제 장부를 올려도 꼬리손실 표만 그대로여서, 화면의 다른 숫자와 서로 다른 장부를
        말했다. 측정한 상관을 여신 판단에 쓰려면 **지금 이 장부**로 다시 돌려야 한다.
    """
    if port.empty:
        return
    rho = loanbook.load_rho(C.out_dir())
    if rho is None:
        st.info("브랜드 상관 측정 산출물(brand_correlation.json)이 없어 꼬리손실을 계산하지 않았습니다.")
        return
    cfg = C.cfg()
    n_sims = int((cfg.get("loanbook") or {}).get("tail_n_sims", 20_000))
    seed = int(cfg.get("seed", 42))
    book = port[[c for c in _TAIL_INPUT if c in port.columns]].copy()
    key = loanbook.book_hash(book, rho["rho_w"], rho["rho_b"], n_sims, seed)
    try:
        with st.spinner(f"지금 장부로 꼬리손실을 다시 계산하는 중… (몬테카를로 {n_sims:,}회)"):
            res = _tail_cached(key, book, rho["rho_w"], rho["rho_b"], n_sims, seed)
    except ValueError as exc:
        st.info(f"꼬리손실을 계산하지 못했습니다 — {exc}")
        return

    i99, c99 = res["independent_p99_mkrw"], res["brand_correlated_p99_mkrw"]
    st.markdown(f"<div style='font-weight:700;font-size:{theme.FS_LG};color:{theme.INK};margin-top:4px'>"
                f"꼬리손실 — 같은 브랜드 차주가 함께 무너지는 해</div>", unsafe_allow_html=True)
    st.caption(f"지금 화면의 장부({label}"
               f"{' · 조정 포함' if '_adj' in port.columns and port['_adj'].abs().sum() > 0 else ''}, "
               f"{res['n_brands']}개 브랜드 · 차주 {res['n_franchisees']:,}명)로 방금 계산한 값입니다. "
               "실행·회수나 LGD 가정을 바꾸면 다시 계산합니다.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("99% 손실 · 차주 독립 가정", f"{_eok(i99):,.1f} 억",
              help="가맹점주 차주를 서로 무관한 소상공인으로 볼 때, 1년 손실이 100번 중 1번꼴로 넘는 금액.")
    m2.metric("99% 손실 · 브랜드 상관 반영", f"{_eok(c99):,.1f} 억",
              delta=f"+{_eok(c99 - i99):,.1f} 억", delta_color="inverse",
              help="같은 브랜드 차주끼리 함께 나빠지는 정도(측정한 상관)를 넣었을 때의 같은 금액.")
    pct = res["understatement_p99_pct"]
    m3.metric("독립 가정의 과소평가 (99%)", f"{pct * 100:+.0f}%" if np.isfinite(pct) else "—")
    mult = res["ul99_multiple"]
    m4.metric("비예상손실(UL) 배수", f"{mult:,.1f}배" if np.isfinite(mult) else "—",
              help="(99% 손실 − 평균 손실)의 비 — 자본 관점에서 쌓아야 할 완충이 몇 배로 커지는가.")

    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        rows = []
        for lvl, nm in (("p95", "95%"), ("p99", "99%"), ("p999", "99.9%")):
            rows.append({
                "신뢰수준": nm,
                "차주 독립 가정": _eok(res[f"independent_{lvl}_mkrw"]),
                "브랜드 상관 반영": _eok(res[f"brand_correlated_{lvl}_mkrw"]),
                "과소평가": _eok(res[f"understatement_{lvl}_mkrw"]),
                "시뮬레이션 오차(±)": _eok(res[f"mc_se_correlated_{lvl}_mkrw"]),
            })
        st.markdown("##### 손실 분위수 (1년)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={
                         **{c: st.column_config.NumberColumn(c, format="%.1f 억")
                            for c in ("차주 독립 가정", "브랜드 상관 반영", "과소평가")},
                         # 오차는 작아서 한 자리로 자르면 '0.0 억'이 된다 — 없는 게 아니라 작은 것이다
                         "시뮬레이션 오차(±)": st.column_config.NumberColumn(format="%.2f 억")})
        st.caption(f"평균 손실 {_eok(res['expected_loss_mkrw']):,.1f}억원은 두 가정에서 같습니다 — 상관은 "
                   "평균이 아니라 **꼬리**를 바꿉니다. 오차는 몬테카를로 표준오차(부트스트랩)이고, 99.9%는 "
                   "시나리오가 적어 오차가 큽니다.")
    with c2:
        st.markdown("##### 이 금액 이상 손실이 날 확률")
        theme.plot(_exceedance_fig(res["samples"]), key="pf_tail_exceed")

    _tail_contribution(res, rho)

    def _ci(pair) -> str:
        # ⚠️ 범위 기호로 '~'를 쓰지 않는다 — 마크다운이 두 '~' 사이를 취소선으로 그린다(실측).
        lo, hi = pair or (None, None)
        return f" (95% 구간 {lo:.3f}–{hi:.3f})" if lo is not None and hi is not None else ""

    st.caption(
        f"상관 — 같은 브랜드 차주끼리 ρ_W {rho['rho_w']:.3f}{_ci(rho.get('rho_w_ci'))}, 다른 브랜드 "
        f"차주끼리 ρ_B {rho['rho_b']:.3f}{_ci(rho.get('rho_b_ci'))}. "
        "**지역별 가맹점 감소가 동시에 일어나는 정도로 잰 대리지표**이며, "
        "차주 부도의 상관을 직접 잰 값이 아닙니다. 모형은 파이프라인과 같은 가맹점(차주) 단위 2단계 요인 "
        "모형으로, 브랜드 리스크를 차주 악화 임계로 쓰고 여신을 차주 수만큼 고르게 나눕니다(차주 수가 "
        f"없으면 가맹점 수). 몬테카를로 {res['n_sims']:,}회 · 시드 {res['seed']} — 같은 장부면 항상 같은 "
        f"값이 나옵니다(계산 {res['elapsed_sec']:.2f}초). 브랜드 리스크는 부도확률이 아니라 구조악화 "
        "확률이라 이 금액은 충당금·규제자본 산출에 쓰지 않습니다.")
    if kind == "estimate":
        st.caption("저장소의 correlation_impact.json(20만 회)은 백테스트 코호트 위험도로 계산한 방법론 "
                   "검증용 산출물이고, 이 화면은 같은 모형을 최신 점수·현재 조정으로 다시 돌린 값입니다.")


def _exceedance_fig(samples: dict) -> go.Figure:
    """초과확률 곡선 — 두 가정의 손실 분포를 꼬리까지 한 그림에 겹친다(세로축 로그)."""
    fig = go.Figure()
    for lab, name, color in (("independent", "차주 독립 가정", theme.INFO_FILL),
                             ("brand_correlated", "브랜드 상관 반영", theme.DANGER_FILL)):
        x = np.sort(np.asarray(samples[lab], dtype=float))[::-1] / 100.0
        n = len(x)
        y = np.arange(1, n + 1) / n
        pick = np.unique(np.clip(np.round(np.logspace(0, np.log10(n), 360)).astype(int) - 1, 0, n - 1))
        fig.add_trace(go.Scatter(
            x=x[pick], y=y[pick], mode="lines", name=name, line={"color": color, "width": 2.6},
            hovertemplate="손실 %{x:,.1f}억원 이상<br>확률 %{y:.2%}<extra>" + name + "</extra>"))
    fig.add_hline(y=0.01, line={"dash": "dot", "color": theme.TEXT_MUTED, "width": 1.2})
    # ⚠️ 로그 축의 주석 좌표는 log10 값이다. add_hline 의 annotation_* 는 그 변환을 하지 않아
    #    '1%' 선의 설명이 그래프 맨 위(10^0.01)에 붙었다(실측) — 주석을 따로 단다.
    fig.add_annotation(x=1, xref="paper", y=np.log10(0.01), yref="y", showarrow=False,
                       xanchor="right", yanchor="bottom", text="99% 신뢰수준 · 100년에 한 번",
                       font={"size": 12, "color": theme.TEXT_SUB})
    fig.update_yaxes(type="log", tickvals=[1, 0.1, 0.01, 0.001], ticktext=["100%", "10%", "1%", "0.1%"],
                     range=[-3.3, 0.02], title="초과 확률")
    fig.update_xaxes(title="1년 손실 (억원)", ticksuffix="")
    fig.update_layout(height=300, margin={"l": 4, "r": 4, "t": 16, "b": 30},
                      legend={"orientation": "h", "y": 1.12, "x": 0})
    return fig


def _tail_contribution(res: dict, rho: dict) -> None:
    """어느 브랜드가 99% 꼬리를 만드는가 — 측정한 ρ 가 여신 판단으로 이어지는 자리.

    왜 이 표가 접힌 참고표가 아니라 본문인가 (심사 지적)
        상관을 측정해 놓고 화면에는 포트폴리오 총량(UL 배수)만 있었다. 심사역의 실제
        질문은 "그래서 어느 브랜드의 한도를 봐야 하는가"다. 성분 ES(Euler 배분)는
        합이 전체 ES 와 정확히 일치하는 유일한 가법 배분이라, 꼬리손실을 브랜드별로
        조작 없이 나눠 그 질문에 답한다. 이제 지금 장부로 계산하므로 조정에도 반응한다.
    """
    df = res["contrib"]
    if df.empty:
        return
    eu = res["euler"]
    st.markdown(f"<div style='font-weight:700;font-size:{theme.FS_LG};color:{theme.INK};"
                f"margin-top:18px'>꼬리손실 기여 상위 — 어느 브랜드가 99% 손실을 만드는가"
                f"</div>", unsafe_allow_html=True)
    top = df.head(10)
    # 비중은 %로 바꿔 한 자리로 맞춘다 — 'percent' 서식은 자릿수가 값마다 달라(5% · 10.42%) 세로로
    # 읽히지 않는다.
    view = pd.DataFrame({
        "브랜드": top["brand_name"],
        "여신 비중": top["exposure_share"] * 100,
        "UL 기여 비중": top["ul_share"] * 100,
        "쏠림 배수": top["concentration_ratio"],
        "꼬리손실 기여(억)": top["ul_contrib_mkrw"] / 100,
    })
    st.dataframe(
        view, hide_index=True, width="stretch",
        column_config={
            "여신 비중": st.column_config.NumberColumn(format="%.1f%%"),
            "UL 기여 비중": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0.0,
                max_value=float(max(view["UL 기여 비중"].max(), 1e-9))),
            "쏠림 배수": st.column_config.NumberColumn(
                format="%.2f배",
                help="UL 기여 비중 ÷ 여신 비중. 1보다 크면 여신 규모에 비해 꼬리손실 "
                     f"기여가 큰 브랜드 — 브랜드 내부 상관(ρ={rho['rho_w']:.3f})과 집중이 만드는 "
                     "위험 쏠림입니다."),
            "꼬리손실 기여(억)": st.column_config.NumberColumn(format="%.2f"),
        })
    st.caption(
        "총손실이 99% 분위수를 넘는 시나리오"
        f"({eu['n_tail_scenarios']:,}개)에서 각 브랜드가 평균적으로 만든 손실"
        "(성분 ES, Euler 배분 — 브랜드별 기여의 합이 전체 꼬리손실과 정확히 일치"
        f", 검산 오차 {eu['component_addup_rel_err']:.0e})에서 그 브랜드의 기대손실을 뺀 값입니다. "
        f"상위 5개 브랜드가 꼬리손실의 {eu['top5_ul_share'] * 100:.0f}%를 만듭니다. 한도 재검토 "
        "우선순위는 여신 잔액이 아니라 이 표의 순서를 따르는 것이 이 도구의 처방입니다.")
    export = pd.DataFrame({
        "브랜드": df["brand_name"], "brand_id": df["brand_id"], "차주 수": df["n_borrowers"],
        "여신(억원)": df["exposure_mkrw"] / 100, "여신 비중(%)": df["exposure_share"] * 100,
        "LGD(%)": df["lgd"] * 100, "기대손실(억원)": df["el_mkrw"] / 100,
        "성분 ES99(억원)": df["es99_mkrw"] / 100, "꼬리손실 기여(억원)": df["ul_contrib_mkrw"] / 100,
        "UL 기여 비중(%)": df["ul_share"] * 100, "쏠림 배수": df["concentration_ratio"],
    })
    st.download_button("꼬리손실 기여 전체 내려받기 (CSV)",
                       export.round(4).to_csv(index=False).encode("utf-8-sig"),
                       file_name="franscore_tail_contribution.csv", mime="text/csv",
                       on_click="ignore")


def _detail_table(port: pd.DataFrame, kind: str, label: str) -> None:
    if port.empty:
        return
    cols = ["brand_name", "industry_mid", "n_stores", "deterioration_1y", "risk_grade",
            "exposure_mkrw", "exposure_share", "lgd", "el_mkrw", "stress_el_mkrw"]
    if kind == "upload":
        cols += ["n_borrowers", "collateral_mix", "match_status"]
    view = port[[c for c in cols if c in port.columns]].copy()
    view["exposure_mkrw"] = view["exposure_mkrw"] / 100
    view["el_mkrw"] = view["el_mkrw"] / 100
    view["stress_el_mkrw"] = view["stress_el_mkrw"] / 100
    view["risk_grade"] = view["risk_grade"].map(C.GRADE_KR).fillna(view["risk_grade"])
    # 비중은 비율(0~1)로 저장돼 있다 — %.2f%% 서식은 값을 그대로 찍으므로
    # 5%가 "0.05%"로 나온다. 표시 직전에 100을 곱한다.
    view["exposure_share"] = view["exposure_share"] * 100
    view["deterioration_1y"] = pd.to_numeric(view["deterioration_1y"], errors="coerce") * 100
    view["lgd"] = pd.to_numeric(view["lgd"], errors="coerce") * 100
    for c in ("collateral_mix", "match_status"):
        if c in view.columns:
            view[c] = view[c].fillna("장부 밖 신규 실행")
    view = view.sort_values("exposure_mkrw", ascending=False)
    lgd_txt = _lgd_label(port)
    st.dataframe(
        view, hide_index=True, width="stretch", height=430,
        column_config={
            "brand_name": st.column_config.TextColumn("브랜드", width="medium"),
            "industry_mid": st.column_config.TextColumn("업종"),
            "n_stores": st.column_config.NumberColumn("가맹점", format="%,.0f"),
            "deterioration_1y": st.column_config.NumberColumn("브랜드 리스크", format="%.1f%%"),
            "risk_grade": st.column_config.TextColumn("등급"),
            "exposure_mkrw": st.column_config.NumberColumn("여신", format="%.1f 억"),
            "exposure_share": st.column_config.ProgressColumn(
                "비중", format="%.2f%%", min_value=0.0,
                max_value=float(view["exposure_share"].max() or 1)),
            "lgd": st.column_config.NumberColumn("LGD", format="%.0f%%"),
            "el_mkrw": st.column_config.NumberColumn(
                "대리 예상손실", format="%.2f 억", help=f"여신 × 브랜드 리스크 × LGD ({lgd_txt})"),
            "stress_el_mkrw": st.column_config.NumberColumn("스트레스", format="%.2f 억"),
            "n_borrowers": st.column_config.NumberColumn("차주 수", format="%,.0f"),
            "collateral_mix": st.column_config.TextColumn("담보 구성", width="medium"),
            "match_status": st.column_config.TextColumn("매칭"),
        })
    # ⚠️ 예전 CSV 는 내부 컬럼명과 백만원 단위 그대로라, 화면(억원)과 숫자가 100배
    #    어긋나 보였다. 화면에 보이는 이름·단위 그대로 내보낸다. 어느 장부로 계산했는지도
    #    첫 열에 남긴다 — 파일만 돌아다니면 추정 금액이 실제 잔액으로 읽힐 수 있다.
    export = view.rename(columns={
        "brand_name": "브랜드", "industry_mid": "업종", "n_stores": "가맹점 수",
        "deterioration_1y": "브랜드 리스크(%)", "risk_grade": "등급",
        "exposure_mkrw": "여신(억원)", "exposure_share": "비중(%)", "lgd": "LGD(%)",
        "el_mkrw": f"대리 예상손실(억원, {lgd_txt})", "stress_el_mkrw": "스트레스 손실(억원)",
        "n_borrowers": "차주 수", "collateral_mix": "담보 구성", "match_status": "매칭 상태"})
    export.insert(0, "장부 기준", label)
    st.download_button("포트폴리오 내려받기 (CSV)",
                       export.round(3).to_csv(index=False).encode("utf-8-sig"),
                       file_name="franscore_portfolio.csv", mime="text/csv")
