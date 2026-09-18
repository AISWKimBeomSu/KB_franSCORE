"""화면 공용 — 데이터 로딩·브랜드 마크·소견 표시·차트.

모든 화면이 같은 산출물을 같은 방식으로 읽도록 로더를 여기 하나로 모은다.
파일 수정시각을 캐시 키에 넣어 파이프라인이 다시 돌면 화면이 자동으로 갱신된다.
"""
from __future__ import annotations

import base64
import colorsys
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import theme
from src.common import load_config

GRADE_KR = {"High": "주의", "Medium": "관찰", "Low": "안정"}
GRADE_ACTION = {
    "High": "신규 취급 전 최근 공시와 본부 재무를 함께 확인하고, "
            "이 브랜드에 나간 여신 총액이 한도 안인지 점검합니다.",
    "Medium": "분기 단위로 점포 수와 계약종료율의 방향을 확인합니다.",
    "Low": "정기 모니터링을 유지합니다.",
}
CATEGORY_ICON = {"성장": "▮", "계약": "▮", "매출": "▮", "재무": "▮",
                 "구조": "▮", "수요": "▮", "평판": "▮"}


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _csv(path: str, m: float) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def _parquet(path: str, m: float, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False)
def _json(path: str, m: float):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def cfg() -> dict:
    return load_config()


def out_dir() -> Path:
    return Path(cfg()["paths"]["outputs"])


def proc_dir() -> Path:
    return Path(cfg()["paths"]["processed"])


# 화면이 반드시 필요로 하는 점수표 컬럼. 하나라도 없으면 그 파일은 못 쓴다.
SCORE_REQUIRED = ("brand_id", "brand_name", "n_stores", "grade", "risk_grade",
                  "deterioration_1y")


def load_scores() -> tuple[pd.DataFrame | None, dict]:
    """서비스가 쓰는 점수표 — **공표 트랙(외식업·점포 30+) 고정**.

    왜 트랙을 고르지 않고 고정하는가 (실제로 두 번 사고가 났다)
        이 저장소에는 트랙이 둘 있다. 공표 트랙(`outputs/`)과 확장 트랙
        (`outputs/extended/`, 전 업종·점포 20+)이다. 확장 트랙은 **체리피킹 방지용
        근거 산출물**이지 서비스 데이터가 아니다 — 좋은 쪽만 골라 보고하지 않았음을
        보이려고 항상 함께 돌려 `track_comparison.csv` 에 남긴다.

        ① 처음에는 확장 트랙을 무조건 우선했다. 확장을 다시 안 돌린 채 공표 트랙만
           재학습하니 화면이 **오래된 점수**를 보여줬다(실측: 화면 7/31, 실제 8/1).
        ② 그래서 '수정시각이 최신인 쪽'으로 바꿨다. 그랬더니 점수 컬럼을
           `deterioration_*` 로 개명하면서 공표 트랙만 재산출한 날, 배포 환경이
           아직 옛 이름을 쓰는 확장 트랙을 골라 화면 전체가 `KeyError` 로 죽었다.
           로컬은 공표 트랙이 더 최신이라 멀쩡했기 때문에 **배포에서만** 터졌다.

        두 사고의 원인은 같다. **어느 데이터로 서비스할지를 파일 타임스탬프가 정하고
        있었다.** 게다가 확장 트랙은 자기 `grade_bands.json` 이 없어 순위 백분위
        등급으로 폴백한다 — 우리가 폐기한 체계다. 모집단도 등급 의미도 다른 표가
        시각에 따라 조용히 화면에 오르는 구조 자체가 결함이었다.

        그래서 서비스는 공표 트랙만 쓴다. 확장 트랙은 문서와 비교표로만 등장한다.
        스키마 검사는 그래도 남긴다 — 공표 트랙 자체가 낡았을 때 조용히 죽지 않고
        무엇이 없는지 말하게 하기 위해서다.
    """
    path = out_dir() / "scores_latest.csv"
    if not path.exists():
        return None, {"rejected": [{"path": str(path), "why": "파일 없음"}]}
    mt = _mtime(path)
    try:
        df = _csv(str(path), mt)
    except (OSError, ValueError) as exc:
        return None, {"rejected": [{"path": str(path), "why": f"읽기 실패: {exc}"}]}
    missing = [c for c in SCORE_REQUIRED if c not in df.columns]
    if missing:
        return None, {"rejected": [{"path": str(path),
                                    "why": f"필수 컬럼 없음: {missing}"}]}
    meta_p = path.parent / "scores_latest_meta.json"
    meta = _json(str(meta_p), _mtime(meta_p)) if meta_p.exists() else {}
    meta["scope_label"] = "외식업"
    meta["scope_dir"] = str(path.parent)
    return df, meta


def load_diagnosis_summary() -> pd.DataFrame | None:
    _, meta = load_scores()
    base = Path(meta.get("scope_dir") or out_dir())
    p = base / "brand_diagnosis_summary.csv"
    if not p.exists():
        p = out_dir() / "brand_diagnosis_summary.csv"
    return _csv(str(p), _mtime(p)) if p.exists() else None


def load_findings(brand_id: str | None = None) -> pd.DataFrame | None:
    _, meta = load_scores()
    base = Path(meta.get("scope_dir") or out_dir())
    p = base / "brand_diagnosis.parquet"
    if not p.exists():
        p = out_dir() / "brand_diagnosis.parquet"
    if not p.exists():
        return None
    df = _parquet(str(p), _mtime(p))
    if brand_id is not None:
        df = df[df["brand_id"].astype(str) == str(brand_id)]
    return df


def load_panel(full: bool = False) -> pd.DataFrame | None:
    name = "panel_full.parquet" if full else "panel.parquet"
    p = proc_dir() / name
    return _parquet(str(p), _mtime(p)) if p.exists() else None


def load_hq_financials() -> pd.DataFrame | None:
    p = proc_dir() / "hq_financials.parquet"
    return _parquet(str(p), _mtime(p)) if p.exists() else None


def hq_source_label(fin: pd.DataFrame) -> str:
    """이 본부 재무표가 **어느 문서에서 왔는지** 화면에 적을 문구.

    ⚠️ 본부 재무는 두 원천에서 온다 — 금감원 감사보고서와 공정위 정보공개서 열람분.
       화면에 '금융감독원 전자공시'만 박아 두면, 외부감사 대상이 아닌 본부의 수치까지
       DART 이름표를 달고 나간다. 확인하러 간 심사역은 그 문서를 찾지 못한다.
       한 브랜드 안에 두 원천이 섞일 수도 있으므로(연도별로 다를 수 있다) 둘 다 적는다.
    """
    from src.ifrmp_web import HQ_SOURCE_TAG
    if fin is None or fin.empty or "source" not in fin.columns:
        return "금융감독원 전자공시"
    kinds = set(fin["source"].dropna().astype(str))
    has_web = HQ_SOURCE_TAG in kinds
    has_dart = bool(kinds - {HQ_SOURCE_TAG})
    if has_web and has_dart:
        return "금융감독원 전자공시 · 공정거래위원회 정보공개서"
    if has_web:
        return "공정거래위원회 정보공개서"
    return "금융감독원 전자공시"


def load_demand() -> dict:
    p = out_dir() / "demand_trends.json"
    if not p.exists():
        return {}
    obj = _json(str(p), _mtime(p))
    return obj if isinstance(obj, dict) else {}


def load_portfolio() -> tuple[pd.DataFrame | None, dict]:
    _, meta = load_scores()
    base = Path(meta.get("scope_dir") or out_dir())
    pf, summ = base / "portfolio.csv", base / "portfolio_summary.json"
    if not pf.exists():
        pf, summ = out_dir() / "portfolio.csv", out_dir() / "portfolio_summary.json"
    df = _csv(str(pf), _mtime(pf)) if pf.exists() else None
    s = _json(str(summ), _mtime(summ)) if summ.exists() else {}
    return df, (s if isinstance(s, dict) else {})


def scored_year() -> str:
    _, meta = load_scores()
    return str(meta.get("scored_year", "-"))


# ---------------------------------------------------------------------------
# 브랜드 마크 (로고)
# ---------------------------------------------------------------------------

_LOGO_CACHE = "data/raw/naver/logos.json"


@st.cache_data(show_spinner=False, max_entries=4096)
def _logo_data_uri(brand_name: str, m: float) -> str:
    """브랜드 로고 파일 → data URI. 없으면 빈 문자열.

    파일 경로는 브랜드명 해시로 **계산**한다 — 색인 파일(logos.json)을 거치지 않는다.
    수집이 백그라운드로 돌면서 색인을 통째로 덮어쓰면 방금 저장한 파일 참조가 사라지는
    경합이 실제로 났다(PNG 556장이 디스크에 있는데 색인은 0건). 경로를 계산하면
    색인과 무관하게 항상 맞고, 수집 도중에도 화면이 정상 동작한다.

    ⚠️ 원본 URL 을 <img src> 로 쓰지 않는 이유: 기업 사이트 상당수가 외부 Referer
       요청을 막아 배포 화면에서 이미지가 깨진다. 파일을 우리가 들고 있으면 상대
       사이트 상태와 무관하게 항상 뜬다. 128px PNG 라 한 장에 5~15KB 다.
    """
    from src.naver import LOGO_DIR, logo_file_name
    p = Path(cfg()["_root"]) / LOGO_DIR / logo_file_name(brand_name)
    if not p.exists():
        return ""
    try:
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
    except OSError:
        return ""


def logo_url(brand_name: str) -> str:
    """브랜드 로고를 화면에 바로 넣을 수 있는 형태로. 없으면 빈 문자열."""
    from src.naver import LOGO_DIR
    d = Path(cfg()["_root"]) / LOGO_DIR
    # 디렉토리 수정시각을 캐시 키에 넣어 수집이 진행되면 화면이 따라오게 한다
    return _logo_data_uri(str(brand_name), _mtime(d))


def _mark_colors(name: str) -> tuple[str, str, str]:
    """브랜드명에서 결정적으로 색을 만든다 (같은 브랜드는 항상 같은 색).

    같은 색조의 밝은 두 단계로 그라데이션을 만들고 글자는 진한 톤으로 둔다.
    단색 사각형에 글자만 얹으면 '자리표시자'로 보이지만, 색조가 브랜드마다 다르고
    질감이 있으면 목록에서 브랜드를 색으로 구분할 수 있다.
    """
    h = int(hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:8], 16)
    hue = (h % 360) / 360.0
    to_hex = lambda t: "#" + "".join(f"{int(v * 255):02X}" for v in t)  # noqa: E731
    return (to_hex(colorsys.hls_to_rgb(hue, 0.90, 0.62)),   # 배경 밝은 쪽
            to_hex(colorsys.hls_to_rgb(hue, 0.80, 0.58)),   # 배경 어두운 쪽
            to_hex(colorsys.hls_to_rgb(hue, 0.28, 0.66)))   # 글자


def _mark_label(brand_name: str) -> str:
    """마크에 넣을 글자 — 한글은 두 자, 영문은 이니셜."""
    s = "".join(ch for ch in str(brand_name) if ch.strip())
    if not s:
        return "?"
    kor = "".join(ch for ch in s if "가" <= ch <= "힣")
    if len(kor) >= 2:
        return kor[:2]
    words = [w for w in re.split(r"[\s()（）\[\]]+", str(brand_name)) if w]
    if len(words) >= 2 and all(w[0].isalpha() for w in words[:2]):
        return (words[0][0] + words[1][0]).upper()
    return s[:2].upper()


def brand_mark_html(brand_name: str, size: int = 64) -> str:
    """브랜드 마크 — 로고가 있으면 이미지, 없으면 브랜드 색 글자 마크.

    로고마다 원본 비율·여백이 제각각이라 그냥 넣으면 어떤 건 꽉 차고 어떤 건
    점처럼 보인다. **같은 크기의 타일 안에 contain 으로 앉히고** 안쪽 여백을
    일정하게 줘서 목록에서 크기가 고르게 보이도록 한다.
    cover 를 쓰면 가로로 긴 로고의 좌우가 잘려 다른 브랜드처럼 보인다.
    """
    url = logo_url(brand_name)
    radius = int(size * 0.26)
    pad = max(5, int(size * 0.13))
    style = (f"width:{size}px;height:{size}px;border-radius:{radius}px;"
             f"flex:0 0 {size}px;box-sizing:border-box;")
    if url:
        return (f"<div class='kb-mark' style='{style}padding:{pad}px'>"
                f"<img src='{url}' alt='{brand_name}'/></div>")
    c1, c2, fg = _mark_colors(brand_name)
    return (f"<div class='kb-mark' style='{style}"
            f"background:linear-gradient(150deg,{c1} 0%,{c2} 100%)'>"
            f"<span class='letter' style='color:{fg};font-size:{int(size * 0.34)}px'>"
            f"{_mark_label(brand_name)}</span></div>")


# ---------------------------------------------------------------------------
# 소견 표시
# ---------------------------------------------------------------------------

def render_findings(findings: pd.DataFrame, limit: int | None = None,
                    show_info: bool = True) -> None:
    """소견 목록을 문장 그대로 보여준다."""
    if findings is None or findings.empty:
        st.caption("표시할 소견이 없습니다.")
        return
    df = findings if show_info else findings[findings["direction"] != "info"]
    rows = df.head(limit) if limit else df
    html = []
    for _, r in rows.iterrows():
        sev = str(r["severity"]) if r["direction"] == "risk" else (
            "Low" if r["direction"] == "mitigant" else "Neutral")
        color_key = {"risk": str(r["severity"]), "mitigant": "Low"}.get(
            str(r["direction"]), "Neutral")
        mark = {"risk": "", "mitigant": "완화요인 · ", "info": "확인 필요 · "}.get(
            str(r["direction"]), "")
        html.append(theme.finding_html(
            title=f"{mark}{r['title']}",
            desc=str(r["detail"]),
            severity=color_key if color_key in theme.GRADE_COLOR else "Low",
            source=f"{r['category']} · {r['source']}"))
        del sev
    st.markdown("".join(html), unsafe_allow_html=True)


def category_summary(findings: pd.DataFrame) -> str:
    """위험 카테고리를 배지 줄로."""
    if findings is None or findings.empty:
        return ""
    risk = findings[findings["direction"] == "risk"]
    chips = []
    for cat, sub in risk.groupby("category", sort=False):
        worst = "High" if (sub["severity"] == "High").any() else (
            "Medium" if (sub["severity"] == "Medium").any() else "Low")
        chips.append(theme.chip(f"{cat} {len(sub)}", worst))
    return " ".join(chips)


# ---------------------------------------------------------------------------
# 차트
# ---------------------------------------------------------------------------

def line_chart(df: pd.DataFrame, x: str, y: str, name: str,
               unit: str = "", fill: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y], mode="lines+markers", name=name,
        line={"color": theme.YELLOW_DEEP, "width": 2.5},
        marker={"size": 7, "color": theme.YELLOW_DEEP,
                "line": {"color": "#FFFFFF", "width": 1.5}},
        fill="tozeroy" if fill else None,
        fillcolor="rgba(255,204,0,0.13)",
        hovertemplate=f"%{{x}}<br><b>%{{y:,.0f}}</b>{unit}<extra></extra>"))
    fig.update_layout(showlegend=False, height=210,
                      margin={"l": 4, "r": 4, "t": 8, "b": 4})
    return fig


def bar_chart(labels, values, colors=None, unit: str = "",
              horizontal: bool = True) -> go.Figure:
    c = colors or theme.YELLOW_DEEP
    fig = go.Figure(go.Bar(
        x=values if horizontal else labels,
        y=labels if horizontal else values,
        orientation="h" if horizontal else "v",
        marker={"color": c, "line": {"width": 0}},
        hovertemplate=f"%{{y}}<br><b>%{{x:,.1f}}</b>{unit}<extra></extra>"
        if horizontal else f"%{{x}}<br><b>%{{y:,.1f}}</b>{unit}<extra></extra>"))
    fig.update_layout(showlegend=False, margin={"l": 4, "r": 4, "t": 8, "b": 4})
    return fig


def refresh_state() -> dict:
    """마지막 갱신 이력.

    ⚠️ 기록보다 **산출물이 더 새로우면 산출물을 믿는다.** 일간 배치(daily_refresh)만
       이 파일을 쓰는데, 배치를 거치지 않고 파이프라인을 직접 재실행하면 점수·소견은
       새것인데 기록은 옛날에 멈춘다. 그러면 화면이 "마지막 갱신 이틀 전"이라고
       말하면서 오늘 산출한 숫자를 띄운다 — 보는 사람에게는 죽은 서비스로 읽힌다.
       실측으로 그 상태였다(기록 08-01 22:01 vs 점수 파일 08-03 05:30).
    """
    p = out_dir() / "refresh_state.json"
    s = out_dir() / "scores_latest.csv"
    rec: dict = {}
    if p.exists():
        try:
            rec = _json(str(p), _mtime(p)) or {}
        except (OSError, ValueError):
            rec = {}
    if not s.exists():
        return rec
    art = datetime.fromtimestamp(_mtime(s)).strftime("%Y-%m-%d %H:%M")
    if not rec:
        return {"finished_at": art, "source": "file_mtime"}
    if str(rec.get("finished_at") or "") < art:
        # 배치 기록은 남기되, 화면에 적는 시각은 실제 산출물 시각으로 바꾼다.
        return {**rec, "finished_at": art, "source": "file_mtime",
                "batch_finished_at": rec.get("finished_at")}
    return rec


def refresh_footer() -> None:
    """화면 하단 갱신 표기.

    이 숫자가 언제 것인지 밝히지 않으면 신뢰할 수 없다. 그리고 **무엇이 매일 바뀌고
    무엇이 안 바뀌는지**를 함께 적는다 — 연 1회 공시로 정해지는 확률을 "실시간"이라
    부르면 화면이 거짓말을 하는 것이 된다.
    """
    st.write("")
    r = refresh_state()
    bits = [f"마지막 갱신 **{r.get('finished_at') or '-'}**"]
    # ⚠️ 시각이 산출물 파일에서 온 것이면, **그 시각과 짝이 맞지 않는 배치 집계는 붙이지
    #    않는다.** 안 그러면 "마지막 갱신 08-03 · 신규 뉴스 18건" 처럼 오늘 시각 옆에
    #    이틀 전 배치의 건수가 나란히 서서, 읽는 사람은 그 18건이 오늘 것이라고 읽는다.
    from_batch = r.get("source") != "file_mtime"
    if from_batch:
        if r.get("news_added"):
            bits.append(f"신규 뉴스 {int(r['news_added']):,}건")
        if r.get("demand_updated"):
            bits.append(f"검색수요 {int(r['demand_updated']):,}개 브랜드")
        if r.get("rescored"):
            bits.append(f"진단 재산출 {int(r['rescored']):,}개 브랜드")
    else:
        bits.append("산출물 파일 시각 기준")
        if r.get("batch_finished_at"):
            bits.append(f"마지막 자동 배치 {r['batch_finished_at']}")
    st.caption(" · ".join(bits))

    with st.expander("갱신 주기와 자료 출처"):
        yr = r.get("disclosure_year")
        st.markdown(
            "| 항목 | 주기 | 출처 |\n|---|---|---|\n"
            "| 진단 소견 · 점검 우선순위 | **매일** | 뉴스 보도, 네이버 검색어트렌드 |\n"
            f"| 브랜드 리스크 확률 | **연 1회** | 공정거래위원회 가맹사업 공시"
            f"{f' ({yr}년 기준)' if yr else ''} |\n"
            "| 가맹본부 재무 | 연 1회 | 금융감독원 전자공시(DART) 감사보고서 · "
            "공정거래위원회 정보공개서 |\n")
        st.caption(
            "리스크 확률은 공시에서 나온 점포 수·계약종료·매출·본부재무로 계산합니다. "
            "그 공시가 1년에 한 번 나오므로 확률도 그때 움직입니다. 매일 갱신되는 것은 "
            "뉴스·검색수요에서 나오는 진단 소견과 그 소견으로 매기는 점검 순서입니다.")
        st.caption(
            "뉴스는 가맹점 100개 이상 브랜드를 대상으로 브랜드명에 "
            "`가맹점`·`본사 분쟁`·`폐점`·`가맹 갈등` 을 붙여 최근 24개월 안에서 "
            "브랜드당 최대 8건까지 모읍니다. 검색수요는 가맹점이 많은 브랜드부터 "
            "차례로 넓혀 갑니다.")

        _capability_note(r)


def _capability_note(r: dict) -> None:
    """마지막 갱신이 **축소 가동**이었으면 그 사실을 화면에 적는다.

    왜 필요한가 (실측)
        일간 배치가 자동화 환경에서 매일 성공으로 끝나고 있었는데, 실행 로그를 열어
        보니 자격증명이 하나도 없어 뉴스는 제목만 보는 경로로, 사건 추출은 규칙기반
        폴백으로 돌고 있었다. 폴백이 제대로 동작한 탓에 아무것도 실패하지 않았고
        화면에도 아무 표시가 없었다. **성능이 줄어든 화면과 온전한 화면이 똑같이
        보이는 것**이 결함이다. 여기서 그 차이를 드러낸다.
    """
    cap = r.get("capability") or {}
    feats = cap.get("features") or []
    if not feats:
        return
    degraded = [f for f in feats if not f.get("full")]
    if not degraded:
        st.caption(f"이 갱신은 자료 연결 {cap.get('n_total', len(feats))}종이 모두 "
                   "정상인 상태에서 산출됐습니다.")
        return
    st.warning(
        f"이 갱신은 **축소 가동**으로 산출됐습니다 — 자료 연결 "
        f"{cap.get('n_total', len(feats))}종 중 {len(degraded)}종이 대체 경로로 "
        "동작했습니다.")
    st.markdown("\n".join(
        f"- **{f['name']}** — 대체: {f['fallback']}  \n  {f['impact']}"
        for f in degraded))
    st.caption("숫자를 지우지 않고 그대로 두되, 어떤 근거로 나온 값인지 함께 밝힙니다.")


RISK_LABEL = "브랜드 리스크"
_RISK_HELP = ("공시·본부재무·검색수요를 학습한 모델이 산출한, 이 브랜드가 향후 1년 안에 "
              "구조적으로 꺾일 가능성입니다. 개별 가맹점의 연체 확률이 아닙니다.")


@st.cache_data(show_spinner=False)
def grade_bands(m: float) -> dict:
    """공표된 등급 밴드와 **검증 실현율**. 없으면 빈 dict.

    등급의 의미는 규칙이 아니라 이 실적표가 정의한다 — 신용평가사가 등급별 장기
    평균부도율을 공표해 등급을 앵커링하는 것과 같은 장치다.
    """
    p = out_dir() / "grade_bands.json"
    if not p.exists():
        return {}
    try:
        return _json(str(p), _mtime(p)) or {}
    except (OSError, ValueError):
        return {}


@st.cache_data(show_spinner=False)
def grade_bounds(m: float) -> dict:
    """등급 컷을 **실제 확률 경계**로 환산한다.

    등급 규칙 자체는 순위 백분위(상위 10% = 주의)다. 그런데 화면에서 "어느 구간이면
    어떤 마크가 붙는가"를 물으면, 사용자가 기대하는 답은 순위가 아니라 **퍼센트 경계**다.
    그래서 이번 산출물에서 그 경계가 실제로 몇 %인지 계산해 함께 보여준다.
    (두 표기를 나란히 두어야 '상위 10%' 라는 상대 기준도 숨기지 않는다.)
    """
    df, _ = load_scores()
    if df is None or df.empty:
        return {}
    g = (cfg().get("portfolio") or {}).get("risk_grades") or {}
    hi, mid = float(g.get("high", 0.90)), float(g.get("medium", 0.70))
    p = pd.to_numeric(df["deterioration_1y"], errors="coerce").dropna()
    return {
        "high_pct": hi, "medium_pct": mid,
        "high_cut": float(p.quantile(hi)) * 100,
        "medium_cut": float(p.quantile(mid)) * 100,
        "min": float(p.min()) * 100, "max": float(p.max()) * 100,
        "counts": {k: int(v) for k, v in df["risk_grade"].value_counts().items()},
    }


# 부문별 진단 — 반증에서 **성립이 확인된 3부문만** 쓴다.
#   본부재무(브랜드 커버리지 11.4%)와 시장수요(지표가 전부 평탄한 순위값, 뉴스 23건)는
#   부문으로 세우면 자료 없는 브랜드가 좋아 보이거나 나빠 보이는 왜곡이 생긴다.
#   창업비용은 2024 공시에 한 건도 없다.
# 점수를 매기지 않는다. 여러 지표를 하나의 숫자로 합치려면 가중치가 필요한데,
#   그 가중치를 정당화할 근거가 우리에게 없다. 관측값과 업종 내 위치만 보여준다.
_SECTIONS: tuple[tuple[str, str, tuple[tuple[str, str, str, str], ...]], ...] = (
    ("존속성", "브랜드가 점포망을 유지·확장하고 있는가", (
        ("n_stores", "가맹점 수", "개", "high_good"),
        ("_growth", "전년 대비 증감률", "%", "high_good"),
        ("_new_rate", "신규 개점률", "%", "neutral"),
        ("_direct_ratio", "직영점 비중", "%", "high_good"),
        ("_age", "업력", "년", "high_good"))),
    ("계약 안정성", "가맹점이 계약을 끝내고 나가는가", (
        ("_end_rate", "계약종료율", "%", "low_good"),
        ("_cancel_rate", "중도해지율", "%", "low_good"),
        ("_change_rate", "명의변경률", "%", "neutral"),
        ("_net_flow", "순증감 (신규−이탈)", "개", "high_good"))),
    ("집중도", "특정 지역에 쏠려 있는가", (
        ("n_regions", "진출 시도 수", "곳", "high_good"),
        ("top_region_share", "최대 지역 비중", "%", "low_good"),
        ("region_hhi", "지역 집중도 (HHI)", "", "low_good"))),
)


def _section_frame(panel: pd.DataFrame, year: int) -> pd.DataFrame:
    """부문 지표를 그 해 코호트 전체에 대해 계산 (업종 내 위치 산출용)."""
    cur = panel[panel["year"] == year].copy()
    prev = panel[panel["year"] == year - 1].set_index("brand_id")["n_stores"]
    base = cur["brand_id"].map(prev)
    num = lambda c: pd.to_numeric(cur.get(c), errors="coerce")     # noqa: E731
    cur["_growth"] = (num("n_stores") / base - 1.0) * 100
    cur["_new_rate"] = num("n_new") / base * 100
    cur["_direct_ratio"] = num("n_direct") / (num("n_direct") + num("n_stores")) * 100
    cur["_age"] = year - pd.to_numeric(cur.get("biz_start_year"), errors="coerce")
    cur["_end_rate"] = num("n_contract_end") / base * 100
    cur["_cancel_rate"] = num("n_contract_cancel") / base * 100
    cur["_change_rate"] = num("n_name_change") / base * 100
    cur["_net_flow"] = (num("n_new") - num("n_contract_end").fillna(0)
                        - num("n_contract_cancel").fillna(0))
    cur["top_region_share"] = num("top_region_share") * 100
    return cur


def section_cards_html(brand_id: str, industry_mid: str | None = None) -> str:
    """부문별 진단 카드. 점수 없이 **관측값 + 업종 내 위치**만."""
    panel = load_panel()
    if panel is None or panel.empty:
        return ""
    year = int(panel["year"].max())
    frame = _section_frame(panel, year)
    row = frame[frame["brand_id"].astype(str) == str(brand_id)]
    if row.empty:
        return ""
    row = row.iloc[0]
    peer = frame
    if industry_mid and "industry_mid" in frame.columns:
        same = frame[frame["industry_mid"] == industry_mid]
        if len(same) >= 20:                       # 표본이 얇으면 업종 비교가 흔들린다
            peer = same
    peer_label = (f"{industry_mid} {len(peer):,}개"
                  if peer is not frame else f"전체 {len(peer):,}개")

    blocks = []
    for title, sub, items in _SECTIONS:
        lines = []
        for col, label, unit, direction in items:
            v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if pd.isna(v):
                lines.append(
                    f"<div style='display:flex;padding:6px 0;border-bottom:1px solid "
                    f"{theme.BORDER};font-size:{theme.FS_MD}'><span style='color:{theme.TEXT_SUB}'>"
                    f"{label}</span><span style='margin-left:auto;color:{theme.TEXT_MUTED}'>"
                    f"공시에 없음</span></div>")
                continue
            s = pd.to_numeric(peer[col], errors="coerce").dropna()
            # 동점을 중간순위로 처리한다. 직영점 0개처럼 같은 값이 몰린 지표에서
            # `(s < v)` 만 쓰면 백분위가 0 으로 찍혀 "하위 0%" 같은 표기가 나온다.
            pct = (float(((s < v).mean() + (s <= v).mean()) / 2 * 100)
                   if len(s) >= 20 else float("nan"))
            good = (direction == "high_good" and pct >= 70) or \
                   (direction == "low_good" and pct <= 30)
            bad = (direction == "high_good" and pct <= 30) or \
                  (direction == "low_good" and pct >= 70)
            col_c = theme.SAFE if good else (theme.DANGER if bad else theme.TEXT)
            # ⚠️ 위치는 **값이 큰 쪽/작은 쪽**으로만 말한다. 예전에는 항상
            #    "상위 {100−pct}%" 로 찍어서, 성장률 −46.1%(하위 11%)가
            #    "업종 내 상위 89%" 로 나와 좋아 보였다. 절반을 기준으로 표현을 바꾼다.
            #    좋고 나쁨은 색이 전달한다 — 지표마다 방향이 달라 문구로 섞으면 오독된다.
            pos = ("—" if not np.isfinite(pct) else
                   f"업종 상위 {100 - pct:.0f}%" if pct >= 50 else f"업종 하위 {pct:.0f}%")
            lines.append(
                f"<div style='display:flex;align-items:baseline;gap:8px;padding:6px 0;"
                f"border-bottom:1px solid {theme.BORDER};font-size:{theme.FS_MD}'>"
                f"<span style='color:{theme.TEXT_SUB}'>{label}</span>"
                f"<span style='margin-left:auto;font-weight:700;color:{col_c};"
                f"font-variant-numeric:tabular-nums'>{v:,.1f}{unit}</span>"
                f"<span style='color:{theme.TEXT_MUTED};font-size:{theme.FS_XS};min-width:92px;"
                f"text-align:right'>{pos}</span></div>")
        blocks.append(
            f"<div style='margin-bottom:16px'>"
            f"<div style='font-weight:700;color:{theme.INK};font-size:{theme.FS_LG}'>{title}</div>"
            f"<div style='color:{theme.TEXT_MUTED};font-size:{theme.FS_SM};margin-bottom:4px'>"
            f"{sub}</div>{''.join(lines)}</div>")
    return ("".join(blocks) +
            f"<div style='color:{theme.TEXT_MUTED};font-size:{theme.FS_XS};margin-top:4px'>"
            f"비교 기준: {peer_label} · {year}년 공시. 부문에 점수를 매기지 않습니다 — "
            f"여러 지표를 하나로 합치려면 가중치가 필요한데 그 가중치를 정당화할 근거가 "
            f"없습니다. 관측값과 업종 내 위치만 보여드립니다.</div>")


def risk_summary_html(row, findings: pd.DataFrame | None) -> str:
    """왜 이 등급이 나왔는지를 **한 화면에 들어오게** 요약한다.

    소견을 아무리 잘 써도 6~10개를 다 읽어야 결론이 나오면 심사역은 안 읽는다.
    부문(카테고리)별로 무거운 것 하나씩만 뽑아 한 줄로 세운다 — 조합이 보여야
    "매출이 줄면서 계약도 끊기고 있다" 같은 판단이 선다.
    """
    if findings is None or findings.empty:
        return ""
    risk = findings[findings["direction"] == "risk"].copy()
    if risk.empty:
        return ""
    order = {"High": 0, "Medium": 1, "Low": 2}
    risk["_s"] = risk["severity"].map(order).fillna(3)
    top = (risk.sort_values(["_s", "weight"], ascending=[True, False])
              .groupby("category", sort=False).head(1)
              .sort_values(["_s", "weight"], ascending=[True, False]).head(4))

    chips = []
    for _, f in top.iterrows():
        sev = str(f.get("severity"))
        col = {"High": theme.DANGER, "Medium": theme.WARN}.get(sev, theme.TEXT_SUB)
        soft = {"High": theme.DANGER_SOFT, "Medium": theme.WARN_SOFT}.get(sev, theme.BG)
        chips.append(
            f"<div style='display:flex;gap:10px;align-items:baseline;padding:7px 0'>"
            f"<span style='flex:0 0 auto;font-size:{theme.FS_XS};font-weight:700;color:{col};"
            f"background:{soft};padding:2px 7px;border-radius:{theme.RADIUS_SM}'>{f['category']}</span>"
            f"<span style='color:{theme.INK};font-weight:600'>{f['title']}</span></div>")

    n_risk = len(risk)
    n_high = int((risk["severity"] == "High").sum())
    cats = " · ".join(dict.fromkeys(risk["category"].astype(str)))
    return (f"<div style='padding:14px 16px;border-radius:{theme.RADIUS_LG};background:{theme.SURFACE};"
            f"border:1px solid {theme.BORDER};box-shadow:var(--shadow-1)'>"
            f"<div style='font-size:{theme.FS_XS};font-weight:700;letter-spacing:.06em;"
            f"color:{theme.TEXT_MUTED};margin-bottom:6px'>이 등급이 나온 이유</div>"
            f"{''.join(chips)}"
            f"<div style='margin-top:8px;padding-top:8px;border-top:1px solid {theme.BORDER};"
            f"font-size:{theme.FS_SM};color:{theme.TEXT_SUB}'>"
            f"위험 소견 <b>{n_risk}건</b>"
            f"{f' (중대 {n_high}건)' if n_high else ''} · 걸린 부문 {cats}</div></div>")


def _int0(v) -> int:
    """결측·문자열 섞인 값을 안전하게 정수로."""
    n = pd.to_numeric(pd.Series([v]), errors="coerce").fillna(0).iloc[0]
    return int(n)


def diagnosis_report(row, findings: pd.DataFrame | None,
                     sections_text: str = "") -> str:
    """진단 보고서 (마크다운). 결재·회람에 그대로 붙일 수 있게 만든다."""
    name = str(row.get("brand_name", "-"))
    grade = str(row.get("grade") or row.get("risk_grade") or "-")
    kr = {"FS1": "안정", "FS2": "관찰", "FS3": "주의"}.get(grade, "")
    state = str(row.get("brand_state") or "-")
    p = pd.to_numeric(pd.Series([row.get("deterioration_1y")]), errors="coerce").iloc[0]
    b = grade_bands(_mtime(out_dir() / "grade_bands.json"))
    when = refresh_state().get("finished_at", "-")
    yr = scored_year()

    L = [f"# {name} 브랜드 리스크 진단 보고서", "",
         f"- **등급** {grade} {kr}",
         f"- **{RISK_LABEL}** {p * 100:.1f}%" if pd.notna(p) else "",
         f"- **브랜드 상태** {state}",
         f"- **업종** {row.get('industry_major', '-')} · {row.get('industry_mid', '-')}",
         f"- **가맹점 수** {_int0(row.get('n_stores')):,}개",
         f"- **기준** {yr}년 공정거래위원회 가맹사업 공시 · 산출 {when}", ""]

    if b.get("pooled"):
        L += ["## 등급의 의미", "",
              "| 등급 | 구간 | 검증 실현 악화율 | 표본 |", "|---|---|---|---|"]
        cuts = b["cuts"]
        rng = {"FS1": f"{cuts[0] * 100:.1f}% 미만",
               "FS2": f"{cuts[0] * 100:.1f}~{cuts[1] * 100:.1f}%",
               "FS3": f"{cuts[1] * 100:.1f}% 이상"}
        for r in b["pooled"]:
            mark = " ←" if r["grade"] == grade else ""
            L.append(f"| {r['grade']} {r['grade_kr']}{mark} | {rng[r['grade']]} | "
                     f"{r['rate'] * 100:.2f}% | {r['n']:,}건 |")
        L += ["", "> 이 확률은 부도확률이 아니라 공정위 공시 지표가 업종 하위구간에 "
                  "진입할 확률입니다.", ""]

    if findings is not None and not findings.empty:
        risk = findings[findings["direction"] == "risk"]
        other = findings[findings["direction"] != "risk"]
        if not risk.empty:
            L += [f"## 위험 소견 ({len(risk)}건)", ""]
            for _, f in risk.iterrows():
                L += [f"### [{f['category']}] {f['title']}  ({f['severity']})",
                      f"{f['detail']}", f"> 출처: {f['source']}", ""]
        if not other.empty:
            L += [f"## 완화요인·확인 필요 ({len(other)}건)", ""]
            for _, f in other.iterrows():
                L += [f"- **{f['title']}** — {f['detail']}"]
            L.append("")

    if sections_text:
        L += ["## 부문별 점검", "", sections_text, ""]

    L += ["---", "",
          "**이 보고서의 사용 범위** — 2선 리스크 관리의 점검 우선순위 산정과 심사 참고 "
          "정보로만 사용합니다. 자동 여신 승인·거절, 한도·금리 산정, 규제자본·충당금 "
          "산출에는 사용할 수 없습니다. 등급은 브랜드의 사업 안정성 평가이며 차주의 "
          "상환능력이나 부도확률이 아닙니다.", ""]
    return "\n".join(x for x in L if x is not None)


@st.cache_data(show_spinner=False)
def watch_rates(m: float) -> dict:
    """요주의 사건수별 다음 해 재발동 실현율 (`tools/watch_base_rates.py` 산출).

    {사건수: {rate, ci_low, ci_high, n}} 형태. 없으면 빈 dict.
    """
    p = out_dir() / "watch_base_rates.json"
    if not p.exists():
        return {}
    try:
        obj = _json(str(p), _mtime(p)) or {}
    except (OSError, ValueError):
        return {}
    out = {}
    for r in obj.get("table") or []:
        out[int(r["n_events_at_t"])] = {
            "rate": float(r["rate"]), "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"]), "n": int(r["n"]),
            "state": str(r["state"]),
        }
    return out


def population_note(row) -> str:
    """이 브랜드의 확률이 **모델이 검증된 구간에서 나온 것인지** 밝힌다.

    학습·보정·평가 표본은 '그 해 악화사건이 없던 브랜드'만으로 만들어졌는데
    점수는 그 게이트 없이 전체에 매겨진다. 2024 코호트에서 주의(FS3) 367개 중
    271개(73.8%)가 모델이 본 적 없는 구간에서 나온다. 두 트랙을 같은 '주의'로
    합쳐 보여주면 심사역은 같은 근거를 가진 것으로 읽는다 — 그건 사실이 아니다.

    ⚠️ 예전에는 여기서 "성능 근거가 없습니다. 순위 참고용으로만 쓰십시오"로 끝냈다.
       그건 정직하긴 해도 **일을 넘긴 것**이다. 심사역이 알고 싶은 것은 '이 브랜드가
       내년에도 나쁠 확률'이고, 그 값은 모형 없이도 셀 수 있다. 라벨 표본이 없을 뿐
       상태 전이는 패널에 그대로 있기 때문이다. 그래서 이제 실현율을 함께 준다.
    """
    state = str(row.get("brand_state") or "") if hasattr(row, "get") else ""
    if state == "건전":
        return (f"<div style='margin-top:8px;font-size:{theme.FS_SM};color:{theme.TEXT_MUTED}'>"
                f"✓ <b>건전</b> — 올해 공시에 악화 사건이 없습니다. "
                f"이 구간에서 모델 성능이 백테스트로 측정됐습니다.</div>")
    if state == "요주의":
        k = row.get("n_events_at_t") if hasattr(row, "get") else None
        rt = watch_rates(_mtime(out_dir() / "watch_base_rates.json"))
        ev = ""
        try:
            r = rt.get(int(k)) if k is not None and str(k) != "nan" else None
        except (TypeError, ValueError):
            r = None
        if r:
            base = rt.get(0)
            mult = (f" — 건전 브랜드({base['rate']*100:.1f}%)의 "
                    f"<b>{r['rate']/base['rate']:.1f}배</b>" if base and base["rate"] else "")
            ev = (f"<div style='margin-top:6px'>올해 악화 사건 <b>{int(k)}건</b>. "
                  f"같은 조건의 과거 브랜드 {r['n']:,}개 중 "
                  f"<b>{r['rate']*100:.1f}%</b>가 다음 해에도 악화 사건을 냈습니다"
                  f"(95% 구간 {r['ci_low']*100:.1f}~{r['ci_high']*100:.1f}%){mult}.</div>")
        return (f"<div style='margin-top:8px;padding:7px 10px;border-radius:{theme.RADIUS_MD};"
                f"background:{theme.WARN_SOFT};border:1px solid #F0DFB8;font-size:{theme.FS_SM};"
                f"color:{theme.TEXT};line-height:1.5'>"
                f"<b>요주의</b> — 올해 공시에 이미 악화 사건이 발동했습니다. 이 구간은 "
                f"학습·평가 표본에 포함되지 않아 <b>위 확률값에는 성능 근거가 없습니다.</b>"
                f"{ev}"
                f"<div style='margin-top:6px;color:{theme.TEXT_SUB}'>이 실현율은 모형이 아니라 "
                f"과거 실적을 그대로 센 값이라 브랜드 간 순위를 매기지 않습니다. "
                f"순위가 필요하면 아래 진단 소견을 보십시오.</div></div>")
    if state == "평가불가":
        return (f"<div style='margin-top:8px;padding:7px 10px;border-radius:{theme.RADIUS_MD};"
                f"background:{theme.BG};border:1px solid {theme.BORDER};font-size:{theme.FS_SM};"
                f"color:{theme.TEXT_SUB};line-height:1.5'>"
                f"<b>평가불가</b> — 올해 공시에서 판정 지표가 관측되지 않았습니다. "
                f"등급을 신뢰할 수 없습니다.</div>")
    return ""


def signal_html(grade: str, deterioration_1y: float | None = None, *, size: int = 13,
                show_value: bool = True) -> str:
    """신호등 — 위험도를 '빨간 숫자'가 아니라 **켜진 등**으로 보여준다.

    빨간 글씨의 확률은 두 가지를 한꺼번에 요구한다: 숫자를 읽고, 그 숫자가 높은지
    낮은지를 스스로 판단하기. 신호등은 판단을 이미 끝낸 상태로 전달한다.
    숫자는 남기되(근거를 감추지 않는다) 보조 정보로 내린다.
    """
    order = ("High", "Medium", "Low")
    on = str(grade) if str(grade) in order else "Low"
    dots = "".join(
        f"<span style='width:{size}px;height:{size}px;border-radius:50%;"
        f"background:{theme.GRADE_FILL[g] if g == on else '#D8D3CC'};"
        f"box-shadow:{f'0 0 0 3px {theme.GRADE_SOFT[g]}' if g == on else 'none'};"
        f"display:inline-block'></span>" for g in order)
    label = (f"<div style='font-size:{theme.FS_MD};font-weight:700;color:{theme.GRADE_COLOR[on]};"
             f"margin-top:6px;letter-spacing:-.01em'>{theme.GRADE_KR[on]}</div>")
    val = ""
    if show_value and deterioration_1y is not None and pd.notna(deterioration_1y):
        val = (f"<div style='font-size:{theme.FS_XS};color:{theme.TEXT_MUTED};margin-top:1px'>"
               f"{RISK_LABEL} {float(deterioration_1y) * 100:.1f}%</div>")
    return (f"<div style='display:inline-flex;flex-direction:column;align-items:center'>"
            f"<div style='display:flex;gap:7px;align-items:center;padding:7px 10px;"
            f"border-radius:{theme.RADIUS_PILL};background:{theme.SURFACE};"
            f"border:1px solid {theme.BORDER}'>{dots}</div>{label}{val}</div>")


def grade_legend_html(bounds: dict) -> str:
    """등급 구간표 — 어떤 구간이면 어떤 등급이고, **그 등급이 실제로 몇 % 악화했는지**.

    실현율을 함께 싣는 것이 핵심이다. 밴드만 보여주면 "10.5% 이상"이 무슨 뜻인지
    독자가 알 수 없고, 부도확률로 오독한다. 검증 실적이 등급의 의미를 정의한다.
    """
    b = grade_bands(_mtime(out_dir() / "grade_bands.json"))
    if b.get("pooled"):
        cuts = b["cuts"]
        rng = {"FS1": f"{cuts[0] * 100:.1f}% 미만",
               "FS2": f"{cuts[0] * 100:.1f} ~ {cuts[1] * 100:.1f}%",
               "FS3": f"{cuts[1] * 100:.1f}% 이상"}
        # 점(도형)과 숫자(글자)는 기준이 달라 색도 다르다 — theme 의 주석 참고
        color = {"FS1": theme.SAFE, "FS2": theme.WARN, "FS3": theme.DANGER}
        fill = {"FS1": theme.SAFE_FILL, "FS2": theme.WARN_FILL, "FS3": theme.DANGER_FILL}
        soft = {"FS1": theme.SAFE_SOFT, "FS2": theme.WARN_SOFT, "FS3": theme.DANGER_SOFT}
        rows = "".join(
            f"<div style='display:flex;align-items:center;gap:10px;padding:9px 2px;"
            f"border-bottom:1px solid {theme.BORDER}'>"
            f"<span style='width:11px;height:11px;border-radius:50%;flex:0 0 auto;"
            f"background:{fill[r['grade']]};box-shadow:0 0 0 3px {soft[r['grade']]}'></span>"
            f"<span style='font-weight:700;color:{theme.INK};min-width:74px'>"
            f"{r['grade']} {r['grade_kr']}</span>"
            f"<span style='color:{theme.TEXT};font-variant-numeric:tabular-nums;"
            f"min-width:104px'>{rng[r['grade']]}</span>"
            f"<span style='margin-left:auto;color:{theme.TEXT_SUB};font-size:{theme.FS_MD}'>"
            f"실제 악화 <b style='color:{color[r['grade']]}'>{r['rate'] * 100:.1f}%</b> "
            f"<span style='color:{theme.TEXT_MUTED}'>(n={r['n']:,})</span></span></div>"
            for r in b["pooled"])
        # ⚠️ 여기서 읽던 키는 `validation_years` 였는데 산출물에 그런 키가 없다
        #    (실제 키는 `calibration_years`). get() 의 기본값이 빈 리스트라 예외 없이
        #    조용히 통과했고, 화면에는 **"(년 검증, 총 1,429건)"** 이라고 찍혔다 —
        #    연도가 빠진 자리가 오타처럼 보이는 문장이 두 화면에 그대로 나갔다.
        # ⚠️ 표기도 '검증'에서 '실적'으로 고친다. 이 표는 컷을 고른 그 자료 위에서
        #    센 실현율이라 산출물 자신이 `selection_caveat` 로 "검증이 아니라 실적표"
        #    라고 적어 두었다 — 화면이 그보다 센 말을 쓰면 안 된다.
        yrs = "·".join(str(y) for y in b.get("calibration_years", []))
        return (f"<div style='font-size:{theme.FS_BASE};line-height:1.5'>{rows}"
                f"<div style='color:{theme.TEXT_MUTED};font-size:{theme.FS_SM};margin-top:9px'>"
                f"구간은 <b>고정</b>입니다 — 순위가 아니라 확률 기준이라, 업계 전체가 "
                f"나빠지면 하위 등급이 늘어납니다. 오른쪽은 그 등급을 받은 브랜드가 "
                f"실제로 다음 해에 악화한 비율입니다({yrs}년 실적, 총 "
                f"{b['n_validation']:,}건). "
                f"이 확률은 <b>부도확률이 아니라</b> 공정위 공시 지표가 업종 하위구간에 "
                f"진입할 확률입니다.</div></div>")
    if not bounds:
        return ""
    rows = [
        ("High", f"{bounds['high_cut']:.1f}% 이상",
         f"상위 {(1 - bounds['high_pct']) * 100:.0f}%", "즉시 점검"),
        ("Medium", f"{bounds['medium_cut']:.1f}% ~ {bounds['high_cut']:.1f}%",
         f"상위 {(1 - bounds['medium_pct']) * 100:.0f}%", "추이 관찰"),
        ("Low", f"{bounds['medium_cut']:.1f}% 미만",
         f"하위 {bounds['medium_pct'] * 100:.0f}%", "정기 점검"),
    ]
    cnt = bounds.get("counts", {})
    cells = "".join(
        f"<div style='display:flex;align-items:center;gap:10px;padding:9px 2px;"
        f"border-bottom:1px solid {theme.BORDER}'>"
        f"<span style='width:11px;height:11px;border-radius:50%;flex:0 0 auto;"
        f"background:{theme.GRADE_FILL[g]};box-shadow:0 0 0 3px {theme.GRADE_SOFT[g]}'></span>"
        f"<span style='font-weight:700;color:{theme.INK};min-width:32px'>{theme.GRADE_KR[g]}</span>"
        f"<span style='color:{theme.TEXT};font-variant-numeric:tabular-nums'>{rng}</span>"
        f"<span style='color:{theme.TEXT_MUTED};font-size:{theme.FS_SM}'>({rank})</span>"
        f"<span style='margin-left:auto;color:{theme.TEXT_SUB};font-size:{theme.FS_SM}'>"
        f"{cnt.get(g, 0):,}개 · {act}</span></div>"
        for g, rng, rank, act in rows)
    return (f"<div style='font-size:{theme.FS_BASE};line-height:1.5'>{cells}"
            f"<div style='color:{theme.TEXT_MUTED};font-size:{theme.FS_SM};margin-top:9px'>"
            f"등급은 <b>순위 기준</b>으로 나눕니다 — 매 산출 시점의 상위 10%가 주의, "
            f"상위 30%까지가 관찰입니다. 위 퍼센트는 이번 산출에서 그 순위 경계가 "
            f"실제로 몇 %였는지를 환산한 값입니다.</div></div>")


def risk_gauge(deterioration_1y: float, rank_pct: float | None = None) -> go.Figure:
    """브랜드 리스크 게이지 — 0~100% 전 구간을 등급 색으로 나눠 보여준다.

    축을 0~50 으로 자르면 '절반이 찼다'는 인상이 실제 위험보다 과장된다.
    확률은 0~100% 가 정의역이므로 축도 그대로 0~100 을 쓴다.
    """
    v = float(deterioration_1y) * 100
    b = grade_bounds(_mtime(out_dir() / "scores_latest.csv"))
    hi = b.get("high_cut", 25.0)
    mid = b.get("medium_cut", 10.0)
    color = (theme.DANGER_FILL if v >= hi else
             theme.WARN_FILL if v >= mid else theme.SAFE_FILL)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={"suffix": "%", "font": {"size": 34, "color": theme.INK}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "dtick": 25,
                     "ticksuffix": "%",
                     "tickfont": {"size": 13, "color": theme.TEXT_SUB}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#F0EEEA",
            "borderwidth": 0,
            "steps": [{"range": [0, mid], "color": theme.SAFE_SOFT},
                      {"range": [mid, hi], "color": theme.WARN_SOFT},
                      {"range": [hi, 100], "color": theme.DANGER_SOFT}],
        }))
    fig.update_layout(height=170, margin={"l": 12, "r": 12, "t": 8, "b": 4})
    del rank_pct
    return fig
