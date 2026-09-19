"""가맹 브랜드 리스크 참고의견서 — 인쇄·회람용 단일 HTML 문서.

왜 필요한가
    화면의 근거는 탭 다섯 개에 흩어져 있고, 기존 내려받기(마크다운)는 표·서식이 없어
    출력물로 쓰기 어렵다. 심사역이 결재 문서에 붙이거나 PDF 로 보관하려면 **한 파일로
    완결된 문서**가 필요하다. 이 모듈은 외부 자원이 전혀 없는(인라인 CSS 만 쓰는) HTML
    한 파일을 만든다 — 메일 첨부·문서관리시스템 보관·브라우저 인쇄(A4, PDF 저장)가
    그대로 된다. 결론(등급·상태·권고)은 인쇄 첫 쪽에 들어오도록 배치했다.

설계 원칙
    · streamlit 을 import 하지 않는다. 화면·배치·테스트가 같은 문서를 만든다.
      그래서 화면 공용(src/views/common.py)의 로더·부문 계산을 가져오지 않고, 같은
      파일·같은 컬럼을 같은 규칙으로 읽도록 여기서 다시 적었다.
    · 모든 동적 문자열은 이스케이프한다(`_e`). 브랜드명에 따옴표가 실제로 있다
      — "꿀스커피GGUL'S COFFEE".
    · 등급은 순위가 아니라 grade_bands.json 의 **고정 확률 구간**으로 설명한다.
    · 확인하지 못한 사실을 단정하지 않는다. 본부 재무가 공시와 매칭되지 않으면
      "확인되지 않음"이라고만 쓴다(원인을 추정해 적지 않는다).
    · 권고 문구는 src/guidance.py 한 곳에서 온다.

사용
    from src.report import build_brand_report_html, load_brand_context
    ctx = load_brand_context("BRD_20220035")          # scores_latest.csv 의 brand_id
    html_doc = build_brand_report_html(ctx)           # str — .html 로 저장·내려받기
"""
from __future__ import annotations

import copy
import functools
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common import ROOT
from src.guidance import (
    CRITICAL_CODES,
    CRITICAL_RATIONALE,
    GRADE_ACTIONS,
    GRADE_CADENCE,
    GRADE_LABEL,
    check_for,
    checklist,
    critical_findings,
    grade_code,
)

SERVICE = "FranSCORE"
DOC_TITLE = "가맹 브랜드 리스크 참고의견서"
SCOPE_LINE = "2선 리스크 관리 참고자료 — 여신 승인·거절, 한도·금리 결정에 사용하지 않음"
DISCLAIMER = ("이 문서는 2선 리스크 관리의 점검 우선순위 산정과 심사 참고 정보로만 사용합니다. "
              "자동 여신 승인·거절, 한도·금리 산정, 규제자본·충당금 산출에 사용할 수 없습니다. "
              "등급은 브랜드의 사업 안정성 평가이며 차주의 상환능력이나 부도확률(PD)이 아닙니다.")
RISK_LABEL = "브랜드 리스크"

N_TREND_YEARS = 5           # 공시 추이 표에 싣는 최근 연도 수
N_HQ_YEARS = 5              # 본부 재무 표에 싣는 최근 결산 수
MIN_RATE_BASE = 10          # 비율을 적는 최소 분모 — src/diagnosis.py 와 같은 값
MIN_PEER = 20               # 업종 비교 최소 표본 — 화면(부문별 점검)과 같은 값

_SEV_LABEL = {"High": "높음", "Medium": "보통", "Low": "낮음"}
_SEV_ORDER = {"High": 0, "Medium": 1, "Low": 2}
_DIR_ORDER = {"risk": 0, "info": 1, "mitigant": 2}

# 규칙 문장을 그대로 싣지 않는 소견.
#   HQ_NO_DATA 의 원문(src/diagnosis.py)은 "외부감사 대상이 아니어서 감사보고서를 제출하지
#   않고"라고 원인을 단정한다. 그러나 관측한 것은 **매칭 실패**뿐이다 — 법인명
#   표기 차이나 사업자번호 불일치로도 같은 결과가 난다. 결재 문서가 확인하지 않은 원인을
#   사실처럼 적으면, 확인하러 간 심사역이 반대 사실을 발견하는 순간 문서 전체의 신뢰가
#   무너진다. 관측 사실만 적는다.
_DETAIL_OVERRIDE = {
    "HQ_NO_DATA": ("금융감독원 전자공시(감사보고서)와 공정거래위원회 정보공개서 열람분에서 이 브랜드 "
                   "가맹본부의 재무를 매칭·확인하지 못했습니다. 매칭되지 않은 원인은 단정하지 않습니다"
                   "(법인명 표기 차이 등으로도 생깁니다). 본부의 자본잠식·적자 여부가 확인되지 않았으므로 "
                   "별도 재무자료로 확인해야 합니다."),
}

# 부문별 점검 — 화면(src/views/common.py `_SECTIONS`)과 같은 3부문·같은 지표·같은 방향.
#   본부재무·시장수요는 커버리지가 낮아 부문으로 세우지 않는다(화면 주석 참고).
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
        ("n_regions", "진출 시·도 수", "곳", "high_good"),
        ("top_region_share", "최대 지역 비중", "%", "low_good"),
        ("region_hhi", "지역 집중도 (HHI)", "", "low_good"))),
)


# ---------------------------------------------------------------------------
# 값 다루기
# ---------------------------------------------------------------------------

def _e(v: object) -> str:
    """HTML 본문·속성에 넣을 문자열.

    먼저 엔티티를 풀고 다시 이스케이프한다. 공시 원천 일부가 이미 HTML 이스케이프된
    이름을 준다(실측: "크래프트한스(Craft Han&#39;s)", "브레댄코(bread&amp;co.DAILY-NEW)").
    그대로 이스케이프하면 인쇄물에 `&#39;` 가 글자로 찍힌다. 풀었다가 다시 이스케이프하므로
    `&lt;script&gt;` 같은 입력도 결국 글자로만 나간다 — 안전성은 같다.
    """
    if v is None:
        return ""
    return html.escape(html.unescape(str(v)), quote=True)


def _num(v: object) -> float | None:
    """None/NaN/문자열을 안전하게 float 로. 실패하면 None."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)                                    # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _int(v: object) -> int | None:
    f = _num(v)
    return None if f is None else round(f)


def _py(v: object) -> object:
    """numpy/pandas 스칼라를 파이썬 값으로, 결측은 None 으로 (ctx 를 dict 로 넘기기 위해)."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _rec(d: dict) -> dict:
    return {str(k): _py(v) for k, v in d.items()}


def _count(v: object, unit: str = "") -> str:
    n = _int(v)
    return "—" if n is None else f"{n:,}{unit}"


def _signed_pct(rate: float | None) -> str:
    return "—" if rate is None else f"{rate * 100:+.1f}%"


def _won_thousand(v: float | None) -> str:
    """천원 단위(공시 평균매출)를 '억/만원'으로 — 소견 문장(src/diagnosis.won)과 같은 표기."""
    if v is None:
        return "공시에 없음"
    krw = v * 1_000
    if abs(krw) >= 1e8:
        return f"{krw / 1e8:,.1f}억원"
    return f"{krw / 1e4:,.0f}만원"


def _finding_records(obj: object) -> list[dict]:
    """소견을 dict 목록으로. load_brand_context 는 목록을 주지만, 화면이 load_findings() 의
    DataFrame 을 그대로 넘겨도 되게 한다."""
    if obj is None:
        return []
    if hasattr(obj, "to_dict") and hasattr(obj, "columns"):
        return [_rec(r) for r in obj.to_dict("records")]   # type: ignore[union-attr]
    return [dict(f) for f in obj if isinstance(f, dict)]   # type: ignore[union-attr]


def _risk_pct(p: float | None, cuts: list[float]) -> float | None:
    """화면에 적을 브랜드 리스크(%) — 반올림이 등급 경계를 넘지 않게 한다.

    점수는 자기 등급 구간 안으로 잘려 있다(src/score.py). 그런데 경계 바로 아래 값
    (예: 15.97%)을 소수 첫째 자리로 반올림하면 "16.0% · FS2 관찰"로 찍혀 숫자와 등급이
    모순돼 보인다. 경계에 닿는 반올림만 내림으로 바꾼다.
    """
    if p is None:
        return None
    v = p * 100
    shown = round(v, 1)
    for c in cuts:
        if v < c * 100 <= shown:
            return math.floor(v * 10) / 10
    return shown


def _band_text(grade: str | None, cuts: list[float]) -> str:
    """등급의 고정 확률 구간. 표기는 기존 보고서(src/views/common.py)와 같다."""
    if len(cuts) != 2 or grade not in GRADE_LABEL:
        return ""
    lo, hi = cuts[0] * 100, cuts[1] * 100
    return {"FS1": f"{lo:.1f}% 미만", "FS2": f"{lo:.1f}~{hi:.1f}%", "FS3": f"{hi:.1f}% 이상"}[grade]


def _fmt_dt(v: object) -> str:
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M")
    return str(v or "")


# ---------------------------------------------------------------------------
# 산출물 읽기 (streamlit 없이 — 화면 로더와 같은 파일·같은 규칙)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def _read_cached(path: str, mtime_ns: int, kind: str) -> object:
    """파일 수정시각을 키에 넣어, 파이프라인이 다시 돌면 자동으로 새로 읽는다."""
    del mtime_ns
    if kind == "csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if kind == "parquet":
        return pd.read_parquet(path)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _load(path: Path, kind: str) -> object:
    """산출물 하나를 읽는다. 없거나 깨졌으면 None — 문서는 그 절을 '없음'으로 쓴다.

    ⚠️ 캐시가 돌려주는 객체는 공유된다. 여기서 받은 표·dict 를 제자리에서 고치지 않는다.
    """
    try:
        mt = path.stat().st_mtime_ns
    except OSError:
        return None
    try:
        return _read_cached(str(path), mt, kind)
    except (OSError, ValueError):
        return None


def _artifact_dirs(base: Path) -> tuple[Path, Path, dict]:
    """config.yaml 의 paths 를 따른다. load_config() 는 디렉터리를 만들고 시크릿을 올리므로
    문서 생성처럼 읽기만 하는 경로에서는 설정 파일만 직접 읽는다."""
    cfg: dict = {}
    p = base / "config.yaml"
    if p.exists():
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            cfg = {}
    paths = cfg.get("paths") or {}
    out = base / str(paths.get("outputs") or "outputs")
    proc = base / str(paths.get("processed") or "data/processed")
    from src import public
    if public.is_public():                      # 공개 배포는 가명 사본에서 읽는다 (src/public.py)
        out, proc = public.data_dirs(out, proc)
    return out, proc, cfg


def _artifact_time(out_dir: Path) -> str | None:
    """산출물이 언제 것인가 — 배치 기록보다 파일이 새로우면 파일을 믿는다(화면과 같은 규칙)."""
    s = out_dir / "scores_latest.csv"
    try:
        art = datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        art = None
    rec = _load(out_dir / "refresh_state.json", "json")
    fin = str(rec.get("finished_at") or "") if isinstance(rec, dict) else ""
    if art and fin < art:
        return art
    return fin or art


def _position(scores: pd.DataFrame, row: dict) -> dict:
    """리스크 순위 — 업종 안(표본 20개 이상일 때)과 전체. '상위 X%' 는 순위÷모집단."""
    p = pd.to_numeric(scores["deterioration_1y"], errors="coerce")
    v = _num(row.get("deterioration_1y"))
    out: dict = {"overall_n": int(p.notna().sum())}
    if v is None:
        return out
    out["overall_rank"] = int((p > v).sum()) + 1
    mid = row.get("industry_mid")
    if mid and "industry_mid" in scores.columns:
        same = p[scores["industry_mid"] == mid].dropna()
        if len(same) >= MIN_PEER:
            out.update(industry=str(mid), industry_n=len(same),
                       industry_rank=int((same > v).sum()) + 1)
    return out


def _section_frame(panel: pd.DataFrame, year: int) -> pd.DataFrame:
    """부문 지표를 그 해 코호트 전체에 대해 계산 — 화면 `_section_frame` 과 같은 식."""
    cur = panel[panel["year"] == year].copy()
    prev_rows = panel[panel["year"] == year - 1]
    prev_rows = prev_rows[~prev_rows["brand_id"].duplicated(keep="last")]
    base = cur["brand_id"].map(prev_rows.set_index("brand_id")["n_stores"])

    def num(c: str) -> pd.Series:
        if c not in cur.columns:
            return pd.Series(np.nan, index=cur.index)
        return pd.to_numeric(cur[c], errors="coerce")

    cur["_growth"] = (num("n_stores") / base - 1.0) * 100
    cur["_new_rate"] = num("n_new") / base * 100
    cur["_direct_ratio"] = num("n_direct") / (num("n_direct") + num("n_stores")) * 100
    cur["_age"] = year - num("biz_start_year")
    cur["_end_rate"] = num("n_contract_end") / base * 100
    cur["_cancel_rate"] = num("n_contract_cancel") / base * 100
    cur["_change_rate"] = num("n_name_change") / base * 100
    cur["_net_flow"] = (num("n_new") - num("n_contract_end").fillna(0)
                        - num("n_contract_cancel").fillna(0))
    cur["top_region_share"] = num("top_region_share") * 100
    return cur.replace([np.inf, -np.inf], np.nan)


def _fmt_metric(v: float, unit: str) -> str:
    if unit == "":
        return f"{v:.3f}"
    if unit in ("개", "곳", "년") and float(v).is_integer():
        return f"{int(v):,}{unit}"
    return f"{v:,.1f}{unit}"


def _sections(panel: pd.DataFrame | None, brand_id: str, year: int | None,
              industry_mid: object) -> dict | None:
    """부문별 점검 — 점수 없이 관측값과 업종 내 위치만(화면과 같은 원칙)."""
    if panel is None or panel.empty:
        return None
    years = set(panel["year"].dropna().astype(int))
    yr = int(year) if year is not None and int(year) in years else max(years)
    frame = _section_frame(panel, yr)
    hit = frame[frame["brand_id"].astype(str) == brand_id]
    if hit.empty:
        return None
    row = hit.iloc[0]
    peer, peer_label = frame, f"전체 {len(frame):,}개"
    if industry_mid and "industry_mid" in frame.columns:
        same = frame[frame["industry_mid"] == industry_mid]
        if len(same) >= MIN_PEER:                      # 표본이 얇으면 업종 비교가 흔들린다
            peer, peer_label = same, f"{industry_mid} {len(same):,}개"
    groups = []
    for title, sub, items in _SECTIONS:
        lines = []
        for col, label, unit, direction in items:
            v = _num(row.get(col))
            if v is None:
                lines.append({"label": label, "value": "공시에 없음", "position": "—", "judgement": ""})
                continue
            s = pd.to_numeric(peer[col], errors="coerce").dropna()
            # 동점은 중간순위 — 같은 값이 몰린 지표(직영점 0개 등)에서 "하위 0%"가 나오지 않게.
            pct = (float(((s < v).mean() + (s <= v).mean()) / 2 * 100)
                   if len(s) >= MIN_PEER else float("nan"))
            good = (direction == "high_good" and pct >= 70) or (direction == "low_good" and pct <= 30)
            bad = (direction == "high_good" and pct <= 30) or (direction == "low_good" and pct >= 70)
            # 위치는 값이 큰 쪽/작은 쪽으로만 말한다. 좋고 나쁨은 '판단' 칸이 따로 말한다.
            pos = ("—" if not np.isfinite(pct) else
                   f"업종 상위 {100 - pct:.0f}%" if pct >= 50 else f"업종 하위 {pct:.0f}%")
            lines.append({"label": label, "value": _fmt_metric(v, unit), "position": pos,
                          "judgement": "양호" if good else ("유의" if bad else "")})
        groups.append({"title": title, "sub": sub, "items": lines})
    return {"year": yr, "peer_label": peer_label, "groups": groups}


def _trend(hist: pd.DataFrame, year: int | None) -> list[dict]:
    """최근 공시 추이 (연도 오름차순). 종료·해지율의 분모는 **연속된** 전년 점포 수."""
    if hist is None or hist.empty:
        return []
    h = hist.sort_values("year")
    if year is not None:
        h = h[h["year"] <= int(year)]
    h = h[~h["year"].duplicated(keep="last")]

    def num(c: str) -> pd.Series:
        if c not in h.columns:
            return pd.Series(np.nan, index=h.index)
        return pd.to_numeric(h[c], errors="coerce")

    consec = h["year"].diff() == 1
    prev = num("n_stores").shift(1).where(consec)
    out = num("n_contract_end").fillna(0) + num("n_contract_cancel").fillna(0)
    rate = (out / prev).where(prev >= MIN_RATE_BASE)
    rows = []
    for (_, r), rt in zip(h.iterrows(), rate, strict=True):
        rows.append({
            "year": int(r["year"]),
            "n_stores": _py(r.get("n_stores")), "n_direct": _py(r.get("n_direct")),
            "n_new": _py(r.get("n_new")), "n_contract_end": _py(r.get("n_contract_end")),
            "n_contract_cancel": _py(r.get("n_contract_cancel")),
            "n_name_change": _py(r.get("n_name_change")),
            "out_rate": _py(rt),
            "avg_sales": _py(r.get("avg_sales")),                    # 천원
            "avg_sales_per_area": _py(r.get("avg_sales_per_area")),  # 천원 / 3.3㎡
            "n_regions": _py(r.get("n_regions")),
        })
    return rows[-N_TREND_YEARS:]


def _hq(proc_dir: Path, company: str | None) -> dict:
    """본부 재무 — 화면과 같은 매칭(법인명 정규화 키). 매칭 실패는 '확인되지 않음'."""
    if not company:
        return {"company": None, "matched": False, "rows": []}
    fin_all = _load(proc_dir / "hq_financials.parquet", "parquet")
    if not isinstance(fin_all, pd.DataFrame) or "key" not in fin_all.columns:
        return {"company": company, "matched": False, "rows": []}
    # 매칭 규칙(법인명 정규화)은 src/dart.py 가 단일 원천이다 — 화면과 같은 키로 찾는다.
    from src.dart import norm_corp
    from src.ifrmp_web import HQ_SOURCE_TAG
    key = norm_corp(company)
    fin = fin_all[fin_all["key"] == key].sort_values("fiscal_year") if key else fin_all.iloc[0:0]
    if fin.empty:
        return {"company": company, "matched": False, "rows": []}
    fin = fin[~fin["fiscal_year"].duplicated(keep="last")].tail(N_HQ_YEARS)
    # 원천이 브랜드·연도마다 다르다 — 한쪽 이름만 적으면 확인하러 간 사람이 문서를 못 찾는다.
    kinds = set(fin["source"].dropna().astype(str)) if "source" in fin.columns else set()
    has_web, has_dart = HQ_SOURCE_TAG in kinds, bool(kinds - {HQ_SOURCE_TAG})
    label = ("금융감독원 전자공시 · 공정거래위원회 정보공개서" if has_web and has_dart else
             "공정거래위원회 정보공개서" if has_web else "금융감독원 전자공시")
    rows = [_rec(r) for r in fin.to_dict("records")]
    rcept = None
    for r in reversed(rows):
        no = str(r.get("rcept_no") or "")
        if no.isdigit() and str(r.get("source") or "") != HQ_SOURCE_TAG:
            rcept = no
            break
    return {"company": company, "matched": True, "rows": rows, "source_label": label,
            "rcept_no": rcept, "web_tag": HQ_SOURCE_TAG}


def _demand(out_dir: Path, brand_id: str) -> dict | None:
    obj = _load(out_dir / "demand_trends.json", "json")
    if not isinstance(obj, dict) or not obj.get("enabled"):
        return None
    d = (obj.get("brands") or {}).get(brand_id)
    if not isinstance(d, dict) or not d:
        return None
    keep = ("term", "category", "period", "brand_yoy", "category_yoy")
    return {**{k: _py(d.get(k)) for k in keep}, "source": obj.get("source")}


def load_brand_context(brand_id: str, root: Path | None = None) -> dict:
    """브랜드 하나의 참고의견서 재료를 산출물에서 모은다.

    Args:
        brand_id: `outputs/scores_latest.csv` 의 brand_id 그대로 (예: "BRD_20220035",
            "NAME:국밥상회냉면@외식"). 화면의 `row["brand_id"]` 를 str() 로 넘기면 된다.
        root: 프로젝트 루트. None 이면 이 저장소 루트. 경로는 root/config.yaml 의 paths 를 따른다.

    Returns:
        `build_brand_report_html` 이 받는 dict. 없는 산출물은 빈 값으로 두며 예외를 내지 않는다.

    Raises:
        FileNotFoundError: scores_latest.csv 가 없을 때.
        KeyError: brand_id 가 점수표에 없을 때.
    """
    base = Path(root) if root is not None else ROOT
    out_dir, proc_dir, cfg = _artifact_dirs(base)
    scores = _load(out_dir / "scores_latest.csv", "csv")
    if not isinstance(scores, pd.DataFrame) or "brand_id" not in scores.columns:
        raise FileNotFoundError(f"{out_dir / 'scores_latest.csv'} 을 읽지 못했습니다")
    bid = str(brand_id)
    hit = scores[scores["brand_id"].astype(str) == bid]
    if hit.empty:
        raise KeyError(f"brand_id {bid!r} 가 scores_latest.csv 에 없습니다")
    brand = _rec(hit.iloc[0].to_dict())

    meta = _load(out_dir / "scores_latest_meta.json", "json")
    year = _int((meta or {}).get("scored_year")) if isinstance(meta, dict) else None
    year = year or _int(brand.get("year"))

    diag = _load(out_dir / "brand_diagnosis.parquet", "parquet")
    findings: list[dict] = []
    if isinstance(diag, pd.DataFrame) and "brand_id" in diag.columns:
        sub = diag[diag["brand_id"].astype(str) == bid]
        if "rank" in sub.columns:
            sub = sub.sort_values("rank")
        findings = [_rec(r) for r in sub.to_dict("records")]
    summ = _load(out_dir / "brand_diagnosis_summary.csv", "csv")
    summary = None
    if isinstance(summ, pd.DataFrame) and "brand_id" in summ.columns:
        s = summ[summ["brand_id"].astype(str) == bid]
        summary = _rec(s.iloc[0].to_dict()) if not s.empty else None

    panel = _load(proc_dir / "panel.parquet", "parquet")
    panel = panel if isinstance(panel, pd.DataFrame) and "brand_id" in panel.columns else None
    hist = (panel[panel["brand_id"].astype(str) == bid].sort_values("year")
            if panel is not None else pd.DataFrame())
    company = None
    if not hist.empty and "company_name" in hist.columns:
        names = hist["company_name"].dropna().astype(str)
        company = names.iloc[-1] if len(names) else None
    # 개요의 공시값은 점수와 같은 해 것이어야 한다 — 패널이 먼저 갱신되면 다음 해 값이 섞인다.
    upto = hist[hist["year"] <= year] if year is not None and not hist.empty else hist
    latest = _rec(upto.iloc[-1].to_dict()) if not upto.empty else {}

    bands = _load(out_dir / "grade_bands.json", "json")
    watch = _load(out_dir / "watch_base_rates.json", "json")
    lab = (cfg.get("label") or {}) if isinstance(cfg, dict) else {}
    q = [_num((ev or {}).get("quantile")) for ev in (lab.get("events") or {}).values()]
    q_low = min([x if x <= 0.5 else 1 - x for x in q if x is not None], default=0.20)

    now = datetime.now()
    return {
        "brand_id": bid,
        "generated_at": now,
        "doc_id": f"FS-{bid}-{now:%Y%m%d}",
        "scored_year": year,
        "artifact_time": _artifact_time(out_dir),
        "brand": brand,
        "latest_disclosure": latest,
        "company_name": company,
        "position": _position(scores, brand),
        "bands": copy.deepcopy(bands) if isinstance(bands, dict) else {},
        "watch": copy.deepcopy(watch) if isinstance(watch, dict) else {},
        "label_rule": {"quantile": q_low,
                       "extreme": _num(lab.get("extreme_quantile")) or 0.05,
                       "min_events": _int(lab.get("min_events")) or 2},
        "findings": findings,
        "diagnosis_summary": summary,
        "sections": _sections(panel, bid, year, brand.get("industry_mid")),
        "trend": _trend(hist, year),
        "hq": _hq(proc_dir, company),
        "demand": _demand(out_dir, bid),
        "localdata": _localdata(out_dir, bid),
    }


def _localdata(out_dir: Path, brand_id: str) -> dict | None:
    """월별 인허가 폐점 신호(src/localdata.py) — 공시보다 늦지 않은 개·폐점 흐름."""
    sig = _load(out_dir / "localdata_signal.csv", "csv")
    if not isinstance(sig, pd.DataFrame) or sig.empty:
        return None
    row = sig[sig["brand_id"].astype(str) == str(brand_id)]
    return row.iloc[0].to_dict() if not row.empty else None


# ---------------------------------------------------------------------------
# 렌더링
# ---------------------------------------------------------------------------

# 인쇄 규칙(@media print)에 대한 설명 — CSS 안에 주석을 두지 않는다. 문서마다 그대로 실려
# 나가고, 본문 문구를 찾는 검사(예: "중대 신호" 가 있는가)를 주석 글자가 통과시켜 버린다(실측).
#   · 결론이 첫 쪽에 들어오도록 인쇄에서는 여백을 줄인다. 화면 여백 그대로면 요약 상자가
#     통째로 둘째 쪽으로 밀려 첫 쪽 절반이 비었다(실측).
#   · 상자 전체가 아니라 상자 안의 한 덩어리(타일 줄·표의 행·항목)만 쪼개지 않는다.
#   · 중대 신호 상자는 통째로 넘긴다. 항목이 많아야 3건(실측)이라 한 쪽을 넘지 않고,
#     제목만 앞 쪽에 남으면 빈 상자처럼 보인다(실측).
_CSS = """
:root{--ink:#26221E;--text:#3D3833;--muted:#6B635A;--line:#E8E4DE;--line-2:#D6D0C7;
  --accent:#FFCC00;--bg:#F5F4F1;--paper:#FFFFFF;--tint:#FAF8F5;
  --fs1:#287B57;--fs1-soft:#EAF5EF;--fs2:#986216;--fs2-soft:#FDF4E3;
  --fs3:#B44C46;--fs3-soft:#FBEDEB;--info:#3D6EA4;--info-soft:#EDF3FA}
*,*::before,*::after{box-sizing:border-box}
html{font-size:16px;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:Pretendard,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;
  line-height:1.6;word-break:keep-all;overflow-wrap:anywhere;-webkit-font-smoothing:antialiased}
.sheet{max-width:210mm;margin:28px auto;background:var(--paper);border:1px solid var(--line);
  border-radius:4px;box-shadow:0 1px 2px rgba(38,34,30,.05),0 12px 32px rgba(38,34,30,.07);
  padding:15mm 15mm 12mm}
p{margin:0}
table{border-collapse:collapse;width:100%}
.muted{color:var(--muted)}
.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
.masthead{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;
  padding-bottom:10px;border-bottom:1px solid var(--line);font-size:.8125rem;color:var(--muted)}
.svc{display:flex;align-items:center;gap:8px;font-weight:800;color:var(--ink);font-size:.9375rem;
  letter-spacing:.01em}
.svc i{display:inline-block;width:11px;height:11px;border-radius:2px;background:var(--accent)}
.svc span{font-weight:500;color:var(--muted);font-size:.8125rem;letter-spacing:0}
.docno b{color:var(--ink);font-variant-numeric:tabular-nums}
h1{font-size:1.75rem;line-height:1.3;letter-spacing:-.02em;margin:18px 0 6px}
.subject{display:flex;align-items:center;flex-wrap:wrap;gap:8px 12px;font-size:1.25rem;font-weight:700;
  letter-spacing:-.01em}
.rule{width:64px;height:3px;background:var(--accent);margin:14px 0}
.meta th,.meta td{border:1px solid var(--line);padding:6px 10px;font-size:.8125rem;text-align:left;
  vertical-align:top}
.meta th{background:var(--tint);color:var(--muted);font-weight:600;white-space:nowrap;width:15%}
.meta td{width:35%;font-variant-numeric:tabular-nums}
.scope{margin-top:12px;padding:9px 12px;border-left:3px solid var(--ink);background:var(--tint);
  font-weight:700;font-size:.9375rem}
.sec{margin-top:26px}
h2{display:flex;align-items:baseline;gap:10px;font-size:1.125rem;letter-spacing:-.01em;margin:0 0 10px;
  padding-bottom:6px;border-bottom:1.5px solid var(--ink)}
h2 .no{font-size:.8125rem;color:var(--muted);font-weight:700;font-variant-numeric:tabular-nums}
h3{font-size:1rem;margin:18px 0 6px}
.lead{color:var(--muted);font-size:.875rem;margin:0 0 8px}
.note{color:var(--muted);font-size:.8125rem;margin-top:6px}
.badge{display:inline-flex;align-items:center;gap:5px;padding:1px 10px;border-radius:999px;
  border:1.5px solid currentColor;font-size:.875rem;font-weight:800;line-height:1.55;white-space:nowrap}
.badge.FS1{color:var(--fs1);background:var(--fs1-soft)}
.badge.FS2{color:var(--fs2);background:var(--fs2-soft)}
.badge.FS3{color:var(--fs3);background:var(--fs3-soft)}
.badge.none{color:var(--muted);background:var(--tint)}
.kv th,.kv td{padding:7px 12px;border-bottom:1px solid var(--line);font-size:.875rem;text-align:left;
  vertical-align:top}
.kv th{color:var(--muted);font-weight:600;white-space:nowrap;width:17%}
.overview{border-top:1px solid var(--line)}
.overview td{width:33%}
.summary{--g:var(--muted);--gs:var(--tint);border:1px solid var(--line);border-left:5px solid var(--g);
  border-radius:4px;overflow:hidden}
.summary.FS1{--g:var(--fs1);--gs:var(--fs1-soft)}
.summary.FS2{--g:var(--fs2);--gs:var(--fs2-soft)}
.summary.FS3{--g:var(--fs3);--gs:var(--fs3-soft)}
.tiles{display:grid;grid-template-columns:1.1fr 1fr 1fr;border-bottom:1px solid var(--line)}
.tile{padding:12px 14px;border-left:1px solid var(--line)}
.tile:first-child{border-left:0;background:var(--gs)}
.tile .k{font-size:.8125rem;font-weight:700;color:var(--muted)}
.tile .v{font-size:1.75rem;font-weight:800;line-height:1.25;margin-top:2px;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.tile .v small{font-size:1rem;font-weight:700;margin-left:3px;letter-spacing:0}
.tile .d{font-size:.8125rem;color:var(--muted);margin-top:4px;line-height:1.45}
.tile .v.grade{color:var(--g)}
.summary .kv th{padding-left:14px}
.summary .kv tr:last-child th,.summary .kv tr:last-child td{border-bottom:0}
.action{padding:11px 14px;background:var(--tint);border-top:1px solid var(--line)}
.action .k{font-size:.8125rem;font-weight:700;color:var(--muted)}
.action .v{font-weight:700;margin-top:2px}
.action .d{font-size:.8125rem;color:var(--muted);margin-top:2px}
.critical{margin-top:14px;border:1px solid #E6C3BF;border-radius:4px;background:var(--fs3-soft);padding:12px 14px}
.critical .head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-weight:800;color:var(--fs3)}
.critical .head .muted{font-weight:600;font-size:.875rem}
.critical ol{margin:8px 0 0;padding-left:1.3em}
.critical li{margin:0 0 9px;padding-left:2px}
.critical li:last-child{margin-bottom:0}
.critical .t{font-weight:700;color:var(--ink)}
.critical .dt{font-size:.875rem;color:var(--text);margin-top:2px}
.critical .why{font-size:.875rem;color:var(--text);margin-top:2px}
.critical .chk{font-size:.875rem;color:var(--text);margin-top:2px}
.critical .why b,.critical .chk b{color:var(--fs3);margin-right:4px}
.critical .src{font-size:.8125rem;color:var(--muted)}
.tag{display:inline-block;font-size:.8125rem;font-weight:700;line-height:1.45;padding:0 7px;
  border-radius:3px;border:1px solid currentColor;white-space:nowrap}
.tag.crit{color:#fff;background:var(--fs3);border-color:var(--fs3)}
.finding{--c:var(--muted);--cs:var(--tint);display:grid;grid-template-columns:4px 1fr;gap:0 12px;
  padding:10px 0;border-bottom:1px solid var(--line)}
.finding .bar{background:var(--c);border-radius:2px}
.finding.sev-High{--c:var(--fs3);--cs:var(--fs3-soft)}
.finding.sev-Medium{--c:var(--fs2);--cs:var(--fs2-soft)}
.finding.sev-Low{--c:#8C837A;--cs:var(--tint)}
.finding.dir-mitigant{--c:var(--fs1);--cs:var(--fs1-soft)}
.finding.dir-info{--c:var(--info);--cs:var(--info-soft)}
.fh{display:flex;align-items:baseline;flex-wrap:wrap;gap:4px 8px}
.fh .tag.sev{color:var(--c);background:var(--cs)}
.finding.sev-Low .fh .tag.sev{color:var(--muted)}
.fh .cat{font-size:.8125rem;font-weight:700;color:var(--muted)}
.fh .t{font-weight:700}
.fd{font-size:.9375rem;color:var(--text);margin-top:3px}
.fs{font-size:.8125rem;color:var(--muted);margin-top:2px}
.grid{font-size:.875rem;font-variant-numeric:tabular-nums}
.grid th,.grid td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.grid thead th{background:var(--tint);color:var(--muted);font-weight:700;white-space:nowrap;
  border-top:1.5px solid var(--ink);border-bottom:1px solid var(--line-2)}
.grid td.num,.grid th.num{text-align:right}
.grid td.nowrap{white-space:nowrap}
.grid thead th.cur{color:var(--ink)}
.grid td.cur{font-weight:700}
.grid tr.group td{background:#FCFBF9;padding-top:9px}
.grid tr.group b{margin-right:8px}
.grid tr.group span{color:var(--muted);font-size:.8125rem}
.grid tr.mine td{background:var(--tint);font-weight:700}
.judge{font-size:.8125rem;font-weight:700;padding:0 6px;border-radius:3px;border:1px solid currentColor}
.judge.good{color:var(--fs1)}
.judge.bad{color:var(--fs3)}
.tablewrap{overflow-x:auto}
.empty{border:1px dashed var(--line-2);border-radius:4px;padding:12px 14px;background:var(--tint)}
.empty .t{font-weight:800}
.empty p{font-size:.9375rem;color:var(--text);margin-top:4px}
.checklist{list-style:none;margin:0;padding:0}
.checklist li{display:grid;grid-template-columns:1.5em 1fr;gap:2px 4px;padding:8px 0;
  border-bottom:1px dashed var(--line-2)}
.box{font-size:1.125rem;line-height:1.3}
.ck{font-weight:600}
.cm{font-size:.8125rem;color:var(--muted);margin-top:2px}
.doclist{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:0 20px;
  grid-auto-flow:row dense}
.doclist li.wide{grid-column:1/-1}
.doclist li{display:grid;grid-template-columns:1.5em 1fr;gap:4px;padding:5px 0;font-size:.9375rem;
  border-bottom:1px dashed var(--line-2)}
.record{margin-top:14px}
.record th,.record td{border:1px solid var(--line-2);padding:6px 10px;font-size:.8125rem;text-align:left}
.record th{background:var(--tint);color:var(--muted);width:13%;white-space:nowrap}
.record td{height:32px}
.record td.memo{height:64px}
.basis p{font-size:.9375rem;color:var(--text)}
.basis ul{margin:4px 0 0;padding-left:1.2em;font-size:.9375rem;color:var(--text)}
.basis li{margin:3px 0}
.disclaimer{margin-top:16px;padding:12px 14px;border:1px solid var(--line-2);border-radius:4px;
  background:var(--tint);font-size:.9375rem}
.disclaimer b{display:block;margin-bottom:3px}
.disclaimer p+p{margin-top:4px}
.foot{margin-top:22px;padding-top:8px;border-top:1px solid var(--line);display:flex;
  justify-content:space-between;gap:6px 16px;flex-wrap:wrap;font-size:.8125rem;color:var(--muted)}
a{color:var(--info)}
@page{size:A4;margin:14mm 13mm 16mm;
  @bottom-left{content:"FranSCORE · 가맹 브랜드 리스크 참고의견서 · 내부 참고용";
    font-family:Pretendard,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;font-size:8pt;color:#6B635A}
  @bottom-right{content:counter(page) " / " counter(pages);
    font-family:Pretendard,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;font-size:8pt;color:#6B635A}}
@media print{
  html{font-size:10pt}
  body{background:#fff}
  .sheet{max-width:none;margin:0;padding:0;border:0;border-radius:0;box-shadow:none}
  *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  h1{margin:12px 0 4px}
  .rule{margin:10px 0}
  .meta th,.meta td{padding:4px 8px}
  .scope{margin-top:8px;padding:6px 10px}
  .sec{margin-top:18px}
  h2{margin-bottom:8px;padding-bottom:4px}
  .kv th,.kv td{padding:5px 10px}
  .tile{padding:9px 12px}
  .tile .v{font-size:1.5rem}
  .action{padding:8px 12px}
  .finding{padding:7px 0}
  .checklist li{padding:6px 0}
  .grid th,.grid td{padding:4px 8px}
  h2,h3{break-after:avoid;page-break-after:avoid}
  .masthead,.meta,.tiles,.action,.critical,.finding,.checklist li,.doclist li,.grid tr,
  .kv tr,.empty,.record,.disclaimer{break-inside:avoid;page-break-inside:avoid}
  thead{display:table-header-group}
  .tablewrap{overflow:visible}
  a{color:inherit;text-decoration:none}
  .foot{display:none}
}
@media (max-width:720px){
  .sheet{margin:0;border:0;border-radius:0;box-shadow:none;padding:20px 16px}
  .tiles{grid-template-columns:1fr}
  .tile{border-left:0;border-top:1px solid var(--line)}
  .tile:first-child{border-top:0}
  .meta th,.meta td,.overview th,.overview td{display:block;width:auto;border-bottom:0}
  .meta tr,.overview tr{display:block;border-bottom:1px solid var(--line)}
  .doclist{grid-template-columns:1fr}
}
"""


def _h2(no: int, title: str) -> str:
    return f"<h2><span class='no'>{no:02d}</span>{_e(title)}</h2>"


def _badge(grade: str | None) -> str:
    if grade in GRADE_LABEL:
        return f"<span class='badge {grade}'>{grade} {GRADE_LABEL[grade]}</span>"
    return "<span class='badge none'>등급 없음</span>"


def _detail(f: dict) -> str:
    return _DETAIL_OVERRIDE.get(str(f.get("code") or ""), str(f.get("detail") or ""))


def _ordered(findings: list[dict]) -> list[dict]:
    """무거운 것부터: 위험 → 확인 필요 → 완화요인, 그 안에서 중대 신호 → 심각도 → 가중치."""
    def key(f: dict) -> tuple:
        return (_DIR_ORDER.get(str(f.get("direction")), 3),
                0 if str(f.get("code")) in CRITICAL_CODES else 1,
                _SEV_ORDER.get(str(f.get("severity")), 3),
                -(_num(f.get("weight")) or 0.0),
                _num(f.get("rank")) or 0.0)
    return sorted(findings, key=key)


def _head(ctx: dict, name: str, grade: str | None, doc_id: str, gen: object) -> str:
    bid = str(ctx.get("brand_id") or (ctx.get("brand") or {}).get("brand_id") or "")
    yr = ctx.get("scored_year")
    art = ctx.get("artifact_time")
    return (
        "<header>"
        f"<div class='masthead'><div class='svc'><i></i>{SERVICE}<span>프랜차이즈 브랜드 리스크</span></div>"
        f"<div class='docno'>문서번호 <b>{_e(doc_id)}</b></div></div>"
        f"<h1>{DOC_TITLE}</h1>"
        f"<div class='subject'><span>{_e(name)}</span>{_badge(grade)}</div>"
        "<div class='rule'></div>"
        "<table class='meta'>"
        f"<tr><th>대상 브랜드</th><td>{_e(name)}</td><th>기준 실적연도</th>"
        f"<td>{_e(yr) + '년 (' + _e(int(yr) + 1) + '년 정보공개서)' if yr else '—'}</td></tr>"
        f"<tr><th>브랜드 ID</th><td>{_e(bid) or '—'}</td><th>작성 시각</th><td>{_e(_fmt_dt(gen))}</td></tr>"
        "<tr><th>문서 구분</th><td>참고의견서 · 내부 참고용</td><th>자료 기준</th>"
        f"<td>{'산출물 ' + _e(art) if art else '—'}</td></tr>"
        "</table>"
        f"<div class='scope'>{SCOPE_LINE}</div>"
        "</header>")


def _overview(ctx: dict, name: str) -> str:
    b = ctx.get("brand") or {}
    latest = ctx.get("latest_disclosure") or {}
    yr = _int(ctx.get("scored_year"))
    ind = " · ".join(str(x) for x in (b.get("industry_major"), b.get("industry_mid")) if x)
    stores = _count(b.get("n_stores"), "개")
    direct = _int(latest.get("n_direct"))
    if direct is not None and stores != "—":
        stores += f" · 직영점 {direct:,}개"
    bsy = _int(latest.get("biz_start_year"))
    start = "—"
    if bsy:
        start = f"{bsy}년"
        if yr and yr >= bsy:
            start += f" (업력 {yr - bsy}년)" if yr > bsy else " (업력 1년 미만)"
    sales = _num(latest.get("avg_sales"))
    sales_txt = (f"{_won_thousand(sales)} ({_e(latest.get('year'))}년 실적)"
                 if sales is not None else "공시에 없음")
    regions = _count(latest.get("n_regions"), "곳")
    d = ctx.get("demand")
    if d:
        cat = f" · '{_e(d.get('category'))}' 카테고리 {_signed_pct(_num(d.get('category_yoy')))}" \
            if d.get("category") else ""
        dem = (f"브랜드 {_signed_pct(_num(d.get('brand_yoy')))}{cat} "
               "<span class='muted'>(최근 12개월 대비 직전 12개월, 네이버 데이터랩)</span>")
    elif str(b.get("eligibility_basis") or "") not in ("", "정규", "nan"):
        dem = ("<span class='muted'>수집되지 않음 — 공시 공백 보정으로 새로 평가된 브랜드라 "
               "다음 수집 때 포함됩니다</span>")
    else:
        dem = "<span class='muted'>수집되지 않음 — 가맹점 수가 많은 브랜드부터 수집합니다</span>"
    rows = [
        ("브랜드명", _e(name), "업종", _e(ind) or "—"),
        ("가맹본부", _e(ctx.get("company_name")) or "—", "가맹사업 개시", _e(start)),
        ("가맹점 수", _e(stores), "진출 시·도", _e(regions)),
        ("가맹점 평균매출", sales_txt, "검색 수요", dem),
    ]
    ld = ctx.get("localdata")
    if ld:
        rows.append(("월별 폐점 신호",
                     f"<b>{_e(ld.get('trend'))}</b> — 최근 3개월 폐업 {_int(ld.get('close_3m')) or 0:,}건 · "
                     f"개업 {_int(ld.get('open_3m')) or 0:,}건",
                     "신호 기준", f"{_e(ld.get('as_of_month'))} · 지자체 인허가({_e(ld.get('scope') or '지역 표본')})"))
    body = "".join(f"<tr><th>{a}</th><td>{b_}</td><th>{c}</th><td>{d_}</td></tr>" for a, b_, c, d_ in rows)
    return f"<section class='sec'>{_h2(1, '브랜드 개요')}<table class='kv overview'>{body}</table></section>"


def _watch_rows(ctx: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in (ctx.get("watch") or {}).get("table") or []:
        k = _int(r.get("n_events_at_t"))
        if k is not None:
            out[k] = r
    return out


def _state_texts(ctx: dict) -> tuple[str, str, str]:
    """(상태 타일 값, 타일 보조문, 요약표 문장)."""
    b = ctx.get("brand") or {}
    state = str(b.get("brand_state") or "")
    k = _int(b.get("n_events_at_t"))
    rates = _watch_rows(ctx)
    base = rates.get(0)
    if state == "요주의":
        r = rates.get(k) if k is not None else None
        tile = f"악화 발생 <small>사건 {k}건</small>" if k is not None else "악화 발생"
        if not r:
            return (tile, "같은 사건 수의 과거 실현율 표본이 없습니다",
                    f"악화 발생 — 올해 공시 악화 사건 {_e(k)}건. 같은 사건 수의 과거 실현율 표본이 없습니다.")
        rate = float(r["rate"])
        mult = (f", 건전 {float(base['rate']) * 100:.1f}%의 {rate / float(base['rate']):.1f}배"
                if base and _num(base.get("rate")) else "")
        return (tile, f"다음 해 재발동 실현율 {rate * 100:.1f}%",
                f"<b>악화 발생</b> — 올해 공시 악화 사건 <b>{k}건</b>. 같은 조건의 과거 브랜드 "
                f"{int(r['n']):,}개 중 <b>{rate * 100:.1f}%</b>가 다음 해에도 악화 사건을 냈습니다"
                f"(95% 구간 {float(r['ci_low']) * 100:.1f}~{float(r['ci_high']) * 100:.1f}%{mult}). "
                "이 구간은 학습·평가 표본 밖이라 위 확률값에는 성능 근거가 없습니다 — "
                "사건 수 실현율과 진단 소견으로 판단하십시오.")
    if state == "건전":
        extra = ""
        if base:
            extra = (f" 같은 조건의 과거 브랜드 {int(base['n']):,}개 중 {float(base['rate']) * 100:.1f}%가 "
                     f"다음 해 악화 사건을 냈습니다(95% 구간 {float(base['ci_low']) * 100:.1f}~"
                     f"{float(base['ci_high']) * 100:.1f}%).")
        sub = f"다음 해 악화 발생률 {float(base['rate']) * 100:.1f}%" if base else "올해 악화 사건 없음"
        return ("건전 <small>사건 0건</small>", sub,
                f"<b>건전</b> — 올해 공시에 악화 사건이 없습니다.{extra} "
                "모형 성능이 백테스트로 측정된 구간입니다.")
    if state == "평가불가":
        return ("평가불가", "판정 지표 미관측",
                "<b>평가불가</b> — 올해 공시에서 판정 지표가 관측되지 않았습니다. 등급을 신뢰할 수 없으므로 "
                "공시 원자료와 진단 소견으로 판단하십시오.")
    return ("—", "상태 정보 없음", "상태 정보가 없습니다.")


def _summary(ctx: dict, grade: str | None, cuts: list[float], crit: list[dict]) -> str:
    b = ctx.get("brand") or {}
    state = str(b.get("brand_state") or "")
    band = _band_text(grade, cuts) or _e(b.get("grade_band"))
    p = _risk_pct(_num(b.get("deterioration_1y")), cuts)
    p_txt = f"{p:.1f}<small>%</small>" if p is not None else "—"
    caveat = {"요주의": " · 악화 발생 구간 — 이 확률값은 성능 근거 없음",
              "평가불가": " · 평가불가 — 신뢰할 수 없음"}.get(state, "")
    basis = str(b.get("eligibility_basis") or "")
    if basis and basis not in ("정규", "nan"):      # src/coverage.py — 공시 공백 보정 브랜드
        caveat += f" · 공시 공백 보정 평가({_e(basis)}) — 과거 검증 표본 밖"
    st_val, st_sub, st_sentence = _state_texts(ctx)
    grade_v = (f"{grade} <small>{GRADE_LABEL[grade]}</small>" if grade in GRADE_LABEL else "—")
    # MODEL_USE_SPEC §4: 밴드는 실현율과 함께만 적는다. 컷을 고른 자료 위의 실적(pooled)이 아니라
    # 시점 밖 검증값을 붙인다 — grade_bands.json 스스로 pooled 를 '검증이 아니라 실적표'라 적었다.
    oot = {str(r.get("grade")): r for r in ((ctx.get("bands") or {}).get("out_of_time") or {}).get("by_grade")
           or []}
    v_txt = ""
    if grade in oot and _num(oot[grade].get("rate")) is not None:
        v_txt = (f" (시점 밖 검증 실현율 {float(oot[grade]['rate']) * 100:.1f}%, "
                 f"n={_int(oot[grade].get('n')) or 0:,})")
    tiles = (
        "<div class='tiles'>"
        f"<div class='tile'><div class='k'>등급</div><div class='v grade'>{grade_v}</div>"
        f"<div class='d'>업종 하위구간 악화 전환 확률 {band or '—'}{v_txt}</div></div>"
        f"<div class='tile'><div class='k'>{RISK_LABEL}</div><div class='v'>{p_txt}</div>"
        f"<div class='d'>향후 1년 공시 지표 악화 전환 추정 확률 · 부도확률 아님{caveat}</div></div>"
        f"<div class='tile'><div class='k'>상태</div><div class='v'>{st_val}</div>"
        f"<div class='d'>{st_sub}</div></div>"
        "</div>")

    rows = []
    if len(cuts) == 2:
        defs = " · ".join(f"{g} {GRADE_LABEL[g]} {_band_text(g, cuts)}" for g in ("FS1", "FS2", "FS3"))
        rows.append(("등급 구간", f"{defs} <span class='muted'>— 고정 확률 구간(순위가 아님). 업계 전체가 "
                                  "나빠지면 하위 등급 브랜드가 늘어납니다.</span>"))
    pos = ctx.get("position") or {}
    bits = []
    if pos.get("industry_rank"):
        bits.append(f"{_e(pos['industry'])} 업종 평가 브랜드 {pos['industry_n']:,}개 중 {pos['industry_rank']:,}위"
                    f"({_rank_side(pos['industry_rank'], pos['industry_n'])})")
    if pos.get("overall_rank"):
        bits.append(f"전체 {pos['overall_n']:,}개 중 {pos['overall_rank']:,}위"
                    f"({_rank_side(pos['overall_rank'], pos['overall_n'])})")
    if bits:
        rows.append(("업종 내 위치", " · ".join(bits) + " <span class='muted'>— 브랜드 리스크가 높은 순서"
                                                     "(1위가 가장 높음)</span>"))
    rows.append(("상태", st_sentence))
    ds = ctx.get("diagnosis_summary") or {}
    if ds:
        n_risk, n_high = _int(ds.get("n_risk")) or 0, _int(ds.get("n_high")) or 0
        cats = str(ds.get("categories") or "").strip()
        rows.append(("진단 소견", f"위험 소견 {n_risk}건" + (f"(심각도 높음 {n_high}건)" if n_high else "")
                     + (f" · 걸린 부문 {_e(cats)}" if cats else "")))
    if crit:
        rows.append(("중대 신호", f"<b>{len(crit)}건</b> — 등급과 무관하게 우선 확인할 사실입니다(아래 참조)."))
    kv = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    action = GRADE_ACTIONS.get(grade or "", "")
    act = ""
    if action:
        caveat = (" 평가불가 상태라 등급에 따른 권고는 참고만 하고, 공시 원자료를 직접 확인하십시오."
                  if state == "평가불가" else "")
        act = (f"<div class='action'><div class='k'>권고 처리 · {grade} {GRADE_LABEL[grade]}</div>"
               f"<div class='v'>{_e(action)}</div>"
               f"<div class='d'>점검 주기 {GRADE_CADENCE.get(grade, '—')}. 확인할 항목은 07 체크리스트에 "
               f"정리했습니다.{caveat}</div></div>")
    return (f"<section class='sec'>{_h2(2, '결론 요약')}"
            f"<div class='summary {grade or ''}'>{tiles}<table class='kv'>{kv}</table>{act}</div>"
            f"{_critical(crit)}</section>")


def _rank_side(rank: int, n: int) -> str:
    """순위를 가까운 쪽 끝에서 센 백분율로. 81/121 을 '상위 67%'라 쓰면 위험해 보이지만
    실제로는 리스크가 낮은 쪽이다 — 절반을 넘으면 '하위'로 바꿔 말한다(부문별 점검과 같은 규칙)."""
    if not n:
        return "—"
    top = rank / n * 100
    side, v = ("상위", top) if top <= 50 else ("하위", (n - rank + 1) / n * 100)
    return f"리스크 {side} {v:.1f}%" if v < 10 else f"리스크 {side} {v:.0f}%"


def _critical(crit: list[dict]) -> str:
    """중대 신호 상자 — 왜 무거운지와 무엇을 확인할지만. 수치가 담긴 소견 문장은 03 에 한 번만 싣는다
    (같은 문단이 한 쪽에 두 번 나오면 읽는 사람은 다른 사실로 오인하거나 건너뛴다)."""
    if not crit:
        return ""
    items = []
    for f in crit:
        code = str(f.get("code"))
        chk = (check_for(code) or {}).get("check", "")
        items.append(
            f"<li><div class='t'>{_e(f.get('title'))}</div>"
            f"<div class='why'><b>왜 중대한가</b>{_e(CRITICAL_RATIONALE.get(code, ''))}</div>"
            + (f"<div class='chk'><b>확인할 것</b>{_e(chk)}</div>" if chk else "")
            + f"<div class='src'>근거 {_e(f.get('category'))} · {_e(f.get('source'))} · 상세는 03 핵심 소견</div>"
            "</li>")
    return (f"<div class='critical'><div class='head'><span class='tag crit'>중대</span>"
            f"중대 신호 {len(crit)}건<span class='muted'>— 본부의 법적·회계적 상태 사건이라 등급과 무관하게 "
            f"우선 확인</span></div><ol>{''.join(items)}</ol></div>")


def _finding_item(f: dict) -> str:
    direction = str(f.get("direction") or "risk")
    sev = str(f.get("severity") or "")
    if direction == "mitigant":
        label, cls = "완화요인", "dir-mitigant"
    elif direction == "info":
        label, cls = "확인 필요", "dir-info"
    else:
        # 심각도 값도 데이터에서 온다 — 모르는 값은 글자는 이스케이프하고 class 는 고정값으로만 쓴다.
        label = f"심각도 {_SEV_LABEL.get(sev) or _e(sev) or '—'}"
        cls = f"sev-{sev}" if sev in _SEV_ORDER else "sev-Low"
    crit = "<span class='tag crit'>중대</span>" if str(f.get("code")) in CRITICAL_CODES else ""
    return (f"<div class='finding {cls}'><div class='bar'></div><div>"
            f"<div class='fh'>{crit}<span class='tag sev'>{label}</span>"
            f"<span class='cat'>{_e(f.get('category'))}</span><span class='t'>{_e(f.get('title'))}</span></div>"
            f"<p class='fd'>{_e(_detail(f))}</p>"
            f"<div class='fs'>근거 {_e(f.get('category'))} · {_e(f.get('source'))}</div></div></div>")


def _findings(ordered: list[dict]) -> str:
    if not ordered:
        return (f"<section class='sec'>{_h2(3, '핵심 소견')}"
                "<p class='lead'>이 브랜드의 진단 소견이 산출되지 않았습니다.</p></section>")
    risk = [f for f in ordered if str(f.get("direction")) == "risk"]
    other = [f for f in ordered if str(f.get("direction")) != "risk"]
    parts = [f"<section class='sec'>{_h2(3, '핵심 소견')}",
             "<p class='lead'>무거운 것부터 싣습니다. 문장의 수치는 이 브랜드의 공시·감사보고서 실측값이며, "
             "보도는 점수에 반영하지 않습니다.</p>"]
    if risk:
        parts.append("".join(_finding_item(f) for f in risk))
    else:
        parts.append("<p class='lead'>위험 소견이 없습니다 — 공시 지표에서 눈에 띄는 악화 신호가 "
                     "확인되지 않았습니다.</p>")
    if other:
        parts.append(f"<h3>완화요인 · 확인 필요 ({len(other)}건)</h3>")
        parts.append("".join(_finding_item(f) for f in other))
    parts.append("</section>")
    return "".join(parts)


def _sections_html(sec: dict | None) -> str:
    head = _h2(4, "부문별 점검")
    if not sec:
        return (f"<section class='sec'>{head}"
                "<p class='lead'>이 브랜드의 부문 지표를 계산할 공시가 없습니다.</p></section>")
    body = []
    for g in sec.get("groups") or []:
        body.append(f"<tr class='group'><td colspan='4'><b>{_e(g['title'])}</b><span>{_e(g['sub'])}</span></td></tr>")
        for it in g["items"]:
            j = it.get("judgement") or ""
            judge = (f"<span class='judge {'good' if j == '양호' else 'bad'}'>{_e(j)}</span>" if j
                     else "<span class='muted'>—</span>")
            body.append(f"<tr><td>{_e(it['label'])}</td><td class='num'>{_e(it['value'])}</td>"
                        f"<td>{_e(it['position'])}</td><td>{judge}</td></tr>")
    return (f"<section class='sec'>{head}"
            "<div class='tablewrap'><table class='grid'><thead><tr><th>지표</th><th class='num'>값</th>"
            "<th>업종 내 위치</th><th>판단</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>"
            f"<p class='note'>비교 기준: {_e(sec.get('peer_label'))} · {_e(sec.get('year'))}년 실적. "
            "부문에 점수를 매기지 않습니다 — 여러 지표를 하나로 합칠 가중치의 근거가 없어 관측값과 업종 내 "
            "위치만 싣습니다. '판단'은 업종 상·하위 30% 안에 들 때만 표시합니다.</p></section>")


_TREND_ROWS: tuple[tuple[str, str, str], ...] = (
    ("n_stores", "가맹점 수", "개"),
    ("n_direct", "직영점 수", "개"),
    ("n_new", "신규 개점", "개"),
    ("n_contract_end", "계약종료", "개"),
    ("n_contract_cancel", "계약해지", "개"),
    ("n_name_change", "명의변경", "건"),
    ("out_rate", "종료·해지율", "%"),
    ("avg_sales", "가맹점 평균매출", "백만원"),
    ("avg_sales_per_area", "면적(3.3㎡)당 평균매출", "백만원"),
    ("n_regions", "진출 시·도", "곳"),
)


def _trend_cell(key: str, v: object) -> str:
    x = _num(v)
    if x is None:
        return "—"
    if key == "out_rate":
        return f"{x * 100:.1f}"
    if key in ("avg_sales", "avg_sales_per_area"):
        return f"{x / 1e3:,.1f}"                        # 공시 원단위 천원 → 백만원
    return f"{round(x):,}"


def _trend_html(trend: list[dict], year: int | None) -> str:
    head = _h2(5, "공시 추이")
    if not trend:
        return f"<section class='sec'>{head}<p class='lead'>공시 이력이 없습니다.</p></section>"
    ys = [int(r["year"]) for r in trend]
    th = "".join(f"<th class='num{' cur' if y == year else ''}'>{y}</th>" for y in ys)
    body = []
    for key, label, unit in _TREND_ROWS:
        cells = "".join(f"<td class='num{' cur' if int(r['year']) == year else ''}'>{_trend_cell(key, r.get(key))}</td>"
                        for r in trend)
        body.append(f"<tr><td>{label}</td><td class='muted'>{unit}</td>{cells}</tr>")
    return (f"<section class='sec'>{head}"
            "<p class='lead'>공정거래위원회 가맹사업 공시, 최근 연도까지. 열은 실적연도(그다음 해 "
            "정보공개서에 실림)이고, 굵은 열이 기준 실적연도입니다.</p>"
            f"<div class='tablewrap'><table class='grid'><thead><tr><th>지표</th><th>단위</th>{th}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>"
            "<p class='note'>종료·해지율 = (계약종료 + 계약해지) ÷ 전년 가맹점 수 — 전년 공시가 없거나 전년 점포가 "
            f"{MIN_RATE_BASE}개 미만이면 표시하지 않습니다. 평균매출은 공시의 천원 단위를 백만원으로 환산했습니다. "
            "'—'는 공시에 기재되지 않은 값입니다.</p></section>")


def _hq_html(hq: dict | None, year: int | None) -> str:
    head = _h2(6, "가맹본부 재무")
    hq = hq or {}
    company = hq.get("company")
    if not hq.get("matched"):
        who = f"<b>{_e(company)}</b>의 재무를" if company else "이 브랜드 가맹본부의 재무를"
        return (f"<section class='sec'>{head}<div class='empty'><div class='t'>확인되지 않음 (공시 매칭 불가)</div>"
                f"<p>{who} 금융감독원 전자공시(감사보고서)와 공정거래위원회 정보공개서 열람분에서 매칭하지 "
                "못했습니다. 본부 재무가 '없다'는 뜻이 아니라 이 자료로는 '확인되지 않았다'는 뜻입니다 — "
                "자본잠식·적자 여부는 별도 재무자료로 확인하십시오(07 체크리스트 참조).</p></div></section>")
    web_tag = hq.get("web_tag")
    body = []
    zero_seen = False
    for r in hq.get("rows") or []:
        li, eq = _num(r.get("liabilities")), _num(r.get("equity"))
        ratio = "자본잠식" if eq is not None and eq <= 0 else (
            f"{li / eq * 100:,.0f}%" if li is not None and eq else "—")
        op = str(r.get("audit_opinion") or "").strip()
        op = "" if op in ("None", "nan") else op
        if _num(r.get("going_concern_flag")) == 1:
            op = f"{op} · 계속기업 불확실성" if op else "계속기업 불확실성"
        src = str(r.get("source") or "")
        src = "정보공개서" if web_tag and src == web_tag else src
        cells = []
        for c in ("assets", "liabilities", "equity", "revenue", "operating_income", "net_income"):
            txt = _mil(r.get(c))
            zero_seen = zero_seen or txt.endswith("*")
            cells.append(f"<td class='num'>{txt}</td>")
        body.append(f"<tr><td class='num'>{_e(_int(r.get('fiscal_year')))}</td>{''.join(cells)}"
                    f"<td class='num'>{_e(ratio)}</td><td>{_e(op) or '—'}</td>"
                    f"<td class='nowrap'>{_e(src) or '—'}</td></tr>")
    notes = ["자본총계가 0 이하이면 부채비율 대신 '자본잠식'으로 적습니다."]
    if zero_seen:
        notes.append("* 추출값이 정확히 0인 칸입니다 — 실제 0이 아니라 추출 누락일 수 있으니 원문으로 확인하십시오.")
    if year:
        notes.append(f"진단 소견은 기준 실적연도 직전({year - 1}년)까지의 결산만 씁니다(시점 안전). "
                     "표에는 확보된 최신 결산까지 함께 싣습니다.")
    rc = hq.get("rcept_no")
    if rc:
        notes.append("최근 감사보고서 원문은 DART 접수번호 "
                     f"<a href='https://dart.fss.or.kr/dsaf001/main.do?rcpNo={_e(rc)}'>{_e(rc)}</a>"
                     "로 찾을 수 있습니다.")
    return (f"<section class='sec'>{head}"
            f"<p class='lead'><b>{_e(company)}</b> · {_e(hq.get('source_label'))} · 단위 백만원</p>"
            "<div class='tablewrap'><table class='grid'><thead><tr><th class='num'>결산연도</th>"
            "<th class='num'>자산</th><th class='num'>부채</th><th class='num'>자본</th><th class='num'>매출</th>"
            "<th class='num'>영업이익</th><th class='num'>순이익</th><th class='num'>부채비율</th>"
            "<th>감사의견</th><th>출처</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>"
            f"<p class='note'>{' '.join(notes)}</p></section>")


def _mil(v: object) -> str:
    """원 → 백만원. 정확히 0 은 '0*' — 실측으로 본부 재무 76행이 정확히 0 이었고(예: 순이익),
    감사보고서 추출 누락이 0 으로 굳은 경우라 실제 0 으로 읽히면 안 된다."""
    x = _num(v)
    if x is None:
        return "—"
    if x == 0:
        return "0*"
    return f"{x / 1e6:,.0f}"


_DISCLOSURE_DOC = re.compile(r"^최근 (3개년 )?정보공개서\((.+)\)$")


def _merge_disclosure_docs(docs: list[str]) -> list[str]:
    """정보공개서 항목을 한 줄로 합친다.

    소견마다 '최근 정보공개서(가맹계약 해지·종료 현황)', '최근 정보공개서(평균매출액)'처럼
    같은 문서의 다른 장을 가리켜, 그대로 나열하면 징구 목록이 같은 문서로 여러 줄이 된다
    (실측: 한 브랜드에서 20줄 중 8줄). 받는 서류는 하나이므로 '볼 장'을 묶어 한 줄로 적는다.
    3개년이 필요한 항목이 하나라도 있으면 3개년으로 받는다(최근분을 포함한다).
    """
    sections: list[str] = []
    three_year = False
    first_at: int | None = None
    rest: list[str] = []
    for d in dict.fromkeys(docs):
        m = _DISCLOSURE_DOC.match(d)
        if not m:
            rest.append(d)
            continue
        three_year = three_year or bool(m.group(1))
        if first_at is None:
            first_at = len(rest)
        if m.group(2) not in sections:          # 괄호 안은 한 덩어리 — '해지·종료 현황'을 쪼개지 않는다
            sections.append(m.group(2))
    if first_at is None:
        return rest
    merged = f"최근 {'3개년 ' if three_year else ''}정보공개서 — 볼 항목: {', '.join(sections)}"
    return [*rest[:first_at], merged, *rest[first_at:]]


def _checklist_html(ordered: list[dict]) -> str:
    head = _h2(7, "확인·징구 체크리스트")
    items = checklist(ordered)
    if not items:
        return f"<section class='sec'>{head}<p class='lead'>소견에서 나온 확인 항목이 없습니다.</p></section>"
    title_by_code: dict[str, list[str]] = {}
    for f in ordered:
        title_by_code.setdefault(str(f.get("code")), []).append(str(f.get("title") or ""))
    lis, docs = [], []
    for it in items:
        titles = list(dict.fromkeys(t for c in it["codes"] for t in title_by_code.get(c, []) if t))
        crit = any(c in CRITICAL_CODES for c in it["codes"])
        meta = [f"관련 소견: {' · '.join(_e(t) for t in titles)}"] if titles else []
        if it.get("cadence"):
            meta.append(f"확인 시점: {_e(it['cadence'])}")
        tag = "<span class='tag crit'>중대</span> " if crit else ""
        lis.append(f"<li><span class='box'>☐</span><div><div class='ck'>{tag}{_e(it['check'])}</div>"
                   f"<div class='cm'>{' · '.join(meta)}</div></div></li>")
        docs.extend(it.get("docs") or [])
    doc_html = ""
    uniq = _merge_disclosure_docs(docs)
    if uniq:
        lis_doc = []
        for d in uniq:
            wide = " class='wide'" if len(d) > 40 else ""     # 긴 줄(합친 정보공개서)은 두 칸을 다 쓴다
            lis_doc.append(f"<li{wide}><span class='box'>☐</span><span>{_e(d)}</span></li>")
        doc_html = f"<h3>권고 징구·확인 서류</h3><ul class='doclist'>{''.join(lis_doc)}</ul>"
    record = ("<table class='record'><tr><th>확인자</th><td></td><th>확인일</th><td></td></tr>"
              "<tr><th>확인 결과</th><td class='memo' colspan='3'></td></tr></table>")
    return (f"<section class='sec'>{head}"
            "<p class='lead'>소견마다 확인할 사실과 필요한 서류입니다. 중요한 순서로 적었습니다.</p>"
            f"<ul class='checklist'>{''.join(lis)}</ul>{doc_html}{record}</section>")


def _basis_html(ctx: dict, grade: str | None, cuts: list[float]) -> str:
    bands = ctx.get("bands") or {}
    yr = _int(ctx.get("scored_year"))
    lr = ctx.get("label_rule") or {}
    q = (_num(lr.get("quantile")) or 0.20) * 100
    ex = (_num(lr.get("extreme")) or 0.05) * 100
    k = _int(lr.get("min_events")) or 2
    parts = [f"<section class='sec basis'>{_h2(8, '산출 기준·한계')}",
             "<h3>라벨 — 공시 지표 기반 구조악화 사건 (부도확률 PD 아님)</h3>",
             f"<p>'악화'는 공정거래위원회 가맹사업 공시 지표의 관측 사건입니다 — 가맹점 순감률·실질 평균매출 증가율이 "
             f"업종×연도 하위 {q:.0f}%에, 계약종료율(종료+해지)이 상위 {q:.0f}%에 드는 사건 가운데 {k}개 이상이 "
             f"함께 발동하거나 1개가 극단({ex:.0f}%)일 때입니다. {RISK_LABEL}는 이런 사건이 향후 1년 안에 일어날 "
             "추정 확률입니다. 상대 분위수 정의라 산업 전체가 나빠져도 발동률이 거의 일정하며, 차주의 "
             "채무불이행 확률(PD)이 아닙니다.</p>"]

    oot = bands.get("out_of_time") or {}
    rows = oot.get("by_grade") or []
    if rows and len(cuts) == 2:
        fit = "·".join(str(y) for y in oot.get("fit_years") or [])
        test = "·".join(str(y) for y in oot.get("test_years") or [])
        trs = []
        for r in rows:
            g = str(r.get("grade"))
            mine = g == grade
            trs.append(
                f"<tr class='{'mine' if mine else ''}'><td>{_e(g)} {_e(GRADE_LABEL.get(g, ''))}"
                f"{' ◀ 이 브랜드' if mine else ''}</td><td>{_band_text(g, cuts)}</td>"
                f"<td class='num'>{float(r['rate']) * 100:.1f}%</td>"
                f"<td class='num'>{float(r['ci_lo']) * 100:.1f}~{float(r['ci_hi']) * 100:.1f}%</td>"
                f"<td class='num'>{int(r['events']):,} / {int(r['n']):,}</td></tr>")
        pooled = bands.get("pooled") or []
        pooled_txt = ""
        if pooled:
            yrs = "·".join(str(y) for y in bands.get("calibration_years") or [])
            pooled_txt = (f" 참고로 {yrs}년 합산 실적은 "
                          + " · ".join(f"{_e(p.get('grade'))} {float(p['rate']) * 100:.1f}%" for p in pooled)
                          + f"(총 {int(bands.get('n_validation') or 0):,}건)입니다 — 구간을 고른 자료 위에서 센 "
                          "값이라 검증이 아니라 실적표입니다.")
        parts.append(
            "<h3>등급 검증 — 시점 밖 실현율</h3>"
            "<div class='tablewrap'><table class='grid'><thead><tr><th>등급</th><th>구간(고정)</th>"
            "<th class='num'>실현 악화율</th><th class='num'>95% 구간</th><th class='num'>악화 / 표본</th>"
            f"</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"
            f"<p class='note'>{_e(fit)}년 자료로만 보정한 뒤 {_e(test)}년에 같은 구간을 적용해, 그 등급을 받은 "
            f"브랜드가 다음 해 실제로 악화한 비율입니다(총 {int(oot.get('n') or 0):,}개, 악화 "
            f"{int(oot.get('events') or 0):,}건).{pooled_txt}</p>")

    rates = _watch_rows(ctx)
    if rates:
        b = ctx.get("brand") or {}
        cur_k = _int(b.get("n_events_at_t")) if str(b.get("brand_state")) in ("건전", "요주의") else None
        trs = []
        for kk in sorted(rates):
            r = rates[kk]
            mine = kk == cur_k
            trs.append(
                f"<tr class='{'mine' if mine else ''}'><td>{_e(r.get('state'))}{' ◀ 이 브랜드' if mine else ''}</td>"
                f"<td class='num'>{kk}건</td><td class='num'>{int(r['n']):,}</td>"
                f"<td class='num'>{float(r['rate']) * 100:.1f}%</td>"
                f"<td class='num'>{float(r['ci_low']) * 100:.1f}~{float(r['ci_high']) * 100:.1f}%</td></tr>")
        w = ctx.get("watch") or {}
        wy = w.get("years") or []
        span = f"{wy[0]}~{wy[-1]}년, " if wy else ""
        parts.append(
            "<h3>상태별 다음 해 악화 사건 실현율</h3>"
            "<div class='tablewrap'><table class='grid'><thead><tr><th>상태</th><th class='num'>올해 사건 수</th>"
            "<th class='num'>표본</th><th class='num'>다음 해 발동</th><th class='num'>95% 구간</th></tr></thead>"
            f"<tbody>{''.join(trs)}</tbody></table></div>"
            f"<p class='note'>과거 실적을 그대로 센 값입니다({span}연속 연도 쌍 "
            f"{int(w.get('n_pairs') or 0):,}건). 모형이 아니므로 같은 사건 수 안에서는 브랜드 간 순위를 "
            "매기지 않습니다.</p>")

    hq = ctx.get("hq") or {}
    hq_src = hq.get("source_label") if hq.get("matched") else None
    demand = ctx.get("demand")
    srcs = [
        "공정거래위원회 가맹사업 공시(정보공개서)"
        + (f" — {yr}년 실적 기준({yr + 1}년 정보공개서)" if yr else "") + ": 가맹점 수·신규개점·계약종료·"
        "계약해지·명의변경·평균매출·지역 분포",
        "가맹본부 재무: " + (_e(hq_src) if hq_src else
                           "금융감독원 전자공시(DART) 감사보고서 · 공정거래위원회 정보공개서 열람분")
        + " — 법인명으로 매칭한 본부만 확인됩니다",
        ("네이버 데이터랩 검색어트렌드" + (f" ({_e(demand.get('period'))}, 상대지수)" if demand else
                                     " — 이 브랜드는 수집되지 않음")),
        "네이버 뉴스 검색 — 보도는 점수에 반영하지 않으며 사실관계는 별도 확인 대상입니다",
    ]
    if ctx.get("artifact_time"):
        srcs.append(f"점수·진단 산출물 갱신 {_e(ctx['artifact_time'])} · 등급·확률은 공시 갱신 시 연 1회, "
                    "진단 소견은 매일 갱신")
    parts.append("<h3>자료 출처와 기준</h3><ul>" + "".join(f"<li>{s}</li>" for s in srcs) + "</ul>")
    limits = [
        "라벨이 업종 내 상대 분위수라 절대적 위험 수준이나 경기 국면을 측정하지 않습니다.",
        "악화 발생(올해 악화 사건이 이미 나타난) 구간에는 모형 판별력 근거가 없습니다 — 위 실현율표로 판단합니다.",
        "본부 재무는 공시와 매칭된 본부만 확인됩니다. 매칭되지 않은 본부는 '재무 없음'이 아니라 "
        "'확인되지 않음'입니다.",
        "영업위약금과 본부→가맹점 대여금은 수집한 공시 항목에 없어 보지 못합니다.",
        "실제 여신 부도·연체 실적으로 검증하지 않았습니다 — 어떤 산출물도 부도확률(PD)이 아닙니다.",
        "외식업 가맹 브랜드(누적 가맹점 30개 이상·3년 연속 공시)에 한정됩니다.",
    ]
    parts.append("<h3>알려진 한계</h3><ul>" + "".join(f"<li>{x}</li>" for x in limits) + "</ul>")
    parts.append(
        f"<div class='disclaimer'><b>사용 범위</b><p>{DISCLAIMER}</p>"
        "<p>개별 가맹점주에 대한 불이익 처분의 단독 근거로 쓸 수 없으며, 실명 브랜드의 등급을 대외 공표·"
        "마케팅 자료에 인용하지 않습니다(내부 참고용). 최종 판단은 심사역·심의기구가 합니다.</p></div>")
    parts.append("</section>")
    return "".join(parts)


def build_brand_report_html(ctx: dict) -> str:
    """참고의견서 HTML 문서 한 편(외부 자원 없음, A4 인쇄 규칙 포함).

    Args:
        ctx: `load_brand_context()` 의 반환값. 화면이 직접 만들 때는 최소한
            `{"brand": {"brand_id", "brand_name", "grade", "deterioration_1y", ...}}` 이면 되고,
            나머지 키가 없으면 해당 절을 '없음'으로 쓴다.

    Returns:
        `<!DOCTYPE html>` 로 시작하는 완결 문서 문자열.
    """
    ctx = ctx or {}
    b = ctx.get("brand") or {}
    name = str(b.get("brand_name") or ctx.get("brand_id") or b.get("brand_id") or "(브랜드 미상)")
    grade = grade_code(b.get("grade")) or grade_code(b.get("risk_grade"))
    raw_cuts = (ctx.get("bands") or {}).get("cuts") or []
    cuts = [float(c) for c in raw_cuts] if len(raw_cuts) == 2 else []
    year = _int(ctx.get("scored_year"))
    ordered = _ordered(_finding_records(ctx.get("findings")))
    crit = critical_findings(ordered)
    gen = ctx.get("generated_at") or datetime.now()
    bid = str(ctx.get("brand_id") or b.get("brand_id") or "")
    doc_id = str(ctx.get("doc_id") or (f"FS-{bid}-{gen:%Y%m%d}" if isinstance(gen, datetime) else f"FS-{bid}"))
    body = "".join([
        _head(ctx, name, grade, doc_id, gen),
        _overview(ctx, name),
        _summary(ctx, grade, cuts, crit),
        _findings(ordered),
        _sections_html(ctx.get("sections")),
        _trend_html(ctx.get("trend") or [], year),
        _hq_html(ctx.get("hq"), year),
        _checklist_html(ordered),
        _basis_html(ctx, grade, cuts),
        f"<footer class='foot'><span>{SERVICE} · {DOC_TITLE} · 문서번호 {_e(doc_id)}</span>"
        "<span>내부 참고용 — 대외 공표 금지</span></footer>",
    ])
    return ("<!DOCTYPE html>\n<html lang='ko'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<meta name='color-scheme' content='light'>"
            f"<title>{DOC_TITLE} — {_e(name)}</title><style>{_CSS}</style></head>"
            f"<body><main class='sheet'>{body}</main></body></html>\n")
