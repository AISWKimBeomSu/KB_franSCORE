"""브랜드 개별 진단 — 왜 이 브랜드가 위험한지를 **그 브랜드의 숫자로** 말한다.

해결하려는 문제
    기존 화면은 위험 근거를 SHAP 상위 3개 피처명으로 표시했다:
        "[업종상대] 계약종료 백분위 · [변화] 점포 증가율 · [추세] 점포 증가율 평균"
    이 표시는 세 가지가 틀렸다.
      1. 상위 10개 브랜드의 근거가 전부 동일하다 — 피처명은 18종뿐이고 모델이 가장
         자주 쓰는 축은 소수라 상위권에서 같은 조합이 반복된다. 브랜드마다 다른
         이야기가 있는데 화면은 그걸 지운다.
      2. 숫자가 없다. "계약종료 백분위가 기여했다"는 말은 그 브랜드의 계약종료율이
         몇 %인지, 업종 평균이 몇 %인지 알려주지 않는다.
      3. 사람이 읽는 문장이 아니다. 데이터분석 산출물이지 심사 소견이 아니다.

이 모듈이 하는 일
    브랜드 하나에 대해 공시 이력·본부 재무·수요 추세·보도를 규칙으로 훑어
    **소견(Finding)** 목록을 만든다. 소견은 그 브랜드의 실제 수치를 담은 한국어 문장이고,
    출처와 심각도를 함께 갖는다. 규칙은 34종이며 브랜드마다 발동 조합이 달라진다.

모델과의 관계 (정직성)
    악화확률은 LightGBM 백테스트로 검증된 값이고 **순위의 기준**이다.
    이 모듈의 소견은 그 순위를 설명하고, 모델이 보지 않는 축(감사의견·보도·검색수요)을
    **가산 감시점수**로 덧붙인다. 가산분은 규칙이지 백테스트된 모형이 아니므로
    산출물에서 `det_component` 와 `rule_component` 를 분리해 기록한다.
    합성 점수의 실효성은 evaluate.py 의 검증 단계에서 과거 라벨로 직접 확인한다.

산출
    outputs/brand_diagnosis.parquet  — 소견 1건 = 1행
    outputs/brand_diagnosis_meta.json

실행
    python run_pipeline.py --step diagnose
"""
from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import get_logger, industry_group_col, load_config

# 본부 재무가 정보공개서 열람분인지 가리는 표식(src/ifrmp_web.py 소유).
# 문자열을 여기서 다시 쓰지 않는다 — 한쪽만 바뀌면 출처가 조용히 뒤바뀐다.
from src.ifrmp_web import HQ_SOURCE_TAG as _IFRMP_TAG

log = get_logger("diagnosis")

# 심각도 → 감시점수 가산치 (규칙 성분). 완화요인은 음수.
SEVERITY_WEIGHT = {"High": 12.0, "Medium": 6.0, "Low": 2.0}
MITIGANT_WEIGHT = {"High": -8.0, "Medium": -4.0, "Low": -1.5}

CATEGORIES = ("성장", "계약", "매출", "재무", "구조", "수요", "평판")


# ---------------------------------------------------------------------------
# 소견
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    code: str
    category: str
    severity: str                 # High / Medium / Low
    direction: str                # risk / mitigant / info
    title: str
    detail: str
    source: str
    evidence: dict = field(default_factory=dict)

    @property
    def weight(self) -> float:
        """감시점수 가산치. `info` 는 0 — '확인할 수 없다'는 위험의 증거가 아니다.

        데이터 공백에 점수를 매기면 은행의 수집 한계가 차주의 위험으로 둔갑한다.
        심사역에게 알려야 할 사실이므로 소견으로는 남기되 점수에는 넣지 않는다.
        """
        if self.direction == "info":
            return 0.0
        table = SEVERITY_WEIGHT if self.direction == "risk" else MITIGANT_WEIGHT
        return table.get(self.severity, 0.0)


# ---------------------------------------------------------------------------
# 숫자 표기 — 화면·문장이 같은 규칙을 쓰도록 한 곳에 둔다
# ---------------------------------------------------------------------------

def _num(v) -> float | None:
    """None/NaN/문자열을 안전하게 float 로. 실패하면 None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def won(thousand_krw: float | None) -> str:
    """천원 단위 값을 '억/만원' 표기로. 공시 avg_sales 가 천원 단위다."""
    v = _num(thousand_krw)
    if v is None:
        return "미기재"
    krw = v * 1_000
    if abs(krw) >= 1e8:
        return f"{krw / 1e8:,.1f}억원"
    if abs(krw) >= 1e4:
        return f"{krw / 1e4:,.0f}만원"
    return f"{krw:,.0f}원"


def won_from_krw(krw: float | None) -> str:
    """원 단위 값(DART 재무)을 '억/만원' 표기로."""
    v = _num(krw)
    if v is None:
        return "미확보"
    if abs(v) >= 1e12:
        return f"{v / 1e12:,.2f}조원"
    if abs(v) >= 1e8:
        return f"{v / 1e8:,.0f}억원"
    if abs(v) >= 1e4:
        return f"{v / 1e4:,.0f}만원"
    return f"{v:,.0f}원"


def pct(x: float | None, digits: int = 1) -> str:
    v = _num(x)
    return "—" if v is None else f"{v * 100:.{digits}f}%"


def signed_pct(x: float | None, digits: int = 1) -> str:
    v = _num(x)
    if v is None:
        return "—"
    return f"{'+' if v >= 0 else ''}{v * 100:.{digits}f}%"


_JOSA = {"은": ("은", "는"), "이": ("이", "가"), "을": ("을", "를"),
         "과": ("과", "와"), "으로": ("으로", "로")}
# 숫자를 한국어로 읽었을 때 받침 유무 (0영 1일 2이 3삼 4사 5오 6육 7칠 8팔 9구)
_DIGIT_BATCHIM = {"0": True, "1": True, "2": False, "3": True, "4": False,
                  "5": False, "6": True, "7": True, "8": True, "9": False}


def josa(word: str, kind: str = "은") -> str:
    """앞말의 받침에 맞는 조사를 고른다.

    화면 문장에 '3,879만원는'·'1.6억원로' 같은 표기가 나오면 사람이 쓴 글로 읽히지
    않는다. 값이 바뀔 때마다 조사가 달라지므로 문자열에 박아둘 수 없다.
    """
    w = str(word or "").rstrip()
    if not w:
        return _JOSA.get(kind, ("", ""))[1]
    ch = w[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        final = (code - 0xAC00) % 28
        has = final != 0
        # '으로/로'는 ㄹ 받침(final==8)일 때 받침 없는 것처럼 '로'를 쓴다
        if kind == "으로" and final == 8:
            has = False
    elif ch.isdigit():
        has = _DIGIT_BATCHIM[ch]
    elif ch.isalpha():
        has = ch.lower() in "lmnr"      # L·M·N·R 로 끝나는 영문은 받침처럼 읽는다
    else:
        return _JOSA.get(kind, ("", ""))[1]
    a, b = _JOSA.get(kind, ("", ""))
    return a if has else b


def _rank_word(p: float) -> str:
    """업종 내 백분위(0~1, 클수록 높은 값) → '상위 N%' 표현."""
    top = (1.0 - p) * 100
    return f"상위 {top:.0f}%"


# ---------------------------------------------------------------------------
# 진단 입력
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    """규칙 하나가 보는 모든 재료. 없는 재료는 None 이며 규칙이 스스로 건너뛴다."""
    brand_id: str
    brand_name: str
    year: int
    cur: pd.Series                      # 대상연도 패널 행
    hist: pd.DataFrame                  # 대상연도까지의 이력 (연도 오름차순)
    ind: dict                           # 업종 벤치마크 (같은 연도·업종그룹)
    industry_label: str
    # 전 연도 벤치마크. 해마다 공시되지 않는 항목(창업비용 등)을 과거 값으로 볼 때
    # **그 값의 연도 분포**와 비교해야 하므로 전체를 들고 있는다.
    bench: dict = field(default_factory=dict)
    hq: pd.DataFrame | None = None      # 본부 재무 (fiscal_year 오름차순, ≤ year-1)
    hq_company: str = ""
    demand: dict | None = None          # 네이버 검색어트렌드 요약
    news: list[dict] | None = None      # 뉴스 신호

    def prev(self, k: int = 1) -> pd.Series | None:
        if len(self.hist) <= k:
            return None
        return self.hist.iloc[-(k + 1)]

    def g(self, col: str, row: pd.Series | None = None) -> float | None:
        r = self.cur if row is None else row
        return _num(r.get(col)) if r is not None else None


# ---------------------------------------------------------------------------
# 파생 계산 (여러 규칙이 공유)
# ---------------------------------------------------------------------------

def _growth(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None or prev <= 0:
        return None
    return cur / prev - 1.0


def _decline_streak(series: Iterable[float | None]) -> int:
    """마지막 값부터 거슬러 올라가며 연속 감소한 햇수."""
    vals = list(series)
    n = 0
    for i in range(len(vals) - 1, 0, -1):
        a, b = _num(vals[i]), _num(vals[i - 1])
        if a is None or b is None or a >= b:
            break
        n += 1
    return n


# ---------------------------------------------------------------------------
# 규칙 — 성장
# ---------------------------------------------------------------------------

def r_store_decline(ctx: Ctx) -> Finding | None:
    prev = ctx.prev()
    if prev is None:
        return None
    cur_n, prev_n = ctx.g("n_stores"), ctx.g("n_stores", prev)
    g = _growth(cur_n, prev_n)
    if g is None or g >= -0.02:
        return None
    ind_g = ctx.ind.get("store_growth_median")
    sev = "High" if g <= -0.15 else "Medium" if g <= -0.07 else "Low"
    cmp_txt = ""
    if ind_g is not None:
        cmp_txt = (f" 같은 {ctx.industry_label} 업종의 중간값은 "
                   f"{signed_pct(ind_g)}입니다.")
    return Finding(
        code="STORE_DECLINE", category="성장", severity=sev, direction="risk",
        title="가맹점이 줄었습니다",
        detail=(f"{ctx.year}년 가맹점이 {int(prev_n):,}개에서 {int(cur_n):,}개로 "
                f"{int(prev_n - cur_n):,}개({signed_pct(g)}) 줄었습니다.{cmp_txt}"),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"prev": prev_n, "cur": cur_n, "growth": g, "industry_median": ind_g})


def r_store_decline_streak(ctx: Ctx) -> Finding | None:
    n = _decline_streak(ctx.hist["n_stores"].tolist())
    if n < 2:
        return None
    first = ctx.hist.iloc[-(n + 1)]
    a, b = _num(first["n_stores"]), ctx.g("n_stores")
    if a is None or b is None or a <= 0:
        return None
    sev = "High" if n >= 3 else "Medium"
    return Finding(
        code="STORE_DECLINE_STREAK", category="성장", severity=sev, direction="risk",
        title=f"{n}년 연속 감소",
        detail=(f"가맹점 수가 {int(first['year'])}년 {int(a):,}개에서 {ctx.year}년 "
                f"{int(b):,}개까지 {n}년 내리 줄었습니다(누적 {signed_pct(b / a - 1)}). "
                "한 해 부진이 아니라 추세로 굳어진 상태입니다."),
        source=f"공정거래위원회 가맹사업 공시 {int(first['year'])}~{ctx.year}",
        evidence={"years": n, "from": a, "to": b})


def r_store_growth(ctx: Ctx) -> Finding | None:
    prev = ctx.prev()
    if prev is None:
        return None
    g = _growth(ctx.g("n_stores"), ctx.g("n_stores", prev))
    if g is None or g < 0.10:
        return None
    return Finding(
        code="STORE_GROWTH", category="성장", severity="Medium", direction="mitigant",
        title="가맹점이 늘고 있습니다",
        detail=(f"{ctx.year}년 가맹점이 {signed_pct(g)} 늘어 "
                f"{int(ctx.g('n_stores')):,}개가 되었습니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"growth": g})


def r_new_open_stall(ctx: Ctx) -> Finding | None:
    prev = ctx.prev()
    if prev is None:
        return None
    base = ctx.g("n_stores", prev)
    new_c, new_p = ctx.g("n_new"), ctx.g("n_new", prev)
    if base is None or base <= 0 or new_c is None or new_p is None:
        return None
    rate = new_c / base
    if new_p < 5 or rate >= 0.05:
        return None
    drop = _growth(new_c, new_p)
    if drop is None or drop > -0.5:
        return None
    return Finding(
        code="NEW_OPEN_STALL", category="성장", severity="Medium", direction="risk",
        title="신규 출점이 멈췄습니다",
        detail=(f"신규 개점이 전년 {int(new_p):,}개에서 {int(new_c):,}개로 줄었습니다"
                f"(기존 점포 대비 {pct(rate)}). 새로 들어오려는 점주가 끊기면 "
                "기존 점포의 이탈을 메울 방법이 없습니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"new_prev": new_p, "new_cur": new_c, "rate": rate})


# ---------------------------------------------------------------------------
# 규칙 — 계약
# ---------------------------------------------------------------------------

MIN_RATE_BASE = 10      # 비율 소견의 최소 분모 — 이보다 적으면 비율이 의미를 잃는다


def r_contract_end_high(ctx: Ctx) -> Finding | None:
    prev = ctx.prev()
    if prev is None:
        return None
    base = ctx.g("n_stores", prev)
    end = ctx.g("n_contract_end")
    if base is None or base < MIN_RATE_BASE or end is None or end <= 0:
        # 점포 5개 중 2개가 끝나면 40% 지만 그것은 위험 신호가 아니라 표본 부족이다
        # (실측: 84개로 성장한 브랜드가 '계약종료 40%'로 최상위에 올라옴).
        return None
    rate = end / base
    p = ctx.ind.get("contract_end_pct_of", lambda _x: None)(rate)
    med = ctx.ind.get("contract_end_median")
    if p is not None and p < 0.75 and rate < 0.12:
        return None
    if p is None and rate < 0.12:
        return None
    sev = "High" if (p or 0) >= 0.92 or rate >= 0.25 else "Medium"
    rank_txt = f" {ctx.industry_label} 업종 {_rank_word(p)}에 해당합니다." if p is not None else ""
    med_txt = f" 업종 중간값은 {pct(med)}입니다." if med is not None else ""
    return Finding(
        code="CONTRACT_END_HIGH", category="계약", severity=sev, direction="risk",
        title="계약을 끝내는 가맹점이 많습니다",
        detail=(f"{ctx.year}년에 계약이 끝난 가맹점이 {int(end):,}개로, 연초 점포의 "
                f"{pct(rate)}입니다.{rank_txt}{med_txt}"),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"n_end": end, "rate": rate, "pctile": p, "industry_median": med})


def r_cancel_high(ctx: Ctx) -> Finding | None:
    prev = ctx.prev()
    if prev is None:
        return None
    base = ctx.g("n_stores", prev)
    cancel = ctx.g("n_contract_cancel")
    if base is None or base < MIN_RATE_BASE or cancel is None or cancel <= 0:
        return None
    rate = cancel / base
    if rate < 0.05:
        return None
    sev = "High" if rate >= 0.12 else "Medium"
    return Finding(
        code="CANCEL_HIGH", category="계약", severity=sev, direction="risk",
        title="중도 해지가 많습니다",
        detail=(f"계약 기간을 채우지 못하고 중도 해지한 가맹점이 {int(cancel):,}개"
                f"({pct(rate)})입니다. 만기 종료와 달리 중도 해지는 점주가 손실을 "
                "감수하고 나가는 것이라 수익성 악화 신호로 봅니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"n_cancel": cancel, "rate": rate})


def r_churn_exceeds_open(ctx: Ctx) -> Finding | None:
    new = ctx.g("n_new")
    end = ctx.g("n_contract_end") or 0.0
    cancel = ctx.g("n_contract_cancel") or 0.0
    out = end + cancel
    if new is None or out <= 0 or out <= new:
        return None
    gap = out - new
    if gap < 3:
        return None
    return Finding(
        code="CHURN_EXCEEDS_OPEN", category="계약", severity="Medium", direction="risk",
        title="나가는 점포가 들어오는 점포보다 많습니다",
        detail=(f"{ctx.year}년 신규 개점 {int(new):,}개에 견줘 계약종료·해지가 "
                f"{int(out):,}개로 {int(gap):,}개 더 많습니다. 순유출 상태입니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"n_new": new, "n_out": out, "gap": gap})


def r_name_change_high(ctx: Ctx) -> Finding | None:
    prev = ctx.prev()
    if prev is None:
        return None
    base = ctx.g("n_stores", prev)
    chg = ctx.g("n_name_change")
    if base is None or base < MIN_RATE_BASE or chg is None or chg <= 0:
        return None
    rate = chg / base
    if rate < 0.08:
        return None
    return Finding(
        code="NAME_CHANGE_HIGH", category="계약", severity="Low", direction="risk",
        title="점주 교체가 잦습니다",
        detail=(f"명의변경이 {int(chg):,}건({pct(rate)})입니다. 점포는 유지되지만 "
                "운영자가 바뀌는 것으로, 폐점 직전 단계에서 자주 나타납니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"n_name_change": chg, "rate": rate})


# ---------------------------------------------------------------------------
# 규칙 — 매출
# ---------------------------------------------------------------------------

def r_sales_decline(ctx: Ctx) -> Finding | None:
    prev = ctx.prev()
    if prev is None:
        return None
    cur_s, prev_s = ctx.g("avg_sales"), ctx.g("avg_sales", prev)
    g = _growth(cur_s, prev_s)
    if g is None or g >= -0.03:
        return None
    sev = "High" if g <= -0.20 else "Medium" if g <= -0.10 else "Low"
    ind_g = ctx.ind.get("sales_growth_median")
    cmp_txt = (f" 같은 업종 중간값은 {signed_pct(ind_g)}입니다." if ind_g is not None else "")
    return Finding(
        code="SALES_DECLINE", category="매출", severity=sev, direction="risk",
        title="가맹점 평균매출이 떨어졌습니다",
        detail=(f"점포당 연매출이 {won(prev_s)}에서 {won(cur_s)}"
                f"{josa(won(cur_s), '으로')} {signed_pct(g)} 줄었습니다.{cmp_txt} "
                "매출이 줄면 점주의 대출 상환 여력이 먼저 나빠집니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"prev": prev_s, "cur": cur_s, "growth": g})


def r_sales_decline_streak(ctx: Ctx) -> Finding | None:
    n = _decline_streak(ctx.hist["avg_sales"].tolist())
    if n < 2:
        return None
    first = ctx.hist.iloc[-(n + 1)]
    a, b = _num(first["avg_sales"]), ctx.g("avg_sales")
    if a is None or b is None or a <= 0:
        return None
    return Finding(
        code="SALES_DECLINE_STREAK", category="매출", severity="High", direction="risk",
        title=f"매출이 {n}년 연속 하락",
        detail=(f"점포당 매출이 {int(first['year'])}년 {won(a)}에서 {ctx.year}년 "
                f"{won(b)}까지 {n}년 내리 줄었습니다(누적 {signed_pct(b / a - 1)})."),
        source=f"공정거래위원회 가맹사업 공시 {int(first['year'])}~{ctx.year}",
        evidence={"years": n, "from": a, "to": b})


def r_sales_low_rank(ctx: Ctx) -> Finding | None:
    s = ctx.g("avg_sales")
    fn = ctx.ind.get("sales_pct_of")
    if s is None or fn is None:
        return None
    p = fn(s)
    if p is None or p > 0.20:
        return None
    med = ctx.ind.get("sales_median")
    return Finding(
        code="SALES_LOW_RANK", category="매출", severity="Medium", direction="risk",
        title="업종 안에서 매출이 낮습니다",
        detail=(f"점포당 매출 {won(s)}{josa(won(s))} {ctx.industry_label} 업종 하위 "
                f"{p * 100:.0f}% 수준입니다"
                + (f" (업종 중간값 {won(med)})." if med is not None else ".")),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"avg_sales": s, "pctile": p, "industry_median": med})


def r_sales_per_area_decline(ctx: Ctx) -> Finding | None:
    prev = ctx.prev()
    if prev is None:
        return None
    g = _growth(ctx.g("avg_sales_per_area"), ctx.g("avg_sales_per_area", prev))
    if g is None or g >= -0.08:
        return None
    # ⚠️ 정합성 검사: 점포당 매출이 늘었는데 면적당 매출만 급락하면 그것은 장사가
    #    안 된 것이 아니라 **면적 기재 기준이 바뀐 것**이다(실측: 매출 +61.7% 인데
    #    면적당 −85.4%). 데이터 아티팩트를 위험 소견으로 내보내면 안 된다.
    sales_g = _growth(ctx.g("avg_sales"), ctx.g("avg_sales", prev))
    if sales_g is not None and sales_g > 0.0:
        return None
    if g <= -0.60 and (sales_g is None or sales_g > -0.30):
        return None
    return Finding(
        code="SALES_PER_AREA_DECLINE", category="매출", severity="Low", direction="risk",
        title="면적당 매출이 떨어졌습니다",
        detail=(f"3.3㎡당 매출이 {signed_pct(g)} 줄었습니다. 점포 크기를 감안해도 "
                "장사가 덜 되고 있다는 뜻입니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"growth": g})


def r_sales_growth(ctx: Ctx) -> Finding | None:
    prev = ctx.prev()
    if prev is None:
        return None
    g = _growth(ctx.g("avg_sales"), ctx.g("avg_sales", prev))
    if g is None or g < 0.08:
        return None
    return Finding(
        code="SALES_GROWTH", category="매출", severity="Medium", direction="mitigant",
        title="점포당 매출이 늘었습니다",
        detail=(f"점포당 연매출이 {signed_pct(g)} 늘어 {won(ctx.g('avg_sales'))}"
                f"{josa(won(ctx.g('avg_sales')), '이')} 되었습니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"growth": g})


# ---------------------------------------------------------------------------
# 규칙 — 구조
# ---------------------------------------------------------------------------

def r_region_concentration(ctx: Ctx) -> Finding | None:
    share = ctx.g("top_region_share")
    nreg = ctx.g("n_regions")
    n = ctx.g("n_stores")
    if share is None or nreg is None or n is None or n < 30:
        return None
    if share < 0.55 or nreg > 6:
        return None
    return Finding(
        code="REGION_CONCENTRATION", category="구조", severity="Medium", direction="risk",
        title="특정 지역에 몰려 있습니다",
        detail=(f"점포의 {pct(share)}가 한 시·도에 몰려 있고 진출 지역은 "
                f"{int(nreg)}곳뿐입니다. 그 지역 상권이 흔들리면 브랜드 전체가 "
                "동시에 흔들립니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"top_region_share": share, "n_regions": nreg})


def r_region_shrink(ctx: Ctx) -> Finding | None:
    prev = ctx.prev()
    if prev is None:
        return None
    cur_r, prev_r = ctx.g("n_regions"), ctx.g("n_regions", prev)
    if cur_r is None or prev_r is None or prev_r - cur_r < 2:
        return None
    return Finding(
        code="REGION_SHRINK", category="구조", severity="Medium", direction="risk",
        title="진출 지역이 줄었습니다",
        detail=(f"영업 중인 시·도가 {int(prev_r)}곳에서 {int(cur_r)}곳으로 "
                f"{int(prev_r - cur_r)}곳 줄었습니다. 특정 지역에서 통째로 철수했다는 뜻입니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"prev": prev_r, "cur": cur_r})


def r_direct_ratio_drop(ctx: Ctx) -> Finding | None:
    prev = ctx.prev()
    if prev is None:
        return None
    dc, nc = ctx.g("n_direct"), ctx.g("n_stores")
    dp, npv = ctx.g("n_direct", prev), ctx.g("n_stores", prev)
    if None in (dc, nc, dp, npv) or nc <= 0 or npv <= 0 or dp < 3:
        return None
    rc, rp = dc / nc, dp / npv
    if rp - rc < 0.05:
        return None
    return Finding(
        code="DIRECT_RATIO_DROP", category="구조", severity="Low", direction="risk",
        title="직영점을 줄이고 있습니다",
        detail=(f"직영점 비중이 {pct(rp)}에서 {pct(rc)}로 낮아졌습니다"
                f"({int(dp)}개 → {int(dc)}개). 본부가 직접 운영하던 점포를 정리하는 것은 "
                "현금이 급할 때 나타나는 움직임입니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"ratio_prev": rp, "ratio_cur": rc})


def r_small_scale(ctx: Ctx) -> Finding | None:
    n = ctx.g("n_stores")
    if n is None or n >= 50:
        return None
    return Finding(
        code="SMALL_SCALE", category="구조", severity="Low", direction="risk",
        title="규모가 작습니다",
        detail=(f"가맹점이 {int(n):,}개로, 점포 몇 곳만 이탈해도 브랜드 지표가 "
                "크게 흔들립니다. 충격을 흡수할 여력이 얇습니다."),
        source=f"공정거래위원회 가맹사업 공시 {ctx.year}",
        evidence={"n_stores": n})


def r_young_brand(ctx: Ctx) -> Finding | None:
    bsy = ctx.g("biz_start_year")
    if bsy is None:
        return None
    age = ctx.year - bsy
    if age > 4:
        return None
    return Finding(
        code="YOUNG_BRAND", category="구조", severity="Low", direction="risk",
        title="가맹사업 업력이 짧습니다",
        detail=(f"{int(bsy)}년에 가맹사업을 시작해 업력이 {int(age)}년입니다. "
                "초기 확장기에는 점포가 빠르게 늘지만, 첫 계약 만기가 돌아오는 "
                "시점에 이탈이 몰리는 경우가 많습니다."),
        source="공정거래위원회 정보공개서",
        evidence={"biz_start_year": bsy, "age": age})


def r_startup_cost_high(ctx: Ctx) -> Finding | None:
    """창업비용이 업종 대비 과다한가.

    ⚠️ 창업비용 공시는 해마다 나오지 않는다 — 실측으로 **2024년은 전 브랜드 결측**이었다.
       대상연도만 보면 이 규칙은 영원히 발동하지 않는다(적대적 감사에서 적발).
       창업비용은 해마다 크게 바뀌지 않는 구조적 속성이므로, **대상연도 이하에서
       가장 최근에 공시된 값**을 쓰고 그 값의 연도를 문장에 명시한다.
       비교 분포도 같은 연도의 것을 써야 사과를 사과와 비교한 것이 된다.
    """
    hist = ctx.hist[ctx.hist["startup_total"].notna()]
    hist = hist[pd.to_numeric(hist["startup_total"], errors="coerce") > 0]
    if hist.empty:
        return None
    row = hist.iloc[-1]
    total, src_year = _num(row["startup_total"]), int(row["year"])
    if total is None:
        return None
    ind = ctx.ind if src_year == ctx.year else ctx.bench.get(
        (src_year, ctx.industry_label), {})
    fn = ind.get("startup_pct_of")
    if fn is None:
        return None
    p = fn(total)
    if p is None or p < 0.85:
        return None
    med = ind.get("startup_median")
    lead = f"{src_year}년 공시 기준 " if src_year != ctx.year else ""
    sales = ctx.g("avg_sales")
    payback = ""
    if sales and sales > 0:
        payback = (f" 점포당 연매출 {won(sales)} 기준으로 창업비 회수에 "
                   f"필요한 매출 규모가 큰 편입니다.")
    return Finding(
        code="STARTUP_COST_HIGH", category="구조", severity="Low", direction="risk",
        title="창업비용이 업종 대비 높습니다",
        detail=(f"{lead}창업비용 합계 {won(total)}{josa(won(total))} "
                f"{ctx.industry_label} 업종 {_rank_word(p)} 수준입니다"
                + (f" (업종 중간값 {won(med)})." if med is not None else ".")
                + payback),
        source=f"공정거래위원회 정보공개서 {src_year}",
        evidence={"startup_total": total, "pctile": p, "industry_median": med,
                  "source_year": src_year})


def r_registration_cancelled(ctx: Ctx) -> Finding | None:
    if not ctx.g("cancel_flag"):
        return None
    kind = str(ctx.cur.get("cancel_type") or "등록취소")
    return Finding(
        code="REGISTRATION_CANCELLED", category="구조", severity="High", direction="risk",
        title=f"정보공개서 {kind}",
        detail=(f"이 브랜드의 정보공개서 등록이 {kind} 처리되었습니다. "
                "가맹점을 새로 모집할 수 없는 상태이며, 기존 가맹점 지원도 "
                "정상적으로 이뤄지는지 확인이 필요합니다."),
        source="공정거래위원회 가맹사업 등록취소 공시",
        evidence={"cancel_type": kind})


# ---------------------------------------------------------------------------
# 규칙 — 본부 재무 (DART 감사보고서)
# ---------------------------------------------------------------------------

HQ_STALE_YEARS = 3      # 최근 결산이 이보다 오래되면 현재 상태로 단정하지 않는다


def _hq_last(ctx: Ctx) -> pd.Series | None:
    if ctx.hq is None or ctx.hq.empty:
        return None
    return ctx.hq.iloc[-1]


def _hq_age(ctx: Ctx, last: pd.Series) -> int:
    return int(ctx.year) - int(last["fiscal_year"])


def _stale_note(ctx: Ctx, last: pd.Series) -> str:
    """오래된 결산이면 그 사실을 문장 끝에 붙인다.

    실측 사례: 2024년 화면에 '투다리 본부가 부분 자본잠식 상태입니다' 가 떴는데
    근거는 **2019년** 감사보고서였다. 5년 전 재무를 현재형으로 단정하면 심사역이
    지금의 사실로 읽는다. 연도만 적어 두는 것으로는 부족하고 명시해야 한다.
    """
    age = _hq_age(ctx, last)
    if age <= HQ_STALE_YEARS:
        return ""
    return (f" 다만 확보된 가장 최근 결산이 {int(last['fiscal_year'])}년으로 "
            f"{age}년 전 자료이므로, 현재 상태는 별도 확인이 필요합니다.")


def _stale_severity(ctx: Ctx, last: pd.Series, sev: str) -> str:
    """오래된 자료에 근거한 소견은 한 단계 낮춰 잡는다."""
    if _hq_age(ctx, last) <= HQ_STALE_YEARS:
        return sev
    return {"High": "Medium", "Medium": "Low"}.get(sev, "Low")


def _hq_source(last: pd.Series) -> str:
    """이 소견의 근거가 **어느 문서**인지 적는다.

    ⚠️ 본부 재무는 이제 두 원천에서 온다 — 금감원 감사보고서와 공정위 정보공개서
       열람분. 그런데 모든 소견이 출처를 '금융감독원 전자공시 감사보고서'로
       박아 두고 있었다. 그 결과 외부감사 대상이 아닌 개인사업자 본부(신전푸드시스
       등)의 수치에까지 DART 이름표가 붙었다 — 심사역이 확인하러 가면 그 문서는
       없다. 근거 문장에서 출처를 틀리는 것은 수치를 틀리는 것과 같은 급의 잘못이다.
    """
    fy = last.get("fiscal_year")
    yr = f"{int(fy)} " if pd.notna(fy) else ""
    if str(last.get("source") or "") == _IFRMP_TAG:
        return f"공정거래위원회 가맹사업정보제공시스템 {yr}정보공개서"
    return f"금융감독원 전자공시 {yr}감사보고서".replace("  ", " ")


def r_hq_equity_negative(ctx: Ctx) -> Finding | None:
    last = _hq_last(ctx)
    if last is None:
        return None
    eq = _num(last.get("equity"))
    if eq is None or eq > 0:
        return None
    return Finding(
        code="HQ_EQUITY_NEGATIVE", category="재무",
        severity=_stale_severity(ctx, last, "High"), direction="risk",
        title="본부가 완전자본잠식 상태입니다",
        detail=(f"{ctx.hq_company}의 {int(last['fiscal_year'])}년 자본총계가 "
                f"{won_from_krw(eq)}{josa(won_from_krw(eq), '으로')} 마이너스입니다. "
                "부채가 자산을 넘어선 상태로, "
                "본부가 물류·판촉·신규출점 지원을 계속할 수 있을지 확인해야 합니다."
                + _stale_note(ctx, last)),
        source=_hq_source(last),
        evidence={"equity": eq, "fiscal_year": int(last["fiscal_year"])})


def r_hq_capital_impaired(ctx: Ctx) -> Finding | None:
    last = _hq_last(ctx)
    if last is None:
        return None
    eq, cs = _num(last.get("equity")), _num(last.get("capital_stock"))
    if eq is None or cs is None or cs <= 0 or eq <= 0 or eq >= cs:
        return None
    return Finding(
        code="HQ_CAPITAL_IMPAIRED", category="재무",
        severity=_stale_severity(ctx, last, "High"), direction="risk",
        title="본부가 부분 자본잠식 상태입니다",
        detail=(f"{ctx.hq_company}의 {int(last['fiscal_year'])}년 자본총계 "
                f"{won_from_krw(eq)}{josa(won_from_krw(eq), '이')} 자본금 "
                f"{won_from_krw(cs)}에 미달합니다"
                f"(잠식률 {pct(1 - eq / cs)}). 누적 결손이 납입자본을 갉아먹고 "
                "있습니다." + _stale_note(ctx, last)),
        source=_hq_source(last),
        evidence={"equity": eq, "capital_stock": cs, "impair": 1 - eq / cs})


def r_hq_operating_loss(ctx: Ctx) -> Finding | None:
    last = _hq_last(ctx)
    if last is None:
        return None
    oi, rev = _num(last.get("operating_income")), _num(last.get("revenue"))
    if oi is None or oi >= 0:
        return None
    margin = (oi / rev) if rev and rev > 0 else None
    n = 0
    if ctx.hq is not None:
        for v in reversed(ctx.hq["operating_income"].tolist()):
            x = _num(v)
            if x is None or x >= 0:
                break
            n += 1
    sev = "High" if n >= 2 else "Medium"
    streak = f" {n}년 연속입니다." if n >= 2 else ""
    mg = f" 영업이익률 {pct(margin)}." if margin is not None else ""
    return Finding(
        code="HQ_OPERATING_LOSS", category="재무",
        severity=_stale_severity(ctx, last, sev), direction="risk",
        title="본부가 영업적자입니다",
        detail=(f"{ctx.hq_company}의 {int(last['fiscal_year'])}년 영업손익이 "
                f"{won_from_krw(oi)}입니다.{mg}{streak} 본업에서 돈을 벌지 못하는 "
                "상태라 가맹점 지원 여력이 줄어듭니다." + _stale_note(ctx, last)),
        source=_hq_source(last),
        evidence={"operating_income": oi, "margin": margin, "streak": n})


def r_hq_high_leverage(ctx: Ctx) -> Finding | None:
    last = _hq_last(ctx)
    if last is None:
        return None
    li, eq = _num(last.get("liabilities")), _num(last.get("equity"))
    if li is None or eq is None or eq <= 0:
        return None
    ratio = li / eq
    if ratio < 4.0:
        return None
    sev = "High" if ratio >= 8.0 else "Medium"
    return Finding(
        code="HQ_HIGH_LEVERAGE", category="재무",
        severity=_stale_severity(ctx, last, sev), direction="risk",
        title="본부 부채비율이 높습니다",
        detail=(f"{ctx.hq_company}의 {int(last['fiscal_year'])}년 부채비율이 "
                f"{ratio * 100:,.0f}%입니다(부채 {won_from_krw(li)} / 자본 "
                f"{won_from_krw(eq)}). 금리가 오르거나 매출이 흔들리면 "
                "이자 부담이 곧바로 손익을 압박합니다." + _stale_note(ctx, last)),
        source=_hq_source(last),
        evidence={"debt_ratio": ratio})


def r_hq_going_concern(ctx: Ctx) -> Finding | None:
    last = _hq_last(ctx)
    if last is None:
        return None
    gc = last.get("going_concern_flag")
    if pd.isna(gc) or not gc:
        return None
    return Finding(
        code="HQ_GOING_CONCERN", category="재무",
        severity=_stale_severity(ctx, last, "High"), direction="risk",
        title="감사보고서에 계속기업 불확실성이 기재되었습니다",
        detail=(f"{ctx.hq_company}의 {int(last['fiscal_year'])}년 감사보고서에 "
                "'계속기업 관련 중요한 불확실성' 문단이 있습니다. 회계감사인이 "
                "이 회사가 사업을 계속할 수 있을지에 의문을 표시했다는 뜻으로, "
                "가장 무거운 재무 경고 신호입니다." + _stale_note(ctx, last)),
        source=_hq_source(last),
        evidence={"fiscal_year": int(last["fiscal_year"])})


def r_hq_audit_opinion(ctx: Ctx) -> Finding | None:
    last = _hq_last(ctx)
    if last is None:
        return None
    op = str(last.get("audit_opinion") or "").strip()
    if op in ("", "적정", "None", "nan"):
        return None
    return Finding(
        code="HQ_AUDIT_OPINION", category="재무",
        severity=_stale_severity(ctx, last, "High"), direction="risk",
        title=f"감사의견 '{op}'",
        detail=(f"{ctx.hq_company}의 {int(last['fiscal_year'])}년 감사의견이 "
                f"'{op}'입니다. 적정의견이 아니라는 것은 재무제표 자체를 "
                "그대로 믿기 어렵다는 뜻입니다." + _stale_note(ctx, last)),
        source=_hq_source(last),
        evidence={"audit_opinion": op})


def r_hq_revenue_decline(ctx: Ctx) -> Finding | None:
    if ctx.hq is None or len(ctx.hq) < 2:
        return None
    a, b = _num(ctx.hq.iloc[-2].get("revenue")), _num(ctx.hq.iloc[-1].get("revenue"))
    g = _growth(b, a)
    if g is None or g >= -0.10:
        return None
    sev = "Medium" if g > -0.25 else "High"
    last = ctx.hq.iloc[-1]
    return Finding(
        code="HQ_REVENUE_DECLINE", category="재무",
        severity=_stale_severity(ctx, last, sev), direction="risk",
        title="본부 매출이 줄었습니다",
        detail=(f"{ctx.hq_company}의 매출이 {int(ctx.hq.iloc[-2]['fiscal_year'])}년 "
                f"{won_from_krw(a)}에서 {int(ctx.hq.iloc[-1]['fiscal_year'])}년 "
                f"{won_from_krw(b)}{josa(won_from_krw(b), '으로')} {signed_pct(g)} 줄었습니다. "
                "본부 매출은 대부분 가맹점에 넘기는 물품·로열티라, 가맹점 사정이 "
                "본부 장부에 먼저 잡힙니다." + _stale_note(ctx, last)),
        source=_hq_source(last),
        evidence={"prev": a, "cur": b, "growth": g})


def r_hq_solid(ctx: Ctx) -> Finding | None:
    last = _hq_last(ctx)
    if last is None:
        return None
    eq, cs = _num(last.get("equity")), _num(last.get("capital_stock"))
    oi, ni = _num(last.get("operating_income")), _num(last.get("net_income"))
    if None in (eq, oi) or eq <= 0 or oi <= 0:
        return None
    if cs is not None and cs > 0 and eq < cs:
        return None
    if ni is not None and ni < 0:
        return None
    return Finding(
        code="HQ_SOLID", category="재무", severity="Medium", direction="mitigant",
        title="본부 재무는 안정적입니다",
        detail=(f"{ctx.hq_company}의 {int(last['fiscal_year'])}년 자본총계 "
                f"{won_from_krw(eq)}, 영업이익 {won_from_krw(oi)}"
                f"{josa(won_from_krw(oi), '으로')} "
                "자본잠식·영업적자가 없습니다." + _stale_note(ctx, last)),
        source=_hq_source(last),
        evidence={"equity": eq, "operating_income": oi})


def r_hq_no_data(ctx: Ctx) -> Finding | None:
    if ctx.hq is not None and not ctx.hq.empty:
        return None
    return Finding(
        code="HQ_NO_DATA", category="재무", severity="Low", direction="info",
        title="본부 재무를 확인할 수 없습니다",
        detail=("이 브랜드의 가맹본부는 외부감사 대상이 아니어서 감사보고서를 "
                "제출하지 않고, 정보공개서 열람분도 아직 확보하지 못했습니다. "
                "본부의 자본잠식·적자 여부를 확인할 방법이 없으므로, 여신 심사 시 "
                "별도 재무자료를 징구해 확인해야 합니다."),
        source="금융감독원 전자공시·공정위 정보공개서 조회 결과 없음",
        evidence={})


# ---------------------------------------------------------------------------
# 규칙 — 수요 (네이버 데이터랩 검색어 트렌드)
# ---------------------------------------------------------------------------

def r_demand_decline(ctx: Ctx) -> Finding | None:
    d = ctx.demand or {}
    g = _num(d.get("brand_yoy"))
    if g is None or g >= -0.15:
        return None
    sev = "High" if g <= -0.40 else "Medium"
    return Finding(
        code="DEMAND_DECLINE", category="수요", severity=sev, direction="risk",
        title="브랜드를 찾는 사람이 줄었습니다",
        detail=(f"네이버에서 '{ctx.brand_name}'을 검색한 양이 최근 1년 동안 "
                f"{signed_pct(g)} 변했습니다. 검색은 매장을 찾기 전 단계라 "
                "매출보다 먼저 움직입니다."),
        source=f"네이버 데이터랩 검색어트렌드 ({d.get('period', '최근 24개월')})",
        evidence={"brand_yoy": g})


def r_demand_underperform(ctx: Ctx) -> Finding | None:
    d = ctx.demand or {}
    bg, cg = _num(d.get("brand_yoy")), _num(d.get("category_yoy"))
    if bg is None or cg is None:
        return None
    gap = bg - cg
    if gap >= -0.15:
        return None
    cat = d.get("category") or ctx.industry_label
    return Finding(
        code="DEMAND_UNDERPERFORM", category="수요", severity="Medium", direction="risk",
        title="업종 흐름보다 더 나쁩니다",
        detail=(f"'{cat}' 카테고리 전체 검색량이 {signed_pct(cg)}인데 "
                f"'{ctx.brand_name}'은 {signed_pct(bg)}입니다. 업종이 어려운 것이 아니라 "
                "이 브랜드가 유독 밀리고 있다는 뜻입니다."),
        source=f"네이버 데이터랩 검색어트렌드 ({d.get('period', '최근 24개월')})",
        evidence={"brand_yoy": bg, "category_yoy": cg, "gap": gap})


def r_category_decline(ctx: Ctx) -> Finding | None:
    d = ctx.demand or {}
    cg = _num(d.get("category_yoy"))
    if cg is None or cg >= -0.15:
        return None
    cat = d.get("category") or ctx.industry_label
    return Finding(
        code="CATEGORY_DECLINE", category="수요", severity="Low", direction="risk",
        title="업종 자체의 수요가 줄고 있습니다",
        detail=(f"'{cat}' 카테고리 검색량이 최근 1년 {signed_pct(cg)}입니다. "
                "이 브랜드만의 문제가 아니라 업종 전체가 축소되는 국면이므로, "
                "같은 업종 여신 전체의 쏠림을 함께 봐야 합니다."),
        source=f"네이버 데이터랩 검색어트렌드 ({d.get('period', '최근 24개월')})",
        evidence={"category_yoy": cg, "category": cat})


def r_demand_growth(ctx: Ctx) -> Finding | None:
    d = ctx.demand or {}
    g = _num(d.get("brand_yoy"))
    if g is None or g < 0.20:
        return None
    return Finding(
        code="DEMAND_GROWTH", category="수요", severity="Medium", direction="mitigant",
        title="브랜드 관심도가 늘고 있습니다",
        detail=(f"네이버 검색량이 최근 1년 {signed_pct(g)} 늘었습니다. "
                "신규 고객 유입이 이어지고 있다는 신호입니다."),
        source=f"네이버 데이터랩 검색어트렌드 ({d.get('period', '최근 24개월')})",
        evidence={"brand_yoy": g})


# ---------------------------------------------------------------------------
# 규칙 — 평판 (뉴스)
# ---------------------------------------------------------------------------

_NEWS_SEVERITY = {"본부분쟁": "High", "재무이슈": "High", "집단폐점": "High", "기타": "Low"}
_NEWS_FRAME = {
    "본부분쟁": "가맹점주와의 분쟁",
    "재무이슈": "본부 재무 관련 사안",
    "집단폐점": "다수 점포 폐점",
    "기타": "브랜드 관련 사안",
}


def r_news_events(ctx: Ctx) -> list[Finding]:
    items = [n for n in (ctx.news or [])
             if str(n.get("event_type", "무관")) not in ("무관", "")]
    if not items:
        return []
    out: list[Finding] = []
    by_type: dict[str, list[dict]] = {}
    for n in items:
        by_type.setdefault(str(n["event_type"]), []).append(n)
    for et, group in by_type.items():
        # 신호 스키마가 원천마다 달라 title 이 비어 있는 건이 있다. 인용할 문장이
        # 하나도 없으면 「」 라는 빈 따옴표가 화면에 나가므로(실측) 그 경우는
        # 건수만 말하고 인용은 생략한다 — 없는 제목을 지어내지 않는다.
        quotes = [t for t in (str(g.get("title") or g.get("evidence_sentence") or "").strip()
                              for g in group) if t]
        if quotes:
            more = f" 외 {len(quotes) - 1}건" if len(quotes) > 1 else ""
            lead = f"「{quotes[0][:70]}」{more} 관련 보도가 확인됩니다."
        else:
            lead = f"{_NEWS_FRAME.get(et, et)}으로 분류된 보도가 {len(group)}건 확인됩니다."
        out.append(Finding(
            code=f"NEWS_{et}", category="평판",
            severity=_NEWS_SEVERITY.get(et, "Low"), direction="risk",
            title=f"보도 확인 — {_NEWS_FRAME.get(et, et)}",
            detail=(f"{lead} 보도는 점수에 반영하지 않으므로 사실관계는 "
                    "별도로 확인해야 합니다."),
            source="네이버 뉴스 검색 · 분류 Gemini",
            evidence={"event_type": et, "n_articles": len(group),
                      "titles": quotes[:5],
                      "urls": [g.get("url") or g.get("source_url") for g in group[:5]]}))
    return out


# ---------------------------------------------------------------------------
# 규칙 등록
# ---------------------------------------------------------------------------

RULES: tuple[Callable[[Ctx], Finding | None], ...] = (
    r_store_decline, r_store_decline_streak, r_store_growth, r_new_open_stall,
    r_contract_end_high, r_cancel_high, r_churn_exceeds_open, r_name_change_high,
    r_sales_decline, r_sales_decline_streak, r_sales_low_rank,
    r_sales_per_area_decline, r_sales_growth,
    r_region_concentration, r_region_shrink, r_direct_ratio_drop,
    r_small_scale, r_young_brand, r_startup_cost_high, r_registration_cancelled,
    r_hq_equity_negative, r_hq_capital_impaired, r_hq_operating_loss,
    r_hq_high_leverage, r_hq_going_concern, r_hq_audit_opinion,
    r_hq_revenue_decline, r_hq_solid, r_hq_no_data,
    r_demand_decline, r_demand_underperform, r_category_decline, r_demand_growth,
)

MULTI_RULES: tuple[Callable[[Ctx], list[Finding]], ...] = (r_news_events,)

# 같은 사실을 두 번 말하지 않게 하는 억제 규칙 (강한 소견이 약한 소견을 덮는다)
_SUPPRESS = {
    "HQ_EQUITY_NEGATIVE": {"HQ_CAPITAL_IMPAIRED", "HQ_HIGH_LEVERAGE", "HQ_SOLID"},
    "HQ_CAPITAL_IMPAIRED": {"HQ_SOLID"},
    "HQ_GOING_CONCERN": {"HQ_SOLID"},
    "HQ_AUDIT_OPINION": {"HQ_SOLID"},
    "HQ_OPERATING_LOSS": {"HQ_SOLID"},
    # 부채비율 1,306% 를 High 로 띄우면서 같은 카드에 "본부 재무는 안정적입니다"
    # 를 함께 내보내고 있었다(실측: 신전떡볶이). 자본잠식·적자가 아니라는 것은
    # 사실이지만, 위험 소견 옆의 "안정적"은 심사역이 읽는 순간 신뢰를 깎는다.
    "HQ_HIGH_LEVERAGE": {"HQ_SOLID"},
    "HQ_REVENUE_DECLINE": {"HQ_SOLID"},
    "STORE_DECLINE_STREAK": set(),
    "SALES_DECLINE_STREAK": {"SALES_DECLINE"},
    "DEMAND_UNDERPERFORM": {"CATEGORY_DECLINE"},
}

_SEV_ORDER = {"High": 0, "Medium": 1, "Low": 2}
_DIR_ORDER = {"risk": 0, "mitigant": 1, "info": 2}


def run_rules(ctx: Ctx) -> list[Finding]:
    """규칙 전체를 적용하고 중복을 정리한 소견 목록 (심각도 → 리스크 우선 정렬)."""
    found: list[Finding] = []
    for rule in RULES:
        try:
            f = rule(ctx)
        except Exception as exc:            # 규칙 하나가 죽어도 진단 전체를 멈추지 않는다
            log.warning("규칙 %s 실패 (%s/%s): %s", rule.__name__, ctx.brand_id, ctx.year, exc)
            continue
        if f is not None:
            found.append(f)
    for mrule in MULTI_RULES:
        try:
            found.extend(mrule(ctx))
        except Exception as exc:
            log.warning("규칙 %s 실패 (%s/%s): %s", mrule.__name__, ctx.brand_id, ctx.year, exc)

    codes = {f.code for f in found}
    drop: set[str] = set()
    for code in codes:
        drop |= _SUPPRESS.get(code, set())
    found = [f for f in found if f.code not in drop]
    found.sort(key=lambda f: (_DIR_ORDER.get(f.direction, 3),
                              _SEV_ORDER.get(f.severity, 3), f.code))
    return found


# ---------------------------------------------------------------------------
# 업종 벤치마크
# ---------------------------------------------------------------------------

def _pct_fn(values: pd.Series):
    """해당 분포에서 값 x 의 백분위를 돌려주는 함수 (0~1). 표본이 적으면 None."""
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(arr) < 10:
        return lambda _x: None
    arr = np.sort(arr)
    n = len(arr)

    def fn(x):
        v = _num(x)
        if v is None:
            return None
        return float(np.searchsorted(arr, v, side="right")) / n

    return fn


def industry_benchmarks(panel: pd.DataFrame, min_group: int = 30) -> dict:
    """(연도, 업종그룹) → 벤치마크 딕셔너리.

    라벨·피처가 쓰는 industry_group_col 과 **같은 그룹 정의**를 써서
    화면 설명과 모델이 서로 다른 업종을 말하는 일이 없게 한다.
    """
    df = panel.copy()
    df["_grp"] = industry_group_col(df, min_group)
    df = df.sort_values(["brand_id", "year"])
    g = df.groupby("brand_id")
    df["_prev_stores"] = g["n_stores"].shift(1)
    df["_prev_sales"] = g["avg_sales"].shift(1)
    df["_store_growth"] = df["n_stores"] / df["_prev_stores"] - 1.0
    df["_sales_growth"] = df["avg_sales"] / df["_prev_sales"] - 1.0
    df["_end_rate"] = df["n_contract_end"] / df["_prev_stores"].replace(0, np.nan)

    out: dict = {}
    # 전부 결측인 업종·연도 그룹에서 median 이 나오는 것은 정상이다(값 없음 → None).
    # numpy 의 "Mean of empty slice" 경고가 로그를 덮어 진짜 경고를 가리므로 잠재운다.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return _benchmark_groups(df, out)


def _benchmark_groups(df: pd.DataFrame, out: dict) -> dict:
    for (year, grp), sub in df.groupby(["year", "_grp"]):
        out[(int(year), str(grp))] = {
            "label": str(grp),
            "n": len(sub),
            "store_growth_median": _num(sub["_store_growth"].median()),
            "sales_growth_median": _num(sub["_sales_growth"].median()),
            "contract_end_median": _num(sub["_end_rate"].median()),
            "sales_median": _num(sub["avg_sales"].median()),
            "startup_median": _num(sub["startup_total"].median())
            if "startup_total" in sub.columns else None,
            "contract_end_pct_of": _pct_fn(sub["_end_rate"]),
            "sales_pct_of": _pct_fn(sub["avg_sales"]),
            "startup_pct_of": _pct_fn(sub["startup_total"])
            if "startup_total" in sub.columns else (lambda _x: None),
        }
    return out


# ---------------------------------------------------------------------------
# 확률 계단(plateau) 보간
# ---------------------------------------------------------------------------

PLATEAU_TOL = 1e-4      # 이보다 가까운 보정값은 같은 계단으로 본다


def smooth_calibrated(p_raw, p_cal, tol: float = PLATEAU_TOL) -> np.ndarray:
    """isotonic 보정으로 생긴 **평탄 구간 내부를 원점수로 선형 보간**한다.

    왜 필요한가
        isotonic 회귀는 계단 함수라 같은 계단에 든 브랜드가 전부 같은 확률로 표시된다
        (실측: 2,510개 중 216개가 42.86% = 3/7 로 동일). 화면에서 이것은 "모델이
        브랜드를 구분하지 못한다"로 읽히고, 실제로 순위 정보를 버리는 것이기도 하다.

    왜 정확 일치가 아니라 허용오차로 묶는가
        evaluate.py 는 동률 분리를 위해 보정값에 순위 기반 미세 노이즈(≤1e-5)를 더한다.
        그 결과 계단 위의 값들이 0.00030007·0.00030008 처럼 **전부 달라져** 정확 일치
        그룹핑으로는 계단을 찾지 못한다(실측: 보간 전후 완전 동일). 노이즈보다 크고
        실제 isotonic 단계 간격보다 작은 tol 로 묶어야 계단이 잡힌다.

    어떻게 (계단 내부 대칭 분산)
        계단 g 의 값을 v_g, 이웃 계단과의 간격을 d⁻·d⁺ 라 할 때, 그 계단에 든 브랜드를
        원점수 순위에 따라 [v_g − d⁻/2, v_g + d⁺/2] 위에 고르게 펼친다.
        - **평균 보존**: 순위를 −0.5~+0.5 로 중심화해 곱하므로 계단 평균이 v_g 그대로다.
          앵커 보간(np.interp) 방식은 최상·최하 계단이 끝값으로 눌려(실측: 상위 10개가
          전부 0.4528 로 그대로 남음) 정작 화면에서 가장 중요한 최상위가 안 펴진다.
        - **단조성**: 각 계단이 간격의 절반까지만 쓰므로 이웃 계단과 겹치지 않는다.
        - **순위 보존**: 계단 내부 순서는 원점수(= 모델 순위)를 따른다.
        양끝 계단은 간격을 한쪽만 알 수 있으므로 반대쪽 간격을 거울처럼 쓴다.

    Returns:
        보간된 확률 배열 (입력과 같은 길이).
    """
    raw = np.asarray(p_raw, dtype=float)
    cal = np.asarray(p_cal, dtype=float)
    if raw.size != cal.size or raw.size == 0:
        return cal
    ok = np.isfinite(raw) & np.isfinite(cal)
    if ok.sum() < 3:
        return cal

    r, c = raw[ok], cal[ok]
    order = np.argsort(c, kind="stable")
    cs = c[order]
    gid_sorted = np.cumsum(np.concatenate([[True], np.diff(cs) > tol])) - 1
    gid = np.empty(len(c), dtype=int)
    gid[order] = gid_sorted
    n_groups = int(gid.max()) + 1
    if n_groups < 2:                        # 계단이 하나뿐 — 펼 근거가 없다
        return cal

    levels = np.array([c[gid == g].mean() for g in range(n_groups)], dtype=float)
    diffs = np.diff(levels)
    gaps_lo = np.empty(n_groups)
    gaps_hi = np.empty(n_groups)
    gaps_lo[1:], gaps_hi[:-1] = diffs, diffs
    gaps_lo[0] = diffs[0]                   # 최하 계단: 위쪽 간격을 거울로
    gaps_hi[-1] = diffs[-1]                 # 최상 계단: 아래쪽 간격을 거울로

    # 계단이 쓸 수 있는 **한쪽 폭**. 세 가지를 동시에 만족해야 한다.
    #   · 이웃 계단과 겹치지 않는다        → 간격의 절반 이하
    #   · [0,1] 을 벗어나지 않는다         → level 과 1-level 이하
    #   · 계단 평균이 보존된다             → 위아래 폭이 **같아야** 한다
    # 마지막 조건 때문에 좌우 중 좁은 쪽으로 맞춘다. 비대칭으로 펼친 뒤 [0,1] 로
    # 자르면 잘려나간 만큼 평균이 위로 밀린다(실측: 최하 계단 0.0100 → 0.0113).
    half = np.minimum.reduce([gaps_lo / 2.0, gaps_hi / 2.0, levels, 1.0 - levels])
    half = np.maximum(half, 0.0)

    out_ok = np.empty(len(c), dtype=float)
    for g in range(n_groups):
        m = gid == g
        k = int(m.sum())
        if k == 1 or half[g] <= 0:
            out_ok[m] = levels[g]
            continue
        # 순위를 −1 ~ +1 로 중심화 (평균 0 → 계단 평균 보존)
        rank = pd.Series(r[m]).rank(method="first").to_numpy() - 1.0
        u = rank / (k - 1) * 2.0 - 1.0
        out_ok[m] = levels[g] + u * half[g]

    out = cal.copy()
    out[ok] = out_ok
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 감시점수
# ---------------------------------------------------------------------------

RULE_CAP = 30.0             # 규칙 성분의 절대 상한 — 검증된 모델 성분을 압도하지 못하게
_DUP_DISCOUNT = 0.35        # 같은 카테고리 두 번째 이후 소견의 반영 비율


def rule_component(findings: list[Finding]) -> float:
    """규칙 소견을 **카테고리별로 체감 합산**한다.

    왜 단순 합이 아닌가
        '계약종료율 높음'·'중도해지 많음'·'나가는 점포가 더 많음'은 사실상 한 가지 사실을
        세 각도에서 말한 것이다. 그대로 더하면 같은 사실이 세 번 계산돼 규칙 성분이
        폭주한다(실측: 단순 합에서 최대 +126점 — 검증된 모델 성분 최대 100점을 넘어섰다).
        카테고리 안에서는 가장 무거운 소견을 온전히 반영하고 나머지는 35%만 더한다.
        서로 다른 카테고리(재무·수요·계약)가 동시에 나빠지는 것은 진짜 누적 위험이므로
        카테고리 사이는 그대로 더한다.
    """
    by_cat: dict[str, list[float]] = {}
    for f in findings:
        w = f.weight
        if w == 0.0:
            continue
        by_cat.setdefault(f.category, []).append(w)
    total = 0.0
    for weights in by_cat.values():
        risk = sorted((w for w in weights if w > 0), reverse=True)
        mit = sorted(w for w in weights if w < 0)
        for group in (risk, mit):
            if group:
                total += group[0] + _DUP_DISCOUNT * sum(group[1:])
    # 하드 캡은 상한에 닿는 순간 변별력을 잃는다 (실측: 브랜드의 75%가 정확히 30.0).
    # tanh 로 부드럽게 포화시키면 ±RULE_CAP 을 넘지 않으면서 순서는 끝까지 유지된다.
    return float(RULE_CAP * np.tanh(total / RULE_CAP))


def watch_components(det_pctile: float, findings: list[Finding]) -> dict:
    """모델 성분과 규칙 성분을 **분리해서** 기록한다.

    det_component  : 백테스트로 검증된 모델 순위 (0~100)
    rule_component: 규칙 소견 가산·감산 — 백테스트된 모형이 아니라 심사 정책이다
    watch_raw     : 둘의 합 (여기서는 자르지 않는다)

    최종 감시점수는 코호트 전체를 본 뒤 순위로 정규화한다(finalize_watch_scores).
    합계를 0~100 으로 바로 자르면 상위권이 전부 100 으로 뭉개져(실측: 상위 12개 전부
    100.0) 애초에 고치려던 '값이 다 똑같다' 문제를 그대로 재현한다.
    """
    base = float(np.clip(det_pctile, 0.0, 1.0) * 100.0)
    add = rule_component(findings)
    return {
        "det_component": round(base, 2),
        "rule_component": round(add, 2),
        "watch_raw": round(base + add, 3),
    }


def finalize_watch_scores(summary: pd.DataFrame) -> pd.DataFrame:
    """코호트 내 순위로 감시점수를 0~100 에 고르게 펼친다.

    감시점수는 '이 브랜드를 몇 번째로 봐야 하는가'를 뜻하므로 절대값이 아니라
    순위가 의미다. 순위 정규화는 동점을 만들지 않아(method='first') 화면에서
    브랜드가 항상 구분된다.
    """
    if summary.empty or "watch_raw" not in summary.columns:
        return summary
    r = summary["watch_raw"].rank(method="first", pct=True)
    summary = summary.copy()
    summary["watch_score"] = (r * 100.0).round(1)
    return summary


# ---------------------------------------------------------------------------
# 배치 실행
# ---------------------------------------------------------------------------

def _hq_frame(hq_all: pd.DataFrame | None, key: str, upto_year: int) -> pd.DataFrame | None:
    """시점 안전: 회계연도 ≤ 대상연도−1 만 본다 (features.py 와 같은 규칙)."""
    if hq_all is None or not key:
        return None
    sub = hq_all[(hq_all["key"] == key) & (hq_all["fiscal_year"] <= upto_year - 1)]
    return sub.sort_values("fiscal_year") if len(sub) else None


def diagnose_cohort(cfg: dict, scores: pd.DataFrame, panel: pd.DataFrame,
                    hq_all: pd.DataFrame | None = None,
                    demand: dict | None = None,
                    news: dict | None = None,
                    min_group: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """점수 코호트 전체를 진단한다.

    Returns:
        (findings_df, summary_df)
        findings_df — 소견 1건 = 1행
        summary_df  — 브랜드 1개 = 1행 (감시점수·보간확률·상위 소견 요약)
    """
    year = int(scores["year"].iloc[0]) if "year" in scores.columns else int(panel["year"].max())
    bench = industry_benchmarks(panel, min_group)
    panel_sorted = panel.sort_values(["brand_id", "year"])
    hist_by_brand = dict(iter(panel_sorted.groupby("brand_id")))
    grp_col = industry_group_col(panel_sorted, min_group)
    grp_by_key = dict(zip(zip(panel_sorted["brand_id"], panel_sorted["year"], strict=False),
                          grp_col, strict=False))

    company_by_brand = (panel_sorted.groupby("brand_id")["company_name"].last().to_dict()
                        if "company_name" in panel_sorted.columns else {})
    try:
        from src.dart import norm_corp
    except Exception:                                   # dart 모듈이 없어도 진단은 돈다
        def norm_corp(s):  # type: ignore[misc]
            return str(s or "")

    rows: list[dict] = []
    summary: list[dict] = []
    for _, s in scores.iterrows():
        bid = str(s["brand_id"])
        hist = hist_by_brand.get(bid)
        if hist is None or hist.empty:
            continue
        hist = hist[hist["year"] <= year]
        if hist.empty:
            continue
        cur = hist.iloc[-1]
        grp = str(grp_by_key.get((bid, int(cur["year"])), cur.get("industry_major") or "전체"))
        ind = bench.get((int(cur["year"]), grp), {})
        company = str(company_by_brand.get(bid, "") or "")
        ctx = Ctx(
            brand_id=bid, brand_name=str(s.get("brand_name") or cur.get("brand_name") or bid),
            year=year, cur=cur, hist=hist, ind=ind,
            industry_label=str(ind.get("label") or grp), bench=bench,
            hq=_hq_frame(hq_all, norm_corp(company), year), hq_company=company,
            demand=(demand or {}).get(bid), news=(news or {}).get(bid))
        findings = run_rules(ctx)
        ws = watch_components(float(s.get("deterioration_rank_pct") or 0.0), findings)

        for i, f in enumerate(findings):
            d = asdict(f)
            d["evidence"] = json.dumps(f.evidence, ensure_ascii=False, default=str)
            rows.append({"brand_id": bid, "brand_name": ctx.brand_name, "year": year,
                         "rank": i, "weight": f.weight, **d})
        risks = [f for f in findings if f.direction == "risk"]
        summary.append({
            "brand_id": bid, "brand_name": ctx.brand_name, "year": year,
            "n_findings": len(findings), "n_risk": len(risks),
            "n_high": sum(1 for f in risks if f.severity == "High"),
            "categories": "·".join(dict.fromkeys(f.category for f in risks)),
            "headline": risks[0].title if risks else "특이 소견 없음",
            # 화면·CSV 는 제목이 아니라 **문장**을 보여준다. 제목만 실으면
            # "계약을 끝내는 가맹점이 많습니다"가 375개 브랜드에 똑같이 찍혀
            # 고치려던 '근거가 다 같다' 문제가 그대로 재현된다.
            "headline_detail": risks[0].detail if risks else
                               "공시 지표에서 눈에 띄는 악화 신호가 확인되지 않았습니다.",
            "top_codes": "|".join(f.code for f in risks[:3]),
            **ws})

    fdf = pd.DataFrame(rows)
    sdf = finalize_watch_scores(pd.DataFrame(summary))
    return fdf, sdf


def run(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """파이프라인 스텝 — scores_latest 를 읽어 진단 산출물을 만든다.

    트랙 분기는 하지 않는다. run_pipeline 이 확장 트랙일 때 이미 cfg['paths'] 를
    outputs/extended 로 바꿔서 넘기므로, 여기서 또 경로를 조립하면 두 벌의 규칙이
    어긋날 수 있다 (기본/확장이 서로 다른 패널을 읽는 사고).
    """
    cfg = cfg or load_config()
    out_dir = Path(cfg["paths"]["outputs"])
    proc = Path(cfg["paths"]["processed"])

    spath = out_dir / "scores_latest.csv"
    if not spath.exists():
        raise FileNotFoundError(f"{spath} 없음 — 먼저 `--step score` 실행")
    scores = pd.read_csv(spath, encoding="utf-8-sig")
    panel = pd.read_parquet(proc / "panel.parquet")

    hq_path = proc / "hq_financials.parquet"
    hq_all = pd.read_parquet(hq_path) if hq_path.exists() else None
    if hq_all is None:
        log.warning("hq_financials.parquet 없음 — 본부 재무 소견은 생성되지 않는다")

    demand = _load_demand(cfg, out_dir)
    news = _load_news(cfg, out_dir, scores)

    fdf, sdf = diagnose_cohort(cfg, scores, panel, hq_all, demand, news)

    fdf.to_parquet(out_dir / "brand_diagnosis.parquet", index=False)
    sdf.to_csv(out_dir / "brand_diagnosis_summary.csv", index=False, encoding="utf-8-sig")

    by_code = fdf["code"].value_counts().to_dict() if len(fdf) else {}
    meta = {
        "year": int(scores["year"].iloc[0]) if "year" in scores.columns else None,
        "n_brands": len(sdf),
        "n_findings": len(fdf),
        "avg_findings_per_brand": round(float(len(fdf) / max(len(sdf), 1)), 2),
        "distinct_headlines": int(sdf["headline"].nunique()) if len(sdf) else 0,
        "distinct_finding_sets": int(
            fdf.groupby("brand_id")["code"].apply(lambda s: "|".join(sorted(s))).nunique())
        if len(fdf) else 0,
        "findings_by_code": by_code,
        "demand_covered": int(sum(1 for v in (demand or {}).values() if v)),
        "news_covered": int(sum(1 for v in (news or {}).values() if v)),
        "hq_covered": int((fdf["code"] != "HQ_NO_DATA").groupby(fdf["brand_id"]).any().sum())
        if len(fdf) else 0,
        "note": ("소견은 규칙 기반이며 각 문장의 수치는 해당 브랜드 공시·감사보고서 실측값이다. "
                 "watch_score = det_component(백테스트 검증 모델 순위) + rule_component(규칙 가산)."),
    }
    (out_dir / "brand_diagnosis_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("진단 완료: 브랜드 %d개 · 소견 %d건 (브랜드당 평균 %.1f건) · "
             "서로 다른 소견조합 %d종 · 서로 다른 대표소견 %d종",
             meta["n_brands"], meta["n_findings"], meta["avg_findings_per_brand"],
             meta["distinct_finding_sets"], meta["distinct_headlines"])
    return fdf, sdf


def _load_demand(cfg: dict, out_dir: Path) -> dict:
    p = out_dir / "demand_trends.json"
    if not p.exists():
        p = Path(cfg["paths"]["outputs"]) / "demand_trends.json"
    if not p.exists():
        log.info("demand_trends.json 없음 — 검색수요 소견은 생성되지 않는다 "
                 "(`--step demand` 로 수집)")
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("demand_trends.json 읽기 실패: %s", exc)
        return {}
    return obj.get("brands", obj) if isinstance(obj, dict) else {}


def _load_news(cfg: dict, out_dir: Path, scores: pd.DataFrame) -> dict:
    p = Path(cfg["paths"]["outputs"]) / "news_signals.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    sigs = obj.get("signals", obj) if isinstance(obj, dict) else obj
    if not isinstance(sigs, list):
        return {}
    # 뉴스 신호는 브랜드명으로 저장되어 있다 → brand_id 로 옮긴다 (정확 일치만).
    name_to_id = dict(zip(scores["brand_name"].astype(str), scores["brand_id"].astype(str),
                          strict=False))
    out: dict[str, list[dict]] = {}
    for s in sigs:
        if not isinstance(s, dict):
            continue
        bid = name_to_id.get(str(s.get("brand", "")))
        if bid:
            out.setdefault(bid, []).append(s)
    return out


if __name__ == "__main__":
    run()
