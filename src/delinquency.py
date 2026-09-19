"""은행 연체 자료로 등급 검증 — FranSCORE 등급이 실제 가맹점주 연체를 가려내는가.

왜 필요한가
    FranSCORE 의 라벨은 '공시 지표 악화'이지 연체·부도가 아니다(docs/MODEL_USE_SPEC.md).
    실무자와 모형검증팀의 첫 질문은 "그래서 이 등급 브랜드의 가맹점주가 실제로 더
    연체하느냐"다. 은행 밖에서는 답할 자료가 없다. 그래서 은행이 자기 자료를 올리면
    **바로** 답이 나오도록 검증 절차를 코드로 고정해 둔다. 기준을 미리 정해 두어야
    결과를 보고 기준을 바꾸는 일(사후 선택)도 막힌다.

검증 설계 — 모형검증팀이 묻는 순서
    ① 서열성    등급이 나쁠수록 연체율이 높은가
                (등급별 연체율 · Clopper-Pearson 95% 구간 · 단조성)
    ② 크기      주의(FS3)와 안정(FS1)의 연체율 차이가 0보다 큰가 (브랜드 군집 부트스트랩)
    ③ 추가 정보 내부등급이 같은 차주끼리도 브랜드 위험이 높으면 더 연체하는가
                (로지스틱 회귀의 브랜드 위험 계수 · 브랜드 군집 강건 표준오차 → 오즈비 95% 구간.
                 효과 크기로 브랜드 단위 교차검증의 표본 밖 AUC 증가분을 함께 낸다)
    ④ 표본      등급마다 연체가 충분히 관측됐는가

왜 대출이 아니라 브랜드를 재표집하나
    같은 브랜드 가맹점주의 연체는 함께 움직인다. 이 저장소가 실측한 브랜드 내 상관은
    ρ_W 0.416 이다(outputs/brand_correlation.json). 대출을 서로 독립으로 보고 구한 구간은
    지나치게 좁다. 브랜드를 통째로 뽑아 다시 계산해야 정직한 구간이 나온다.

시점 정합 (point-in-time)
    대출은 **취급 당시 볼 수 있었던 등급**에 맞춘다. 등급 이력의 연도는 실적연도다(t년 실적은
    t+1년 정보공개서로, 빨라야 그해 중반에 공개된다). 그래서 취급 시점에 확실히 볼 수 있던
    것은 '취급연도 − 2년' 실적의 등급이고, 이것을 기본값으로 둔다(outputs/grade_history.csv,
    src/score.py build_grade_history). 하반기 취급이 대부분이고 공개 시점을 확인했다면
    '− 1년'을 고를 수 있다. 최신 등급을 과거 대출에 붙이면 대출 이후에 나온 정보로 과거를
    설명하는 셈이라 결과가 부풀려진다. 취급일 열이 없으면 최신 등급을 쓰되 그 한계를 표시한다.

    브랜드 매칭도 **이력 전체**(과거에 평가됐던 브랜드 포함)로 한다. 최신 목록으로만 찾으면
    그사이 사라진 브랜드 — 연체가 가장 많을 곳 — 가 빠져 결과가 좋게 나온다(생존 편향).

개인정보
    파일은 디스크에 쓰지 않고 메모리에서만 처리한다. 브랜드·연체·취급일·내부등급·금액 열만
    읽고 나머지 열은 해석하지 않는다.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import batch
from src.brand_search import normalize

GRADES = ("FS1", "FS2", "FS3")
GRADE_KR = {"FS1": "안정", "FS2": "관찰", "FS3": "주의"}
STATE_KR = {"건전": "건전", "요주의": "악화 발생", "평가불가": "평가불가"}

# 열 이름 단서 (앞에 있을수록 우선). 비교는 공백을 빼고 소문자로 한다.
DPD_HINTS = ("연체일수", "최장연체일수", "dpd", "dayspastdue")
OUTCOME_HINTS = ("연체여부", "부도여부", "부실여부", "연체", "부도", "부실", "default", "bad")
DATE_HINTS = ("취급일", "실행일", "대출일", "신규일", "취급연도", "대출연도", "실행연도",
              "origination")
INTERNAL_HINTS = ("내부등급", "신용등급", "css", "cb점수", "신용점수", "행동평점", "bss")
SYNTHETIC_COLUMN = "자료구분"
SYNTHETIC_MARK = "가상"

# 표본 충분성 기준 — 등급별 연체가 이보다 적으면 연체율 구간이 넓어 판단을 미룬다
MIN_DEFAULTS_PER_GRADE = 10
MIN_DEFAULTS_TOTAL = 30

_YES = {"1", "1.0", "y", "yes", "예", "o", "true", "t", "연체", "부도", "부실", "bad"}
_NO = {"0", "0.0", "n", "no", "아니오", "아니요", "x", "false", "f", "정상", "good", "-"}

EXCL_BRAND = "브랜드 확인 필요 (동명·유사 이름)"
EXCL_NOT_FOUND = "평가 이력에 없는 브랜드"
EXCL_OUTCOME = "연체 여부를 읽지 못함"
EXCL_DATE = "취급일을 읽지 못함"
EXCL_PIT = "취급 당시 등급 없음"
MATCH_PAST_NAME = "과거 등록명 일치"
_CONFIDENT = {batch.MATCH_EXACT, batch.MATCH_ALIAS, MATCH_PAST_NAME}


@dataclass
class Columns:
    """업로드 표에서 쓰는 열. 화면이 자동 인식값을 보여 주고 사용자가 고칠 수 있다."""
    brand: str
    outcome: str | None
    outcome_kind: str = "flag"          # "flag"(연체 여부) | "dpd"(연체일수)
    date: str | None = None
    internal: str | None = None
    amount: str | None = None


@dataclass
class Prepared:
    loans: pd.DataFrame                 # 검증에 쓰는 대출 (행마다 취급 당시 등급이 붙어 있다)
    excluded: pd.DataFrame              # 제외한 행과 사유
    pit: bool                           # 취급 시점 등급으로 맞췄는가
    lag_years: int
    synthetic: bool
    notes: list[str] = field(default_factory=list)


# ── 열 인식 ──────────────────────────────────────────────────────────────────

def _key(c) -> str:
    return str(c).strip().lower().replace(" ", "")


def _find(cols: list[str], hints: tuple[str, ...], exclude: tuple = ()) -> str | None:
    pool = [c for c in cols if c not in exclude]
    for h in hints:                                   # 정확히 같은 이름 먼저
        for c in pool:
            if _key(c) == h:
                return c
    for h in hints:                                   # 그다음 포함
        for c in pool:
            if h in _key(c):
                return c
    return None


def detect_columns(df: pd.DataFrame) -> Columns:
    cols = [str(c) for c in df.columns]
    brand = batch.detect_brand_column(df) or cols[0]
    dpd = _find(cols, DPD_HINTS, (brand,))
    flag = _find(cols, OUTCOME_HINTS, (brand, dpd))
    kind, outcome = ("flag", flag) if flag else ("dpd", dpd) if dpd else ("flag", None)
    # 연체 '여부' 열인데 값이 2 이상 숫자면 사실상 연체일수다
    if kind == "flag" and outcome is not None:
        v = pd.to_numeric(df[outcome], errors="coerce")
        if v.notna().mean() > 0.9 and v.max() > 1:
            kind = "dpd"
    date = _find(cols, DATE_HINTS, (brand, outcome))
    internal = _find(cols, INTERNAL_HINTS, (brand, outcome, date))
    amount = batch.detect_amount_column(df.drop(columns=[c for c in (brand, outcome, date, internal)
                                                         if c], errors="ignore"))
    return Columns(brand=brand, outcome=outcome, outcome_kind=kind, date=date,
                   internal=internal, amount=amount)


# ── 값 해석 ──────────────────────────────────────────────────────────────────

def parse_outcome(s: pd.Series, kind: str = "flag", dpd_threshold: int = 90) -> pd.Series:
    """연체 여부 → 1.0 / 0.0 / NaN(해석 불가).

    빈칸을 '정상'으로 읽지 않는다. 추출 과정에서 빠진 값일 수 있어, 조용히 정상으로
    세면 연체율이 낮게 나온다 — 제외하고 건수를 보여 준다.
    """
    if kind == "dpd":
        v = pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")
        return (v >= dpd_threshold).astype(float).where(v.notna())
    t = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[t.isin(_YES)] = 1.0
    out[t.isin(_NO)] = 0.0
    out[s.isna()] = np.nan
    return out


def parse_year(s: pd.Series) -> pd.Series:
    """취급일·취급연도 → 연도(정수, 해석 불가는 NaN). 엑셀 날짜 일련번호도 읽는다."""
    t = s.astype(str).str.strip()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    yr = pd.to_numeric(t.str.extract(r"^(\d{4})(?:년)?$")[0], errors="coerce")
    out = out.fillna(yr)
    ymd = pd.to_datetime(t.str.extract(r"^(\d{8})$")[0], format="%Y%m%d", errors="coerce")
    out = out.fillna(ymd.dt.year)
    serial = pd.to_numeric(t, errors="coerce")
    xl = serial.where((serial > 20000) & (serial < 80000))
    out = out.fillna((pd.Timestamp("1899-12-30") + pd.to_timedelta(xl, unit="D")).dt.year)
    rest = t.where(out.isna()).str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    dt = pd.to_datetime(rest, errors="coerce", format="mixed")
    out = out.fillna(dt.dt.year)
    return out.where((out >= 2000) & (out <= 2100))


# ── 이력 · 매칭 ──────────────────────────────────────────────────────────────

def load_history(outputs: Path) -> pd.DataFrame:
    """시점별 등급 이력. 없으면 최신 등급만으로 대신한다(이때는 시점 정합이 불가능하다)."""
    p = outputs / "grade_history.csv"
    if p.exists():
        h = pd.read_csv(p, encoding="utf-8-sig")
    else:
        h = pd.read_csv(outputs / "scores_latest.csv", encoding="utf-8-sig")
        h["basis"] = "운영 등급 — 라벨 미확정"
    h["brand_id"] = h["brand_id"].astype(str)
    h["year"] = h["year"].astype(int)
    return h


def _search_frames(hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(브랜드별 최신 이름 표, 과거 이름 전체 표)."""
    latest = (hist.sort_values("year").groupby("brand_id", as_index=False).tail(1)
              [["brand_id", "brand_name", "n_stores"]].reset_index(drop=True))
    names = hist[["brand_id", "brand_name"]].drop_duplicates().copy()
    names["norm"] = names["brand_name"].astype(str).map(normalize)
    return latest, names


def _match(q: str, ctx: batch.Context, names: pd.DataFrame,
           exact: dict[str, list[str]] | None = None) -> dict:
    # 정규화 이름이 브랜드 하나에만 정확히 맞으면 검색 없이 확정한다 — match_one 의
    # '정확 일치' 판정과 같은 결과이고, 수천 건 장부에서 검색을 브랜드마다 돌리지 않게 한다.
    hit = (exact or {}).get(normalize(q), [])
    if len(hit) == 1:
        nm = ctx.scores.loc[ctx.scores["brand_id"] == hit[0], "brand_name"]
        return {"status": batch.MATCH_EXACT, "brand_id": hit[0],
                "brand_name": str(nm.iloc[0]), "candidates": ""}
    m = batch.match_one(q, ctx)
    if m["status"] in _CONFIDENT:
        return m
    # 등록명이 바뀐 브랜드 — 은행 장부에는 옛 이름이 남아 있을 수 있다
    ids = names.loc[names["norm"] == normalize(q), "brand_id"].unique()
    if len(ids) == 1:
        nm = ctx.scores.loc[ctx.scores["brand_id"] == ids[0], "brand_name"]
        return {"status": MATCH_PAST_NAME, "brand_id": str(ids[0]),
                "brand_name": str(nm.iloc[0]) if len(nm) else q, "candidates": ""}
    return m


def prepare(df: pd.DataFrame, cols: Columns, hist: pd.DataFrame, *, lag_years: int = 2,
            dpd_threshold: int = 90, max_stale_years: int = 1,
            all_brands: pd.DataFrame | None = None, outputs: Path | None = None) -> Prepared:
    """업로드 표 → 검증용 대출표(취급 당시 등급 부착) + 제외 목록."""
    if cols.outcome is None:
        raise ValueError("연체 여부(또는 연체일수) 열을 찾지 못했습니다.")
    latest, names = _search_frames(hist)
    ctx = batch.Context(scores=latest, findings=None, summary=None, all_brands=all_brands,
                        outputs=outputs or Path("outputs"))

    raw = df.reset_index(drop=True)
    base = pd.DataFrame({"row": np.arange(len(raw)) + 2,        # 엑셀 행 번호(머리글 1행)
                         "brand_input": raw[cols.brand].astype(str).str.strip()})
    base["default"] = parse_outcome(raw[cols.outcome], cols.outcome_kind, dpd_threshold)
    pit = cols.date is not None
    base["orig_year"] = parse_year(raw[cols.date]) if pit else np.nan
    base["internal"] = (pd.to_numeric(raw[cols.internal].astype(str).str.replace(",", ""),
                                      errors="coerce") if cols.internal else np.nan)
    base["amount"] = (pd.to_numeric(raw[cols.amount].astype(str).str.replace(",", ""),
                                    errors="coerce") if cols.amount else np.nan)

    exact: dict[str, list[str]] = {}
    for bid, nm in zip(latest["brand_id"], latest["brand_name"].astype(str), strict=True):
        exact.setdefault(normalize(nm), []).append(str(bid))
    cache: dict[str, dict] = {}
    for q in base["brand_input"].unique():
        cache[q] = _match(q, ctx, names, exact)
    base["match"] = base["brand_input"].map(lambda q: cache[q]["status"])
    base["brand_id"] = base["brand_input"].map(lambda q: cache[q]["brand_id"])
    base["brand_name"] = base["brand_input"].map(lambda q: cache[q]["brand_name"])

    reason = pd.Series("", index=base.index, dtype=object)
    reason[base["default"].isna()] = EXCL_OUTCOME
    reason[(reason == "") & pit & base["orig_year"].isna()] = EXCL_DATE
    confident = base["match"].isin(_CONFIDENT)
    reason[(reason == "") & ~confident & base["brand_id"].notna()] = EXCL_BRAND
    reason[(reason == "") & base["brand_id"].isna()] = EXCL_NOT_FOUND

    ok = base[reason == ""].copy()
    max_year = int(hist["year"].max())
    ok["grade_year"] = (ok["orig_year"] - lag_years) if pit else float(max_year)
    h = (hist[["brand_id", "year", "grade", "deterioration_step", "brand_state", "n_events_at_t"]]
         .rename(columns={"year": "hist_year"}).sort_values("hist_year"))
    ok["_i"] = ok.index                       # merge_asof 는 인덱스를 새로 매긴다 — 원래 행을 들고 간다
    ok = ok.sort_values("grade_year")
    ok["grade_year"] = ok["grade_year"].astype(int)
    h["hist_year"] = h["hist_year"].astype(int)
    joined = pd.merge_asof(ok, h, left_on="grade_year", right_on="hist_year", by="brand_id",
                           direction="backward", tolerance=max_stale_years)
    missing = joined["grade"].isna()
    first_year = int(hist["year"].min())
    pit_detail: dict[int, str] = {}
    for i, gy in zip(joined.loc[missing, "_i"], joined.loc[missing, "grade_year"], strict=True):
        reason[i] = EXCL_PIT
        pit_detail[int(base.at[i, "row"])] = (
            f"{int(gy)}년 실적 등급은 모형 학습 연도라 검증에 쓰지 않습니다" if gy < first_year
            else f"{int(gy)}년 실적 기준으로는 평가 대상이 아니었습니다")
    loans = (joined[~missing].drop(columns="_i").rename(columns={"deterioration_step": "risk"})
             .sort_values("row").reset_index(drop=True))
    loans["default"] = loans["default"].astype(int)

    excl = base[reason != ""].copy()
    excl["reason"] = reason[reason != ""]
    excl["detail"] = excl["row"].map(pit_detail).fillna("")
    for i in excl.index[excl["reason"].isin([EXCL_BRAND, EXCL_NOT_FOUND])]:
        m = cache.get(excl.at[i, "brand_input"], {})
        if m.get("status") == batch.MATCH_SAME_NAME:
            excl.at[i, "detail"] = ("같은 이름으로 등록된 브랜드가 여럿입니다 — 가맹본부(법인명)로 "
                                    "구분해 이름을 바꿔 다시 올려 주십시오.")
        elif m.get("status") == batch.MATCH_FUZZY:
            excl.at[i, "detail"] = f"가장 가까운 이름: {m.get('brand_name')}" + (
                f" · 다른 후보: {m['candidates']}" if m.get("candidates") else "")
        elif m.get("status") == batch.MATCH_NOT_SCORED:
            excl.at[i, "detail"] = "공시에는 있으나 평가된 적이 없는 브랜드입니다"
        elif m.get("candidates"):
            excl.at[i, "detail"] = f"비슷한 이름: {m['candidates']}"
    excl = excl[["row", "brand_input", "reason", "detail"]].sort_values("row").reset_index(drop=True)

    synthetic = (SYNTHETIC_COLUMN in raw.columns and
                 raw[SYNTHETIC_COLUMN].astype(str).str.contains(SYNTHETIC_MARK).any())
    notes = []
    if not pit:
        notes.append(f"취급일 열이 없어 모든 대출에 최신({max_year}년 실적) 등급을 붙였습니다. "
                     "대출 이후에 나온 정보로 과거를 설명하게 되어 결과가 실제보다 좋게 나올 수 "
                     "있습니다 — 취급일 열을 넣어 다시 검증하십시오.")
    return Prepared(loans=loans, excluded=excl, pit=pit, lag_years=lag_years,
                    synthetic=bool(synthetic), notes=notes)


# ── 통계 ─────────────────────────────────────────────────────────────────────

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    if n == 0:
        return (np.nan, np.nan)
    lo = float(beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC (동점은 평균 순위 — 같은 브랜드 대출은 점수가 같다)."""
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _cluster_index(codes: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(codes, kind="stable")
    cuts = np.flatnonzero(np.diff(codes[order])) + 1
    return np.split(order, cuts)


def _fold_auc(y: np.ndarray, internal: np.ndarray, risk: np.ndarray,
              groups: np.ndarray, folds: int = 5) -> tuple[float, float]:
    """내부등급만 / 내부등급+브랜드 위험 — 브랜드 단위 교차검증의 **폴드별** 표본 밖 AUC 평균.

    같은 브랜드 대출이 학습과 평가에 함께 들어가면 브랜드 효과를 '외워서' 맞힌다 —
    그래서 브랜드를 묶어 나눈다. 그리고 AUC 는 폴드 안에서 재고 평균한다. 폴드마다
    절편이 다른 예측을 한데 모아 AUC 를 재면 효과가 없는데도 증가분이 양수로 치우친다
    (귀무 모의실험 40회: 모아서 잰 증가분 평균 +0.0012, 72%가 양수 — 실측).
    """
    import warnings

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    z = (internal - internal.mean()) / (internal.std() or 1.0)
    xb = z.reshape(-1, 1)
    xa = np.column_stack([z, _logit(risk)])
    ab, aa = [], []
    k = int(min(folds, len(np.unique(groups))))
    # macOS Accelerate BLAS 는 정상 입력에서도 matmul 부동소수 경고를 쏟는다(numpy 2.x 알려진
    # 현상 — 입력은 표준화값과 ±9.2 안의 로짓이라 실제 넘침이 없다).
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for tr, te in GroupKFold(n_splits=k).split(xb, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            ab.append(auc(y[te], LogisticRegression().fit(xb[tr], y[tr]).predict_proba(xb[te])[:, 1]))
            aa.append(auc(y[te], LogisticRegression().fit(xa[tr], y[tr]).predict_proba(xa[te])[:, 1]))
    return (float(np.mean(ab)) if ab else np.nan, float(np.mean(aa)) if aa else np.nan)


def _logistic_irls(X: np.ndarray, y: np.ndarray, iters: int = 60) -> tuple[np.ndarray, np.ndarray,
                                                                               np.ndarray]:
    """비벌점 로지스틱 회귀(뉴턴-랩슨). 반환 (계수, 예측확률, 헤시안)."""
    beta = np.zeros(X.shape[1])
    ridge = 1e-9 * np.eye(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        h = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(h + ridge, X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
    return beta, p, X.T @ (X * (p * (1 - p))[:, None])


def cluster_wald(y: np.ndarray, internal: np.ndarray, risk: np.ndarray,
                 groups: np.ndarray) -> dict:
    """연체 ~ 내부등급 + 브랜드 위험 — 브랜드 위험 계수의 **브랜드 군집 강건** 검정.

    '내부등급이 같은 차주끼리도 브랜드 위험이 높으면 더 연체하는가'를 직접 묻는 검정이다.
    표준오차는 브랜드 단위 샌드위치 추정(같은 브랜드 대출의 공통 충격을 반영)이고, 작은
    군집 수 보정 G/(G−1)을 곱한다. 두 변수는 표준화해 오즈비를 '1표준편차당'으로 읽는다.

    보정 확인(브랜드 200개 × 30건, 브랜드 효과 0인 귀무 모의실험, 명목 단측 2.5%):
    브랜드 충격 표준편차 0.3 → 오탐 3.0% · 0.8 → 4.7~6.0% · 1.5 → 1.3%. 대출을 독립으로 본
    표준오차는 4.0% · 8.0% 로 더 나쁘다. 브랜드 재표집 부트스트랩도 시험했으나(0.8 → 4.7%)
    나아지지 않아, 빠르고 해석이 쉬운 이쪽을 쓴다.
    """
    zi = (internal - internal.mean()) / (internal.std() or 1.0)
    lr = _logit(risk)
    zr = (lr - lr.mean()) / (lr.std() or 1.0)
    X = np.column_stack([np.ones(len(y)), zi, zr])
    with np.errstate(all="ignore"):
        beta, p, h = _logistic_irls(X, y.astype(float))
        u = X * (y - p)[:, None]
        g = pd.factorize(groups)[0]
        n_g = int(g.max()) + 1
        sums = np.zeros((n_g, X.shape[1]))
        np.add.at(sums, g, u)
        hinv = np.linalg.inv(h)
        v = hinv @ (sums.T @ sums) @ hinv * (n_g / max(n_g - 1, 1))
    se = float(np.sqrt(max(v[2, 2], 0.0)))
    b = float(beta[2])
    from scipy.stats import norm
    zstat = b / se if se > 0 else np.nan
    return {"beta": b, "se": se, "or": float(np.exp(b)), "or_lo": float(np.exp(b - 1.96 * se)),
            "or_hi": float(np.exp(b + 1.96 * se)),
            "p": float(2 * norm.sf(abs(zstat))) if np.isfinite(zstat) else np.nan,
            "n_clusters": n_g}


def _pct(a: list[float], q: float) -> float:
    a = np.asarray([v for v in a if np.isfinite(v)])
    return float(np.percentile(a, q)) if len(a) else np.nan


def _rate_table(frame: pd.DataFrame, key: str, order: list, label: dict | None = None,
                amount: bool = False) -> pd.DataFrame:
    rows = []
    for v in order:
        m = frame[key] == v
        n, k = int(m.sum()), int(frame.loc[m, "default"].sum())
        lo, hi = clopper_pearson(k, n)
        row = {"구분": (label or {}).get(v, v), "대출 수": n, "연체 수": k,
               "연체율(%)": round(k / n * 100, 2) if n else np.nan,
               "95% 구간 하한(%)": round(lo * 100, 2) if n else np.nan,
               "95% 구간 상한(%)": round(hi * 100, 2) if n else np.nan,
               "브랜드 수": int(frame.loc[m, "brand_id"].nunique())}
        if amount:
            a = frame.loc[m, "amount"]
            tot = float(a.sum())
            row["금액 가중 연체율(%)"] = (round(float(a[frame.loc[m, "default"] == 1].sum()) / tot * 100, 2)
                                   if tot > 0 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(prep: Prepared, *, n_boot: int = 400, seed: int = 42, folds: int = 5) -> dict:
    """검증 결과 전체. 화면·엑셀이 같은 dict 를 쓴다."""
    L = prep.loans
    if L.empty:
        raise ValueError("검증에 쓸 수 있는 대출이 없습니다 — 제외 사유를 확인하십시오.")
    y = L["default"].to_numpy(int)
    gi = L["grade"].map({g: i for i, g in enumerate(GRADES)}).to_numpy()
    risk = L["risk"].to_numpy(float)
    has_amount = bool(L["amount"].notna().mean() > 0.9)

    by_grade = _rate_table(L, "grade", list(GRADES),
                           {g: f"{g} {GRADE_KR[g]}" for g in GRADES}, amount=has_amount)
    states = [s for s in ("건전", "요주의", "평가불가") if (L["brand_state"] == s).any()]
    by_state = _rate_table(L, "brand_state", states, STATE_KR)
    years = sorted(L["grade_year"].unique())
    by_year = pd.concat([_rate_table(L[L["grade_year"] == yv], "grade", list(GRADES),
                                     {g: f"{g} {GRADE_KR[g]}" for g in GRADES})
                         .assign(**{"등급 기준 실적연도": int(yv)}) for yv in years],
                        ignore_index=True)

    rates = [(y[gi == i].mean() if (gi == i).any() else np.nan) for i in range(3)]
    monotone = (bool(rates[0] < rates[1] < rates[2]) if all(np.isfinite(rates)) else None)
    diff = rates[2] - rates[0] if np.isfinite(rates[2]) and np.isfinite(rates[0]) else np.nan
    ratio = rates[2] / rates[0] if np.isfinite(diff) and rates[0] > 0 else np.nan
    auc_brand = auc(y, risk)

    # 내부등급 — 대출의 90% 이상에 값이 있을 때만 추가 정보 검증을 한다
    inc = None
    strata = None
    has_internal = bool(L["internal"].notna().mean() >= 0.9)
    if has_internal:
        mi = L["internal"].notna().to_numpy()
        codes_i = pd.factorize(L.loc[mi, "brand_id"])[0]
        yi, ii, ri = y[mi], L.loc[mi, "internal"].to_numpy(float), risk[mi]
        base_auc, aug_auc = _fold_auc(yi, ii, ri, codes_i, folds)
        inc = {"n": int(mi.sum()), "base": base_auc, "aug": aug_auc, "delta": aug_auc - base_auc,
               **cluster_wald(yi, ii, ri, codes_i)}
        # 내부등급 방향(숫자가 클수록 나쁜가)을 연체와의 상관으로 정해 층 이름을 붙인다
        worse_high = np.corrcoef(ii, yi)[0, 1] >= 0 if yi.std() > 0 else True
        try:
            band = pd.qcut(pd.Series(ii).rank(method="first"), 3, labels=False)
        except ValueError:
            band = None
        if band is not None:
            names = (["우량 1/3", "중간 1/3", "열위 1/3"] if worse_high
                     else ["열위 1/3", "중간 1/3", "우량 1/3"])
            sub = L.loc[mi].assign(층=[names[int(b)] for b in band])
            piv = []
            for s_name in ["우량 1/3", "중간 1/3", "열위 1/3"]:
                ss = sub[sub["층"] == s_name]
                row = {"내부등급 층": s_name, "대출 수": len(ss)}
                for g in GRADES:
                    m = ss["grade"] == g
                    n, k = int(m.sum()), int(ss.loc[m, "default"].sum())
                    row[f"{g} {GRADE_KR[g]} 연체율(%)"] = round(k / n * 100, 2) if n else np.nan
                    row[f"{g} 대출 수"] = n
                piv.append(row)
            strata = pd.DataFrame(piv)

    # 브랜드 군집 부트스트랩
    rng = np.random.default_rng(seed)
    codes = pd.factorize(L["brand_id"])[0]
    clusters = _cluster_index(codes)
    nb = len(clusters)
    bd, ba = [], []
    for _ in range(n_boot):
        idx = np.concatenate([clusters[j] for j in rng.integers(0, nb, nb)])
        yy, gg = y[idx], gi[idx]
        r0 = yy[gg == 0].mean() if (gg == 0).any() else np.nan
        r2 = yy[gg == 2].mean() if (gg == 2).any() else np.nan
        bd.append(r2 - r0)
        ba.append(auc(yy, risk[idx]))

    rates = [float(r) for r in rates]
    rank = {"rates": rates, "monotone": monotone, "diff": float(diff), "ratio": float(ratio),
            "diff_lo": _pct(bd, 2.5), "diff_hi": _pct(bd, 97.5)}
    auc_res = {"est": auc_brand, "lo": _pct(ba, 2.5), "hi": _pct(ba, 97.5)}

    k_by = by_grade.set_index("구분")["연체 수"]
    sample_ok = bool(k_by.min() >= MIN_DEFAULTS_PER_GRADE and y.sum() >= MIN_DEFAULTS_TOTAL)
    verdict = _verdict(rank, inc, sample_ok, int(k_by.min()), int(y.sum()))

    brands = (L.groupby(["brand_id", "brand_name"], as_index=False)
              .agg(대출_수=("default", "size"), 연체_수=("default", "sum"),
                   최근_등급=("grade", "last"))
              .assign(연체율=lambda d: (d["연체_수"] / d["대출_수"] * 100).round(2))
              .sort_values(["연체_수", "대출_수"], ascending=False)
              .rename(columns={"brand_name": "브랜드", "대출_수": "대출 수", "연체_수": "연체 수",
                               "최근_등급": "취급 당시 등급(최근)", "연체율": "연체율(%)"})
              .drop(columns="brand_id").reset_index(drop=True))

    return {"n": len(L), "defaults": int(y.sum()), "rate": float(y.mean()),
            "n_brands": int(L["brand_id"].nunique()), "n_excluded": len(prep.excluded),
            "years": [int(v) for v in years], "pit": prep.pit, "lag_years": prep.lag_years,
            "synthetic": prep.synthetic, "n_boot": n_boot, "has_amount": has_amount,
            "by_grade": by_grade, "by_state": by_state, "by_year": by_year, "strata": strata,
            "brands": brands, "rank": rank, "auc": auc_res, "incremental": inc,
            "verdict": verdict, "notes": list(prep.notes)}


def _verdict(rank: dict, inc: dict | None, sample_ok: bool, min_k: int, total_k: int) -> dict:
    """판정 — 기준은 코드에 고정한다(결과를 보고 바꾸지 않는다)."""
    lo = rank["diff_lo"]
    if rank["monotone"] is None or not np.isfinite(lo):
        r = ("판단 불가", "등급 중 대출이 없는 구간이 있어 서열을 비교할 수 없습니다.")
    elif rank["monotone"] and lo > 0:
        r = ("확인", "안정 → 관찰 → 주의 순으로 연체율이 높아지고, 주의−안정 차이의 95% 구간이 "
                    "0보다 큽니다.")
    elif lo > 0:
        r = ("부분 확인", "주의 등급 연체율은 안정보다 유의하게 높지만 관찰 구간에서 순서가 "
                        "뒤집힙니다.")
    elif rank["monotone"]:
        r = ("방향만 일치", "순서는 맞지만 주의−안정 차이의 95% 구간이 0을 포함합니다 — "
                          "표본을 늘려 다시 보십시오.")
    else:
        r = ("확인 안 됨", "등급 순서대로 연체율이 높아지지 않습니다.")

    if inc is None:
        i = ("평가 생략", "내부등급(또는 CB점수) 열이 없어 기존 등급 대비 추가 정보는 보지 "
                         "않았습니다.")
    elif np.isfinite(inc["or_lo"]) and inc["or_lo"] > 1:
        i = ("확인", f"내부등급이 같아도 브랜드 위험이 1표준편차 높으면 연체 오즈가 {inc['or']:.2f}배"
                    f"(95% 구간 {inc['or_lo']:.2f}~{inc['or_hi']:.2f})입니다. 표본 밖 AUC "
                    f"{inc['base']:.3f} → {inc['aug']:.3f}.")
    elif inc["or"] > 1:
        i = ("방향만 일치", f"브랜드 위험 오즈비 {inc['or']:.2f}배지만 95% 구간"
                          f"({inc['or_lo']:.2f}~{inc['or_hi']:.2f})이 1을 포함합니다.")
    else:
        i = ("확인 안 됨", "내부등급을 통제하면 브랜드 위험과 연체의 양(+)의 관계가 보이지 않습니다.")

    s = (("충분", f"등급별 연체가 최소 {min_k}건, 전체 {total_k}건 관측됐습니다.") if sample_ok else
         ("부족", f"등급별 연체가 최소 {min_k}건, 전체 {total_k}건입니다 — 등급별 "
                  f"{MIN_DEFAULTS_PER_GRADE}건·전체 {MIN_DEFAULTS_TOTAL}건 이상일 때 판정을 "
                  "신뢰할 수 있습니다."))

    if not sample_ok:
        overall = "표본이 부족해 판정을 보류합니다. 관찰 기간이나 대상 상품을 넓혀 다시 검증하십시오."
    elif r[0] == "확인" and i[0] == "확인":
        overall = ("등급이 연체를 서열화하고 내부등급에 없는 정보를 더합니다 — 심사 보조지표 "
                   "채택을 검토할 근거가 됩니다(모형검증 절차 필요).")
    elif r[0] in ("확인", "부분 확인") and i[0] != "확인":
        overall = ("등급이 연체를 서열화하지만 내부등급과 겹치는 정보이거나 추가분이 "
                   "불확실합니다 — 사후관리 우선순위·편중 관리 참고용으로 적합합니다.")
    else:
        overall = "이 자료에서는 등급의 연체 설명력이 확인되지 않았습니다 — 참고 지표로만 두십시오."
    return {"rank": r, "incremental": i, "sample": s, "overall": overall}


# ── 반출 · 양식 · 시연 자료 ──────────────────────────────────────────────────

def to_excel(res: dict, prep: Prepared, meta: dict) -> bytes:
    """요약 · 등급별 · 상태별 · 취급연도별 · 내부등급 층별 · 브랜드별 · 제외 목록 · 안내."""
    v = res["verdict"]
    summary = pd.DataFrame([
        ("자료 구분", "가상 시연 자료 — 실제 성과가 아님" if res["synthetic"] else "업로드 자료"),
        ("검증 대출 수", res["n"]), ("연체 수", res["defaults"]),
        ("전체 연체율(%)", round(res["rate"] * 100, 2)), ("브랜드 수", res["n_brands"]),
        ("제외 행 수", res["n_excluded"]),
        ("등급 기준", f"취급연도 − {res['lag_years']}년 실적 등급" if res["pit"] else "최신 등급(시점 정합 아님)"),
        ("서열성", f"{v['rank'][0]} — {v['rank'][1]}"),
        ("추가 정보", f"{v['incremental'][0]} — {v['incremental'][1]}"),
        ("표본", f"{v['sample'][0]} — {v['sample'][1]}"),
        ("종합", v["overall"]),
        ("브랜드 위험 AUC", f"{res['auc']['est']:.3f} [{res['auc']['lo']:.3f}, {res['auc']['hi']:.3f}]"),
        *([("브랜드 위험 오즈비(1표준편차, 내부등급 통제)",
            f"{res['incremental']['or']:.2f} [{res['incremental']['or_lo']:.2f}, "
            f"{res['incremental']['or_hi']:.2f}] · p={res['incremental']['p']:.4f}"),
           ("표본 밖 AUC (내부등급 → +브랜드 위험)",
            f"{res['incremental']['base']:.3f} → {res['incremental']['aug']:.3f}")]
          if res["incremental"] else []),
    ], columns=["항목", "값"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="요약")
        w.sheets["요약"].column_dimensions["A"].width = 20
        w.sheets["요약"].column_dimensions["B"].width = 110
        res["by_grade"].to_excel(w, index=False, sheet_name="등급별")
        res["by_state"].to_excel(w, index=False, sheet_name="브랜드 상태별")
        res["by_year"].to_excel(w, index=False, sheet_name="실적연도별")
        if res["strata"] is not None:
            res["strata"].to_excel(w, index=False, sheet_name="내부등급 층별")
        res["brands"].to_excel(w, index=False, sheet_name="브랜드별")
        prep.excluded.rename(columns={"row": "원본 행", "brand_input": "입력 브랜드",
                                      "reason": "제외 사유", "detail": "상세"}).to_excel(
            w, index=False, sheet_name="제외 목록")
        notes = [
            f"기준: 공정거래위원회 가맹사업 공시 · 등급 이력 {', '.join(map(str, res['years']))}년 실적 · "
            f"산출 {meta.get('generated', '-')}",
            "FranSCORE 등급은 브랜드의 공시 지표 악화 확률로 매긴 것이며 차주의 부도확률(PD)이 아닙니다.",
            "구간은 브랜드를 통째로 재표집한 군집 부트스트랩 "
            f"{res['n_boot']}회 결과입니다(같은 브랜드 가맹점주 연체는 함께 움직이므로).",
            "추가 정보 판정은 '연체 ~ 내부등급 + 브랜드 위험' 로지스틱 회귀의 브랜드 위험 계수를 "
            "브랜드 군집 강건 표준오차로 검정한 것입니다. AUC 증가분은 브랜드 단위 5겹 교차검증의 "
            "폴드별 표본 밖 AUC 평균입니다.",
            "업로드 자료는 서버에 저장하지 않았습니다. 이 파일에는 원본 행 번호와 브랜드명만 남습니다.",
            *res["notes"],
        ]
        pd.DataFrame({"안내": notes}).to_excel(w, index=False, sheet_name="안내")
        w.sheets["안내"].column_dimensions["A"].width = 120
    return buf.getvalue()


def template_bytes(names: list[str] | None = None) -> bytes:
    """업로드 양식 — 필수 열(브랜드·연체 여부)과 권장 열(취급일·내부등급·금액).

    names 를 주면 예시 브랜드를 그 이름으로 쓴다 — 공개 배포는 가명(src/public.py).
    """
    ex = [*(names or []), "메가커피", "교촌치킨", "빽다방"][:3]
    sample = pd.DataFrame({
        "대출번호": ["L-0001", "L-0002", "L-0003"],
        "브랜드명": ex,
        "취급일": ["2023-03-15", "2023-07-02", "2024-01-20"],
        "내부등급": [4, 7, 5],
        "대출금액(백만원)": [80, 120, 60],
        "연체여부": ["N", "Y", "N"],
    })
    guide = pd.DataFrame({"열": ["브랜드명 (필수)", "연체여부 (필수)", "취급일 (권장)",
                                "내부등급 (권장)", "대출금액 (선택)"],
                          "설명": ["공시 등록명이나 통칭. 이름이 정확히 같지 않으면 검증에서 빼고 "
                                 "목록으로 보여 줍니다.",
                                 "Y/N, 1/0, 예/아니오. 대신 '연체일수' 열을 주면 90일 이상을 연체로 봅니다. "
                                 "관찰 기간(예: 취급 후 12개월)은 은행 기준으로 통일해 주십시오.",
                                 "날짜 또는 연도. 취급 당시 볼 수 있던 공시 등급에 맞추는 데 씁니다 — 없으면 최신 "
                                 "등급을 쓰고 결과가 낙관적일 수 있다고 표시합니다.",
                                 "은행 내부 신용등급이나 CB점수(숫자). 있으면 '내부등급만 쓴 모형 대비 추가 "
                                 "변별력'을 검증합니다.",
                                 "있으면 금액 가중 연체율을 함께 냅니다."]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        sample.to_excel(w, index=False, sheet_name="대출목록")
        for letter, width in zip("ABCDEF", (12, 18, 14, 10, 16, 10), strict=True):
            w.sheets["대출목록"].column_dimensions[letter].width = width
        guide.to_excel(w, index=False, sheet_name="작성 안내")
        w.sheets["작성 안내"].column_dimensions["A"].width = 18
        w.sheets["작성 안내"].column_dimensions["B"].width = 110
    return buf.getvalue()


def sample_frame(hist: pd.DataFrame, n_loans: int = 6000, seed: int = 7,
                 lag_years: int = 2) -> pd.DataFrame:
    """**가상** 시연 자료 — 화면이 어떤 결과를 내는지 보여 주기 위한 것이다.

    ⚠️ 이 자료는 연체 확률에 브랜드 위험을 **일부러 심어** 만든다(내부등급 효과 + 브랜드 효과 +
       브랜드·연도 공통 충격). 그러니 이 자료로 나온 '확인' 판정은 FranSCORE 성능의 근거가
       아니다 — 절차가 어떻게 판정하는지 보여 주는 시연일 뿐이다. 모든 행에 '자료구분=가상'을
       달아, 파일을 내려받아 다시 올려도 화면이 가상 자료임을 표시하게 한다.
    """
    rng = np.random.default_rng(seed)
    # 최신 실적연도 등급으로 취급한 대출은 아직 관찰 기간(12개월)이 끝나지 않았다 — 뺀다
    years = sorted(int(v) for v in hist["year"].unique())
    years = years[:-1] if len(years) > 1 else years
    z_all = _logit(hist["deterioration_step"].to_numpy(float))
    mu, sd = float(z_all.mean()), float(z_all.std() or 1.0)
    per = [n_loans // len(years)] * len(years)
    per[-1] += n_loans - sum(per)
    rows = []
    for gy, n in zip(years, per, strict=True):
        h = hist[hist["year"] == gy].reset_index(drop=True)
        w = np.sqrt(pd.to_numeric(h["n_stores"], errors="coerce").fillna(1).clip(lower=1).to_numpy())
        pick = rng.choice(len(h), size=n, p=w / w.sum())
        z = (_logit(h["deterioration_step"].to_numpy(float)) - mu) / sd
        shock = rng.normal(0.0, 0.5, size=len(h))          # 브랜드·연도 공통 충격 (군집)
        zi = z[pick]
        internal = np.clip(np.rint(rng.normal(5 + 0.6 * zi, 2.0)), 1, 10).astype(int)
        lp = -3.75 + 0.32 * (internal - 5) + 0.45 * zi + shock[pick]
        default = rng.random(n) < 1 / (1 + np.exp(-lp))
        oy = gy + lag_years
        month = rng.integers(1, 13, size=n)
        day = rng.integers(1, 29, size=n)
        amount = np.round(rng.lognormal(np.log(70), 0.5, size=n)).astype(int)
        for j in range(n):
            rows.append({"브랜드명": str(h.at[int(pick[j]), "brand_name"]),
                         "취급일": f"{oy}-{int(month[j]):02d}-{int(day[j]):02d}",
                         "내부등급(1=최우량)": int(internal[j]),
                         "대출금액(백만원)": int(amount[j]),
                         "연체여부(취급 후 12개월 내 90일+)": "Y" if default[j] else "N"})
    out = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.insert(0, "대출번호", [f"S-{i:05d}" for i in range(1, len(out) + 1)])
    out[SYNTHETIC_COLUMN] = "가상 데이터 — 시연용 (실제 대출 아님)"
    return out


def sample_bytes(hist: pd.DataFrame, n_loans: int = 6000, seed: int = 7) -> bytes:
    df = sample_frame(hist, n_loans, seed)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="가상대출")
        pd.DataFrame({"안내": [
            "이 파일은 화면 시연을 위한 가상 자료입니다. 실제 대출·연체 기록이 아닙니다.",
            "연체 확률에 브랜드 위험을 일부러 반영해 만들었으므로, 이 자료로 나온 판정은 "
            "FranSCORE 성능의 근거가 아닙니다.",
        ]}).to_excel(w, index=False, sheet_name="안내")
    return buf.getvalue()
