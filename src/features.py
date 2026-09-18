"""M2 피처 생성 (INTERFACES.md §2).

행 (brand_id, t)의 피처는 **t 이하 연도 값만** 사용한다 (⛔ t+1 절대 금지).
- 브랜드 자기 이력: 그룹 내 shift/rolling (과거 방향만).
- 횡단면 통계(업종 중앙값·rank pct·윈저라이즈 경계)는 **해당 연도(t) 행만** 사용
  — 연도별 계산이므로 t+1 값이 바뀌어도 (brand, t) 행은 불변 (누출 테스트 대상).

피처 그룹 (계약 §2):
- f_lvl_*    수준: n_stores, avg_sales(log1p), direct_ratio, brand_age(관측연차)
- f_chg_*    변화율: store_growth_rate, sales_growth, contract_end_rate,
             신규개점률(n_new/n_stores_{t-1}), 명의변경률(n_name_change/n_stores_{t-1})
- f_trd_*    추세: 최근 2~3년 store_growth·sales_growth 평균/기울기/표준편차
             (rolling window=3, 유효 관측 <2면 NaN 허용)
- f_ind_*    업종 상대위치: store_growth·real_sales_growth·contract_end_rate의
             업종그룹×연도 내 분위수(rank pct) + 업종 대분류 더미
- f_struct_* 라벨 차원 밖: direct_ratio 변화, 직영점수 변화율, 신규개점-종료 격차

구현상 해석 (계약 모호점 — 명시):
1. 윈저라이즈는 **연도별 횡단면** 분위수로 클립 (t+1 분포 미사용 — 시점누출 방지).
   더미(f_ind_major_*)를 제외한 모든 f_* 연속 피처에 적용.
2. 직영점수 변화율 분모는 (n_direct_{t-1} + 1) — 직영 0점포 브랜드의 0나눗셈 방지
   (스무딩 가정, 로그에 기록).
3. 신규개점-종료 격차는 (n_new − 종료 − 해지) / n_stores_{t-1} 로 정규화.
4. 파생 지표 정의는 labels.compute_derived_metrics 공용 (단일 정의 원칙).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import get_logger, load_config, make_synthetic_panel, set_seed
from src.dart import norm_corp
from src.labels import compute_derived_metrics

log = get_logger("features")

_TRD_WINDOW = 3   # 최근 2~3년 추세 창
_TRD_MIN_OBS = 2  # 유효 관측 최소 개수 (미만이면 NaN)


def _slope(a: np.ndarray) -> float:
    """rolling 창 내 선형추세 기울기 (NaN 제외, 유효 관측 <2면 NaN)."""
    m = ~np.isnan(a)
    if m.sum() < 2:
        return np.nan
    x = np.arange(len(a), dtype=float)[m]
    return float(np.polyfit(x, a[m], 1)[0])


# 가맹본부 재무에서 파생 피처를 만들 때 쓰는 원천 컬럼 (누출 테스트의 교란 대상 목록).
# ⚠️ src/dart.py 가 컬럼을 추가하면 여기와 tests/test_sanity.py 의 교란표에 함께 추가할 것.
HQ_VALUE_COLS = ("assets", "liabilities", "equity", "capital_stock", "revenue",
                 "operating_income", "net_income", "going_concern_flag")

# 본부 재무의 허용 노후도(년). 최근 결산이 이보다 오래되면 붙이지 않는다 —
# 5년 전 재무제표를 '현재 본부 상태'로 쓰면 오히려 잘못된 안심/경보를 준다.
_HQ_MAX_STALENESS = 3


def _load_hq_financials(cfg: dict) -> pd.DataFrame | None:
    p = Path(cfg["paths"]["processed"]) / "hq_financials.parquet"
    if not p.exists():
        log.info("hq_financials.parquet 없음 — 본부 재무 피처(f_hq_*)를 생성하지 않는다 "
                 "(`python -m src.dart` 로 수집).")
        return None
    return pd.read_parquet(p)


def _attach_hq_features(feat: pd.DataFrame, df: pd.DataFrame,
                        hq_fin: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    """패널 (brand, t) 에 **회계연도 ≤ t-1** 의 가장 최근 본부 재무를 붙여 f_hq_* 를 만든다.

    ⛔ 시점 안전 — 왜 t-1 인가:
        회계연도 t 의 감사보고서는 t+1 년 3~4월에 공시되고, 공정위 브랜드 통계 t년치도
        t+1 년에 공개된다. 따라서 회계연도 t 를 써도 형식상 누출은 아니다. 그럼에도
        **t-1 로 한 해 더 물린다** — 두 공시의 시차가 해마다 흔들릴 수 있고, 여신 모형에서
        '아슬아슬하게 안 새는 설계'는 나중에 반드시 문제가 되기 때문이다. 한 해를 버리는
        대신 누출 가능성을 구조적으로 0 으로 만든다.
    """
    if hq_fin is None or hq_fin.empty or "company_name" not in df.columns:
        return feat

    fin = hq_fin.copy()
    for c in HQ_VALUE_COLS:
        if c not in fin.columns:
            fin[c] = np.nan
        fin[c] = pd.to_numeric(fin[c], errors="coerce")
    fin["fiscal_year"] = pd.to_numeric(fin["fiscal_year"], errors="coerce")
    fin = fin.dropna(subset=["key", "fiscal_year"]).sort_values(["key", "fiscal_year"])

    # 재무 자체의 시계열 파생 (같은 법인 안에서 연속 회계연도일 때만)
    g = fin.groupby("key", sort=False)
    prev_rev = g["revenue"].shift(1)
    prev_yr = g["fiscal_year"].shift(1)
    consec = (fin["fiscal_year"] - prev_yr).eq(1)
    fin["rev_growth"] = ((fin["revenue"] - prev_rev) / prev_rev.abs()).where(
        consec & prev_rev.abs().gt(0))
    # 연속 적자 연수 — 흑자를 만나면 0 으로 리셋되는 누적 카운트.
    # ⚠️ 두 가지를 반드시 끊어줘야 한다(적대적 리뷰 지적):
    #   ① 회계연도 갭 — 결측 연도를 건너뛰고 이어 세면 '가짜 연속 적자'가 만들어진다.
    #   ② 순이익 결측 — '모른다'를 '흑자'로 취급해 연속 카운트를 리셋하면, 확인하지 못한
    #      것을 확인해서 괜찮았다고 말하는 셈이다. 결측 행의 streak 는 결측으로 둔다.
    loss = fin["net_income"].lt(0)
    known = fin["net_income"].notna()
    reset = (~consec) | (~known) | (~loss)
    blocks = reset.groupby(fin["key"]).cumsum()
    fin["loss_streak"] = (loss.astype(float)
                          .groupby([fin["key"], blocks]).cumsum()
                          .where(known))

    left = df[["brand_id", "year", "company_name"]].copy()
    left["key"] = left["company_name"].map(norm_corp)
    left["_asof"] = pd.to_numeric(left["year"], errors="coerce") - 1
    left = left.dropna(subset=["_asof"]).sort_values("_asof")

    m = pd.merge_asof(left, fin.rename(columns={"fiscal_year": "_fy"}).sort_values("_fy"),
                      left_on="_asof", right_on="_fy", by="key", direction="backward")
    max_stale = float((cfg.get("dart") or {}).get("max_staleness_years",
                                                  _HQ_MAX_STALENESS))
    m["_age"] = m["_asof"] - m["_fy"]
    stale = m["_age"] > max_stale
    m.loc[stale, ["_fy", "_age", *HQ_VALUE_COLS, "rev_growth", "loss_streak"]] = np.nan

    def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        b = b.where(b.abs() > 0)
        return a / b

    out = pd.DataFrame(index=m.index)
    # 자기자본비율은 부채비율보다 안정적이다 — 자본이 음수(완전잠식)여도 발산하지 않고
    # 음수로 이어져 그대로 '더 나쁨'을 뜻한다. 부채비율은 자본→0 에서 무한대로 튄다.
    out["f_hq_equity_ratio"] = safe_div(m["equity"], m["assets"])
    out["f_hq_liab_ratio"] = safe_div(m["liabilities"], m["assets"])
    out["f_hq_op_margin"] = safe_div(m["operating_income"], m["revenue"])
    out["f_hq_net_margin"] = safe_div(m["net_income"], m["revenue"])
    out["f_hq_roa"] = safe_div(m["net_income"], m["assets"])
    out["f_hq_revenue_log"] = np.log1p(m["revenue"].clip(lower=0))
    out["f_hq_revenue_growth"] = m["rev_growth"]
    out["f_hq_loss_streak"] = m["loss_streak"]
    out["f_hq_op_loss"] = m["operating_income"].lt(0).astype(float).where(
        m["operating_income"].notna())
    # 자본잠식: 자본총계 < 자본금(부분) / 자본총계 ≤ 0(완전). 주석의 '자본잠식' 문자열이
    # 아니라 **숫자로** 판정한다 (문자열은 피투자회사 잠식까지 잡아 오탐이 난다).
    out["f_hq_capital_impaired"] = (
        m["equity"].lt(m["capital_stock"]).astype(float)
        .where(m["equity"].notna() & m["capital_stock"].notna()))
    out["f_hq_equity_negative"] = m["equity"].le(0).astype(float).where(m["equity"].notna())
    out["f_hq_going_concern"] = m["going_concern_flag"]
    # 재무 확보 여부 자체가 신호다(외부감사 대상 = 일정 규모 이상). 결측을 0 으로 덮지 않고
    # 별도 지표로 분리해, 모형이 '무정보'와 '나쁜 값'을 구분할 수 있게 한다.
    out["f_hq_has_financials"] = m["_fy"].notna().astype(float)
    out["f_hq_data_age"] = m["_age"]

    m = pd.concat([m[["brand_id", "year"]], out], axis=1)
    # (brand_id, year) 가 중복이면 merge 가 행을 조용히 불려 피처 행수가 패널과 어긋난다.
    # 그렇게 되면 이후 라벨 조인까지 오염되므로 여기서 즉시 실패시킨다.
    dup = m.duplicated(["brand_id", "year"]).sum()
    if dup:
        raise ValueError(f"본부 재무 결합 키 (brand_id, year) 중복 {dup}건 — 패널 무결성 확인 필요")
    n_before = len(feat)
    res = feat.merge(m, on=["brand_id", "year"], how="left")
    if len(res) != n_before:
        raise ValueError(f"본부 재무 결합으로 행수 변화 {n_before} → {len(res)}")
    n_cov = int(res["f_hq_has_financials"].fillna(0).sum())
    log.info("본부 재무 피처: %d/%d 행에 결합 (%.1f%%), 자본잠식 %d행 / 계속기업의문 %d행",
             n_cov, len(res), 100.0 * n_cov / max(len(res), 1),
             int(res["f_hq_capital_impaired"].fillna(0).sum()),
             int(res["f_hq_going_concern"].fillna(0).sum()))
    return res


def build_features(panel: pd.DataFrame, cfg: dict,
                   hq_fin: pd.DataFrame | None = None) -> pd.DataFrame:
    """계약 §2: [brand_id, year] + f_* 컬럼 반환. (brand, t) 행은 t 이하 연도만 사용.

    hq_fin: 가맹본부 재무 (src/dart.py 산출). None 이면 processed 에서 읽는다.
            테스트는 합성 재무를 직접 넘겨 누출 검증 범위에 포함시킨다.
    """
    if hq_fin is None:
        hq_fin = _load_hq_financials(cfg)
    df = compute_derived_metrics(panel)
    gb = df.groupby("brand_id", sort=False)
    feat = df[["brand_id", "year"]].copy()

    # --- f_lvl_ 수준 -------------------------------------------------------
    feat["f_lvl_n_stores"] = df["n_stores"]
    feat["f_lvl_avg_sales_log"] = np.log1p(df["avg_sales"])
    feat["f_lvl_direct_ratio"] = df["direct_ratio"]
    feat["f_lvl_brand_age"] = (gb.cumcount() + 1).astype(float)  # 관측연차
    # 명세 3.1 "면적당매출" 수준 (3.3㎡당 평균매출액)
    feat["f_lvl_sales_per_area_log"] = np.log1p(df["avg_sales_per_area"])
    # 명세 3.1 업력 — 가맹사업개시연도 기준 실제 업력 (관측연차와 별개 신호).
    # 하한 0 클립: 패널 연도(실적연도)가 공시연도−1이라, 공시 등록 연도에 개시한 브랜드는
    # 산술상 -1이 되는데 이는 '업력 0년(개시 원년)'을 뜻한다 (리뷰 지적 반영).
    if "biz_start_year" in df.columns:
        feat["f_lvl_biz_age"] = (df["year"] - df["biz_start_year"]).astype(float).clip(lower=0.0)
    # 본부 규모 (라벨 차원 밖)
    if "emp_cnt" in df.columns:
        feat["f_lvl_emp_cnt_log"] = np.log1p(pd.to_numeric(df["emp_cnt"], errors="coerce"))

    # --- f_chg_ 변화율 -----------------------------------------------------
    feat["f_chg_store_growth"] = df["store_growth_rate"]
    feat["f_chg_sales_growth"] = df["sales_growth"]
    feat["f_chg_contract_end_rate"] = df["contract_end_rate"]
    feat["f_chg_new_open_rate"] = df["n_new"] / df["_prev_stores"]
    feat["f_chg_name_change_rate"] = df["n_name_change"] / df["_prev_stores"]
    # 명세 3.1 면적당매출 전년比 변화율
    feat["f_chg_sales_per_area_growth"] = df["sales_per_area_growth"]

    # --- f_trd_ 추세 (브랜드 자기 이력, 과거 방향 rolling만) ---------------
    # 연속 관측 구간 ID: 브랜드 경계 또는 연도 갭(비연속)에서 새 구간 시작.
    # (df는 compute_derived_metrics에서 (brand_id, year) 정렬 보장)
    _year_gap = gb["year"].diff()
    _run_id = (_year_gap.ne(1) | _year_gap.isna()).cumsum()
    for metric, tag in [("store_growth_rate", "store_growth"),
                        ("sales_growth", "sales_growth"),
                        ("sales_per_area_growth", "sales_per_area")]:
        # 롤링 창은 **연속 관측 구간(run) 안에서만** 계산 (리뷰 지적 반영: 행 위치 기반
        # 롤링은 관측 공백을 건너뛰어 4년+ 전 값을 '최근 추세'에 섞는다 — 실측 3.9% 행)
        s = df.groupby(_run_id, sort=False)[metric]
        feat[f"f_trd_{tag}_mean"] = s.transform(
            lambda x: x.rolling(_TRD_WINDOW, min_periods=_TRD_MIN_OBS).mean())
        feat[f"f_trd_{tag}_std"] = s.transform(
            lambda x: x.rolling(_TRD_WINDOW, min_periods=_TRD_MIN_OBS).std())
        feat[f"f_trd_{tag}_slope"] = s.transform(
            lambda x: x.rolling(_TRD_WINDOW, min_periods=1).apply(_slope, raw=True))

    # --- f_ind_ 업종 상대위치 (업종그룹×연도 내 rank pct — 해당 연도 행만) --
    grp = df.groupby(["industry_group", "year"])
    feat["f_ind_store_growth_pct"] = grp["store_growth_rate"].rank(pct=True)
    feat["f_ind_real_sales_growth_pct"] = grp["real_sales_growth"].rank(pct=True)
    feat["f_ind_contract_end_pct"] = grp["contract_end_rate"].rank(pct=True)
    # 업종 더미는 **범주가 2개 이상일 때만** 만든다. 단일 범주 트랙(기본=외식 전용)에서
    # 더미는 전 행 1.0인 상수라 정보가 전혀 없고 '죽은 피처'로 오해된다(리뷰 지적).
    # ⚠️ "분산 0이면 사후에 버린다"는 데이터 의존 필터로 처리하지 않는 이유:
    #    그 방식은 피처 **스키마**가 t+1 이후 값에까지 의존하게 만들어, "과거 행은 미래
    #    교란에 불변"이라는 시점누출 원칙을 스키마 차원에서 위반한다(테스트로 실제 검출됨).
    #    반면 업종 범주 개수는 브랜드의 정적 속성이라 미래 값과 무관하다.
    n_major = int(df["industry_major"].nunique(dropna=True))
    if n_major >= 2:
        dummies = pd.get_dummies(df["industry_major"], prefix="f_ind_major", dtype=float)
        feat = pd.concat([feat, dummies], axis=1)
    else:
        only = df["industry_major"].dropna().unique()
        log.info("업종 단일 범주(%s) — 상수가 될 업종 더미는 생성하지 않음",
                 only[0] if len(only) else "미상")

    # --- f_struct_ 라벨 차원 밖 구조 신호 ----------------------------------
    # 명세 3.1 핵심: "GBM이 persistence를 이길 근거" — 라벨(점포/매출/계약종료)과
    # 다른 차원의 정보. 직영/가맹 비중, 지역 분산, 개폐점 격차.
    feat["f_struct_direct_ratio_chg"] = df["direct_ratio"] - df["_prev_direct_ratio"]
    feat["f_struct_direct_growth"] = (df["n_direct"] - df["_prev_direct"]) / (df["_prev_direct"] + 1.0)
    feat["f_struct_open_close_gap"] = (
        df["n_new"] - df["n_contract_end"] - df["n_contract_cancel"]) / df["_prev_stores"]

    # 명세 3.1 "지역 분산(집중도)" — 브랜드가 몇 개 시도에 퍼져 있고 얼마나 몰려 있는가.
    # (지역 집중이 심한 브랜드는 지역 경기·상권 충격에 취약 → 라벨 차원 밖 구조 위험)
    g2 = df.groupby("brand_id", sort=False)
    for src, tag in [("n_regions", "n_regions"), ("region_hhi", "region_hhi"),
                     ("top_region_share", "top_region_share")]:
        if src in df.columns:
            feat[f"f_struct_{tag}"] = pd.to_numeric(df[src], errors="coerce")
            prev = g2[src].shift(1).where(df["_consec"]) if "_consec" in df.columns \
                else g2[src].shift(1)
            feat[f"f_struct_{tag}_chg"] = feat[f"f_struct_{tag}"] - pd.to_numeric(prev, errors="coerce")
    # 지역당 평균 점포수 (밀도) — 같은 점포수라도 소수 지역 집중 여부를 구분
    if "n_regions" in df.columns:
        nreg = pd.to_numeric(df["n_regions"], errors="coerce")
        feat["f_struct_stores_per_region"] = df["n_stores"] / nreg.where(nreg > 0)

    # --- f_hq_ 가맹본부 재무 (DART 전자공시) --------------------------------
    # 브랜드 지표는 결과이고 본부 재무는 원인에 가깝다. 본부가 자본잠식·영업적자에 빠지면
    # 물류·판촉·신규출점 지원이 끊기고 그 여파가 1~2년 뒤 가맹점 지표에 나타난다.
    feat = _attach_hq_features(feat, df, hq_fin, cfg)

    # --- 상수(제로분산) 피처 점검 — '제거'가 아니라 '보고' -------------------
    # 상수 컬럼은 모형에 정보가 없다(LGBM은 분할 불가, 로지스틱은 절편에 흡수). 다만
    # 값 분산을 보고 컬럼을 **버리면** 피처 집합이 전체 연도(=미래 포함) 값에 의존하게
    # 되므로 버리지 않고 경고만 남긴다. 상수가 될 수 있는 구조적 원인(단일 업종 더미)은
    # 위에서 생성 단계에 제거했으므로, 여기서 걸리는 것이 있다면 데이터 이상 신호다.
    const_cols = [c for c in feat.columns if c.startswith("f_")
                  and feat[c].nunique(dropna=True) <= 1]
    if const_cols:
        log.warning("상수(제로분산) 피처 %d개 감지 — 정보 없음, 데이터 점검 필요: %s",
                    len(const_cols), const_cols)

    # --- 윈저라이즈: 연도별 횡단면 분위수 클립 (t+1 미사용 — 누출 방지) ----
    # ⚠️ **이진 플래그는 제외해야 한다.** 윈저라이즈는 연도별 상하위 pct 를 잘라내는데,
    #    발생률이 pct(1%)보다 낮은 해에는 상위 분위수 자체가 0 이 되어 해당 플래그가
    #    **전부 0 으로 지워진다**. 실측에서 계속기업의문은 6만 행 중 70행(0.1%)이라
    #    가장 중요한 위험 신호가 통째로 사라질 뻔했다(적대적 리뷰 지적).
    #    0/1 값은 이상치가 존재할 수 없으므로 윈저라이즈 대상이 아니다.
    fcols = [c for c in feat.columns if c.startswith("f_")]
    binary_cols = [c for c in fcols
                   if set(pd.unique(feat[c].dropna())) <= {0.0, 1.0}]
    cont_cols = [c for c in fcols
                 if not c.startswith("f_ind_major_") and c not in binary_cols]
    if binary_cols:
        log.info("윈저라이즈 제외(이진 플래그) %d개: %s", len(binary_cols), binary_cols)
    pct = float(cfg["sample"]["winsorize_pct"])
    for c in cont_cols:
        lo = feat.groupby("year")[c].transform(lambda s: s.quantile(pct))
        hi = feat.groupby("year")[c].transform(lambda s: s.quantile(1.0 - pct))
        feat[c] = feat[c].clip(lower=lo, upper=hi)

    feat = feat.sort_values(["brand_id", "year"]).reset_index(drop=True)

    log.info(
        "features: %d rows x %d features (winsorize per-year pct=%.3f on %d continuous cols)",
        len(feat), len(fcols), pct, len(cont_cols),
    )
    log.info(
        "features: assumptions — direct_growth denominator=(prev_direct+1) smoothing; "
        "open_close_gap normalized by prev_stores; year-gap rows get NaN change metrics."
    )
    _write_summary(feat, fcols, cfg)
    return feat


def _write_summary(feat: pd.DataFrame, fcols: list[str], cfg: dict) -> None:
    """피처 목록·결측률·분포 요약 → outputs/feature_summary.csv."""
    rows = []
    for c in fcols:
        s = feat[c]
        rows.append({
            "feature": c,
            "group": c.split("_")[1],  # lvl/chg/trd/ind/struct
            "n": int(s.size),
            "n_missing": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()) if s.notna().any() else np.nan,
            "p50": float(s.quantile(0.5)) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
        })
    summary = pd.DataFrame(rows)
    out_dir = Path(cfg["paths"]["outputs"])  # ⚠️ 호출 시점에 해석 (demo 격리 대응)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8")
    log.info("features: summary -> %s", path)


# ---------------------------------------------------------------------------
# 단독 실행 (합성 패널 스모크 — 산출물은 outputs/_smoke 격리)
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    smoke_cfg = {**cfg, "paths": {**cfg["paths"], "outputs": cfg["paths"]["demo_outputs"]}}
    panel = make_synthetic_panel(cfg)
    feat = build_features(panel, smoke_cfg)
    fcols = [c for c in feat.columns if c.startswith("f_")]
    groups = pd.Series([c.split("_")[1] for c in fcols]).value_counts().to_dict()
    print("=== features smoke (synthetic panel; outputs isolated to _smoke) ===")
    print(f"rows={len(feat)}  n_features={len(fcols)}  by_group={groups}")
    miss = feat[fcols].isna().mean().sort_values(ascending=False)
    print("top-5 missing rates:")
    print(miss.head(5).to_string())


if __name__ == "__main__":
    main()
