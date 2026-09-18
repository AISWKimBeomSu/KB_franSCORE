"""M4 포트폴리오 — 예시(합성) 여신 포트폴리오의 집중도·예상손실 산출.

INTERFACES.md §4 `portfolio.build_portfolio(cfg)` 구현.

⚠️ 정직성 원칙: 본 모듈의 exposure(여신액)는 실여신 데이터가 아니라
   seed 고정 난수로 합성한 **예시값**이다(방법론 실증용).
   그 사실을 portfolio.csv 컬럼과 portfolio_summary.json의 'assumptions'에 명시한다.

입력 (호출 시점에 cfg['paths']에서 해석 — 데모 러너가 경로를 교체할 수 있음):
  - data/processed/predictions.parquet : test split, p_calibrated 우선 else p_lgbm
  - data/processed/panel.parquet       : n_stores (exposure 스케일링)
    (panel 부재 시 features_matrix.parquet의 f_lvl_n_stores로 폴백, 둘 다 없으면 명확한 에러)

산출:
  - outputs/portfolio.csv           : 브랜드별 exposure·등급·시나리오별 EL
  - outputs/portfolio_summary.json  : 총 exposure·HHI·상위10집중·시나리오별 EL·
                                      스트레스 EL·헤드라인 문장·가정 명세

단위: exposure·EL은 백만원(MKRW), 표시용 변환은 억원(= 백만원/100).
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import get_logger, load_config, set_seed

log = get_logger("portfolio")

MKRW_PER_EKW = 100.0  # 1억원 = 100백만원

# 익스포저 산출 근거별 고지 문구 — 화면·요약JSON이 같은 문장을 쓰도록 한 곳에 둔다.
_EXPOSURE_STATEMENT = {
    "actual_loan_book": (
        "여신 exposure는 은행이 제공한 **실제 여신 데이터**입니다 "
        "(config.portfolio.exposure_source)."),
    "disclosed_startup_cost": (
        "여신 exposure는 **공정위 공시 창업비용(15110265) 실측값**에서 유도했습니다: "
        "exposure = 가맹점수 × 브랜드별 창업비용 × 대출조달비율 × 은행 점유율. "
        "창업비용과 가맹점수는 실측이며, 대출조달비율·은행 점유율은 가정입니다 "
        "(실제 여신 잔액은 은행 내부 자료로 공개되지 않습니다). "
        "따라서 절대 금액은 시나리오이나, 브랜드 간 상대 규모·집중도는 실제 구조를 반영합니다."),
    "synthetic_lognormal": (
        "여신 exposure는 실제 데이터가 아니라 seed 고정 난수로 합성한 예시값입니다 "
        "(창업비용 원천을 찾지 못한 경우의 폴백 — 집중도 수치는 가정의 산물)."),
}


def _to_ekw(mkrw: float) -> float:
    """백만원 → 억원 변환 (표시용)."""
    return float(mkrw) / MKRW_PER_EKW


def _pick_pd_column(test: pd.DataFrame) -> tuple[pd.Series, str]:
    """악화확률 컬럼 선택: p_calibrated(값 존재 시) 우선, 결측은 p_lgbm으로 보충."""
    if "p_calibrated" in test.columns and test["p_calibrated"].notna().any():
        p = test["p_calibrated"].astype(float)
        if "p_lgbm" in test.columns:
            p = p.fillna(test["p_lgbm"].astype(float))
        return p.clip(0.0, 1.0), "p_calibrated"
    if "p_lgbm" in test.columns:
        return test["p_lgbm"].astype(float).clip(0.0, 1.0), "p_lgbm"
    raise KeyError("predictions.parquet에 p_calibrated/p_lgbm 컬럼이 없습니다.")


def _load_store_counts(processed: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """점포수 소스 로드.

    반환: (stores[brand_id, year, n_stores], meta[브랜드 메타 or None], 소스 파일명)
    panel.parquet 우선, 없으면 features_matrix.parquet의 f_lvl_n_stores 폴백.
    """
    panel_path = processed / "panel.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        # startup_total(공시 창업비용)도 함께 가져온다 — 여신 익스포저 산출의 실측 기반
        keep = ["brand_id", "year", "n_stores"] + (
            ["startup_total"] if "startup_total" in panel.columns else [])
        stores = panel[keep].copy()
        meta_cols = [c for c in ["brand_id", "brand_name", "industry_major", "industry_mid"]
                     if c in panel.columns]
        meta = (panel.sort_values(["brand_id", "year"])
                     .groupby("brand_id", as_index=False).tail(1)[meta_cols])
        return stores, meta, "panel.parquet"

    fm_path = processed / "features_matrix.parquet"
    if fm_path.exists():
        fm = pd.read_parquet(fm_path)
        if "f_lvl_n_stores" not in fm.columns:
            raise KeyError(
                f"{fm_path}에 f_lvl_n_stores 컬럼이 없어 exposure 스케일링 불가.")
        stores = (fm[["brand_id", "year", "f_lvl_n_stores"]]
                  .rename(columns={"f_lvl_n_stores": "n_stores"}))
        return stores, None, "features_matrix.parquet"

    raise FileNotFoundError(
        f"점포수 소스를 찾지 못했습니다: {panel_path} 또는 {fm_path} 필요 "
        "(panel 또는 features 스텝을 먼저 실행하세요).")


def _attach_store_counts(test: pd.DataFrame, stores: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """(brand_id, year) 정확 매칭 → 실패 시 브랜드별 최신 관측치로 폴백."""
    stores = stores.dropna(subset=["n_stores"])
    merged = test.merge(stores, on=["brand_id", "year"], how="left")
    n_fallback = 0
    missing = merged["n_stores"].isna()
    if missing.any():
        last = (stores.sort_values(["brand_id", "year"])
                      .groupby("brand_id")["n_stores"].last())
        merged.loc[missing, "n_stores"] = merged.loc[missing, "brand_id"].map(last)
        n_fallback = int(missing.sum() - merged["n_stores"].isna().sum())
    # 창업비용은 연도별 공시 누락이 흔하므로 브랜드별 최신 관측치로 보완 (과거 정보만 사용)
    if "startup_total" in merged.columns:
        miss_c = merged["startup_total"].isna()
        if miss_c.any() and "startup_total" in stores.columns:
            last_c = (stores.dropna(subset=["startup_total"])
                            .sort_values(["brand_id", "year"])
                            .groupby("brand_id")["startup_total"].last())
            merged.loc[miss_c, "startup_total"] = merged.loc[miss_c, "brand_id"].map(last_c)
    return merged, n_fallback


def _build_exposure(port: pd.DataFrame, cfg: dict, rng) -> tuple[pd.DataFrame, dict]:
    """브랜드별 여신 익스포저 산출. **실제 여신 > 공시 창업비용 > 합성** 순으로 시도.

    ── 여신 익스포저란 ─────────────────────────────────────────────────────────
    은행이 그 브랜드의 **가맹점주들에게 빌려준 대출 잔액의 합**이다. 브랜드가 무너지면
    그 금액이 한꺼번에 부실해질 수 있으므로, 브랜드 단위 집중 리스크의 크기 그 자체다.

    ── 층 ①: 실제 여신 (은행 내부 데이터) ─────────────────────────────────────
    `config.portfolio.exposure_source` 에 CSV 경로를 주면 그것을 쓴다.
    필요 컬럼: brand_id(또는 brand_name) + exposure_mkrw [+ n_borrowers]
    은행 내부에서 돌릴 때는 이 경로만 채우면 나머지 파이프라인이 그대로 동작한다.

    ── 층 ②: 공시 창업비용 기반 추정 (기본값, **실측 기반**) ───────────────────
    실제 여신은 은행 내부 자료라 공개 데이터에 존재하지 않는다. 그러나 정보공개서는
    **가맹점 1곳 개설 총 투자액(가맹금+교육비+보증금+기타)을 브랜드별로 공시**한다
    (15110265, 천원). 이를 관측 기반으로 삼아:

        익스포저 = 가맹점수 × 창업비용(공시 실측) × 대출조달비율 × 은행 점유율

    **창업비용·가맹점수는 실측**이고 뒤의 두 비율만 가정이다. 난수를 쓰지 않으므로
    브랜드 간 상대 규모(집중도·HHI)가 **실제 업종·브랜드 구조를 반영**한다.

    ── 층 ③: 합성 (원천이 전혀 없을 때만) ─────────────────────────────────────
    """
    pcfg = cfg["portfolio"]
    src = pcfg.get("exposure_source")
    n = len(port)

    if src:                                        # 층 ①
        p = Path(src)
        if not p.is_absolute():
            p = Path(cfg.get("_root", ".")) / p
        if p.exists():
            real = pd.read_csv(p)
            key = "brand_id" if "brand_id" in real.columns else "brand_name"
            if key not in port.columns or "exposure_mkrw" not in real.columns:
                raise ValueError(f"{p}: '{key}' 와 'exposure_mkrw' 컬럼이 필요합니다")
            cols = [key, "exposure_mkrw"] + (
                ["n_borrowers"] if "n_borrowers" in real.columns else [])
            port = port.drop(columns=["exposure_mkrw"], errors="ignore").merge(
                real[cols].drop_duplicates(key), on=key, how="left")
            hit = int(port["exposure_mkrw"].notna().sum())
            port["exposure_mkrw"] = port["exposure_mkrw"].fillna(0.0)
            log.info("여신 익스포저: **실제 여신 데이터** 사용 (%s, %d/%d 매칭)", p, hit, n)
            return port, {"basis": "actual_loan_book", "source_file": str(p),
                          "matched_brands": hit, "is_synthetic": False}
        log.warning("exposure_source=%s 파일이 없어 공시 창업비용 기반으로 대체", p)

    ltv = float(pcfg.get("loan_to_startup_cost", 0.6))
    share = float(pcfg.get("bank_share", 0.15))
    if "startup_total" in port.columns and port["startup_total"].notna().any():   # 층 ②
        cost = pd.to_numeric(port["startup_total"], errors="coerce")
        n_hit = int(cost.notna().sum())
        # 결측은 같은 업종 중앙값 → 없으면 전체 중앙값 (난수 미사용).
        # 업종 컬럼은 이 시점에 아직 붙지 않았을 수 있으므로 방어적으로 처리한다.
        if "industry_major" in port.columns:
            cost = cost.fillna(port.assign(_c=cost)
                                   .groupby("industry_major")["_c"].transform("median"))
        cost = cost.fillna(cost.median())
        stores = pd.to_numeric(port["n_stores"], errors="coerce").fillna(0.0)
        port["startup_cost_mkrw"] = cost / 1000.0             # 천원 → 백만원
        port["exposure_mkrw"] = stores * port["startup_cost_mkrw"] * ltv * share
        log.info("여신 익스포저: **공시 창업비용 기반** (실측 %d/%d 브랜드, 점포당 중앙값 "
                 "%.1f백만원) × 대출조달비율 %.0f%% × 은행 점유율 %.0f%%",
                 n_hit, n, float(port["startup_cost_mkrw"].median()), 100 * ltv, 100 * share)
        return port, {"basis": "disclosed_startup_cost", "is_synthetic": False,
                      "startup_cost_observed_brands": n_hit, "n_brands": n,
                      "loan_to_startup_cost": ltv, "bank_share": share,
                      "source_api": "15110265 브랜드별 창업 금액 현황",
                      "assumption_note": ("창업비용·가맹점수는 공정위 공시 실측값. "
                                          "대출조달비율·은행 점유율은 가정(실제 여신은 미공개).")}

    scale = float(pcfg.get("exposure_scale_per_store_mkrw", 50))                 # 층 ③
    sigma = float(pcfg.get("exposure_lognorm_sigma", 0.5))
    port["exposure_mkrw"] = (port["n_stores"].to_numpy(dtype=float) * scale
                             * rng.lognormal(mean=0.0, sigma=sigma, size=n))
    log.warning("여신 익스포저: 창업비용 원천이 없어 **합성 난수** 사용")
    return port, {"basis": "synthetic_lognormal", "is_synthetic": True,
                  "scale_per_store_mkrw": scale, "sigma": sigma}


def build_portfolio(cfg: dict) -> None:
    """예시 여신 포트폴리오 합성 → 집중도·시나리오별 예상손실 산출 (INTERFACES.md §4)."""
    set_seed(cfg["seed"])
    pcfg = cfg["portfolio"]
    # 경로는 반드시 호출 시점에 해석 (데모 러너가 cfg['paths']를 교체함)
    processed = Path(cfg["paths"]["processed"])
    outputs = Path(cfg["paths"]["outputs"])
    outputs.mkdir(parents=True, exist_ok=True)

    # ---------------- 1) 예측확률 (test split) ----------------
    pred_path = processed / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"predictions.parquet 없음: {pred_path} — model 스텝을 먼저 실행하세요.")
    pred = pd.read_parquet(pred_path)
    test = pred[pred["split"].astype(str) == "test"].copy()
    if test.empty:
        raise ValueError("predictions.parquet에 split=='test' 행이 없습니다.")
    test_year = int(test["year"].max())
    if test["year"].nunique() > 1:
        log.warning("test split에 연도가 %d개 존재 — 계약(§3)상 단일 연도. 최종 연도 %d만 사용.",
                    test["year"].nunique(), test_year)
        test = test[test["year"] == test_year]
    test = test.drop_duplicates(subset="brand_id", keep="last").reset_index(drop=True)

    p, det_col = _pick_pd_column(test)
    test["deterioration_1y"] = p.values
    log.info("test 연도=%d, 브랜드 %d개, 악화확률 컬럼=%s", test_year, len(test), det_col)

    # ---------------- 2) 점포수 결합 (exposure 스케일) ----------------
    stores, meta, store_src = _load_store_counts(processed)
    test, n_store_fallback = _attach_store_counts(test[["brand_id", "year", "deterioration_1y"]], stores)
    if n_store_fallback:
        log.info("점포수 (brand,year) 정확 매칭 실패 %d건 → 브랜드별 최신 관측치로 폴백", n_store_fallback)
    n_drop = int((test["n_stores"].isna() | (test["n_stores"] <= 0)).sum())
    if n_drop:
        log.warning("점포수 결측/비양수 %d개 브랜드는 포트폴리오 후보에서 제외", n_drop)
    test = test[test["n_stores"].notna() & (test["n_stores"] > 0)].reset_index(drop=True)
    if test.empty:
        raise ValueError("점포수가 유효한 test 브랜드가 없어 포트폴리오를 구성할 수 없습니다.")

    # ---------------- 3) 위험등급 (test 전체 분포 내 순위 기반) ----------------
    # 값 임계 컷(p >= q0.90)은 보정확률 동률 블록이 경계에 걸리면 High가 설계 비율
    # (상위 10%)을 크게 초과한다(리뷰 확정 결함). 순위(pct rank, method='first') 기반으로
    # 부여해 등급 비율이 항상 설계값과 일치하게 한다. 컷 값은 참고용으로만 기록.
    grades_cfg = pcfg["risk_grades"]
    test = test.copy()
    test["deterioration_rank_pct"] = test["deterioration_1y"].rank(pct=True, method="first")
    hi_cut = float(np.quantile(test["deterioration_1y"], float(grades_cfg["high"])))
    med_cut = float(np.quantile(test["deterioration_1y"], float(grades_cfg["medium"])))

    # ---------------- 4) 예시 포트폴리오 + 여신 익스포저 ----------------
    rng = np.random.default_rng(cfg["seed"])
    n_sample = min(int(pcfg["n_brands_sampled"]), len(test))
    eligible = test.sort_values("brand_id").reset_index(drop=True)  # 입력 행순서 무관 결정성
    take = np.sort(rng.choice(len(eligible), size=n_sample, replace=False))
    port = eligible.iloc[take].reset_index(drop=True)
    port, expo_meta = _build_exposure(port, cfg, rng)
    port["exposure_ekw"] = port["exposure_mkrw"] / MKRW_PER_EKW

    total_exposure = float(port["exposure_mkrw"].sum())
    port["exposure_share"] = port["exposure_mkrw"] / total_exposure

    port["risk_grade"] = np.select(
        [port["deterioration_rank_pct"] > float(grades_cfg["high"]),
         port["deterioration_rank_pct"] > float(grades_cfg["medium"])],
        ["High", "Medium"], default="Low")

    # ---------------- 5) 집중도 ----------------
    shares_desc = port["exposure_share"].sort_values(ascending=False)
    hhi = float((shares_desc ** 2).sum())
    top10_share = float(shares_desc.head(10).sum())
    top1_share = float(shares_desc.iloc[0])

    # --------- 6) 스트레스: 위험 상위 stress_top_pct 브랜드 악화확률 × multiplier (cap 1.0) ---
    stress_top_pct = float(pcfg["stress_top_pct"])
    stress_mult = float(pcfg["stress_pd_multiplier"])
    n_stress = max(1, math.ceil(len(port) * stress_top_pct))
    stress_idx = port["deterioration_1y"].nlargest(n_stress).index
    port["is_stressed"] = port.index.isin(stress_idx)
    port["det_stressed"] = np.where(
        port["is_stressed"], np.minimum(port["deterioration_1y"] * stress_mult, 1.0), port["deterioration_1y"])

    # ------------- 7) 시나리오별 예상손실 EL = exposure × 악화확률 × LGD -------------
    lgd_scenarios = sorted(float(x) for x in pcfg["lgd_scenarios"])
    el_rows = []
    for lgd in lgd_scenarios:
        key = f"lgd{round(lgd * 100):02d}"
        port[f"el_{key}_mkrw"] = port["exposure_mkrw"] * port["deterioration_1y"] * lgd
        port[f"stress_el_{key}_mkrw"] = port["exposure_mkrw"] * port["det_stressed"] * lgd
        el = float(port[f"el_{key}_mkrw"].sum())
        sel = float(port[f"stress_el_{key}_mkrw"].sum())
        el_rows.append({
            "lgd": lgd,
            "el_mkrw": el,
            "el_ekw": _to_ekw(el),
            "el_pct_of_exposure": el / total_exposure,
            "stress_el_mkrw": sel,
            "stress_el_ekw": _to_ekw(sel),
            "stress_el_pct_of_exposure": sel / total_exposure,
        })

    # ---------------- 8) 브랜드 메타 결합 + CSV 저장 ----------------
    if meta is not None:
        port = port.merge(meta, on="brand_id", how="left")
    if "brand_name" not in port.columns:
        port["brand_name"] = port["brand_id"]
    else:
        port["brand_name"] = port["brand_name"].fillna(port["brand_id"])
    # 익스포저의 출처를 행마다 명시 — 실여신/공시추정/합성 중 무엇인지 숨기지 않는다
    port["exposure_basis"] = expo_meta["basis"]
    port["exposure_is_synthetic"] = bool(expo_meta.get("is_synthetic", True))

    lead = [c for c in ["brand_id", "brand_name", "industry_major", "industry_mid",
                        "year", "n_stores", "deterioration_1y", "risk_grade",
                        "startup_cost_mkrw", "exposure_mkrw", "exposure_ekw", "exposure_share",
                        "is_stressed", "det_stressed"] if c in port.columns]
    rest = [c for c in port.columns if c not in lead]
    port = port[lead + rest].sort_values("exposure_mkrw", ascending=False).reset_index(drop=True)

    csv_path = outputs / "portfolio.csv"
    port.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("portfolio.csv 저장: %s (%d 브랜드)", csv_path, len(port))

    # ---------------- 9) 요약 JSON (헤드라인 + 가정 명세) ----------------
    mid = el_rows[len(el_rows) // 2]  # 중간 LGD 시나리오를 헤드라인에 사용
    basis_tag = {"actual_loan_book": "[실제 여신]",
                 "disclosed_startup_cost": "[공시 창업비용 기반 추정 여신]",
                 "synthetic_lognormal": "[합성 예시 여신]"}.get(expo_meta["basis"], "[여신]")
    headline = (
        f"{basis_tag} {test_year}년 test 브랜드 {len(port)}개로 구성한 포트폴리오의 "
        f"총 여신은 {_to_ekw(total_exposure):,.0f}억원이며, "
        f"중간 시나리오(LGD {mid['lgd']:.0%}) 기준 1년 예상손실은 "
        f"{mid['el_ekw']:,.1f}억원(여신의 {mid['el_pct_of_exposure']:.2%}), "
        f"위험 상위 {stress_top_pct:.0%} 브랜드 동시 악화 스트레스 시 "
        f"{mid['stress_el_ekw']:,.1f}억원으로 확대됩니다. "
        f"상위 10개 브랜드 여신 집중도는 {top10_share:.1%}, HHI는 {hhi:.3f}입니다.")

    grade_counts = port["risk_grade"].value_counts().to_dict()
    grade_exposure = port.groupby("risk_grade")["exposure_mkrw"].sum().to_dict()

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        # 익스포저가 실제로 합성 난수인지 여부. 하드코딩 True 로 두면 공시 창업비용 실측을
        # 쓰고 있는데도 "합성"이라고 알리게 되어(실제로 그런 상태였다) 정직 표기가 거꾸로
        # 틀린다. 실측 원천을 쓸 때는 그렇게 말해야 한다 — 과소 주장도 부정확한 주장이다.
        "synthetic": bool(expo_meta.get("is_synthetic", True)),
        "test_year": test_year,
        "n_brands": len(port),
        "det_column_used": det_col,
        "total_exposure_mkrw": total_exposure,
        "total_exposure_ekw": _to_ekw(total_exposure),
        "concentration": {
            "hhi": hhi,
            "top10_share": top10_share,
            "top1_share": top1_share,
        },
        "risk_grades": {
            "high_quantile": float(grades_cfg["high"]),
            "medium_quantile": float(grades_cfg["medium"]),
            "high_pd_cut": hi_cut,
            "medium_pd_cut": med_cut,
            "counts": {str(k): int(v) for k, v in grade_counts.items()},
            "exposure_mkrw": {str(k): float(v) for k, v in grade_exposure.items()},
        },
        "expected_loss_scenarios": el_rows,
        "stress": {
            "top_pct": stress_top_pct,
            "stress_multiplier": stress_mult,
            "prob_cap": 1.0,
            "n_stressed_brands": int(n_stress),
        },
        "headline": headline,
        "assumptions": {
            "exposure": expo_meta,
            "synthetic_exposure": bool(expo_meta.get("is_synthetic", True)),
            "statement_ko": _EXPOSURE_STATEMENT.get(
                expo_meta["basis"], "여신 exposure 산출 근거 미상"),
            "seed": int(cfg["seed"]),
            "units": {"exposure": "백만원(MKRW)", "display": "억원 = 백만원 / 100"},
            # ⚠️ 이 문장은 pd_* → deterioration_* 일괄 개명(156건) 때 조사가 깨진 채
            #    산출물에 실려 나갔다("부도확률(PD)이 사용한 것으로"). 뜻이 통하지 않을 뿐
            #    아니라, 우리 산출물을 PD 라고 주장하는 것처럼 읽혀 MODEL_USE_SPEC 의
            #    금지 용도와 정면으로 충돌한다. 명세와 같은 말을 하도록 다시 쓴다.
            "risk_definition": (
                "악화확률은 모델이 예측한 '1년 내 구조악화 전환 확률'입니다. 공정거래위원회 "
                "공시 지표가 업종×연도 하위 구간에 진입하는 사건의 확률이며, 차주의 "
                "채무불이행 확률(PD)이 아닙니다. EL 은 그 확률을 쓴 대리 손실 지표입니다."),
            "el_formula": "EL_i = exposure_i × 악화확률_i × LGD (LGD 시나리오 3종)",
            "stress_rule": (
                f"악화확률 상위 {stress_top_pct:.0%} 브랜드({n_stress}개)의 악화확률 × {stress_mult} "
                "(상한 1.0) 동시 악화 가정"),
            "risk_grade_basis": (
                "test 전체 브랜드 예측확률 분포 내 **순위(pct rank)** 기준 — "
                f"High = 상위 {1 - float(grades_cfg['high']):.0%}, "
                f"Medium = 상위 {1 - float(grades_cfg['medium']):.0%} 이내. "
                "값 임계 컷은 보정확률 동률 블록이 경계에 걸릴 때 등급 비율이 설계값을 "
                "초과할 수 있어(개발 중 실측 사례 확인) 순위 기준으로 부여한다."),
            "risk_grade_cut_values_reference": {
                "high_pd_cut": round(hi_cut, 6), "medium_pd_cut": round(med_cut, 6),
                "note": "참고용 값 — 등급 판정에는 순위를 사용한다",
            },
            "store_count_source": store_src,
            "store_count_fallback_rows": int(n_store_fallback),
            "brands_dropped_no_stores": int(n_drop),
        },
    }

    json_path = outputs / "portfolio_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("portfolio_summary.json 저장: %s", json_path)
    log.info("헤드라인: %s", headline)


if __name__ == "__main__":
    build_portfolio(load_config())
