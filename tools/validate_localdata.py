"""월별 인허가 신호가 공정위 공시와 같은 것을 재는가 — 동시 타당도 점검.

왜 필요한가
    인허가(지자체 식품접객업) 기록은 매일 갱신되지만, 사업장명을 브랜드에 이어 붙여 만든 신호다.
    매칭이 틀리거나 인허가 폐업이 가맹 계약 종료와 다른 사건이라면, 월별 신호는 빠르기만 하고
    엉뚱한 것을 잰다. 그래서 공시가 있는 해(2023·2024년 실적)에 한해 두 원천을 브랜드별로 맞대 본다.
      ① 폐점률   인허가 폐업(명의 이전 제외) ÷ 전년 말 영업 점포  vs  공시 (계약종료+해지) ÷ 전년 가맹점 수
      ② 순증감률 인허가 연말 영업 점포 증감                        vs  공시 가맹점 수 증감
      ③ 매칭 완전성 인허가에서 찾은 연말 영업 점포 ÷ 공시 가맹점 수 (중앙값)
    두 원천 모두 가맹점 30곳 이상인 브랜드만 쓴다(작은 분모의 잡음을 피한다).

실행: python tools/validate_localdata.py --csv data/raw/localdata/*.csv
산출: outputs/localdata_validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import localdata as L  # noqa: E402
from src.common import get_logger  # noqa: E402

log = get_logger("validate_localdata")
MIN_STORES = 30


def _spearman(x: pd.Series, y: pd.Series, n_boot: int = 1000, seed: int = 7) -> dict:
    from scipy.stats import spearmanr
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    xv, yv = x[ok].to_numpy(), y[ok].to_numpy()
    rho = float(spearmanr(xv, yv).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(xv), len(xv))
        boots.append(spearmanr(xv[i], yv[i]).statistic)
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    return {"n": int(ok.sum()), "rho": round(rho, 3), "ci_lo": round(float(np.percentile(boots, 2.5)), 3),
            "ci_hi": round(float(np.percentile(boots, 97.5)), 3)}


def annual_from_flows(flows: pd.DataFrame, year: int) -> pd.DataFrame:
    f = flows.copy()
    f["month"] = f["month"].astype(str)
    prev_end, end = f"{year - 1}-12", f"{year}-12"
    cur = f[f["month"].str.startswith(str(year))]
    g = cur.groupby("brand_id").agg(ld_close=("n_close", "sum"), ld_open=("n_open", "sum"))
    act = f[f["month"].isin([prev_end, end])].pivot_table(index="brand_id", columns="month",
                                                          values="n_active_end", aggfunc="sum")
    g["ld_active_prev"] = act.get(prev_end)
    g["ld_active_end"] = act.get(end)
    return g


def ftc_annual(panel: pd.DataFrame, year: int) -> pd.DataFrame:
    p = panel[["brand_id", "year", "n_stores", "n_contract_end", "n_contract_cancel"]].copy()
    cur = p[p["year"] == year].set_index("brand_id")
    prev = p[p["year"] == year - 1].set_index("brand_id")["n_stores"]
    out = pd.DataFrame(index=cur.index)
    out["ftc_stores"] = cur["n_stores"]
    out["ftc_stores_prev"] = prev.reindex(cur.index)
    out["ftc_churn"] = cur[["n_contract_end", "n_contract_cancel"]].sum(axis=1, min_count=1)
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="인허가 신호 ↔ 공정위 공시 동시 타당도")
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--brands", default=str(ROOT / "outputs" / "scores_latest.csv"))
    ap.add_argument("--panel", default=str(ROOT / "data" / "processed" / "panel.parquet"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "localdata_validation.json"))
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    args = ap.parse_args(argv)

    records = L.load_records(args.csv)
    matched = L.match_brands(records, pd.read_csv(args.brands, encoding="utf-8-sig"))
    months = (int(matched["closed_on"].pipe(pd.to_datetime, errors="coerce").dt.year.max() or 2026)
              - (min(args.years) - 1)) * 12 + 12
    flows = L.monthly_brand_flows(matched, months=months)
    panel = pd.read_parquet(args.panel)

    res: dict = {"as_of_month": str(flows["month"].max()), "n_records": len(records),
                 "n_matched": int(matched["brand_id"].notna().sum()),
                 "n_brands_matched": int(matched["brand_id"].nunique()), "min_stores": MIN_STORES,
                 "years": {}}
    for y in args.years:
        ld, ftc = annual_from_flows(flows, y), ftc_annual(panel, y)
        j = ld.join(ftc, how="inner")
        j = j[(j["ld_active_prev"] >= MIN_STORES) & (j["ftc_stores_prev"] >= MIN_STORES)]
        j["ld_close_rate"] = j["ld_close"] / j["ld_active_prev"]
        j["ftc_churn_rate"] = j["ftc_churn"] / j["ftc_stores_prev"]
        j["ld_growth"] = j["ld_active_end"] / j["ld_active_prev"] - 1
        j["ftc_growth"] = j["ftc_stores"] / j["ftc_stores_prev"] - 1
        cover = (j["ld_active_end"] / j["ftc_stores"]).replace([np.inf, -np.inf], np.nan).dropna()
        res["years"][str(y)] = {
            "close_vs_churn": _spearman(j["ld_close_rate"], j["ftc_churn_rate"]),
            "growth_vs_growth": _spearman(j["ld_growth"], j["ftc_growth"]),
            "coverage_median": round(float(cover.median()), 3),
            "coverage_iqr": [round(float(cover.quantile(0.25)), 3), round(float(cover.quantile(0.75)), 3)],
        }
        log.info("%d년: 폐점률 ρ=%s · 순증감 ρ=%s · 매칭 완전성 중앙값 %.1f%%", y,
                 res["years"][str(y)]["close_vs_churn"], res["years"][str(y)]["growth_vs_growth"],
                 100 * res["years"][str(y)]["coverage_median"])
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("저장: %s", args.out)
    return res


if __name__ == "__main__":
    main()
