"""M2 라벨 생성 — '현재 양호(t) → 미래 악화 전환(t+1)' 라벨 (INTERFACES.md §2).

계약 요약
---------
- ``label=1`` ⇔ t년에 양호(healthy)했던 브랜드가 t+1년에 악화 사건을 발동. 행의 year는 t.
- 악화 사건 3종 (각각 **t+1년** 지표, 업종그룹×연도 내 분위수, config ``label.events``):
    ① store_growth_rate 하위 quantile **그리고** |Δ점포| ≥ ``min_abs_store_change``
    ② real_sales_growth 하위 quantile
    ③ contract_end_rate 상위 quantile
- 발동: ``min_events``개 이상 **or** 1개가 극단(``extreme_quantile`` 하위/상위 5%).
- healthy_at_t: t년(t−1→t 지표)에 사건이 하나도 발동하지 않음.
  ``healthy_gate_at_t=true``면 healthy_at_t=True 행만 표본으로 반환.
- 게이트: n_stores_t ≥ ``min_stores_at_t``. t+1 행이 없으면 표본 제외.
- 분위수 임계값은 사건 지표의 t+1년 **실측 분포**로 계산 (라벨 정의 — 피처 아님, 누출 아님).

구현상 해석 (계약 모호점 — 정직성 원칙에 따라 명시)
---------------------------------------------------
1. 분위수 경계 비교는 **엄격 부등호**(<, >). 그룹 전원이 동일값이면 아무도
   '하위/상위 20%'가 아니라고 본다 (동률 경계값 미발동).
2. 극단(extreme)은 해당 사건의 발동 조건(점포 사건의 |Δ점포| 게이트 포함)을 함께
   통과한 경우에만 인정한다 — 극단 ⊂ 사건.
3. 지표가 NaN(관측 첫해, 연도 갭, 매출 결측 등)이면 그 사건은 '발동 안 함' 처리.
   따라서 관측 첫해는 사건 판정 불가 → healthy로 간주된다.
4. |Δ점포| 게이트는 계약 §2의 식 그대로 **절대값** 기준(성장·감소 무관 |Δ| ≥ 3).
5. 연도 갭(전년 미관측)·전년 점포수 0이면 전년 기반 지표는 NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import (
    get_logger,
    industry_group_col,
    load_config,
    make_synthetic_panel,
    set_seed,
)

log = get_logger("labels")

# 사건 metric → 반환 컬럼 접미사 (계약 §2 반환 스키마 고정)
_METRIC_SHORT = {
    "store_growth_rate": "store",
    "real_sales_growth": "sales",
    "contract_end_rate": "contract",
}
_EV_COLS = ["ev_store", "ev_sales", "ev_contract"]


# ---------------------------------------------------------------------------
# 파생 지표 (계약 §1 — features.py와 공용, 단일 정의 원칙)
# ---------------------------------------------------------------------------

def compute_derived_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    """계약 §1 파생 지표를 (brand_id, year) 행마다 붙여 반환한다.

    추가 컬럼:
      store_growth_rate, delta_stores, contract_end_rate, sales_growth,
      real_sales_growth, direct_ratio, industry_group,
      _prev_stores, _prev_direct, _prev_direct_ratio (내부용)

    연도 갭(전년 미관측)이거나 전년 점포수 ≤ 0이면 전년 기반 지표는 NaN.
    """
    df = panel.sort_values(["brand_id", "year"]).reset_index(drop=True).copy()
    g = df.groupby("brand_id", sort=False)
    consec = g["year"].shift(1).eq(df["year"] - 1)

    prev_stores = g["n_stores"].shift(1).where(consec)
    prev_stores = prev_stores.where(prev_stores > 0)  # 분모 0 방지
    prev_sales = g["avg_sales"].shift(1).where(consec)
    prev_sales = prev_sales.where(prev_sales > 0)
    prev_direct = g["n_direct"].shift(1).where(consec)
    prev_spa = g["avg_sales_per_area"].shift(1).where(consec)
    prev_spa = prev_spa.where(prev_spa > 0)

    df["_consec"] = consec
    df["_prev_stores"] = prev_stores
    df["_prev_direct"] = prev_direct
    df["delta_stores"] = df["n_stores"] - prev_stores
    df["store_growth_rate"] = (df["n_stores"] - prev_stores) / prev_stores
    df["contract_end_rate"] = (df["n_contract_end"] + df["n_contract_cancel"]) / prev_stores
    df["sales_growth"] = df["avg_sales"] / prev_sales - 1.0
    # 면적당매출 성장률 (명세 3.1 필수 시계열 — 점포 규모 효과를 통제한 매출 신호)
    df["sales_per_area_growth"] = df["avg_sales_per_area"] / prev_spa - 1.0

    denom = df["n_direct"] + df["n_stores"]
    df["direct_ratio"] = np.where(denom > 0, df["n_direct"] / denom, np.nan)
    g2 = df.groupby("brand_id", sort=False)
    df["_prev_direct_ratio"] = g2["direct_ratio"].shift(1).where(consec)

    # 업종그룹 (해당 연도 industry_mid 크기 ≥ 30이면 mid, 아니면 major)
    df["industry_group"] = industry_group_col(df)
    med = df.groupby(["industry_group", "year"])["sales_growth"].transform("median")
    df["real_sales_growth"] = df["sales_growth"] - med
    return df


# ---------------------------------------------------------------------------
# 사건 플래그
# ---------------------------------------------------------------------------

def _event_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """각 행(=그 연도 지표)에 대해 사건/극단 발동 플래그를 계산한다.

    분위수 임계값 풀 = 업종그룹×연도 셀 내 **비교 가능한 규모의 동료군**
    (전년 점포수 ≥ min_stores_at_t — 해당 지표 시점에 이미 알 수 있는 정보라 실시간 안전).
    1~9개 점포 영세 브랜드의 초변동성(±100% 성장 등)이 임계값을 극단화해
    '업종 내 상대 악화'라는 라벨의 업무 의미를 왜곡하는 것을 방지한다.
    임계값은 풀에서 계산하되 셀 내 모든 행에 적용된다.
    """
    lab = cfg["label"]
    xq = float(lab["extreme_quantile"])
    pool = (df["_prev_stores"] >= float(lab["min_stores_at_t"])).fillna(False)
    out = pd.DataFrame(index=df.index)
    for name, spec in lab["events"].items():
        metric = spec["metric"]
        short = _METRIC_SHORT[metric]
        q = float(spec["quantile"])
        side = spec["side"]
        pooled = df[metric].where(pool)  # 풀 밖 행은 임계값 계산에서 제외(적용은 전 행)
        grp = pooled.groupby([df["industry_group"], df["year"]])
        thr = grp.transform(lambda s, q=q: s.quantile(q))
        if side == "lower":
            xthr = grp.transform(lambda s, xq=xq: s.quantile(xq))
            fired = df[metric] < thr
            extreme = df[metric] < xthr
        elif side == "upper":
            xthr = grp.transform(lambda s, xq=xq: s.quantile(1.0 - xq))
            fired = df[metric] > thr
            extreme = df[metric] > xthr
        else:
            raise ValueError(f"label.events.{name}: unknown side '{side}'")
        if metric == "store_growth_rate":
            # 노이즈 게이트: 절대 점포수 변화 (극단 판정에도 동일 적용)
            gate = (df["delta_stores"].abs() >= float(lab["min_abs_store_change"])).fillna(False)
            fired = fired & gate
            extreme = extreme & gate
        fired = fired.fillna(False).astype(bool)
        extreme = (extreme.fillna(False).astype(bool)) & fired
        out[f"ev_{short}"] = fired
        out[f"ex_{short}"] = extreme
    return out


# ---------------------------------------------------------------------------
# 라벨 빌드
# ---------------------------------------------------------------------------

def brand_state(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """브랜드×연도의 **t년 시점 상태**. t+1 관측을 요구하지 않는다.

    [brand_id, year, state, n_events, n_obs_metrics] 반환.
      state = "건전"   — t년 악화사건 0건, 판정 지표 1개 이상 관측
              "요주의" — t년 악화사건 발동, 판정 지표 관측됨
              "평가불가" — 판정 지표가 하나도 관측되지 않음

    왜 필요한가
        모형은 `healthy_at_t=True` 행으로만 학습·보정·평가됐다. 그런데 점수는 자격
        조건만 걸고 전체에 매겨진다. 어느 브랜드가 그 모집단 안에 있는지 알려면
        **점수 산출 연도 자체의 상태**가 필요한데, labels.parquet 은 t+1 이 있어야
        행이 생기므로 최신 연도가 아예 없다. 그래서 t+1 없이 상태만 뽑는 경로를 둔다.

        ⚠️ 이전에는 '직전 연도 라벨 표본에 있었는가'로 대신 판정했다. 그건 t−1년
        상태여서 t년 상태를 과소 보고한다(2024 기준 50.9% vs 실제 61.9%).
    """
    df = compute_derived_metrics(panel)
    ev = _event_flags(df, cfg)
    df = pd.concat([df, ev], axis=1)
    ex_cols = [c for c in ev.columns if c.startswith("ex_")]
    n_events = df[_EV_COLS].sum(axis=1).astype(int)
    deteriorated = (n_events >= int(cfg["label"]["min_events"])) | df[ex_cols].any(axis=1)
    obs_cols = ["store_growth_rate", "real_sales_growth", "contract_end_rate"]
    n_obs = df[obs_cols].notna().sum(axis=1)

    state = np.where(n_obs < 1, "평가불가",
                     np.where(n_events == 0, "건전", "요주의"))
    return pd.DataFrame({
        "brand_id": df["brand_id"].to_numpy(), "year": df["year"].to_numpy(),
        "state": state, "n_events": n_events.to_numpy(),
        "n_obs_metrics": n_obs.to_numpy(), "deteriorated_at_t": deteriorated.to_numpy(),
    })


def build_labels(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """계약 §2: [brand_id, year, label, healthy_at_t, ev_store, ev_sales,
    ev_contract, n_events, extreme_fired] 반환. 행의 year는 t (피처 시점),
    ev_*/n_events/extreme_fired는 **t+1년** 사건이다."""
    lab = cfg["label"]

    df = compute_derived_metrics(panel)
    ev = _event_flags(df, cfg)
    df = pd.concat([df, ev], axis=1)
    ex_cols = [c for c in ev.columns if c.startswith("ex_")]
    df["n_events"] = df[_EV_COLS].sum(axis=1).astype(int)
    df["extreme_fired"] = df[ex_cols].any(axis=1)
    df["deteriorated"] = (df["n_events"] >= int(lab["min_events"])) | df["extreme_fired"]
    # '양호' 판정에는 검증 가능성 요건을 둔다 (적대적 리뷰 확정 결함 수정):
    # t년 사건 지표 3종이 전부 결측이면 "사건 미발동"이 아니라 "판정 불가"다.
    # (구버전은 관측 첫해·공백 복귀 행이 무조건 양호로 간주되어, 전환 표본의 27%가
    #  검증 불가 상태였고 그 구간의 양성률이 2배였다 → 전환 예측 주장 왜곡)
    obs_cols = ["store_growth_rate", "real_sales_growth", "contract_end_rate"]
    df["n_obs_metrics"] = df[obs_cols].notna().sum(axis=1)
    df["healthy"] = (df["n_events"] == 0) & (df["n_obs_metrics"] >= 1)

    # t 행에 t+1년 사건을 붙인다 (t+1 미관측 → inner merge에서 자동 제외)
    nxt = df[["brand_id", "year", *_EV_COLS, "n_events", "extreme_fired", "deteriorated"]].copy()
    nxt["year"] = nxt["year"] - 1
    cur = df[["brand_id", "year", "n_stores", "healthy"]].rename(columns={"healthy": "healthy_at_t"})
    out = cur.merge(nxt, on=["brand_id", "year"], how="inner")

    n_with_next = len(out)
    out = out[out["n_stores"] >= float(lab["min_stores_at_t"])]
    n_after_stores = len(out)
    if lab.get("healthy_gate_at_t", True):
        out = out[out["healthy_at_t"]]
    n_after_healthy = len(out)

    out = out.copy()
    out["label"] = out["deteriorated"].astype(int)
    out = out[["brand_id", "year", "label", "healthy_at_t", *_EV_COLS, "n_events", "extreme_fired"]]
    out = out.sort_values(["brand_id", "year"]).reset_index(drop=True)

    log.info(
        "labels: %d rows with t+1 -> %d after min_stores_at_t(%s) -> %d after healthy gate(%s)",
        n_with_next, n_after_stores, lab["min_stores_at_t"], n_after_healthy, lab.get("healthy_gate_at_t", True),
    )
    if len(out):
        log.info(
            "labels: positive_rate=%.4f | event rates(t+1): store=%.4f sales=%.4f contract=%.4f | extreme=%.4f",
            out["label"].mean(), out["ev_store"].mean(), out["ev_sales"].mean(),
            out["ev_contract"].mean(), out["extreme_fired"].mean(),
        )
    log.info(
        "labels: rule interpretation — strict quantile inequality; extreme subset of event "
        "(store gate applies); NaN metric never fires; |delta stores| gate is absolute value; "
        "healthy_at_t requires >=1 observable t-metric (판정 불가 행은 표본 제외)."
    )
    # ⚠️ 이 시점의 `out` 은 **자격 필터 이전** 프레임이다. 학습·검증 표본은
    #    run_pipeline 이 eligible_t 로 한 번 더 거른 뒤 확정된다(7,163 → 3,635).
    #    두 프레임의 양성률이 다르므로(11.4% vs 9.2%) 어느 쪽 표인지 파일에 적는다.
    write_composition(out, cfg, scope=PREGATE_SCOPE, filename="label_composition_pregate.csv")
    return out


PREGATE_SCOPE = "라벨 산출 가능 전체 (실시간 자격 필터 이전)"
SAMPLE_SCOPE = "학습·검증 표본 (실시간 자격 통과)"


def write_composition(sample: pd.DataFrame, cfg: dict, scope: str,
                      filename: str = "label_composition.csv") -> None:
    """사건별 발동률·중복 분포·연도별 양성률.

    **scope 를 파일 안에 적는다.** 예전에는 이 표가 자격 필터 이전 프레임(7,163행,
    양성률 11.4%)을 기록하면서 파일 이름만 label_composition.csv 였다. 그래서
    IMPLEMENTATION 은 "양성률 9.2%"(실제 학습 표본 3,635행)라 인용하고 OPERATIONS 는
    "연도별 10.1~12.6%"(자격 이전 표)라 인용하는 모순이 생겼다. 같은 파일을 두 문서가
    다른 뜻으로 읽고 있었고, 파일 자신은 어느 쪽인지 말하지 않았다.
    """
    rows: list[dict] = []
    n = len(sample)
    rows.append({"section": "scope", "key": "표본 범위", "value": np.nan, "n": 0, "scope": scope})
    rows.append({"section": "overall", "key": "n_sample", "value": float(n), "n": n})
    rows.append({
        "section": "overall", "key": "positive_rate",
        "value": float(sample["label"].mean()) if n else np.nan, "n": int(sample["label"].sum()) if n else 0,
    })
    for c in _EV_COLS:
        rows.append({
            "section": "event_rate", "key": c,
            "value": float(sample[c].mean()) if n else np.nan, "n": int(sample[c].sum()) if n else 0,
        })
    counts = sample["n_events"].value_counts() if n else {}
    for k in range(0, len(_EV_COLS) + 1):
        cnt = int(counts.get(k, 0))
        rows.append({"section": "overlap", "key": f"{k}_events", "value": (cnt / n) if n else np.nan, "n": cnt})
    if n:
        for year, grp in sample.groupby("year"):
            rows.append({
                "section": "year_positive_rate", "key": str(int(year)),
                "value": float(grp["label"].mean()), "n": len(grp),
            })
    comp = pd.DataFrame(rows, columns=["section", "key", "value", "n", "scope"])
    comp["scope"] = comp["scope"].fillna(scope)
    out_dir = Path(cfg["paths"]["outputs"])  # ⚠️ 호출 시점에 해석 (demo 격리 대응)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    comp.to_csv(path, index=False, encoding="utf-8")
    log.info("labels: composition(%s) -> %s", scope, path)


# ---------------------------------------------------------------------------
# 단독 실행 (합성 패널 스모크 — 산출물은 outputs/_smoke 격리)
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    # 합성 스모크 산출물이 제출 지표(outputs/)를 오염시키지 않도록 격리
    smoke_cfg = {**cfg, "paths": {**cfg["paths"], "outputs": cfg["paths"]["demo_outputs"]}}
    panel = make_synthetic_panel(cfg)
    labels = build_labels(panel, smoke_cfg)
    print("=== labels smoke (synthetic panel; outputs isolated to _smoke) ===")
    print(f"rows={len(labels)}  positive_rate={labels['label'].mean():.4f}")
    print(labels.groupby("year")["label"].agg(n="count", positive_rate="mean").to_string())


if __name__ == "__main__":
    main()
