"""워크포워드(확장창) 백테스트 — 검정력 보강.

배경 (정직성 기록):
    명세 §4.1의 단일 시간분할(train ≤T-2 / valid T-1 / test T)은 test 표본이
    한 연도(양성 수십 건 규모 — 현재 수치는 outputs/metrics.csv 참조)뿐이라,
    페어드 부트스트랩에서 기준모형 대비 우위가 표본 변동 범위를 넘기 어렵다.
    이는 모형 결함이 아니라 **단일 연도 검정력 부족**이다.

해결 (우회가 아닌 표준 방법):
    확장창 워크포워드 — 여신 리스크 백테스트의 표준 관행.
        fold 1: train ≤2019, valid 2020, test 2021
        fold 2: train ≤2020, valid 2021, test 2022
        fold 3: train ≤2021, valid 2022, test 2023
    각 fold는 **자기 시점 이전 데이터만** 사용하고(시점 누출 없음), 매 fold마다
    모형을 처음부터 재학습한다. 각 test 연도의 out-of-sample 예측을 풀링하면
    표본이 수 배로 늘어 기준모형 대비 우위를 실제로 검정할 수 있다.

    단일분할 결과(명세 준수, 주 지표)는 그대로 유지하고, 본 모듈은 **검정력 보강용
    보조 근거**로 병기한다. 두 결과를 모두 공개한다.

⚠️ 집계 방식 (실측으로 발견한 결함과 그 수정 — 정직 기록):
    처음에는 fold별 OOS 예측을 한데 모아 **원점수(raw probability) 전역 상위 10%** 로
    Lift@10 을 계산했다. 그런데 LightGBM의 풀링 Lift(2.150)가 **모든 개별 fold 값
    (2.62 / 2.25 / 2.50)보다 낮게** 나오는 모순이 관측됐다. 원인을 추적한 결과:

      fold마다 모형을 **처음부터 재학습**하므로 출력 확률의 **스케일이 fold마다 다르다**
      (실측: LightGBM 최대확률이 2021 fold 0.80 / 2022 fold 0.84 인데 2023 fold는 0.57).
      원점수를 그대로 섞으면 점수가 낮게 나온 fold의 브랜드는 전역 상위 10%에 거의
      못 들어간다 — 실측으로 2023 fold는 행의 35.9%를 차지하는데 전역 상위 10%의
      4.4%만 차지했다(편향배수 0.12).

    즉 원점수 풀링은 **재학습 모형에 불리하고 고정공식 기준모형(persistence·단일변수)에
    유리한** 인위적 편향을 만든다(기준모형은 매년 같은 변환이라 스케일이 안 흔들린다).
    순위 기반 지표를 서로 다른 모형이 매긴 점수로 전역 정렬하는 것 자체가 잘못이다.

    수정: 상위 k% 지표는 **fold 안에서** 계산하는 것을 원칙으로 한다.
      · `macro_avg_folds`  fold별 지표의 단순평균 — **워크포워드 주 지표**
        (운영 의미: "매년 상위 10%를 점검하면 얻는 리프트". 은행은 연도를 섞어
         한 줄로 세우지 않으므로 이 정의가 실제 사용과 일치한다.)
      · `pooled_oos_rank`  fold 내 백분위로 정규화한 뒤 풀링 (스케일 드리프트 제거)
      · `pooled_oos_raw`   결함이 있는 원래 방식 — **삭제하지 않고 함께 공개**한다.
        진단 근거는 outputs/walkforward_pool_bias.csv 에 수치로 남긴다.

산출물:
    outputs/walkforward_metrics.csv       fold별 + 3가지 집계 지표
    outputs/walkforward_predictions.parquet  fold별 OOS 예측 (원점수)
    outputs/walkforward_pool_bias.csv     원점수 풀링의 fold 편향 진단 (결함 증빙)
    outputs/walkforward_delta_ci.csv      기준모형 대비 Lift 차이 페어드 부트스트랩
                                          (연도블록 재표집 → fold별 Lift → 매크로 평균)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import get_logger, load_config, set_seed
from src.evaluate import MODEL_COLS, _metric_row
from src.model import (
    _baseline_scores,
    _numeric_feature_frame,
    _sanitize_feature_names,
    fit_lgbm_valid_selected,
)

log = get_logger("backtest")

MIN_TRAIN_YEARS = 2  # fold 성립 최소 학습 연도 수


def _join_sample(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    feats = features.drop_duplicates(subset=["brand_id", "year"]).copy()
    lab = labels.dropna(subset=["label"]).drop_duplicates(subset=["brand_id", "year"]).copy()
    lab["label"] = lab["label"].astype(int)
    overlap = (set(feats.columns) & set(lab.columns)) - {"brand_id", "year"}
    if overlap:
        feats = feats.drop(columns=sorted(overlap))
    df = feats.merge(lab, on=["brand_id", "year"], how="inner")
    return df.sort_values(["brand_id", "year"]).reset_index(drop=True)


def _fit_fold(df: pd.DataFrame, tr: np.ndarray, va: np.ndarray, cfg: dict) -> dict:
    """한 fold 학습 → 전 행 예측 점수 딕셔너리 반환 (기준모형 3종 + LightGBM)."""
    y = df["label"].to_numpy()
    X_orig = _numeric_feature_frame(df, log)
    safe_names, _ = _sanitize_feature_names(X_orig.columns)
    X = X_orig.set_axis(safe_names, axis=1)

    p_pers, p_single = _baseline_scores(df, list(X_orig.columns), cfg)

    log_cfg = (cfg.get("model") or {}).get("logistic") or {}
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=float(log_cfg.get("C", 1.0)),
            max_iter=int(log_cfg.get("max_iter", 2000)),
            class_weight=log_cfg.get("class_weight", "balanced"),
            random_state=int(cfg["seed"]))),
    ])
    pipe.fit(X.loc[tr], y[tr])
    p_logit = pipe.predict_proba(X)[:, 1]

    # fold별로 동일 절차 재적용 (valid 기반 복잡도 선택 — 각 fold의 valid만 사용)
    clf, best_iter, _, _ = fit_lgbm_valid_selected(X, y, tr, va, cfg, log)
    p_lgbm = clf.predict_proba(X)[:, 1]
    return {
        "p_lgbm": np.clip(p_lgbm, 0, 1), "p_persistence": np.clip(p_pers, 0, 1),
        "p_single": np.clip(p_single, 0, 1), "p_logistic": np.clip(p_logit, 0, 1),
        "best_iteration": int(best_iter),
    }


def _within_fold_rank(pooled: pd.DataFrame, col: str) -> np.ndarray:
    """fold 안에서의 백분위 순위(0~1). fold 간 점수 스케일 차이를 제거한다."""
    return pooled.groupby("fold_test_year")[col].rank(pct=True, method="average").to_numpy()


def _pool_bias(pooled: pd.DataFrame, k_pct: float) -> pd.DataFrame:
    """원점수 전역 상위 k%에 각 fold가 얼마나 들어가는지 — 스케일 드리프트 진단표.

    편향배수 = (상위k% 내 해당 fold 비중) / (전체 행 중 해당 fold 비중).
    1.0 이면 중립, 1보다 크게 벗어나면 그 fold가 과대/과소 대표된다는 뜻이다.
    이 표가 'pooled_oos_raw 를 주 지표로 쓰면 안 되는' 객관적 근거다.
    """
    n = len(pooled)
    if n == 0:
        return pd.DataFrame()
    k = max(1, round(k_pct * n))
    share_rows = pooled["fold_test_year"].value_counts(normalize=True)
    rows = []
    for name, col in MODEL_COLS.items():
        share_top = pooled.nlargest(k, col)["fold_test_year"].value_counts(normalize=True)
        for fy in sorted(pooled["fold_test_year"].unique()):
            r = float(share_rows.get(fy, 0.0))
            t = float(share_top.get(fy, 0.0))
            rows.append({"model": name, "fold_test_year": int(fy),
                         "row_share": round(r, 4), "top_k_share": round(t, 4),
                         "bias_ratio": round(t / r, 3) if r > 0 else np.nan})
    return pd.DataFrame(rows)


def _macro_lift(y: np.ndarray, p: np.ndarray, fold: np.ndarray, k_pct: float) -> float:
    """fold별 Lift@k 의 단순평균 (계산 불가 fold는 제외). 전부 불가면 NaN."""
    vals = []
    for fy in np.unique(fold):
        m = fold == fy
        v = _metric_row(y[m], p[m], k_pct)["lift_at_10"]
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def _paired_delta_ci(pooled: pd.DataFrame, k_pct: float, n_boot: int, seed: int) -> pd.DataFrame:
    """LightGBM − 기준모형의 **fold 평균 Lift@k** 차이에 대한 부트스트랩 CI·승률.

    설계:
      · 재표집 단위는 **연도(=fold) 블록**. 같은 연도 행끼리 상관되므로 행 단위로
        섞으면 CI가 과소추정된다.
      · 각 재표집 표본에서 지표를 **fold 안에서 계산한 뒤 평균**한다.
        (원점수를 전역 정렬하면 fold 간 확률 스케일 차이가 순위를 왜곡한다 —
         모듈 상단 '집계 방식' 참조. 여기서 같은 실수를 반복하지 않기 위한 것이다.)
      · 페어드: 동일 재표집 표본 안에서 LightGBM과 기준모형의 차이를 취한다.

    ⚠️ 한계(정직 기록): 연도 블록이 소수(현재 3개)라 서로 다른 블록 조합의 가짓수가
    작다. CI·승률은 **보수적 근사**이며 정밀한 p-값으로 읽어선 안 된다. 페어드 설계라
    방향(우열) 판정에는 유효하다.
    """
    y_all = pooled["y_true"].to_numpy(dtype=float)
    if len(y_all) == 0:
        return pd.DataFrame()
    years = pooled["fold_test_year"].to_numpy()
    uniq_years = np.unique(years)
    idx_by_year = {int(yy): np.flatnonzero(years == yy) for yy in uniq_years}
    scores = {name: pooled[col].to_numpy(dtype=float) for name, col in MODEL_COLS.items()}
    rng = np.random.default_rng(seed)
    baselines = ["persistence", "single", "logistic"]
    deltas: dict[str, list[float]] = {b: [] for b in baselines}
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq_years, size=len(uniq_years), replace=True)
        idx = np.concatenate([idx_by_year[int(yy)] for yy in pick])
        # 같은 연도를 두 번 뽑아도 fold 경계가 유지되도록 블록 순번으로 fold를 재부여
        blk = np.concatenate([np.full(idx_by_year[int(yy)].size, j)
                              for j, yy in enumerate(pick)])
        yb = y_all[idx]
        m = _macro_lift(yb, scores["lgbm"][idx], blk, k_pct)
        if math.isnan(m):
            continue
        for b in baselines:
            bl = _macro_lift(yb, scores[b][idx], blk, k_pct)
            if not math.isnan(bl):
                deltas[b].append(float(m - bl))
    rows = []
    for b in baselines:
        arr = np.asarray(deltas[b], dtype=float)
        if arr.size == 0:
            continue
        rows.append({"comparison": f"lgbm - {b}", "mean_delta_lift": float(arr.mean()),
                     "ci_lo": float(np.quantile(arr, 0.025)),
                     "ci_hi": float(np.quantile(arr, 0.975)),
                     "p_win": float((arr > 0).mean()), "n_boot": int(arr.size),
                     "metric": "macro_avg_fold_lift_at_10",
                     "bootstrap_unit": "year_block"})
    return pd.DataFrame(rows)


def run_walkforward(cfg: dict) -> pd.DataFrame:
    """확장창 워크포워드 실행 → fold별·풀링 지표 저장 및 반환."""
    set_seed(cfg["seed"])
    proc = Path(cfg["paths"]["processed"])
    out_dir = Path(cfg["paths"]["outputs"])
    k_pct = float((cfg.get("evaluate") or {}).get("k_pct", 0.10))

    features = pd.read_parquet(proc / "features.parquet")
    labels = pd.read_parquet(proc / "labels.parquet")
    df = _join_sample(features, labels)
    years = sorted(int(y) for y in df["year"].unique())
    if len(years) < MIN_TRAIN_YEARS + 2:
        log.warning("워크포워드 fold 성립 불가 (라벨 연도 %s) - 생략", years)
        return pd.DataFrame()

    fold_rows: list[dict] = []
    pooled_parts: list[pd.DataFrame] = []
    # test 연도는 앞에 train(≥MIN_TRAIN_YEARS) + valid(1) 가 확보되는 연도부터
    for i in range(MIN_TRAIN_YEARS + 1, len(years)):
        test_year = years[i]
        valid_year = years[i - 1]
        train_years = years[:i - 1]
        tr = df["year"].isin(train_years).to_numpy()
        va = df["year"].eq(valid_year).to_numpy()
        te = df["year"].eq(test_year).to_numpy()
        if te.sum() == 0 or tr.sum() == 0 or len(np.unique(df["label"].to_numpy()[va])) < 2:
            log.warning("fold test=%d 건너뜀 (표본/클래스 부족)", test_year)
            continue
        scores = _fit_fold(df, tr, va, cfg)
        y = df["label"].to_numpy()
        fold_pred = pd.DataFrame({
            "brand_id": df["brand_id"].astype(str).to_numpy()[te],
            "year": df["year"].astype(int).to_numpy()[te],
            "y_true": y[te].astype(int),
            "p_lgbm": scores["p_lgbm"][te],
            "p_persistence": scores["p_persistence"][te],
            "p_single": scores["p_single"][te],
            "p_logistic": scores["p_logistic"][te],
            "fold_test_year": test_year,
            "is_new_brand": ~df["brand_id"].isin(set(df.loc[tr, "brand_id"])).to_numpy()[te],
        })
        pooled_parts.append(fold_pred)
        for name, col in MODEL_COLS.items():
            row = _metric_row(fold_pred["y_true"], fold_pred[col], k_pct)
            fold_rows.append({"scope": f"fold_test_{test_year}", "model": name,
                              "train_years": f"{train_years[0]}~{train_years[-1]}",
                              "valid_year": valid_year, **row})
        log.info("fold test=%d: train %s / valid %d (n_test=%d, 양성 %d, best_iter=%d)",
                 test_year, f"{train_years[0]}~{train_years[-1]}", valid_year,
                 int(te.sum()), int(y[te].sum()), scores["best_iteration"])

    if not pooled_parts:
        log.warning("워크포워드 fold 없음 - 생략")
        return pd.DataFrame()

    pooled = pd.concat(pooled_parts, ignore_index=True)
    pooled.to_parquet(out_dir / "walkforward_predictions.parquet", index=False)

    fold_df = pd.DataFrame(fold_rows)
    agg_rows: list[dict] = []
    for name, col in MODEL_COLS.items():
        sub = fold_df[fold_df["model"] == name]
        # (A) 주 지표 — fold별 지표의 단순평균
        agg_rows.append({
            "scope": "macro_avg_folds", "model": name, "train_years": "expanding",
            "valid_year": -1, "n": int(sub["n"].sum()),
            **{k: float(sub[k].mean()) for k in
               ("lift_at_10", "precision_at_10", "pr_auc", "roc_auc", "brier", "base_rate")},
        })
        # (B) fold 내 백분위로 정규화한 풀링 (스케일 드리프트 제거).
        #     brier 는 확률 자체의 정확도라 순위값에 적용하면 의미가 없어 제외한다.
        rank_row = _metric_row(pooled["y_true"], _within_fold_rank(pooled, col), k_pct)
        rank_row["brier"] = float("nan")
        agg_rows.append({"scope": "pooled_oos_rank", "model": name,
                         "train_years": "expanding", "valid_year": -1, **rank_row})
        # (C) 원점수 풀링 — 결함이 있으나 투명성을 위해 함께 공개
        agg_rows.append({"scope": "pooled_oos_raw", "model": name,
                         "train_years": "expanding", "valid_year": -1,
                         **_metric_row(pooled["y_true"], pooled[col], k_pct)})
    metrics = pd.concat([fold_df, pd.DataFrame(agg_rows)], ignore_index=True)
    metrics.to_csv(out_dir / "walkforward_metrics.csv", index=False, encoding="utf-8-sig")

    bias = _pool_bias(pooled, k_pct)
    bias.to_csv(out_dir / "walkforward_pool_bias.csv", index=False, encoding="utf-8-sig")
    worst = bias.loc[bias["bias_ratio"].sub(1.0).abs().idxmax()] if not bias.empty else None
    if worst is not None:
        log.info("원점수 풀링 편향 진단(참고): 최대 이탈 %s fold=%d 편향배수=%.2f "
                 "(1.0이 중립) — 주 지표는 macro_avg_folds 사용",
                 worst["model"], int(worst["fold_test_year"]), float(worst["bias_ratio"]))

    n_boot = int((cfg.get("evaluate") or {}).get("bootstrap_n", 200) or 200)
    ci = _paired_delta_ci(pooled, k_pct, n_boot, int(cfg["seed"]))
    # 빈 결과여도 항상 새로 기록 — 이전 실행의 낡은 파일이 남아 현재 트랙 결과로
    # 오인되는 것을 방지 (리뷰 지적 반영)
    ci_cols = ["comparison", "mean_delta_lift", "ci_lo", "ci_hi", "p_win", "n_boot",
               "metric", "bootstrap_unit"]
    (ci if not ci.empty else pd.DataFrame(columns=ci_cols)) \
        .to_csv(out_dir / "walkforward_delta_ci.csv", index=False, encoding="utf-8-sig")

    show = ["lift_at_10", "precision_at_10", "pr_auc", "roc_auc", "brier", "n", "base_rate"]
    print()
    print("=== 워크포워드 확장창 백테스트 — 주 지표: fold 평균(macro_avg_folds) ===")
    print(metrics[metrics["scope"] == "macro_avg_folds"].set_index("model")[show]
          .round(3).to_string())
    print()
    print("--- 참고 ① fold 내 백분위 정규화 후 풀링 (pooled_oos_rank) ---")
    print(metrics[metrics["scope"] == "pooled_oos_rank"].set_index("model")["lift_at_10"]
          .round(3).to_string())
    print("--- 참고 ② 원점수 풀링 (pooled_oos_raw) — fold별 확률 스케일 차이로 왜곡됨. "
          "진단: outputs/walkforward_pool_bias.csv ---")
    print(metrics[metrics["scope"] == "pooled_oos_raw"].set_index("model")["lift_at_10"]
          .round(3).to_string())
    if not ci.empty:
        print()
        print("--- 기준모형 대비 fold평균 Lift@10% 차이 (연도블록 부트스트랩 95% CI) ---")
        print(ci[["comparison", "mean_delta_lift", "ci_lo", "ci_hi", "p_win"]].round(3)
              .to_string(index=False))
    print()
    log.info("워크포워드 완료: fold %d개, 풀링 OOS n=%d (양성 %d)",
             pooled["fold_test_year"].nunique(), len(pooled), int(pooled["y_true"].sum()))
    return metrics


if __name__ == "__main__":
    run_walkforward(load_config())
