"""M3 - 평가: 지표·보정·SHAP·라벨 분해·신규브랜드 분리·ablation.

계약(docs/INTERFACES.md §3). 입력은 model.train_all이 저장한 파일만 사용(재학습 최소화):
- data/processed/predictions.parquet  (+ 여기서 p_calibrated 컬럼을 추가 저장)
- data/processed/features_matrix.parquet / labels_joined.parquet
- outputs/model_lgbm.txt / outputs/split_years.json

산출물(전부 cfg['paths']['outputs'] - 호출 시점에 경로 해석):
- metrics.csv           (model × split) Lift@10%·Precision@10%·PR-AUC·ROC-AUC·Brier·n·base_rate
- calibration.png/.csv  valid로 isotonic(표본<calibration_min_samples면 sigmoid) 학습 → test 곡선
- calibrator.joblib     {"method", "model", "fitted_on": "valid", ...}
- shap_summary.png / shap_values.parquet (test 행별 상위 |SHAP| 10개, long 포맷)
- label_decomposition.csv  사건유형(ev_*)별 recall@10%
- newbrand_split.csv    기존 vs 신규 브랜드(test) 성능 분리
- ablation.csv          피처그룹 누적(lvl→+chg→+trd→+ind(+struct)) test Lift@10%

정직성:
- 보정기는 valid에서만 학습, test에는 적용만 (시점누출 없음).
- ablation 재학습도 train 학습·valid early stopping·test 평가 (동일 시간분할).
- Lift@10% = Precision@10% / base_rate, k = max(1, floor(n×k_pct)) (소표본 가드).
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # 반드시 pyplot import 전에 (Windows/headless)
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common import get_logger, load_config, set_seed

log = get_logger("evaluate")

MODEL_COLS = {
    "persistence": "p_persistence",
    "single": "p_single",
    "logistic": "p_logistic",
    "lgbm": "p_lgbm",
}
METRIC_ORDER = ["lift_at_10", "precision_at_10", "pr_auc", "roc_auc", "brier", "n", "base_rate"]


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def _metric_row(y, p, k_pct: float) -> dict:
    """Lift@k·Precision@k·PR-AUC·ROC-AUC·Brier·n·base_rate. 소표본 가드 k=max(1,floor(n·k_pct))."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    if (~ok).any():
        log.warning("점수 NaN %d행 제외 후 지표 계산", int((~ok).sum()))
    y, p = y[ok], p[ok]
    out = {"lift_at_10": np.nan, "precision_at_10": np.nan, "pr_auc": np.nan,
           "roc_auc": np.nan, "brier": np.nan, "n": len(y), "base_rate": np.nan}
    if len(y) == 0:
        return out
    base = float(y.mean())
    out["base_rate"] = base
    k = max(1, math.floor(len(y) * k_pct))
    order = np.argsort(-p, kind="stable")
    prec = float(y[order[:k]].mean())
    out["precision_at_10"] = prec
    out["lift_at_10"] = prec / base if base > 0 else np.nan
    out["brier"] = float(np.mean((p - y) ** 2))
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    return out


def _bootstrap_ci(y, p, k_pct: float, n_boot: int, seed: int,
                  alpha: float = 0.05) -> dict:
    """지표별 bootstrap 백분위 신뢰구간 (COULD 항목).

    test 표본을 복원추출로 n_boot회 재표집해 각 지표의 분포를 얻는다.
    기준모형 대비 우위가 표본 변동 범위 안에서도 유지되는지 판단하는 근거.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    keys = ["lift_at_10", "precision_at_10", "pr_auc", "roc_auc", "brier"]
    if len(y) == 0:
        return {f"{k}_{s}": np.nan for k in keys for s in ("lo", "hi")}
    rng = np.random.default_rng(seed)
    n = len(y)
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        row = _metric_row(y[idx], p[idx], k_pct)
        for k in keys:
            v = row[k]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                draws[k].append(float(v))
    out: dict[str, float] = {}
    for k in keys:
        arr = np.asarray(draws[k], dtype=float)
        if arr.size == 0:
            out[f"{k}_lo"] = out[f"{k}_hi"] = np.nan
        else:
            out[f"{k}_lo"] = float(np.quantile(arr, alpha / 2))
            out[f"{k}_hi"] = float(np.quantile(arr, 1 - alpha / 2))
    out["n_boot"] = int(n_boot)
    return out


def _paired_bootstrap(te: pd.DataFrame, k_pct: float, n_boot: int, seed: int,
                      out_dir: Path) -> None:
    """페어드 부트스트랩: 같은 재표집 표본에서 LightGBM과 각 기준모형의 Lift@10% 차이 분포.

    "LightGBM이 기준모형을 이긴다"는 주장이 표본 변동을 넘어서는지 직접 검정한다
    (동일 표본에서 두 모형을 비교하므로 표본 변동이 상쇄되어 검정력이 높다).
    산출: outputs/lift_delta_bootstrap.csv — 차이의 95% CI와 승률 P(delta>0).
    """
    y = te["y_true"].to_numpy(dtype=float)
    if len(y) == 0:
        return
    baselines = [c for c in ("persistence", "single", "logistic") if MODEL_COLS.get(c) in te.columns]
    p_main = te[MODEL_COLS["lgbm"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas: dict[str, list[float]] = {b: [] for b in baselines}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        main = _metric_row(yb, p_main[idx], k_pct)["lift_at_10"]
        if main is None or (isinstance(main, float) and math.isnan(main)):
            continue
        for b in baselines:
            pb = te[MODEL_COLS[b]].to_numpy(dtype=float)[idx]
            base_lift = _metric_row(yb, pb, k_pct)["lift_at_10"]
            if base_lift is not None and not (isinstance(base_lift, float) and math.isnan(base_lift)):
                deltas[b].append(float(main - base_lift))
    rows = []
    for b in baselines:
        arr = np.asarray(deltas[b], dtype=float)
        if arr.size == 0:
            continue
        rows.append({
            "comparison": f"lgbm - {b}",
            "mean_delta_lift": float(arr.mean()),
            "ci_lo": float(np.quantile(arr, 0.025)),
            "ci_hi": float(np.quantile(arr, 0.975)),
            "p_win": float((arr > 0).mean()),   # 재표집에서 LightGBM이 이긴 비율
            "n_boot": int(arr.size),
        })
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "lift_delta_bootstrap.csv",
                                  index=False, encoding="utf-8-sig")
        log.info("lift_delta_bootstrap.csv 저장 — 기준모형 대비 Lift 차이 CI·승률")


# ---------------------------------------------------------------------------
# 보정
# ---------------------------------------------------------------------------

def _fit_calibrator(p: np.ndarray, y: np.ndarray, method: str):
    if method == "isotonic":
        m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        m.fit(p, y)
        return m
    # sigmoid(Platt): 원확률 단일변수 로지스틱 (단조 변환 - 방법 로그에 기록)
    m = LogisticRegression(C=1e6, max_iter=1000)
    m.fit(p.reshape(-1, 1), y)
    return m


def _apply_calibrator(m, p: np.ndarray, method: str, cfg: dict | None = None) -> np.ndarray:
    """보정확률 산출 + **확률 상·하한 적용**.

    왜 상·하한이 필요한가 (실측 결함):
        isotonic 은 계단함수라 양 끝 구간이 검증표본의 경험률로 **포화**된다. 실측에서
        보정확률이 정확히 1.000(3행)과 1.4e-08(하위 830행)까지 나왔다. 1년 악화확률
        "100%"·"7천만분의 1"은 수십 건의 검증 관측이 뒷받침할 수 있는 추정이 아니고,
        그대로 `EL = exposure × 악화확률 × LGD` 에 들어가 손실 추정까지 과신하게
        만든다. 여신 실무가 **PD(부도확률)** 에 하한(바젤 IRB 0.03%)과 상한을 두는 것과
        같은 이유로 준용한다 — 우리 산출물이 PD 라는 뜻은 아니다.

    ⚠️ 클립은 **단조 변환**이라 순위가 바뀌지 않는다 — Lift@k·Precision@k·PR-AUC·ROC-AUC
       같은 순위 기반 지표는 영향을 받지 않고, 과신만 억제된다(Brier·EL에만 반영).
    """
    if method == "identity" or m is None:
        out = np.clip(p, 0.0, 1.0)
    elif method == "isotonic":
        out = np.clip(m.predict(p), 0.0, 1.0)
    else:
        out = np.clip(m.predict_proba(p.reshape(-1, 1))[:, 1], 0.0, 1.0)
    ecfg = (cfg or {}).get("evaluate") or {}
    floor = float(ecfg.get("prob_floor", 0.0))
    cap = float(ecfg.get("prob_cap", 1.0))
    return np.clip(out, floor, cap)


def _reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    """분위수 빈 신뢰도표: bin, mean_pred, frac_pos, count."""
    if len(p) == 0:
        return pd.DataFrame(columns=["bin", "mean_pred", "frac_pos", "count"])
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.array([0.0, float(np.median(p)), 1.0]))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    d = pd.DataFrame({"y": y, "p": p, "bin": idx})
    g = d.groupby("bin")
    return pd.DataFrame({
        "bin": g.size().index.to_numpy(),
        "mean_pred": g["p"].mean().to_numpy(),
        "frac_pos": g["y"].mean().to_numpy(),
        "count": g.size().to_numpy(),
    })


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def evaluate_all(cfg: dict) -> None:
    """predictions.parquet 등 아티팩트를 읽어 계약 §3의 평가 산출물 전부 생성."""
    set_seed(cfg["seed"])
    out_dir = Path(cfg["paths"]["outputs"])
    proc_dir = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_cfg = cfg.get("evaluate") or {}
    k_pct = float(ev_cfg.get("k_pct", 0.10))

    pred_path = proc_dir / "predictions.parquet"
    preds = pd.read_parquet(pred_path)

    # --- 1) metrics.csv -----------------------------------------------------
    rows = []
    for split in ("train", "valid", "test"):
        sub = preds[preds["split"] == split]
        if sub.empty:
            continue
        for name, col in MODEL_COLS.items():
            rows.append({"model": name, "split": split,
                         **_metric_row(sub["y_true"], sub[col], k_pct)})
    metrics = pd.DataFrame(rows)[["model", "split", *METRIC_ORDER]]
    metrics.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    log.info("metrics.csv 저장 (%d행: %d모형 × 분할)", len(metrics), len(MODEL_COLS))

    # --- 2) calibration -----------------------------------------------------
    va = preds[preds["split"] == "valid"]
    min_n = int(ev_cfg.get("calibration_min_samples", 300))
    method = str(ev_cfg.get("calibration_method", "isotonic"))
    if len(va) < min_n:
        log.info("valid n=%d < %d - 보정 방법 sigmoid로 전환", len(va), min_n)
        method = "sigmoid"
    y_va = va["y_true"].to_numpy(dtype=float)
    p_va = va["p_lgbm"].to_numpy(dtype=float)
    if len(va) == 0 or len(np.unique(y_va)) < 2:
        log.warning("valid 비었거나 단일 클래스 - 보정 생략(identity, 한계 기록)")
        method, calibrator = "identity", None
    else:
        calibrator = _fit_calibrator(p_va, y_va, method)
    p_cal_all = _apply_calibrator(calibrator, preds["p_lgbm"].to_numpy(dtype=float), method, cfg)
    _ecfg = cfg.get("evaluate") or {}
    _floor, _cap = float(_ecfg.get("prob_floor", 0.0)), float(_ecfg.get("prob_cap", 1.0))
    log.info("확률 상·하한 적용: [%.4g, %.4g] — 하한 도달 %d행 / 상한 도달 %d행 "
             "(isotonic 양끝 포화 억제, 순위 불변)",
             _floor, _cap, int((p_cal_all <= _floor).sum()), int((p_cal_all >= _cap).sum()))
    joblib.dump(
        {"method": method, "model": calibrator, "fitted_on": "valid",
         "n_valid": len(va), "input_col": "p_lgbm"},
        out_dir / "calibrator.joblib",
    )
    # 동률 붕괴 방지 (적대적 리뷰 확정 결함 수정): isotonic 계단으로 보정확률이 소수의
    # 유일값으로 붕괴하면 상위 10% 경계·등급 컷이 대규모 동률 블록에 걸려 하류 순위가
    # 임의화된다. 원점수(p_lgbm) 순위 기반 미세 오프셋(≤1e-5)을 더해 순위 정보를 보존한다.
    # (isotonic은 p_lgbm에 단조이므로 보정 일관성 유지, 표시 확률에는 사실상 무영향)
    tie_break = preds["p_lgbm"].rank(method="first").to_numpy(dtype=float)
    tie_break = tie_break / max(len(tie_break), 1) * 1e-5
    preds["p_calibrated"] = np.clip(np.clip(p_cal_all, 0.0, 1.0) + tie_break, 0.0, 1.0)
    preds.to_parquet(pred_path, index=False)  # p_calibrated 추가 저장 (계약 §3)
    log.info("calibration: method=%s - valid(n=%d)로 학습, 전체에 적용·test로 곡선 (원점수 순위 tie-break 적용)",
             method, len(va))

    # 보정 점수 기준 지표도 metrics.csv에 추가 — 하류(포트폴리오·대시보드)가 실제로 쓰는
    # 점수의 성능을 보고 지표와 일치시킨다 (리뷰 결함 수정: 보고 Lift가 배포 점수로 달성 가능해야 함)
    cal_rows = []
    for split in ("train", "valid", "test"):
        sub = preds[preds["split"] == split]
        if sub.empty:
            continue
        cal_rows.append({"model": "lgbm_calibrated", "split": split,
                         **_metric_row(sub["y_true"], sub["p_calibrated"], k_pct)})

    # ⚠️ **공정 비교**: 기준모형에도 *완전히 동일한* 보정 절차를 적용한 행을 함께 낸다.
    #    (적대적 감사 확정 지적) LightGBM만 보정해 놓고 "Brier 최우수"라고 하면 불공정하다 —
    #    기준모형의 원점수는 애초에 확률로 해석되도록 만들어진 값이 아니기 때문이다.
    #    같은 valid로 같은 방법(isotonic/sigmoid)을 적합해 붙인 뒤 비교해야 주장이 성립한다.
    #    결과가 불리해지더라도 그대로 공개한다.
    for name, col in MODEL_COLS.items():
        if name == "lgbm":
            continue
        if calibrator is None or len(va) == 0 or len(np.unique(y_va)) < 2:
            continue
        cal_b = _fit_calibrator(va[col].to_numpy(dtype=float), y_va, method)
        p_b = _apply_calibrator(cal_b, preds[col].to_numpy(dtype=float), method, cfg)
        tb = preds[col].rank(method="first").to_numpy(dtype=float)
        p_b = np.clip(p_b + tb / max(len(tb), 1) * 1e-5, 0.0, 1.0)
        for split in ("train", "valid", "test"):
            m = (preds["split"] == split).to_numpy()
            if not m.any():
                continue
            cal_rows.append({"model": f"{name}_calibrated", "split": split,
                             **_metric_row(preds.loc[m, "y_true"], p_b[m], k_pct)})
    log.info("공정 비교: 기준모형 3종에도 동일 보정(%s, valid 적합) 적용해 비교 행 추가", method)

    metrics = pd.concat([metrics, pd.DataFrame(cal_rows)[["model", "split", *METRIC_ORDER]]],
                        ignore_index=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    te = preds[preds["split"] == "test"]
    n_bins = int(ev_cfg.get("n_bins_calibration", 10))
    cal_tabs = []
    for series, col in (("raw", "p_lgbm"), ("calibrated", "p_calibrated")):
        t = _reliability_table(te["y_true"].to_numpy(dtype=float),
                               te[col].to_numpy(dtype=float), n_bins)
        t.insert(0, "series", series)
        cal_tabs.append(t)
    cal_tab = pd.concat(cal_tabs, ignore_index=True) if cal_tabs else \
        pd.DataFrame(columns=["series", "bin", "mean_pred", "frac_pos", "count"])
    cal_tab.to_csv(out_dir / "calibration.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Perfectly calibrated")
    for series, marker in (("raw", "o"), ("calibrated", "s")):
        t = cal_tab[cal_tab["series"] == series]
        if not t.empty:
            ax.plot(t["mean_pred"], t["frac_pos"], marker=marker, label=f"LightGBM ({series})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(f"Calibration on test (fit on valid, method={method})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=150)
    plt.close(fig)

    # --- 3) SHAP ------------------------------------------------------------
    fm = pd.read_parquet(proc_dir / "features_matrix.parquet")
    # LightGBM C++ 파일 API는 Windows 비ASCII 경로에서 실패 → Python I/O로 로드(UTF-8)
    with open(out_dir / "model_lgbm.txt", encoding="utf-8") as f:
        booster = lgb.Booster(model_str=f.read())
    feat_names = booster.feature_name()
    name_map: dict[str, str] = {}
    sy_path = out_dir / "split_years.json"
    if sy_path.exists():
        with open(sy_path, encoding="utf-8") as f:
            name_map = json.load(f).get("feature_name_map") or {}
    fm_te = fm[fm["split"] == "test"].reset_index(drop=True)
    missing = [c for c in feat_names if c not in fm_te.columns]
    if missing:
        raise ValueError(f"features_matrix에 부재한 모델 피처: {missing}")
    X_te = fm_te[feat_names].astype(float)
    if len(X_te) == 0:
        log.warning("test 행 없음 - SHAP 산출물은 빈 스키마로 저장")
        pd.DataFrame(columns=["brand_id", "year", "feature", "shap_value", "feature_value"]) \
            .to_parquet(out_dir / "shap_values.parquet", index=False)
    else:
        explainer = shap.TreeExplainer(booster)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # shap의 LightGBM 출력형식 변경 UserWarning 억제
            sv = explainer.shap_values(X_te)
        if isinstance(sv, list):
            sv = sv[-1]  # 이진분류: 양성 클래스 기여
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., -1]
        top_k = min(10, sv.shape[1])  # 행당 상위 |SHAP| 10개로 용량 제한
        order = np.argsort(-np.abs(sv), axis=1)[:, :top_k]
        top_sv = np.take_along_axis(sv, order, axis=1)
        top_val = np.take_along_axis(X_te.to_numpy(), order, axis=1)
        feat_arr = np.asarray(feat_names, dtype=object)[order]
        shap_long = pd.DataFrame({
            "brand_id": np.repeat(fm_te["brand_id"].to_numpy(), top_k),
            "year": np.repeat(fm_te["year"].to_numpy(), top_k),
            "feature": [name_map.get(str(fn), str(fn)) for fn in feat_arr.ravel()],
            "shap_value": top_sv.ravel().astype(float),
            "feature_value": top_val.ravel().astype(float),
        })
        shap_long.to_parquet(out_dir / "shap_values.parquet", index=False)

        plt.figure(figsize=(9, 6))
        # 그림 라벨은 ASCII-safe 피처명 사용 (한글 폰트 이슈 회피)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # shap 내부 numpy 전역 RNG FutureWarning 억제
            shap.summary_plot(sv, X_te, feature_names=feat_names, max_display=15,
                              show=False, plot_size=None)
        plt.gcf().suptitle("SHAP summary (test split)", fontsize=11)
        plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("SHAP 저장: shap_values.parquet(%d행), shap_summary.png", len(shap_long))

    # --- 4) label_decomposition.csv ----------------------------------------
    ldc_path = out_dir / "label_decomposition.csv"
    lab = None
    for cand in (proc_dir / "labels_joined.parquet", proc_dir / "labels.parquet"):
        if cand.exists():
            lab = pd.read_parquet(cand)
            log.info("label_decomposition 라벨 소스: %s", cand)
            break
    if lab is None:
        log.warning("labels_joined/labels.parquet 없음 - 사건유형 분해 생략 (한계 기록)")
        pd.DataFrame([{"event_type": "unavailable",
                       "note": "labels parquet not found; decomposition skipped"}]) \
            .to_csv(ldc_path, index=False, encoding="utf-8-sig")
    else:
        lab_cols = ["brand_id", "year"] + [c for c in lab.columns if c.startswith("ev_")]
        m = te.merge(lab[lab_cols].drop_duplicates(subset=["brand_id", "year"]),
                     on=["brand_id", "year"], how="left")
        n = len(m)
        k = max(1, math.floor(n * k_pct)) if n else 0
        flag = np.zeros(n, dtype=bool)
        if n:
            order = np.argsort(-m["p_lgbm"].to_numpy(dtype=float), kind="stable")
            flag[order[:k]] = True
        pos = m["y_true"].to_numpy() == 1
        ld_rows = [{"event_type": "any", "n_pos": int(pos.sum()),
                    "n_caught": int((pos & flag).sum()),
                    "recall_at_10": float((pos & flag).sum() / pos.sum()) if pos.sum() else np.nan,
                    "n_test": n, "k": k}]
        for c in [c for c in lab.columns if c.startswith("ev_")]:
            emask = pos & (pd.to_numeric(m[c], errors="coerce").fillna(0).to_numpy() > 0)
            ld_rows.append({"event_type": c, "n_pos": int(emask.sum()),
                            "n_caught": int((emask & flag).sum()),
                            "recall_at_10": float((emask & flag).sum() / emask.sum()) if emask.sum() else np.nan,
                            "n_test": n, "k": k})
        pd.DataFrame(ld_rows).to_csv(ldc_path, index=False, encoding="utf-8-sig")

    # --- 5) newbrand_split.csv ---------------------------------------------
    nb_rows = []
    for flag_val in (False, True):
        sub = te[te["is_new_brand"] == flag_val]
        if sub.empty:
            continue
        for name, col in MODEL_COLS.items():
            nb_rows.append({"is_new_brand": flag_val, "model": name,
                            **_metric_row(sub["y_true"], sub[col], k_pct)})
    nb = pd.DataFrame(nb_rows, columns=["is_new_brand", "model", *METRIC_ORDER])
    nb.to_csv(out_dir / "newbrand_split.csv", index=False, encoding="utf-8-sig")

    # --- 6) ablation.csv (피처그룹 누적 재학습 - 동일 시간분할) ---------------
    mm = fm.merge(preds[["brand_id", "year", "y_true"]], on=["brand_id", "year"], how="inner")
    fcols = [c for c in fm.columns if c.startswith("f_")]
    stages = [
        ("lvl", ["f_lvl_"]),
        ("lvl+chg", ["f_lvl_", "f_chg_"]),
        ("lvl+chg+trd", ["f_lvl_", "f_chg_", "f_trd_"]),
        ("lvl+chg+trd+ind+struct", ["f_lvl_", "f_chg_", "f_trd_", "f_ind_", "f_struct_"]),
        # 본부 재무(f_hq_)를 **마지막 단계로 분리**해야 이 층이 실제로 기여했는지가 보인다.
        # 함께 묶으면 "추가했는데 성능은 그대로"인 경우를 숨기게 된다.
        ("전체(+본부재무)", ["f_lvl_", "f_chg_", "f_trd_", "f_ind_", "f_struct_", "f_hq_"]),
    ]
    tr_m = (mm["split"] == "train").to_numpy()
    va_m = (mm["split"] == "valid").to_numpy()
    te_m = (mm["split"] == "test").to_numpy()
    y_all = mm["y_true"].to_numpy()
    ab_params = dict((cfg.get("model") or {}).get("lightgbm") or {})
    ab_params.setdefault("random_state", int(cfg["seed"]))
    # 본모델이 valid로 선택한 복잡도를 ablation 재학습에도 동일 적용 (리뷰 확정 결함 수정:
    # 기본 파라미터로 재학습하면 ablation이 출하 모델과 다른 모형을 비교하게 됨)
    sel_path = out_dir / "split_years.json"
    if sel_path.exists():
        try:
            _sel = json.loads(sel_path.read_text(encoding="utf-8")).get("selected_complexity")
            if isinstance(_sel, dict) and _sel:
                ab_params.update(_sel)
                log.info("ablation: 본모델 선택 복잡도 적용 %s", _sel)
        except (OSError, json.JSONDecodeError):
            pass
    ab_params["n_estimators"] = min(300, int(ab_params.get("n_estimators", 300)))  # 소형 재학습
    esr = int((cfg.get("model") or {}).get("early_stopping_rounds", 50))
    ab_metric = "auc" if len(np.unique(y_all[va_m])) > 1 else "binary_logloss"
    ab_rows, prev_cols = [], None
    for stage, prefs in stages:
        cols = [c for c in fcols if any(c.startswith(p) for p in prefs)]
        if not cols or cols == prev_cols:
            log.info("ablation 단계 '%s' 스킵 (신규 피처 없음)", stage)
            continue
        prev_cols = cols
        ab_clf = LGBMClassifier(**ab_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # lightgbm eval_set deprecation 경고 국소 억제
            ab_clf.fit(
                mm.loc[tr_m, cols], y_all[tr_m],
                eval_set=[(mm.loc[va_m, cols], y_all[va_m])],
                eval_metric=ab_metric,
                # first_metric_only=True: 본모델과 동일하게 auc 기준 조기종료 (리뷰 확정 결함 수정)
                callbacks=[lgb.early_stopping(esr, first_metric_only=True, verbose=False),
                           lgb.log_evaluation(period=0)],
            )
        p_te = ab_clf.predict_proba(mm.loc[te_m, cols])[:, 1]
        mrow = _metric_row(y_all[te_m], p_te, k_pct)
        ab_rows.append({"stage": stage, "n_features": len(cols),
                        "lift_at_10": mrow["lift_at_10"],
                        "precision_at_10": mrow["precision_at_10"],
                        "pr_auc": mrow["pr_auc"], "roc_auc": mrow["roc_auc"]})
    pd.DataFrame(ab_rows, columns=["stage", "n_features", "lift_at_10",
                                   "precision_at_10", "pr_auc", "roc_auc"]) \
        .to_csv(out_dir / "ablation.csv", index=False, encoding="utf-8-sig")
    log.info("ablation.csv 저장 (%d단계)", len(ab_rows))

    # --- 6b) bootstrap 신뢰구간 (COULD) -------------------------------------
    n_boot = int(ev_cfg.get("bootstrap_n", 0) or 0)
    if n_boot > 0:
        te_b = preds[preds["split"] == "test"]
        boot_rows = []
        cols = dict(MODEL_COLS)
        if "p_calibrated" in preds.columns:
            cols["lgbm_calibrated"] = "p_calibrated"
        for name, col in cols.items():
            row = _metric_row(te_b["y_true"], te_b[col], k_pct)
            ci = _bootstrap_ci(te_b["y_true"], te_b[col], k_pct, n_boot, int(cfg["seed"]))
            boot_rows.append({"model": name, "split": "test",
                              "lift_at_10": row["lift_at_10"],
                              "lift_at_10_lo": ci["lift_at_10_lo"], "lift_at_10_hi": ci["lift_at_10_hi"],
                              "precision_at_10": row["precision_at_10"],
                              "precision_at_10_lo": ci["precision_at_10_lo"],
                              "precision_at_10_hi": ci["precision_at_10_hi"],
                              "pr_auc": row["pr_auc"],
                              "pr_auc_lo": ci["pr_auc_lo"], "pr_auc_hi": ci["pr_auc_hi"],
                              "roc_auc": row["roc_auc"],
                              "roc_auc_lo": ci["roc_auc_lo"], "roc_auc_hi": ci["roc_auc_hi"],
                              "brier": row["brier"],
                              "brier_lo": ci["brier_lo"], "brier_hi": ci["brier_hi"],
                              "n_boot": n_boot})
        boot = pd.DataFrame(boot_rows)
        boot.to_csv(out_dir / "metrics_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
        log.info("metrics_bootstrap_ci.csv 저장 (%d모형 × %d회 재표집, 95%% 백분위 CI)",
                 len(boot_rows), n_boot)
        # 페어드 부트스트랩: 동일 재표집에서 LightGBM − 최강 기준모형 Lift 차이 분포
        _paired_bootstrap(te_b, k_pct, n_boot, int(cfg["seed"]), out_dir)

    # --- 7) 콘솔 비교표 ------------------------------------------------------
    tab = metrics[metrics["split"] == "test"].set_index("model") \
        .reindex(["persistence", "single", "logistic", "lgbm"])
    disp = tab[["lift_at_10", "precision_at_10", "pr_auc", "roc_auc", "brier"]].round(3)
    n_test = int(tab["n"].max()) if len(tab) else 0
    base = float(tab["base_rate"].max()) if len(tab) else float("nan")
    print()
    print("=== Baselines vs LightGBM (test split) ===")
    print(disp.to_string())
    print(f"(test n={n_test}, base_rate={base:.3f}, k=top {round(k_pct * 100)}%)")
    print()
    log.info("evaluate_all 완료 - 산출물 위치: %s", out_dir)


if __name__ == "__main__":
    evaluate_all(load_config())
