"""M3 - 모델 학습: 기준모형 3종(persistence/single/logistic) + LightGBM 주모형.

계약(docs/INTERFACES.md §3):
- 표본 = features x labels (inner join, on brand_id+year)
- 완전 시간분할(auto_last_two): 라벨 최대 연도 T → test={T}, valid={T-1}, train={≤T-2}
  결정 연도는 outputs/split_years.json에 기록.
- 기준모형 확률 컬럼명 고정: p_persistence / p_single / p_logistic. 주모형 p_lgbm.
- 저장물:
    data/processed/predictions.parquet
        [brand_id, year, split, y_true, p_lgbm, p_persistence, p_single,
         p_logistic, is_new_brand]
    data/processed/features_matrix.parquet  (brand_id/year/split + 수치 피처 X
        - evaluate.py가 SHAP·ablation에 사용. 피처명은 ASCII-safe로 통일)
    data/processed/labels_joined.parquet    (표본에 조인된 라벨 - ev_* 포함,
        evaluate.py의 label_decomposition용)
    outputs/model_lgbm.txt (booster) / outputs/model_logistic.joblib
    outputs/split_years.json (분할 연도 + 피처명 매핑 + 표본 통계)

정직성 가정(로그에도 기록):
- persistence 기준모형: healthy gate 표본에는 t년 발동 사건이 없으므로,
  t년 사건 지표 3종의 업종내 rank pct를 '심도'로 통일(악화 방향=1)하여
  score = 0.5·(발동수/가용지표수) + 0.5·(평균 심도),
  발동 = 심도 ≥ (1 − 해당 사건 quantile) [config label.events 기준, 기본 0.8].
- 과튜닝 금지: 그리드서치 없음. config 하이퍼파라미터 그대로, valid early stopping만.
- LightGBM 피처명은 ASCII-safe로 치환(한글 더미 등 대비). 매핑은 split_years.json의
  feature_name_map(safe→원본)에 보존.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import get_logger, load_config, set_seed

log = get_logger("model")

PREDICTION_COLUMNS = [
    "brand_id", "year", "split", "y_true",
    "p_lgbm", "p_persistence", "p_single", "p_logistic", "is_new_brand",
]

# (사건명, 지표 컬럼 탐색 부분문자열 - 우선순위 순, 악화 방향)
# 방향 low = 지표값이 낮을수록 악화(성장률류), high = 높을수록 악화(계약종료율)
_EVENT_SPECS = [
    ("store", ["store_growth"], "low"),
    ("sales", ["real_sales_growth", "sales_growth"], "low"),
    ("end", ["contract_end"], "high"),
]


def _sanitize_feature_names(cols) -> tuple[list[str], dict[str, str]]:
    """피처명을 ASCII-safe(영숫자+_)로 치환. 비ASCII 문자는 uXXXX 코드포인트로
    대체해 유일성을 보존하고, 충돌 시 _2, _3 접미사로 해소.

    반환: (safe 이름 리스트, {safe: 원본} 매핑)
    """
    safe_names: list[str] = []
    mapping: dict[str, str] = {}
    seen: set[str] = set()
    for c in cols:
        c = str(c)
        s = "".join(
            ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else f"u{ord(ch):04X}"
            for ch in c
        )
        base, k = s, 2
        while s in seen:
            s = f"{base}_{k}"
            k += 1
        seen.add(s)
        safe_names.append(s)
        mapping[s] = c
    return safe_names, mapping


def _numeric_feature_frame(df: pd.DataFrame, logger) -> pd.DataFrame:
    """f_* 컬럼만 골라 float64 프레임으로. 수치 변환이 전혀 안 되는 컬럼은 제외."""
    feat_cols = [c for c in df.columns if str(c).startswith("f_")]
    if not feat_cols:
        raise ValueError("f_* 피처 컬럼이 없습니다 (features 스키마 확인).")
    keep: dict[str, pd.Series] = {}
    dropped: list[str] = []
    for c in feat_cols:
        s = df[c]
        if s.dtype == bool or pd.api.types.is_numeric_dtype(s):
            keep[c] = s.astype("float64")
        else:
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().any():
                keep[c] = coerced.astype("float64")
            else:
                dropped.append(c)
    if dropped:
        logger.warning("수치 변환 불가 피처 제외: %s", dropped)
    return pd.DataFrame(keep, index=df.index)


def _pick_metric_col(feat_cols: list[str], subs: list[str], prefix: str) -> str | None:
    """prefix로 시작하고 subs 중 하나를 포함하는 첫 컬럼(우선순위: subs 순서→사전순)."""
    cands = [c for c in feat_cols if c.startswith(prefix)]
    for s in subs:
        hit = sorted(c for c in cands if s in c)
        if hit:
            return hit[0]
    return None


def _baseline_scores(df: pd.DataFrame, feat_cols: list[str], cfg: dict):
    """p_persistence / p_single 계산 (0~1).

    - 각 사건 지표의 '심도'를 업종내 rank pct 방향 통일로 정의(악화=1).
      f_ind_* rank 피처가 있으면 그대로 사용(계약: 업종그룹×연도 내 분위수),
      없으면 f_chg_*/f_* 원지표를 연도 내 rank pct로 재계산(연도 횡단면 - 시점누출 아님).
    - p_persistence = 0.5·(발동수/가용지표수) + 0.5·(평균 심도),
      발동 임계 = config label.events quantile 기준(기본 심도 0.8 = 상하위 20%).
    - p_single = 계약종료율 심도(= 업종내 rank pct) 단일변수.
    """
    ev_cfg = (cfg.get("label") or {}).get("events") or {}
    thr_map = {
        "store": 1.0 - float((ev_cfg.get("store_net_decline") or {}).get("quantile", 0.20)),
        "sales": 1.0 - float((ev_cfg.get("sales_decline") or {}).get("quantile", 0.20)),
        "end": float((ev_cfg.get("contract_end_spike") or {}).get("quantile", 0.80)),
    }
    sev: dict[str, pd.Series] = {}
    for name, subs, direction in _EVENT_SPECS:
        col = None
        for prefix in ("f_ind_", "f_chg_", "f_"):
            col = _pick_metric_col(feat_cols, subs, prefix)
            if col is not None:
                break
        if col is None:
            log.warning("기준모형: '%s' 사건 지표 컬럼 미발견 - 해당 심도 제외", name)
            continue
        if col.startswith("f_ind_"):
            r = df[col].clip(0.0, 1.0)  # 이미 rank pct 가정 (계약 §2)
            src = f"{col} (rank pct 그대로)"
        else:
            r = df.groupby("year")[col].rank(pct=True)
            src = f"{col} (연도내 rank pct 재계산)"
        sev[name] = (1.0 - r) if direction == "low" else r
        log.info("baseline 심도[%s] <- %s", name, src)
    if not sev:
        raise ValueError("기준모형 계산용 사건 지표 피처를 하나도 찾지 못했습니다.")
    sev_df = pd.DataFrame(sev, index=df.index).clip(0.0, 1.0)
    filled = sev_df.fillna(0.5)  # 결측 심도는 중립(0.5) 처리 - 가정 기록
    thr = pd.Series({k: thr_map[k] for k in sev_df.columns})
    fired = filled.ge(thr, axis=1)
    n_avail = len(sev_df.columns)
    p_persistence = (0.5 * fired.sum(axis=1) / n_avail + 0.5 * filled.mean(axis=1)).clip(0.0, 1.0)
    if "end" in filled.columns:
        p_single = filled["end"].clip(0.0, 1.0)
    else:
        log.warning("p_single: contract_end 지표 미발견 - persistence 점수로 대체 (한계 기록)")
        p_single = p_persistence.copy()
    log.info(
        "baseline 정의: p_persistence = 0.5*(발동수/%d) + 0.5*평균심도, 발동임계=%s",
        n_avail, {k: round(v, 2) for k, v in thr_map.items() if k in sev_df.columns},
    )
    return p_persistence.to_numpy(), p_single.to_numpy()


def fit_lgbm_valid_selected(X: pd.DataFrame, y: np.ndarray, tr, va, cfg: dict, logger):
    """LightGBM 학습 — 사전 선언된 복잡도 후보 중 **valid AUC로만** 선택.

    명세 §4.1 "과튜닝 금지" 준수 방식:
      · 후보는 config.model.complexity_candidates 에 사전 고정(3개). 그리드서치 아님.
      · 선택 기준은 valid 분할의 AUC. **test 분할은 선택에 일절 사용하지 않는다.**
      · 조기종료도 valid로만, first_metric_only=True (auc 기준).
    반환: (선택된 모델, best_iteration, 선택 로그 문자열)
    """
    base = dict((cfg.get("model") or {}).get("lightgbm") or {})
    base.setdefault("random_state", int(cfg["seed"]))
    esr = int((cfg.get("model") or {}).get("early_stopping_rounds", 50))
    cands = (cfg.get("model") or {}).get("complexity_candidates") or [{}]
    eval_metric = "auc" if len(np.unique(y[va])) > 1 else "binary_logloss"
    if eval_metric != "auc":
        logger.warning("valid 단일 클래스 - early stopping 지표를 binary_logloss로 전환")

    results = []
    for cand in cands:
        params = {**base, **(cand or {})}
        m = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # lightgbm 4.7 eval_set deprecation (동작 동일)
            m.fit(X.loc[tr], y[tr], eval_set=[(X.loc[va], y[va])], eval_metric=eval_metric,
                  callbacks=[lgb.early_stopping(esr, first_metric_only=True, verbose=False),
                             lgb.log_evaluation(period=0)])
        bi = int(m.best_iteration_ or params.get("n_estimators", 0))
        if len(np.unique(y[va])) > 1:
            score = float(roc_auc_score(y[va], m.predict_proba(X.loc[va])[:, 1]))
        else:
            score = float("-inf")
        results.append({"cand": cand or {}, "model": m, "best_iter": bi, "valid_auc": score})

    best = max(results, key=lambda r: r["valid_auc"])
    note = " | ".join(f"{r['cand']}→valid_auc={r['valid_auc']:.4f}(it={r['best_iter']})"
                      for r in results)
    logger.info("LightGBM 복잡도 선택(valid AUC 기준, test 미사용): %s", note)
    logger.info("LightGBM 선택: %s, best_iteration=%d, valid_auc=%.4f",
                best["cand"], best["best_iter"], best["valid_auc"])
    return best["model"], best["best_iter"], note, dict(best["cand"])


def train_all(features: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> dict:
    """기준모형 3종 + LightGBM 학습, 예측·아티팩트 저장. 계약 §3 참조."""
    set_seed(cfg["seed"])
    out_dir = Path(cfg["paths"]["outputs"])
    proc_dir = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # --- 조인 표본 -----------------------------------------------------------
    feats = features.drop_duplicates(subset=["brand_id", "year"]).copy()
    if len(feats) < len(features):
        log.warning("features 중복 (brand_id,year) %d행 제거", len(features) - len(feats))
    lab = labels.dropna(subset=["label"]).drop_duplicates(subset=["brand_id", "year"]).copy()
    if len(lab) < len(labels):
        log.info("labels: label NaN/중복 %d행 제외", len(labels) - len(lab))
    lab["label"] = lab["label"].astype(int)
    overlap = (set(feats.columns) & set(lab.columns)) - {"brand_id", "year"}
    if overlap:
        log.warning("features/labels 컬럼 충돌 %s - features 쪽 제거", sorted(overlap))
        feats = feats.drop(columns=sorted(overlap))
    df = feats.merge(lab, on=["brand_id", "year"], how="inner")
    if df.empty:
        raise ValueError("features x labels 조인 결과가 비었습니다.")
    df = df.sort_values(["brand_id", "year"]).reset_index(drop=True)
    log.info("조인 표본 n=%d (features %d, labels %d), 양성률=%.3f",
             len(df), len(feats), len(lab), df["label"].mean())

    # --- 완전 시간분할 (auto_last_two) --------------------------------------
    years = sorted(int(y) for y in df["year"].unique())
    if len(years) < 3:
        raise ValueError(f"시간분할에 라벨 연도 3개 이상 필요 (현재 {years})")
    test_year, valid_year = years[-1], years[-2]
    train_years = years[:-2]
    for key, val in (("test_year", test_year), ("valid_year", valid_year)):
        fixed = (cfg.get("split") or {}).get(key)
        if fixed is not None and int(fixed) != val:
            log.warning("config split.%s=%s 와 auto 결정 %s 불일치 - auto 사용", key, fixed, val)
    split = np.select(
        [df["year"].eq(test_year), df["year"].eq(valid_year)],
        ["test", "valid"], default="train",
    )
    df["split"] = split
    tr = df["split"].eq("train").to_numpy()
    va = df["split"].eq("valid").to_numpy()
    te = df["split"].eq("test").to_numpy()
    y = df["label"].to_numpy()
    log.info("시간분할: train=%s(n=%d) valid=%d(n=%d) test=%d(n=%d)",
             train_years, tr.sum(), valid_year, va.sum(), test_year, te.sum())

    # --- 피처 행렬 (ASCII-safe 이름 통일) ------------------------------------
    X_orig = _numeric_feature_frame(df, log)
    safe_names, name_map = _sanitize_feature_names(X_orig.columns)
    if any(s != o for s, o in zip(safe_names, X_orig.columns, strict=False)):
        log.info("피처명 %d개 ASCII-safe 치환 (매핑은 split_years.json)",
                 sum(s != o for s, o in zip(safe_names, X_orig.columns, strict=False)))
    X = X_orig.set_axis(safe_names, axis=1)

    # --- 기준모형 ① persistence ② single -------------------------------------
    p_pers, p_single = _baseline_scores(df, list(X_orig.columns), cfg)

    # --- 기준모형 ③ logistic (median impute + 표준화 파이프라인) --------------
    log_cfg = (cfg.get("model") or {}).get("logistic") or {}
    logistic_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=float(log_cfg.get("C", 1.0)),
            max_iter=int(log_cfg.get("max_iter", 2000)),
            class_weight=log_cfg.get("class_weight", "balanced"),
            random_state=int(cfg["seed"]),
        )),
    ])
    logistic_pipe.fit(X.loc[tr], y[tr])
    p_logit = logistic_pipe.predict_proba(X)[:, 1]

    # --- 주모형 LightGBM (valid 기반 복잡도 선택 + valid early stopping) -----
    clf, best_iter, sel_note, sel_cand = fit_lgbm_valid_selected(X, y, tr, va, cfg, log)
    p_lgbm = clf.predict_proba(X)[:, 1]

    # --- is_new_brand: train 연도에 등장하지 않은 brand_id --------------------
    train_brands = set(df.loc[tr, "brand_id"])
    is_new = ~df["brand_id"].isin(train_brands)

    # --- 저장 ---------------------------------------------------------------
    preds = pd.DataFrame({
        "brand_id": df["brand_id"].astype(str).to_numpy(),
        "year": df["year"].astype(int).to_numpy(),
        "split": df["split"].to_numpy(),
        "y_true": y.astype(int),
        "p_lgbm": np.clip(p_lgbm, 0.0, 1.0),
        "p_persistence": np.clip(p_pers, 0.0, 1.0),
        "p_single": np.clip(p_single, 0.0, 1.0),
        "p_logistic": np.clip(p_logit, 0.0, 1.0),
        "is_new_brand": is_new.to_numpy(),
    })[PREDICTION_COLUMNS]
    pred_path = proc_dir / "predictions.parquet"
    preds.to_parquet(pred_path, index=False)

    fm = pd.concat(
        [df[["brand_id", "year", "split"]].reset_index(drop=True),
         X.reset_index(drop=True)],
        axis=1,
    )
    fm_path = proc_dir / "features_matrix.parquet"
    fm.to_parquet(fm_path, index=False)

    label_cols = [c for c in lab.columns if c not in ("brand_id", "year") and c in df.columns]
    lj = df[["brand_id", "year", "split", *label_cols]].copy()
    lj_path = proc_dir / "labels_joined.parquet"
    lj.to_parquet(lj_path, index=False)

    lgbm_path = out_dir / "model_lgbm.txt"
    # LightGBM C++ 파일 API는 Windows 비ASCII 경로에서 실패 → Python I/O로 저장(UTF-8)
    with open(lgbm_path, "w", encoding="utf-8") as f:
        f.write(clf.booster_.model_to_string())
    logit_path = out_dir / "model_logistic.joblib"
    joblib.dump(logistic_pipe, logit_path)

    split_info = {
        "mode": (cfg.get("split") or {}).get("mode", "auto_last_two"),
        "train_years": train_years,
        "valid_year": valid_year,
        "test_year": test_year,
        "n_train": int(tr.sum()),
        "n_valid": int(va.sum()),
        "n_test": int(te.sum()),
        "base_rate": {
            "train": float(y[tr].mean()) if tr.sum() else None,
            "valid": float(y[va].mean()) if va.sum() else None,
            "test": float(y[te].mean()) if te.sum() else None,
        },
        "n_features": len(safe_names),
        "eval_metric": "auc",                      # 조기종료·복잡도 선택 모두 valid AUC 기준
        "best_iteration": best_iter,
        "complexity_selection": sel_note,          # valid AUC 기반 후보 비교 로그 (test 미사용)
        "selected_complexity": sel_cand,           # 선택된 후보 (ablation 등 재학습이 동일 사용)
        "feature_name_map": name_map,  # safe -> 원본 피처명
    }
    sy_path = out_dir / "split_years.json"
    with open(sy_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    log.info("저장 완료: %s, %s, %s, %s, %s, %s",
             pred_path, fm_path, lj_path, lgbm_path, logit_path, sy_path)

    return {
        "predictions": preds,
        "split_years": split_info,
        "feature_names": safe_names,
        "paths": {
            "predictions": str(pred_path),
            "features_matrix": str(fm_path),
            "labels_joined": str(lj_path),
            "model_lgbm": str(lgbm_path),
            "model_logistic": str(logit_path),
            "split_years": str(sy_path),
        },
    }


if __name__ == "__main__":
    cfg = load_config()
    proc = Path(cfg["paths"]["processed"])
    fpath, lpath = proc / "features.parquet", proc / "labels.parquet"
    if not (fpath.exists() and lpath.exists()):
        log.error("입력 없음: %s / %s - 선행 스텝(features·labels)을 먼저 실행하세요.", fpath, lpath)
        raise SystemExit(1)
    train_all(pd.read_parquet(fpath), pd.read_parquet(lpath), cfg)
