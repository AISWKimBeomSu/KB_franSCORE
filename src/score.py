"""M3.5 운영 점수 산출 (serving) — **아직 라벨이 없는 최신 코호트**를 점수화한다.

왜 이 모듈이 필요한가 (자체 감사 critical 지적):
    model.py 는 features × labels 를 inner join 하므로 **라벨이 확정된 과거 연도만** 점수를
    갖는다. 라벨은 t+1년 실적으로 만들어지고 그 실적은 t+2년 공시로 확정되므로, 정의상
    **가장 최근 연도에는 라벨이 없다** — 그런데 은행이 정작 알고 싶은 것은 바로 그 최신
    코호트다. 이 경로가 없으면 이 저장소는 예측기가 아니라 **백테스트 하네스**에 그친다.

    이 모듈은 학습을 하지 않는다. 학습이 남긴 산출물만 **읽어서** 점수를 낸다:
      outputs/model_lgbm.txt   (LightGBM booster)
      outputs/calibrator.joblib (valid에서 적합된 보정기 + 방법)
      outputs/split_years.json  (피처명 ASCII 치환 매핑 — 학습 때와 같은 이름으로 정렬)
    따라서 운영에서 재학습 없이 매 공시 주기마다 이 스텝만 돌리면 된다.

⛔ 시점 안전:
    점수 대상 행의 피처는 features.py 계약상 **t 이하 연도만** 사용한다(누출 테스트로 보증).
    보정기는 과거 valid에서 적합된 것을 **그대로 적용만** 한다(재적합 금지 — 재적합하면
    최신 코호트의 결과를 보고 보정하는 셈이라 실전에서 불가능한 정보를 쓰게 된다).

산출: outputs/scores_latest.csv  (심사역이 그대로 받아 쓰는 점검 큐)
      brand_id·brand_name·year·deterioration_1y·risk_grade·deterioration_rank_pct·n_stores·상위 위험요인 3종

실행: 프로젝트 루트에서 `python -m src.score`  또는  `python run_pipeline.py --step score`
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common import get_logger, load_config, set_seed
from src.diagnosis import smooth_calibrated
from src.evaluate import _apply_calibrator
from src.feature_names import feature_korean_name
from src.model import _numeric_feature_frame, _sanitize_feature_names

log = get_logger("score")

TOP_FACTORS = 3   # 심사역 화면·CSV에 싣는 위험요인 개수


def _load_artifacts(out_dir: Path) -> tuple[lgb.Booster, dict, dict]:
    """학습이 남긴 산출물 로드 (booster, 보정기 번들, 피처명 매핑)."""
    mp = out_dir / "model_lgbm.txt"
    if not mp.exists():
        raise FileNotFoundError(f"{mp} 없음 — 먼저 `--step model` 실행")
    booster = lgb.Booster(model_str=mp.read_text(encoding="utf-8"))

    cal_path = out_dir / "calibrator.joblib"
    cal = joblib.load(cal_path) if cal_path.exists() else {"method": "identity", "model": None}
    if not cal_path.exists():
        log.warning("calibrator.joblib 없음 — 원점수를 그대로 사용(보정 미적용)")

    name_map: dict[str, str] = {}
    sy = out_dir / "split_years.json"
    if sy.exists():
        name_map = json.loads(sy.read_text(encoding="utf-8")).get("feature_name_map") or {}
    return booster, cal, name_map


def _load_bands(cfg: dict) -> dict | None:
    """등급 밴드 정의. 없으면 None (호출부가 순위 등급으로 폴백)."""
    p = Path(cfg["paths"]["outputs"]) / "grade_bands.json"
    if not p.exists():
        return None
    try:
        b = json.loads(p.read_text(encoding="utf-8"))
        return b if b.get("cuts") and b.get("grades") else None
    except (OSError, ValueError) as exc:
        log.warning("grade_bands.json 읽기 실패: %s", exc)
        return None


def _load_deploy_calibrator(cfg: dict):
    p = Path(cfg["paths"]["outputs"]) / "calibrator_deploy.joblib"
    if not p.exists():
        return None
    try:
        return (joblib.load(p) or {}).get("model")
    except Exception as exc:
        log.warning("배포용 보정기 읽기 실패: %s", exc)
        return None


def _brand_state(cfg: dict, brand_ids: pd.Series,
                 target_year: int) -> tuple[pd.Series, pd.Series]:
    """점수 산출 연도의 (브랜드 상태, 악화사건 수).

    왜 이 표시가 필요한가 (실측 근거)
        라벨은 `healthy_gate_at_t=true` 로 만들어진다 — 학습·보정·평가 표본은
        **그 해에 악화사건이 하나도 없던 브랜드만**이다. 그런데 점수 산출은
        `eligible_t` 만 걸고 게이트를 걸지 않는다. 그 결과 2024 코호트 1,442개 중
        절반 가까이가 모델이 학습한 적 없는 상태이고, **주의 등급의 대부분이
        거기서 나온다**. 이들에 대한 성능 근거는 원리적으로 만들 수 없다 —
        labels.parquet 에 행 자체가 없어 어떤 백테스트도 불가능하다.

        ⚠️ 이전 판은 '직전 연도 라벨 표본에 있었는가'라는 프록시를 썼다. 그건
        t−1년 상태라 t년 상태를 **과소 보고**한다(2024: 50.9% vs 실제 61.9%).
        이제 `labels.brand_state()` 로 산출연도 상태를 직접 판정한다.

    사건 수를 함께 내보내는 이유
        요주의 구간에는 모형 판별력 근거가 없다. 대신 `outputs/watch_base_rates.csv`
        가 **사건 수별 다음 해 재발동 실현율**을 공표한다(1건 24.1% / 2건 45.9% /
        3건 64.8%, 건전 9.4%). 화면이 그 표를 이 브랜드에 적용하려면 사건 수가
        산출물에 있어야 한다. 등급 확률이 답이 아닌 구간에서 답을 주는 값이다.
    """
    idx = brand_ids.index
    panel_p = Path(cfg["paths"]["processed"]) / "panel.parquet"
    if not panel_p.exists():
        log.warning("panel.parquet 없음 — 브랜드 상태 표시를 생략합니다")
        return pd.Series("미상", index=idx), pd.Series(pd.NA, index=idx, dtype="Int64")
    from src.labels import brand_state
    st = brand_state(pd.read_parquet(panel_p), cfg)
    st = st[st["year"] == target_year].copy()
    st["brand_id"] = st["brand_id"].astype(str)
    st = st.drop_duplicates("brand_id").set_index("brand_id")
    key = brand_ids.astype(str)
    return (key.map(st["state"]).fillna("미상"),
            key.map(st["n_events"]).astype("Int64"))


def score_latest(cfg: dict, year: int | None = None) -> pd.DataFrame:
    """최신(또는 지정) 연도의 자격 브랜드를 점수화해 outputs/scores_latest.csv 로 저장."""
    set_seed(cfg["seed"])
    proc, out_dir = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["outputs"])
    booster, cal, name_map = _load_artifacts(out_dir)

    feats = pd.read_parquet(proc / "features.parquet")
    panel = pd.read_parquet(proc / "panel.parquet")
    keep = [c for c in ("brand_id", "year", "brand_name", "industry_major", "industry_mid",
                        "n_stores", "avg_sales", "eligible_t") if c in panel.columns]
    df = feats.merge(panel[keep], on=["brand_id", "year"], how="left")

    target_year = int(year if year is not None else df["year"].max())
    cohort = df[df["year"] == target_year].copy()
    # (아래 _in_model_population 이 이 target_year 를 기준으로 모집단 소속을 표시한다)
    if "eligible_t" in cohort.columns:
        n0 = len(cohort)
        cohort = cohort[cohort["eligible_t"].fillna(False).astype(bool)]
        log.info("자격 필터: %d → %d행 (점포 %s+ 누적 & %s년 연속 — 과거 정보만 사용)",
                 n0, len(cohort), cfg["sample"]["min_stores"],
                 cfg["sample"]["min_consecutive_years"])
    if cohort.empty:
        raise ValueError(f"{target_year}년 자격 브랜드가 없습니다 (features/panel 확인)")

    # 학습 때와 **완전히 같은** 피처 행렬 구성 (이름·순서·dtype)
    X_orig = _numeric_feature_frame(cohort, log)
    safe, _ = _sanitize_feature_names(X_orig.columns)
    X = X_orig.set_axis(safe, axis=1)
    trained = list(booster.feature_name())
    missing = [c for c in trained if c not in X.columns]
    extra = [c for c in X.columns if c not in trained]
    if missing:
        # 결측 피처를 0으로 채우면 조용히 틀린 점수가 나간다 — 명시적으로 실패시킨다.
        raise ValueError(f"학습 피처 {len(missing)}개가 현재 피처에 없습니다: {missing[:8]} "
                         f"— features 재생성 또는 모델 재학습이 필요합니다.")
    if extra:
        log.info("학습에 없던 피처 %d개는 제외: %s", len(extra), extra[:5])
    X = X[trained]

    p_raw = np.clip(booster.predict(X), 0.0, 1.0)
    # 배포용 보정기가 있으면 그것을 쓴다 — 등급 밴드가 그 척도 위에서 도출·검증됐기
    # 때문이다. 척도가 다르면 공표한 '등급별 실현 악화율'과 화면 확률이 어긋난다.
    # (기존 보정기는 valid 한 해 689행 적합, 배포용은 워크포워드 OOS 2,063행 적합.)
    bands = _load_bands(cfg)
    deploy_cal = _load_deploy_calibrator(cfg) if bands else None
    if deploy_cal is not None:
        p_cal_step = np.clip(deploy_cal.predict(p_raw), 0.0, 1.0)
        log.info("배포용 보정기 적용 (워크포워드 OOS 적합) — 예측평균 %.4f", p_cal_step.mean())
    else:
        p_cal_step = _apply_calibrator(
            cal.get("model"), p_raw, str(cal.get("method", "identity")), cfg)
    # isotonic 은 계단 함수라 같은 계단에 든 브랜드가 **화면에 같은 확률로** 표시된다
    # (실측: 2,510개 중 216개가 42.86%). 계단 내부를 원점수로 선형 보간해 편다.
    # 1e-5 짜리 미세 tie-break 는 소수점 첫째 자리에서 여전히 같은 값이라 소용이 없었다.
    p_cal = smooth_calibrated(p_raw, p_cal_step)
    # ⚠️ 보간은 계단 **양끝을 넘어설 수 있다**. 실측: 설정 하한 prob_floor(0.0003) 미만
    #    27행(정확히 0.0 이 1행). 확률 0.0 은 "절대 악화하지 않는다"는 뜻이라 어떤
    #    통계 모형도 주장할 수 없는 값이다 — 설정 한계 안으로 되돌린다.
    #
    #    상한은 **일부러 건드리지 않는다.** 최상위 계단의 수준값이 곧 보정기 최댓값
    #    (0.45283)이라, 평균을 보존하며 펴면 그 위로 나가는 것이 수학적으로 불가피하다.
    #    상한을 계단 최댓값으로 자르면 최상위 100개가 다시 한 값으로 뭉쳐(실측) 계단
    #    문제가 되살아난다. 초과폭은 최대 2.74%p 이며, 이 값은 '보정된 확률'이 아니라
    #    **계단 안에서의 표시 순서를 만든 값**이다 — 그 사실을 문서와 화면에 밝힌다.
    #    (계단 내부 순서의 판별력은 검증 결과 out-of-sample 로 유의하지 않다.)
    ecfg = cfg.get("evaluate") or {}
    lo, hi = float(ecfg.get("prob_floor", 0.0)), float(ecfg.get("prob_cap", 1.0))
    n_viol = int(((p_cal < lo - 1e-12) | (p_cal > hi + 1e-12)).sum())
    p_cal = np.clip(p_cal, lo, hi)
    n_before = int(pd.Series(p_cal_step).round(4).nunique())
    n_after = int(pd.Series(p_cal).round(4).nunique())
    log.info("보정확률 계단 보간: 서로 다른 값 %d → %d개 (소수점 4자리 기준, n=%d) "
             "· 범위 [%.4f, %.4f] 로 클립 (위반 %d행)",
             n_before, n_after, len(p_cal), lo, hi, n_viol)

    res = cohort[["brand_id", "year"]].copy()
    for c in ("brand_name", "industry_major", "industry_mid", "n_stores"):
        if c in cohort.columns:
            res[c] = cohort[c].to_numpy()
    res["deterioration_1y"] = p_cal
    res["score_raw"] = p_raw
    # 보정기가 실제로 산출한 계단값. deterioration_1y 는 이 값을 계단 안에서 편 결과이므로,
    # **감사 가능한 원본**을 함께 실어 둘을 대조할 수 있게 한다.
    res["deterioration_step"] = p_cal_step
    res["deterioration_rank_pct"] = res["deterioration_1y"].rank(method="first", pct=True)
    res["brand_state"], res["n_events_at_t"] = _brand_state(
        cfg, res["brand_id"], target_year)
    # 모델이 실제로 학습·검증한 모집단 = 건전 상태 브랜드
    res["in_model_population"] = res["brand_state"].eq("건전")

    # 등급 — 관측 악화율로 앵커링된 **절대 컷** (tools/derive_grade_bands.py 산출).
    #
    # 왜 순위 백분위를 버렸나: `deterioration_rank_pct >= 0.90 → High` 는 매 산출 시점에 항상
    # 정확히 10%를 주의로 만든다. 산업 전체가 나빠져도 등급 분포가 고정되고, 위험이
    # 전혀 변하지 않은 브랜드의 등급이 남의 변화 때문에 바뀐다(무변화 쌍 570개 중 19쌍 실측).
    # 무엇보다 등급별 확률이 정의되지 않아 어떤 검증도 성립하지 않는다.
    if bands:
        cuts = [float(c) for c in bands["cuts"]]
        # ⚠️ 등급은 **계단값(deterioration_step)** 으로 매긴다. 보간값(deterioration_1y)이 아니다.
        #
        #    컷은 tools/derive_grade_bands.py:218 이 `deploy.predict(...)` 출력, 즉
        #    보정기의 계단값 위에서 도출했다(grade_bands.json 의 anchored_on 도 같은 말).
        #    공표 실현율 FS1 2.22% / FS2 11.69% / FS3 30.22% 도 그 척도로 매긴 등급의
        #    실적이다. 그런데 예전 코드는 그 컷을 **보간값**에 적용했다 — 컷이 도출된
        #    적 없는 척도다.
        #
        #    결과(실측): 계단 0.159823 에 98개 브랜드가 앉아 있는데 컷이 0.16 이라,
        #    보간이 그중 48개를 컷 위로 밀어 올려 주의(FS3)로 만들었다. 그 브랜드들의
        #    **실제 추정 확률은 컷 아래**다. 게다가 계단 내부 순서는 이 저장소가 직접
        #    측정해 out-of-sample 판별력이 없다고 결론 낸 값이다 — 신호가 없는 양이
        #    등급 경계를 넘기고 있었다. 검증 산출물(이행행렬·등급별 이항검정)은 계단값
        #    으로 매기므로, 화면 등급과 검증 등급이 48개 브랜드에서 어긋나 있었다.
        idx = np.digitize(res["deterioration_step"].to_numpy(), cuts)
        res["grade"] = np.array(bands["grades"], dtype=object)[idx]
        # 하위 호환: 기존 화면·포트폴리오·큐가 쓰는 High/Medium/Low 를 함께 유지
        res["risk_grade"] = np.array(["Low", "Medium", "High"], dtype=object)[idx]
        # 화면에 쓰는 보간값이 자기 등급의 구간을 벗어나면, 사용자는 "16.4%인데 관찰"
        # 이라는 모순을 본다. 보간은 **계단 안의 표시 순서**를 만들려고 넣은 것이므로
        # 자기 구간 안에 머물러야 한다. 구간 안으로 되돌리고 순서는 그대로 둔다.
        edges = [0.0, *cuts, 1.0]
        lo_b = np.array(edges[:-1], dtype=float)[idx]
        hi_b = np.array(edges[1:], dtype=float)[idx]
        n_out = int(((res["deterioration_1y"] < lo_b) | (res["deterioration_1y"] >= hi_b)).sum())
        res["deterioration_1y"] = np.clip(res["deterioration_1y"], lo_b, np.nextafter(hi_b, 0.0))
        if n_out:
            log.info("보간값을 등급 구간 안으로 되돌림: %d행", n_out)
        res["grade_band"] = [
            f"{cuts[0] * 100:.1f}% 미만" if i == 0 else
            (f"{cuts[0] * 100:.1f}~{cuts[1] * 100:.1f}%" if i == 1
             else f"{cuts[1] * 100:.1f}% 이상") for i in idx]
        log.info("절대 등급 적용 (컷 %s): %s", cuts,
                 res["grade"].value_counts().to_dict())
    else:
        log.warning("grade_bands.json 없음 — 순위 백분위 등급으로 대체합니다 "
                    "(`python tools/derive_grade_bands.py` 실행 권장)")
        g = cfg["portfolio"]["risk_grades"]
        hi, mid = float(g["high"]), float(g["medium"])
        res["risk_grade"] = np.where(res["deterioration_rank_pct"] >= hi, "High",
                                     np.where(res["deterioration_rank_pct"] >= mid, "Medium", "Low"))
        res["grade"] = res["risk_grade"].map({"High": "FS3", "Medium": "FS2", "Low": "FS1"})

    # 상위 위험요인 (SHAP) — 심사역이 '왜'를 바로 볼 수 있게 CSV에 함께 싣는다
    try:
        contrib = booster.predict(X, pred_contrib=True)[:, :-1]  # 마지막 열은 base value
        order = np.argsort(-np.abs(contrib), axis=1)[:, :TOP_FACTORS]
        cols = np.array(trained)
        for i in range(TOP_FACTORS):
            idx = order[:, i]
            # 원시 피처명(f_ind_contract_end_pct)이 그대로 나가면 아무도 읽을 수 없다 →
            # 화면·CSV·메모가 공유하는 한국어 표시명으로 변환해서 저장한다.
            res[f"factor{i + 1}"] = [feature_korean_name(name_map.get(str(c), str(c)))
                                     for c in cols[idx]]
            res[f"factor{i + 1}_shap"] = np.round(contrib[np.arange(len(idx)), idx], 5)
    except Exception as exc:
        log.warning("SHAP 기여도 산출 생략: %s", exc)

    log.info("브랜드 상태: %s", res["brand_state"].value_counts().to_dict())
    high_mask = res["deterioration_rank_pct"] >= float(cfg["portfolio"]["risk_grades"]["high"])
    log.info("최상위 등급 %d개의 상태 구성: %s", int(high_mask.sum()),
             res.loc[high_mask, "brand_state"].value_counts().to_dict())

    res = res.sort_values("deterioration_1y", ascending=False).reset_index(drop=True)
    dest = out_dir / "scores_latest.csv"
    res.to_csv(dest, index=False, encoding="utf-8-sig")

    meta = {
        "scored_year": target_year,
        "n_scored": len(res),
        "model_file": "model_lgbm.txt",
        "calibration_method": str(cal.get("method")),
        "calibration_fitted_on": str(cal.get("fitted_on", "valid")),
        "prob_floor": float((cfg.get("evaluate") or {}).get("prob_floor", 0.0)),
        "prob_cap": float((cfg.get("evaluate") or {}).get("prob_cap", 1.0)),
        "grade_counts": res["risk_grade"].value_counts().to_dict(),
        "deterioration_min": float(res["deterioration_1y"].min()),
        "deterioration_max": float(res["deterioration_1y"].max()),
        "note": ("라벨 미확정 코호트의 사전 점수. 보정기는 과거 valid 적합본을 적용만 했으며 "
                 "재적합하지 않았다. 성능 근거는 metrics.csv·walkforward_metrics.csv 참조."),
    }
    (out_dir / "scores_latest_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("운영 점수 산출: %d년 %d개 브랜드 → %s (High %d / Medium %d / Low %d), "
             "악화확률 %.4f~%.4f", target_year, len(res), dest.name,
             meta["grade_counts"].get("High", 0), meta["grade_counts"].get("Medium", 0),
             meta["grade_counts"].get("Low", 0), meta["deterioration_min"], meta["deterioration_max"])
    log.info("⚠️ 이 코호트는 아직 라벨이 없어 사후 검증이 불가하다 — 성능 근거는 "
             "과거 백테스트(metrics.csv·walkforward_metrics.csv)에 있다.")
    return res


if __name__ == "__main__":
    score_latest(load_config())
