"""등급 밴드 도출 — 순위 백분위가 아니라 **관측된 악화율로 앵커링된 절대 컷**.

왜 바꾸는가
    기존 등급은 `deterioration_rank_pct >= 0.90 → High` 라는 순위 규칙이었다. 그러면 매 산출
    시점에 **항상 정확히 10%가 주의**가 되어, 산업 전체가 나빠져도 등급 분포가 고정된다.
    등급별 확률이 정의되지 않으므로 어떤 검증도 성립하지 않는다.

어떻게 앵커링하는가 (신용평가사 관행)
    평가사는 등급을 규칙이 아니라 **실적표**로 정의한다 — 등급별 장기 평균부도율을
    공표하고, 그 표가 등급의 의미가 된다. 여기서도 같은 방식을 쓴다.
    컷은 한 번 도출해 고정하고, 등급별 실현 악화율을 함께 공표한다.

보정 자료 (src/calibration.py 참조)
    **운영 모형 점수**를 쓰고 **학습 연도는 뺀다**. 워크포워드 예측은 fold 마다
    재학습해 원점수 척도가 다르고(실측 평균 0.2826/0.3306/0.1698), 그 위에서 보정하면
    체계적 과소예측이 난다(Z=+3.41, HL p=0.00027 기각). 학습 연도 점수는 in-sample
    이라 더 나쁘다(Z=+17.8).
    검증은 **선행 연도로 적합해 후행 연도에 적용**한다. 가장 이른 연도는 적합에만
    쓰이고 채점되지 않는다. 배포용은 학습 연도 밖 전부로 적합하므로 검증치가
    배포 성능의 보수적 하한이 된다.

컷 선택 규칙 (사후 조정 없이 재현 가능해야 한다)
    격자 탐색으로 아래를 **모두** 만족하는 후보 중 인접 등급 신뢰구간 분리폭이
    가장 큰 것을 고른다.
      1. 등급별 표본 n >= MIN_GRADE_N
      2. 실현 악화율이 등급 순으로 단조 증가
      3. 인접 등급의 Wilson 95% 신뢰구간이 겹치지 않음
      4. 검증에 쓰는 **모든 연도에서** 단조성이 유지됨

⚠️ 등급 3개인 이유
    4등급 이상은 진짜 OOS(보정연도 제외)에서 인접 신뢰구간이 겹치고, 연도를 하나씩
    빼고 재도출하면 최상위 등급이 붕괴한다(실측). 평가사도 18~20 notch 로 부여하되
    부도율 통계는 6개 버킷으로만 공표한다 — 표본이 지지하지 않는 세분화는 접는다.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.common import get_logger, load_config  # noqa: E402

log = get_logger("bands")

GRADES = ["FS1", "FS2", "FS3"]
GRADE_KR = {"FS1": "안정", "FS2": "관찰", "FS3": "주의"}
MIN_GRADE_N = 150          # 등급별 최소 표본 (풀링 기준)
MIN_CAL_N = 200            # 보정에 필요한 최소 선행 표본
GRID = np.round(np.arange(0.02, 0.32, 0.005), 4)


def wilson(k: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson 95% 구간 (소표본에서 보수적)."""
    if n == 0:
        return (0.0, 0.0)
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def calibration_frame(cfg: dict) -> pd.DataFrame:
    """**운영 모형 점수 + 라벨**, 학습 연도를 제외하고 모은다.

    왜 워크포워드 예측을 쓰지 않는가: fold 마다 재학습해 원점수 척도가 다르다
    (실측 평균 0.2826 / 0.3306 / 0.1698). 그 위에서 보정하면 체계적 과소예측이 난다.
    왜 학습 연도를 빼는가: 그 해 점수는 in-sample 이라 분리력이 과장돼 있다
    (써 봤더니 Spiegelhalter Z=+17.8 로 훨씬 나빠졌다).
    """
    from src.validation import _production_scores_with_ids
    out_dir = Path(cfg["paths"]["outputs"])
    wf = pd.read_parquet(out_dir / "walkforward_predictions.parquet")
    truth = wf.set_index(["brand_id", "year"])["y_true"]
    train_years = set(json.loads(
        (out_dir / "split_years.json").read_text(encoding="utf-8")).get("train_years", []))

    rows = []
    for yr in sorted({int(v) for v in wf["year"].unique()}):
        if yr in train_years:
            log.info("%s년: 학습 연도 — 보정에서 제외 (in-sample)", yr)
            continue
        ids, raw = _production_scores_with_ids(cfg, yr)
        if raw is None:
            continue
        d = pd.DataFrame({"brand_id": ids, "year": yr, "p_prod": raw})
        d["y_true"] = [truth.get((b, yr), np.nan) for b in ids]
        d = d.dropna(subset=["y_true"])
        rows.append(d)
        log.info("%s년: 보정 가능 %d행 · 악화 %d건 (%.2f%%) · 원점수 평균 %.4f",
                 yr, len(d), int(d["y_true"].sum()), d["y_true"].mean() * 100,
                 d["p_prod"].mean())
    if not rows:
        raise SystemExit("보정에 쓸 수 있는 연도가 없습니다 (학습 연도만 존재)")
    return pd.concat(rows, ignore_index=True)


def time_consistent_oos(cfg: dict, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """**선행 연도로 적합해 후행 연도에 적용**한 확률. 검증용.

    가장 이른 연도는 적합에만 쓰이고 검증표에는 들어가지 않는다 — 자기 자신을
    채점하지 않기 위해서다.
    """
    from src import calibration
    years = sorted(frame["year"].unique())
    if len(years) < 2:
        raise SystemExit(f"검증하려면 학습 연도 밖 관측이 2개 이상 필요합니다 (현재 {years})")
    out, meta = [], {}
    for y in years[1:]:
        prior = frame[frame["year"] < y]
        if len(prior) < MIN_CAL_N:
            log.info("%s년: 선행 보정표본 %d행 (< %d) → 검증표 제외", y, len(prior), MIN_CAL_N)
            continue
        cal, cv = calibration.fit(prior["p_prod"], prior["y_true"],
                                  fitted_on=f"{sorted(prior['year'].unique())}")
        cur = frame[frame["year"] == y].copy()
        cur["p_cal"] = cal.predict(cur["p_prod"])
        out.append(cur)
        meta[str(y)] = {**cal.describe(), "cv_logloss": cv}
        log.info("%s년: 선행 %d행으로 보정(α=%g) → 적용 %d행 (예측평균 %.4f, 실제 %.4f)",
                 y, len(prior), cal.alpha, len(cur), cur["p_cal"].mean(),
                 cur["y_true"].mean())
    if not out:
        raise SystemExit("검증 가능한 연도가 없습니다")
    return pd.concat(out, ignore_index=True), meta


def search_cuts(oos: pd.DataFrame) -> tuple[list[float], list[dict], float]:
    """규칙을 만족하는 컷 중 인접 구간 분리폭이 가장 큰 것."""
    y = oos["y_true"].to_numpy()
    p = oos["p_cal"].to_numpy()
    years = sorted(oos["year"].unique())
    best = None
    for c1 in GRID:
        for c2 in GRID[c1 < GRID]:
            g = np.digitize(p, [c1, c2])
            rows, ok = [], True
            for i in range(3):
                m = g == i
                n, k = int(m.sum()), int(y[m].sum())
                if n < MIN_GRADE_N:
                    ok = False
                    break
                lo, hi = wilson(k, n)
                rows.append({"grade": GRADES[i], "n": n, "events": k,
                             "rate": k / n, "ci_lo": lo, "ci_hi": hi})
            if not ok or not (rows[0]["rate"] < rows[1]["rate"] < rows[2]["rate"]):
                continue
            if not (rows[0]["ci_hi"] < rows[1]["ci_lo"] and rows[1]["ci_hi"] < rows[2]["ci_lo"]):
                continue
            if not all(
                (lambda r: r[0] < r[1] < r[2])(
                    [float(y[(g == i) & (oos["year"].to_numpy() == yr)].mean())
                     for i in range(3)])
                for yr in years
            ):
                continue
            sep = min(rows[1]["ci_lo"] - rows[0]["ci_hi"],
                      rows[2]["ci_lo"] - rows[1]["ci_hi"])
            if best is None or sep > best[2]:
                best = ([float(c1), float(c2)], rows, float(sep))
    if best is None:
        raise SystemExit(
            "조건을 만족하는 3등급 컷이 없습니다. 표본이 등급 분리를 지지하지 않으므로 "
            "등급 수를 줄이거나 순위 등급을 유지해야 합니다 — 억지로 만들지 마십시오.")
    return best


def oot_cut_procedure(frame: pd.DataFrame) -> dict:
    """**컷 선택 절차 자체**의 시점 밖 검증.

    왜 필요한가 (심사 지적 — 순환논증)
        배포 컷은 2022+2023 풀에서 '인접 CI 분리폭 최대화'로 골랐고, 격자 탐색의
        연도별 단조 조건이 2023 라벨을 이미 본다. 그 뒤 2023 실현율을 '시점 밖
        확인'으로 제시하면 **컷을 고를 때 본 데이터로 컷을 검증**하는 셈이다.
        비중첩·단조는 선택 제약이지 독립 검증이 아니다.

    무엇을 검증하는가
        "이 절차(보정 적합 → 격자 탐색)가 미래에도 유효한 등급을 만드는가"를
        검증한다. 선행 연도만으로 보정기를 적합하고 컷을 고른 뒤, **한 번도 보지
        않은 후행 연도**에 그대로 적용해 단조성·실현율을 채점한다. 검증 대상은
        특정 컷 값이 아니라 **절차**다 — 배포 컷은 검증된 절차를 전체 자료에
        적용한 결과다(모형을 검증 후 전체 자료로 재적합하는 표준 관행과 같다).
    """
    from src import calibration
    years = sorted(frame["year"].unique())
    if len(years) < 2:
        return {"feasible": False, "why": f"연도 2개 이상 필요 (현재 {years})"}
    anchor_years, test_year = years[:-1], years[-1]
    anchor = frame[frame["year"].isin(anchor_years)]
    test = frame[frame["year"] == test_year].copy()

    cal, _ = calibration.fit(anchor["p_prod"], anchor["y_true"],
                             fitted_on=str(anchor_years))
    a = anchor.copy()
    a["p_cal"] = cal.predict(a["p_prod"])
    try:
        cuts_a, _rows_a, sep_a = search_cuts(a)
    except SystemExit as exc:
        return {"feasible": False, "anchor_years": [int(y) for y in anchor_years],
                "why": f"선행 연도만으로는 조건 만족 컷 없음: {exc}"}

    test["p_cal"] = cal.predict(test["p_prod"])
    gi = np.digitize(test["p_cal"].to_numpy(), cuts_a)
    yv = test["y_true"].to_numpy()
    by_grade = []
    for i, g in enumerate(GRADES):
        m = gi == i
        n, k = int(m.sum()), int(yv[m].sum())
        lo, hi = wilson(k, n)
        by_grade.append({"grade": g, "n": n, "events": k,
                         "rate": round(k / n, 6) if n else None,
                         "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)})
    rates = [r["rate"] for r in by_grade]
    monotone = all(r is not None for r in rates) and rates[0] < rates[1] < rates[2]
    sep_test = min(by_grade[1]["ci_lo"] - by_grade[0]["ci_hi"],
                   by_grade[2]["ci_lo"] - by_grade[1]["ci_hi"])
    return {"feasible": True,
            "anchor_years": [int(y) for y in anchor_years],
            "test_year": int(test_year),
            "cuts_from_anchor": [float(c) for c in cuts_a],
            "anchor_min_separation": round(sep_a, 6),
            "test_by_grade": by_grade,
            "test_monotone": bool(monotone),
            "test_min_separation": round(float(sep_test), 6),
            "note": ("컷 선택이 검증 연도를 전혀 보지 않은 상태의 성적. "
                     "분리폭이 음수여도 단조가 유지되면 절차는 유효하다 — "
                     "비중첩까지 요구하면 검증이 아니라 선택 제약의 재확인이 된다.")}


def null_permutation_test(scored: pd.DataFrame, n_iter: int = 200,
                          seed: int = 42) -> dict:
    """라벨 무작위 치환 자료에 같은 격자 탐색을 돌린다 — 다중비교의 크기를 잰다.

    격자 60값 × 1,770쌍에서 '분리폭 최대'를 고르는 선택에는 다중비교가 내재한다.
    라벨을 연도 안에서 무작위로 섞으면(연도별 기저율 보존) 등급과 결과의 진짜 연관은
    사라진다. 그 귀무 자료에서 ① 조건 만족 컷이 나올 확률 ② 나왔을 때의 분리폭
    분포를 재면, 관측된 분리폭이 선택 효과만으로 설명되는지 답할 수 있다.
    """
    rng = np.random.default_rng(seed)
    found, seps = 0, []
    base = scored[["year", "p_cal", "y_true"]].copy()
    for _ in range(n_iter):
        perm = base.copy()
        perm["y_true"] = (perm.groupby("year")["y_true"]
                          .transform(lambda s: rng.permutation(s.to_numpy())))
        try:
            _, _, sep = search_cuts(perm)
            found += 1
            seps.append(float(sep))
        except SystemExit:
            continue
    return {"n_iter": n_iter,
            "n_qualifying_cut_found": found,
            "p_qualifying_by_chance": round(found / n_iter, 4),
            "null_sep_max": round(max(seps), 6) if seps else None,
            "null_sep_p95": round(float(np.percentile(seps, 95)), 6) if seps else None,
            "note": ("무작위 라벨에서 조건(표본·단조·비중첩·연도별 단조)을 모두 만족하는 "
                     "컷이 나올 확률과, 나왔을 때의 분리폭. 관측 분리폭이 귀무 최대값보다 "
                     "크면 선택 효과만으로는 설명되지 않는다.")}


def per_year_table(oos: pd.DataFrame, cuts: list[float]) -> pd.DataFrame:
    g = np.digitize(oos["p_cal"].to_numpy(), cuts)
    rows = []
    for yr in sorted(oos["year"].unique()):
        for i in range(3):
            m = (g == i) & (oos["year"].to_numpy() == yr)
            n, k = int(m.sum()), int(oos["y_true"].to_numpy()[m].sum())
            lo, hi = wilson(k, n)
            rows.append({"year": int(yr), "grade": GRADES[i], "grade_kr": GRADE_KR[GRADES[i]],
                         "n": n, "events": k, "realized_rate": round(k / n, 6) if n else None,
                         "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["outputs"])
    if not (out_dir / "walkforward_predictions.parquet").exists():
        raise SystemExit("walkforward_predictions.parquet 없음 — "
                         "`python run_pipeline.py --step evaluate` 를 먼저 실행")
    from src import calibration
    frame = calibration_frame(cfg)

    # 배포용 보정기 — 학습 연도 밖 관측 **전부**로 적합한다.
    deploy, deploy_cv = calibration.fit(frame["p_prod"], frame["y_true"],
                                        fitted_on=f"{sorted(frame['year'].unique())}")
    import joblib
    joblib.dump({"model": deploy, "method": "isotonic+eb_shrinkage",
                 "fitted_on": deploy.fitted_on, "alpha": deploy.alpha,
                 "n": deploy.n, "events": deploy.events, "cv_logloss": deploy_cv},
                out_dir / "calibrator_deploy.joblib")
    log.info("배포용 보정기: %s", deploy.describe())

    # 컷은 **배포 보정기 척도 위에서** 도출한다. 화면에 뜨는 확률이 그 척도이므로
    # 다른 척도에서 뽑은 컷을 얹으면 공표 실현율과 화면이 어긋난다.
    # ⚠️ 이 표는 보정기가 적합된 자료 위에서 계산되므로 **보정 in-sample** 이다.
    #    그 사실을 산출물에 적고, 아래에서 시점 밖 확인을 따로 붙인다.
    scored = frame.copy()
    scored["p_cal"] = deploy.predict(scored["p_prod"])
    cuts, rows, sep = search_cuts(scored)
    log.info("채택 컷 %s · 인접 신뢰구간 최소 분리 %.2f%%p", cuts, sep * 100)
    for r in rows:
        log.info("  %s %s  n=%d  악화 %d건  실현율 %.2f%%  95%%CI [%.1f, %.1f]",
                 r["grade"], GRADE_KR[r["grade"]], r["n"], r["events"],
                 r["rate"] * 100, r["ci_lo"] * 100, r["ci_hi"] * 100)

    # ── 검증 3종 ──────────────────────────────────────────────────────────
    # ⚠️ 위 표(pooled)는 컷을 **고른 자료 위에서** 계산된다. 비중첩·단조는 격자
    #    탐색의 선택 제약이므로 이 표는 검증이 아니라 **실적표(앵커)** 다.
    #    검증은 아래 세 개가 맡는다:
    #    ① cut_procedure_oot — 컷 선택이 검증 연도를 전혀 보지 않은 절차 검증
    #    ② null_test         — 무작위 라벨에서 같은 조건이 우연히 만족될 확률
    #    ③ out_of_time       — 확률값 자체의 시점 밖 보정 확인 (기존)
    proc = oot_cut_procedure(frame)
    log.info("절차 검증(선행→후행): %s",
             json.dumps({k: v for k, v in proc.items() if k != "test_by_grade"},
                        ensure_ascii=False))
    null_t = null_permutation_test(scored)
    log.info("귀무검정: 무작위 라벨에서 조건 만족 %d/%d (p=%.3f) · 귀무 분리폭 최대 %s "
             "vs 관측 %.4f", null_t["n_qualifying_cut_found"], null_t["n_iter"],
             null_t["p_qualifying_by_chance"], null_t["null_sep_max"], sep)

    # 시점 밖 확인 — 앞 연도로만 적합한 보정기로 뒤 연도를 매겨 **같은 컷**의
    # 등급별 실현율이 유지되는지 본다 (확률 보정의 시점 밖 성능).
    oot, cal_meta = time_consistent_oos(cfg, frame)
    oot_rows = []
    gi = np.digitize(oot["p_cal"].to_numpy(), cuts)
    yv = oot["y_true"].to_numpy()
    for i, g in enumerate(GRADES):
        m = gi == i
        n, k = int(m.sum()), int(yv[m].sum())
        lo, hi = wilson(k, n)
        oot_rows.append({"grade": g, "grade_kr": GRADE_KR[g], "n": n, "events": k,
                         "rate": round(k / n, 6) if n else None,
                         "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)})
    log.info("시점 밖 확인 (%s → %s): %s",
             sorted(set(frame["year"]) - set(oot["year"])), sorted(set(oot["year"])),
             " / ".join(f"{r['grade']} {(r['rate'] or 0) * 100:.2f}%(n={r['n']})"
                        for r in oot_rows))

    per_year = per_year_table(scored, cuts)
    per_year.to_csv(out_dir / "grade_validation.csv", index=False, encoding="utf-8-sig")

    bands = {
        "cuts": cuts,
        "grades": GRADES,
        "grade_kr": GRADE_KR,
        "anchored_on": ("운영 모형 점수(학습 연도 제외) + 등온회귀·계단별 축소 보정. "
                        "컷은 배포 보정기 척도에서 도출."),
        "selection_caveat": ("pooled 실현율표는 컷을 고른 자료 위에서 계산된 **실적표**다. "
                             "비중첩·단조는 격자 탐색의 선택 제약이므로 이 표 자체는 "
                             "검증이 아니다 — 검증은 cut_procedure_oot(절차의 시점 밖 "
                             "검증)와 null_test(라벨 치환 귀무검정)가 맡는다."),
        "calibration_data_note": ("보정 자료 중 2022년은 운영 모형의 valid 연도다 — "
                                  "복잡도 선택과 조기종료가 그 해 라벨을 봤으므로 그 해 "
                                  "점수의 분리력은 순수 OOS 보다 유리할 수 있다. 2023년은 "
                                  "모형 선택에 쓰이지 않은 순수 OOS 다. 절차 검증"
                                  "(cut_procedure_oot)의 채점 연도는 2023이므로 이 우려의 "
                                  "영향권 밖이다."),
        "cut_procedure_oot": proc,
        "null_test": null_t,
        "calibration_years": [int(y) for y in sorted(frame["year"].unique())],
        "out_of_time": {
            "fit_years": [int(y) for y in sorted(set(frame["year"]) - set(oot["year"]))],
            "test_years": [int(y) for y in sorted(oot["year"].unique())],
            "n": len(oot), "events": int(oot["y_true"].sum()), "by_grade": oot_rows},
        "calibrators": cal_meta,
        "deploy_calibrator_detail": deploy.describe(),
        "n_validation": len(scored),
        "n_events": int(scored["y_true"].sum()),
        "min_ci_separation": round(sep, 6),
        "pooled": [{**r, "grade_kr": GRADE_KR[r["grade"]],
                    "rate": round(r["rate"], 6),
                    "ci_lo": round(r["ci_lo"], 6), "ci_hi": round(r["ci_hi"], 6)}
                   for r in rows],
        "deploy_calibrator": "calibrator_deploy.joblib",
        "label_note": ("악화 = 공정거래위원회 가맹사업 공시 지표(가맹점수·계약종료·"
                       "평균매출·면적당매출)가 업종×연도 하위 구간에 진입하는 사건이다. "
                       "차주의 채무불이행 확률이 아니다."),
        "rule": {"min_grade_n": MIN_GRADE_N,
                 "criteria": ["등급별 최소 표본", "실현율 단조 증가",
                              "인접 Clopper-Pearson 95% 구간 비중첩",
                              "검증 전 연도에서 단조성 유지"]},
    }
    (out_dir / "grade_bands.json").write_text(
        json.dumps(bands, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("저장: grade_bands.json · grade_validation.csv · calibrator_deploy.joblib")
    print(json.dumps({k: v for k, v in bands.items() if k != "pooled"},
                     ensure_ascii=False, indent=2))
    print(per_year.to_string(index=False))


if __name__ == "__main__":
    main()
