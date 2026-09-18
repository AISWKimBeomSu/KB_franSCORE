"""모형 검증 산출물 — 판별력 · 정확성 · 안정성.

왜 이 모듈이 필요한가
    `docs/OPERATIONS.md` 는 PSI 경보 임계 0.10 / 적색 0.25 를 규정하는데, 저장소
    어디에도 PSI 구현이 없었다. 규정만 있고 측정이 없으면 그 규정은 장식이다.
    은행 실무자가 "검증표를 보여달라"고 했을 때 내놓을 것을 여기서 만든다.

무엇을 만드는가 (은행 모형검증 3축)
    판별력  AUC · Gini · KS · Lift@10  (연도별 + 풀링, 부트스트랩 신뢰구간)
    정확성  Hosmer-Lemeshow · Spiegelhalter Z · ECE · 등급별 정확 이항검정
    안정성  PSI(피처·점수) · 등급 이행행렬(**이탈 열 포함**)

정직성 원칙
    · 신뢰구간 없는 점추정치는 싣지 않는다. 표본이 작다는 사실을 지표와 함께 보인다.
    · 이행행렬에 **이탈(WR) 열**을 반드시 둔다. 악화한 브랜드는 다음 해 표본에서
      사라지므로, 그 열이 없으면 이행행렬이 "다들 잘 버텼다"는 거짓말을 한다.
      신용평가사 이행행렬에 WR(등급철회) 열이 항상 있는 것과 같은 이유다.
    · 계산 근거(표본 수·분모)를 모든 표에 함께 싣는다.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.common import get_logger, load_config

log = get_logger("validation")

PSI_WARN, PSI_ALERT = 0.10, 0.25      # docs/OPERATIONS.md 규정 임계
PSI_BINS = 10
PSI_EPS = 1e-6                        # 빈 구간의 log 발산 방지
N_BOOT = 2000
HL_BINS = 10


# ---------------------------------------------------------------------------
# 안정성 — PSI
# ---------------------------------------------------------------------------

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = PSI_BINS) -> tuple[float, int]:
    """Population Stability Index. (값, 유효 구간 수) 반환.

    구간 경계는 **기준(expected) 분포의 분위수**로 잡는다. 실제(actual) 로 잡으면
    드리프트가 경계에 흡수돼 PSI 가 과소평가된다.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < bins * 2 or len(a) < bins:
        return float("nan"), 0
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # 상수에 가까운 피처
        return 0.0, len(edges) - 1
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e, bins=edges)[0] / len(e)
    a_pct = np.histogram(a, bins=edges)[0] / len(a)
    e_pct = np.clip(e_pct, PSI_EPS, None)
    a_pct = np.clip(a_pct, PSI_EPS, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))), len(edges) - 1


def psi_band(v: float) -> str:
    if not np.isfinite(v):
        return "산출불가"
    return "적색" if v >= PSI_ALERT else ("경보" if v >= PSI_WARN else "정상")


def feature_psi(train: pd.DataFrame, deploy: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in train.columns or c not in deploy.columns:
            continue
        v, nb = psi(train[c].to_numpy(), deploy[c].to_numpy())
        rows.append({"feature": c, "psi": round(v, 6) if np.isfinite(v) else None,
                     "band": psi_band(v), "n_bins": nb,
                     "n_train": int(train[c].notna().sum()),
                     "n_deploy": int(deploy[c].notna().sum()),
                     "train_missing_pct": round(train[c].isna().mean(), 4),
                     "deploy_missing_pct": round(deploy[c].isna().mean(), 4)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")


# ---------------------------------------------------------------------------
# 판별력
# ---------------------------------------------------------------------------

def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUC (동점은 0.5로)."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _ks(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(stats.ks_2samp(pos, neg).statistic)


def _lift_at(y: np.ndarray, p: np.ndarray, frac: float = 0.10) -> float:
    n = max(1, round(len(p) * frac))
    idx = np.argsort(-p, kind="stable")[:n]
    base = y.mean()
    return float(y[idx].mean() / base) if base > 0 else float("nan")


def discrimination(y: np.ndarray, p: np.ndarray, *, seed: int = 42) -> dict:
    """판별력 지표 + 부트스트랩 95% 신뢰구간."""
    rng = np.random.default_rng(seed)
    point = {"auc": _auc(y, p), "gini": 2 * _auc(y, p) - 1,
             "ks": _ks(y, p), "lift_at_10": _lift_at(y, p)}
    boot = {k: [] for k in point}
    n = len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        yb, pb = y[i], p[i]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        boot["auc"].append(_auc(yb, pb))
        boot["gini"].append(2 * _auc(yb, pb) - 1)
        boot["ks"].append(_ks(yb, pb))
        boot["lift_at_10"].append(_lift_at(yb, pb))
    out = {"n": int(n), "events": int(y.sum()), "base_rate": float(y.mean())}
    for k, v in point.items():
        arr = np.asarray([x for x in boot[k] if np.isfinite(x)])
        lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan))
        out[k] = round(float(v), 6)
        out[f"{k}_lo"] = round(float(lo), 6)
        out[f"{k}_hi"] = round(float(hi), 6)
    return out


# ---------------------------------------------------------------------------
# 정확성 (보정)
# ---------------------------------------------------------------------------

def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, bins: int = HL_BINS) -> dict:
    """예측확률 구간별 관측/기대 비교. 유의(p<0.05)하면 보정이 깨진 것.

    ⚠️ 구간을 **순위 등분할**로 만든다. 등온회귀 산출은 계단이라 같은 값이 대량으로
       묶이는데, 값 분위로 자르면 경계가 계단값과 겹쳐 그룹 크기가 극단적으로
       불균등해지고 통계량이 왜곡된다. 실측으로 같은 자료에서 값 분위 p=0.00000 vs
       순위 등분할 p=0.018 — 세 자릿수 차이가 오직 구간 방식에서 나왔다.
       동점이 많은 예측에 대한 표준 처리다.
    """
    if len(np.unique(p)) < 3:
        return {"statistic": None, "df": None, "p_value": None, "bins": 0,
                "note": "예측확률의 서로 다른 값이 적어 산출 불가"}
    idx = pd.qcut(pd.Series(p).rank(method="first"), bins,
                  labels=False, duplicates="drop").to_numpy()
    stat, used = 0.0, 0
    for b in np.unique(idx):
        m = idx == b
        n, o, e = int(m.sum()), float(y[m].sum()), float(p[m].sum())
        if n == 0 or e <= 0 or e >= n:
            continue
        stat += (o - e) ** 2 / (e * (1 - e / n))
        used += 1
    df = max(1, used - 2)
    return {"statistic": round(stat, 4), "df": int(df),
            "p_value": round(float(1 - stats.chi2.cdf(stat, df)), 6), "bins": used,
            "verdict": "보정 적정(기각 못함)" if 1 - stats.chi2.cdf(stat, df) >= 0.05
                       else "보정 부적정(기각)"}


def spiegelhalter_z(y: np.ndarray, p: np.ndarray) -> dict:
    """전체 보정 수준 검정. |Z| 가 크면 예측이 체계적으로 높거나 낮다."""
    num = float(np.sum((y - p) * (1 - 2 * p)))
    den = float(np.sqrt(np.sum(((1 - 2 * p) ** 2) * p * (1 - p))))
    if den == 0:
        return {"z": None, "p_value": None}
    z = num / den
    return {"z": round(z, 4), "p_value": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 6),
            "verdict": "보정 적정" if abs(z) < 1.96 else
                       ("과소예측" if z > 0 else "과대예측")}


def ece(y: np.ndarray, p: np.ndarray, bins: int = HL_BINS) -> float:
    """Expected Calibration Error — 구간별 |관측−예측| 의 표본가중 평균."""
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return float("nan")
    q[0], q[-1] = -np.inf, np.inf
    idx = np.digitize(p, q[1:-1])
    tot = 0.0
    for b in np.unique(idx):
        m = idx == b
        tot += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def grade_binomial(y: np.ndarray, p: np.ndarray, grade: np.ndarray,
                   order: list[str]) -> pd.DataFrame:
    """등급별 정확 이항검정 — 예측 평균확률 대비 실현율이 유의하게 다른가."""
    rows = []
    for g in order:
        m = grade == g
        n, k = int(m.sum()), int(y[m].sum())
        if n == 0:
            continue
        exp = float(p[m].mean())
        pv = float(stats.binomtest(k, n, exp).pvalue) if 0 < exp < 1 else float("nan")
        lo = float(stats.beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
        hi = float(stats.beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
        rows.append({"grade": g, "n": n, "events": k,
                     "realized": round(k / n, 6), "predicted": round(exp, 6),
                     "ratio": round((k / n) / exp, 4) if exp > 0 else None,
                     "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                     "p_value": round(pv, 6) if np.isfinite(pv) else None,
                     "verdict": "정합" if not np.isfinite(pv) or pv >= 0.05 else "불일치"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 안정성 — 등급 이행행렬
# ---------------------------------------------------------------------------

def transition_matrix(graded: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """연초 등급 → 연말 등급 이행. **이탈 열을 반드시 포함한다.**

    graded: [brand_id, year, grade]
    악화한 브랜드는 다음 해 자격 표본에서 사라진다(healthy gate·자격조건). 이탈을
    빼고 대각선만 보면 "다들 등급을 유지했다"는 거짓 인상을 준다.
    """
    rows = []
    years = sorted(graded["year"].unique())
    for a, b in itertools.pairwise(years):
        ga = graded[graded["year"] == a].set_index("brand_id")["grade"]
        gb = graded[graded["year"] == b].set_index("brand_id")["grade"]
        for g in order:
            src = ga[ga == g].index
            if len(src) == 0:
                continue
            nxt = gb.reindex(src)
            row = {"from_year": int(a), "to_year": int(b), "from_grade": g,
                   "n_start": len(src)}
            for h in order:
                row[h] = int((nxt == h).sum())
            row["이탈"] = int(nxt.isna().sum())
            for h in [*order, "이탈"]:
                row[f"{h}_pct"] = round(row[h] / len(src), 4)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def _graded_history(cfg: dict, wf: pd.DataFrame, cuts: list[float],
                    grades: list[str]) -> pd.DataFrame:
    """이행행렬용 연도별 등급 — **운영 모형 하나로** 전 연도를 매긴다.

    ⚠️ 워크포워드 예측(`p_lgbm`)을 그대로 쓰면 안 된다. fold 마다 재학습하므로
       원점수 척도가 다르다(fold 평균 0.2826 / 0.3306 / 0.1698). 그 위에 공통 컷을
       얹으면 브랜드가 옮겨간 것이 아니라 **모형이 바뀐 것**을 이행으로 읽는다.
       실측: FS3 유지율이 42.2% → 6.0% 로 붕괴하는데, 원인은 브랜드가 아니라 척도였다.

    운영 모형을 과거 코호트에 적용하면 그 연도에 대해 in-sample 이 될 수 있다.
    다만 이 표의 목적은 성능이 아니라 **등급 배정의 안정성**이므로 그 편이 맞다 —
    모형을 고정해야 브랜드의 이동만 남는다. 성능은 grade_bands.json 의 시점 정합
    검증표가 담당한다.
    """
    # ⚠️ 반드시 **배포 보정기**를 쓴다. 다른 보정기로 확률을 만들면 그 척도에서 도출된
    #    컷이 맞지 않아 최상위 등급이 통째로 비어 버린다(실측: FS3 행이 사라졌다).
    import joblib
    cal_p = Path(cfg["paths"]["outputs"]) / "calibrator_deploy.joblib"
    if not cal_p.exists():
        return pd.DataFrame(columns=["brand_id", "year", "grade", "p_cal"])
    deploy = joblib.load(cal_p)["model"]

    rows = []
    for yr in sorted({int(v) for v in wf["year"].unique()}):
        ids, raw = _production_scores_with_ids(cfg, yr)
        if raw is None:
            continue
        p = np.clip(deploy.predict(raw), 0.0, 1.0)
        rows.append(pd.DataFrame({
            "brand_id": ids, "year": yr,
            "grade": np.array(grades, dtype=object)[np.digitize(p, cuts)], "p_cal": p}))
    if not rows:
        return pd.DataFrame(columns=["brand_id", "year", "grade", "p_cal"])
    return pd.concat(rows, ignore_index=True)


def _production_scores_with_ids(cfg: dict, year: int) -> tuple[np.ndarray, np.ndarray | None]:
    """운영 모형 원점수 + 해당 brand_id."""
    import lightgbm as lgb

    from src.model import _numeric_feature_frame, _sanitize_feature_names
    out_dir, proc = Path(cfg["paths"]["outputs"]), Path(cfg["paths"]["processed"])
    mp = out_dir / "model_lgbm.txt"
    if not mp.exists():
        return np.array([]), None
    booster = lgb.Booster(model_str=mp.read_text(encoding="utf-8"))
    feats = pd.read_parquet(proc / "features.parquet")
    panel = pd.read_parquet(proc / "panel.parquet")
    cols = [c for c in ("brand_id", "year", "eligible_t") if c in panel.columns]
    df = feats.merge(panel[cols], on=["brand_id", "year"], how="left")
    sub = df[df["year"] == int(year)]
    if "eligible_t" in sub.columns:
        sub = sub[sub["eligible_t"].fillna(False).astype(bool)]
    if sub.empty:
        return np.array([]), None
    X = _numeric_feature_frame(sub, log)
    X = X.set_axis(_sanitize_feature_names(X.columns)[0], axis=1)
    trained = list(booster.feature_name())
    if any(c not in X.columns for c in trained):
        return np.array([]), None
    return (sub["brand_id"].to_numpy(),
            np.clip(booster.predict(X[trained]), 0.0, 1.0))


def _production_raw_scores(cfg: dict, year: int) -> np.ndarray | None:
    """운영 모형(model_lgbm.txt)을 지정 연도 자격 코호트에 적용한 **원점수**.

    PSI 는 한 모형의 점수 분포가 시간에 따라 옮겨갔는지를 재는 지표다. 그러려면
    비교 양쪽이 같은 모형이어야 한다 — 그래서 워크포워드(fold별 재학습) 점수가
    아니라 운영 모형을 과거 코호트에 다시 적용해 얻는다.
    """
    import lightgbm as lgb

    from src.model import _numeric_feature_frame, _sanitize_feature_names
    out_dir, proc = Path(cfg["paths"]["outputs"]), Path(cfg["paths"]["processed"])
    mp = out_dir / "model_lgbm.txt"
    if not mp.exists():
        return None
    booster = lgb.Booster(model_str=mp.read_text(encoding="utf-8"))
    feats = pd.read_parquet(proc / "features.parquet")
    panel = pd.read_parquet(proc / "panel.parquet")
    cols = [c for c in ("brand_id", "year", "eligible_t") if c in panel.columns]
    df = feats.merge(panel[cols], on=["brand_id", "year"], how="left")
    sub = df[df["year"] == int(year)]
    if "eligible_t" in sub.columns:
        sub = sub[sub["eligible_t"].fillna(False).astype(bool)]
    if sub.empty:
        return None
    X = _numeric_feature_frame(sub, log)
    X = X.set_axis(_sanitize_feature_names(X.columns)[0], axis=1)
    trained = list(booster.feature_name())
    if any(c not in X.columns for c in trained):
        return None
    return np.clip(booster.predict(X[trained]), 0.0, 1.0)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    out_dir = Path(cfg["paths"]["outputs"])
    vdir = out_dir / "validation"
    vdir.mkdir(parents=True, exist_ok=True)

    bands = json.loads((out_dir / "grade_bands.json").read_text(encoding="utf-8"))
    cuts, grades = [float(c) for c in bands["cuts"]], list(bands["grades"])
    wf = pd.read_parquet(out_dir / "walkforward_predictions.parquet")
    graded = _graded_history(cfg, wf, cuts, grades)

    summary: dict = {"n_oos": len(wf), "n_events": int(wf["y_true"].sum()),
                     "years": [int(y) for y in sorted(wf["year"].unique())],
                     "cuts": cuts, "grades": grades}

    # ── 판별력 ────────────────────────────────────────────────────────
    rows = []
    for y in [None, *sorted(wf["year"].unique())]:
        sub = wf if y is None else wf[wf["year"] == y]
        d = discrimination(sub["y_true"].to_numpy(), sub["p_lgbm"].to_numpy())
        rows.append({"scope": "풀링" if y is None else int(y), **d})
    disc = pd.DataFrame(rows)
    disc.to_csv(vdir / "discrimination.csv", index=False, encoding="utf-8-sig")
    summary["discrimination_pooled"] = rows[0]
    log.info("판별력(풀링): AUC %.3f [%.3f, %.3f] · KS %.3f · Lift@10 %.2f",
             rows[0]["auc"], rows[0]["auc_lo"], rows[0]["auc_hi"],
             rows[0]["ks"], rows[0]["lift_at_10"])

    # ── 정확성 ────────────────────────────────────────────────────────
    # ⚠️ 보정 검정은 반드시 **시점 밖 확률**로 해야 한다. 배포 보정기는 이 자료로
    #    적합됐으므로 같은 데이터에 검정을 걸면 HL p=1.0, ECE=0.0000 처럼 완벽하게
    #    나온다(실측). 그건 보정이 좋다는 뜻이 아니라 자기 자신을 채점한 결과다.
    from tools.derive_grade_bands import calibration_frame, time_consistent_oos
    tc, _ = time_consistent_oos(cfg, calibration_frame(cfg))
    tc["grade"] = np.array(grades, dtype=object)[np.digitize(tc["p_cal"].to_numpy(), cuts)]
    y, p = tc["y_true"].to_numpy(), tc["p_cal"].to_numpy()
    cal = {"basis": ("시점 밖 보정 — 운영 모형 점수(학습 연도 제외)를 선행 연도로 "
                     "적합해 후행 연도에 적용"),
           "years": [int(v) for v in sorted(tc["year"].unique())],
           "n": len(tc), "events": int(y.sum()),
           "hosmer_lemeshow": hosmer_lemeshow(y, p),
           "spiegelhalter": spiegelhalter_z(y, p),
           "ece": round(ece(y, p), 6),
           "mean_predicted": round(float(p.mean()), 6),
           "observed": round(float(y.mean()), 6)}
    gb = grade_binomial(y, p, tc["grade"].to_numpy(), grades)
    gb.to_csv(vdir / "grade_binomial.csv", index=False, encoding="utf-8-sig")
    (vdir / "calibration_tests.json").write_text(
        json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["calibration"] = cal
    log.info("정확성: HL p=%s (%s) · Spiegelhalter Z=%s (%s) · ECE %.4f",
             cal["hosmer_lemeshow"].get("p_value"), cal["hosmer_lemeshow"].get("verdict"),
             cal["spiegelhalter"].get("z"), cal["spiegelhalter"].get("verdict"), cal["ece"])

    # ── 안정성: 이행행렬 ──────────────────────────────────────────────
    tm = transition_matrix(graded[["brand_id", "year", "grade"]], grades)
    tm.to_csv(vdir / "transition_matrix.csv", index=False, encoding="utf-8-sig")
    if not tm.empty:
        summary["transition_pairs"] = int(tm[["from_year", "to_year"]].drop_duplicates().shape[0])
        summary["mean_exit_rate"] = round(float(tm["이탈"].sum() / tm["n_start"].sum()), 4)
        log.info("이행행렬: 코호트 %d쌍 · 평균 이탈률 %.1f%%",
                 summary["transition_pairs"], summary["mean_exit_rate"] * 100)

    # ── 안정성: PSI ───────────────────────────────────────────────────
    proc = Path(cfg["paths"]["processed"])
    feats = pd.read_parquet(proc / "features.parquet")
    labels = pd.read_parquet(proc / "labels.parquet")
    sy = json.loads((out_dir / "split_years.json").read_text(encoding="utf-8"))
    train_years = {int(v) for v in sy.get("train_years", [])}
    fcols = [c for c in feats.columns if c.startswith("f_")]

    # 피처 PSI 를 **두 질문으로 나눠** 낸다. 하나로 합치면 원인을 못 가린다.
    #
    #  (A) 대표성 — 개발표본(healthy 게이트 통과, 학습연도) vs 적용모집단(배포 코호트).
    #      CRR 제174조(c)·SR 26-2 가 요구하는 use-population 적합성 점검이다.
    #      여기서 크게 나오는 것은 **드리프트가 아니라 게이트로 생긴 모집단 차이**다.
    #  (B) 안정성 — 같은 자격 조건의 과거 코호트 vs 배포 코호트. 시간에 따른 진짜 이동.
    scores = pd.read_csv(out_dir / "scores_latest.csv", encoding="utf-8-sig")
    target_year = int(scores["year"].iloc[0])
    panel = pd.read_parquet(proc / "panel.parquet")
    elig = panel[["brand_id", "year"] + (["eligible_t"] if "eligible_t" in panel else [])]
    ef = feats.merge(elig, on=["brand_id", "year"], how="left")
    if "eligible_t" in ef.columns:
        ef = ef[ef["eligible_t"].fillna(False).astype(bool)]

    tr_keys = labels[labels["year"].isin(train_years)][["brand_id", "year"]]
    train_f = feats.merge(tr_keys, on=["brand_id", "year"], how="inner")
    dep_f = ef[ef["year"] == target_year]

    fp = feature_psi(train_f, dep_f, fcols)          # (A) 대표성
    fp.insert(1, "comparison", "대표성(개발표본→적용모집단)")
    prev_year = max((y for y in ef["year"].unique() if y < target_year), default=None)
    if prev_year is not None:
        fp2 = feature_psi(ef[ef["year"] == prev_year], dep_f, fcols)   # (B) 안정성
        fp2.insert(1, "comparison", f"안정성({prev_year}→{target_year} 동일 자격조건)")
        fp = pd.concat([fp, fp2], ignore_index=True)
    fp.to_csv(vdir / "psi_features.csv", index=False, encoding="utf-8-sig")
    fp_repr = fp[fp["comparison"].str.startswith("대표성")]
    fp_stab = fp[fp["comparison"].str.startswith("안정성")]

    # 점수 PSI — **같은 모형의 시점 간 비교**여야 한다.
    #
    # ⚠️ 처음에는 워크포워드 OOS 점수와 2024 배포 점수를 비교했다. 그건 틀렸다.
    #    워크포워드는 fold 마다 **재학습한 서로 다른 모형** 3개의 점수를 섞은 것이고
    #    2024 는 단일 운영 모형이다. 다른 모형끼리 비교하면 PSI 가 모형 차이를 재게
    #    되어 원점수에서도 1.41 이 나온다(실측). PSI 는 한 모형의 점수 분포가 시간에
    #    따라 옮겨갔는지를 재는 지표이므로, 운영 모형을 과거 코호트에 적용해 비교한다.
    #
    #    보정 계단값으로 재는 것도 피한다 — 고유값이 14~22개뿐이라 분위 구간이
    #    퇴화해 값이 폭주한다(4.37). 연속인 원점수로 잰다.
    target_year = int(scores["year"].iloc[0])
    base_raw = _production_raw_scores(cfg, target_year)
    score_rows = []
    for yr in sorted({int(v) for v in wf["year"].unique()}):
        prev_raw = _production_raw_scores(cfg, yr)
        if prev_raw is None or base_raw is None:
            continue
        v, nb = psi(prev_raw, base_raw)
        score_rows.append({"base_year": yr, "target_year": target_year,
                           "psi": round(v, 6), "band": psi_band(v), "n_bins": nb,
                           "n_base": len(prev_raw), "n_target": len(base_raw)})
    if score_rows:
        pd.DataFrame(score_rows).to_csv(vdir / "psi_score.csv", index=False,
                                        encoding="utf-8-sig")

    def _pack(d: pd.DataFrame) -> dict:
        return {"n_features": len(d), "bands": d["band"].value_counts().to_dict(),
                "worst": d.dropna(subset=["psi"]).head(5)[
                    ["feature", "psi", "band"]].to_dict("records")}

    summary["psi"] = {
        "score": {"basis": "운영 모형 원점수, 과거 코호트 → 배포 코호트 (동일 모형·시점 간)",
                  "by_year": score_rows},
        "features_representativeness": _pack(fp_repr),
        "features_stability": _pack(fp_stab) if len(fp_stab) else None,
        "n_train": len(train_f), "n_deploy": len(dep_f),
        "note": ("대표성 PSI 가 크면 개발표본과 적용모집단이 다르다는 뜻이고(게이트 효과), "
                 "안정성 PSI 가 크면 시간에 따른 실제 이동이다. 둘을 합치면 원인을 못 가린다."),
        "thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT}}
    log.info("PSI(점수·운영모형 시점간): %s",
             {r["base_year"]: f"{r['psi']:.4f}({r['band']})" for r in score_rows})
    log.info("PSI(피처) 대표성 %s · 안정성 %s",
             summary["psi"]["features_representativeness"]["bands"],
             (summary["psi"]["features_stability"] or {}).get("bands"))

    (vdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("검증 산출물 저장: %s", vdir)
    return summary


if __name__ == "__main__":
    run()
