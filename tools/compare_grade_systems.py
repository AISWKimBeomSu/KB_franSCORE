"""신·구 등급체계 효과성 비교 — 절대 컷이 순위 백분위보다 실제로 나은가.

"바꿨더니 좋아졌다"는 주장은 측정 없이는 못 한다. 두 체계를 **같은 데이터·같은
모형**에 적용해 다섯 가지로 비교한다.

  1. 등급 분포의 경기 대응성   산업이 나빠진 해에 하위 등급이 늘어나는가
  2. 등급별 실현 악화율 단조성 상위 등급일수록 실제로 덜 악화하는가 (연도별로도)
  3. 인접 등급 신뢰구간 분리   등급이 통계적으로 구분되는가
  4. 위험 무변화 시 등급 이동  확률이 그대로인데 남의 변화로 등급이 바뀌는가
  5. 점검 큐 효율             상위 등급이 실제 악화 건을 얼마나 포착하는가

⚠️ 두 체계 모두 **운영 모형 하나**로 점수를 낸 뒤 등급만 다르게 매긴다. 워크포워드
   예측을 그대로 쓰면 fold 마다 재학습해 원점수 척도가 달라(평균 0.28/0.33/0.17)
   등급 이동이 브랜드가 아니라 모형 때문에 생긴다 — 실측으로 확인한 함정이다.
"""
from __future__ import annotations

import contextlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.common import get_logger, load_config  # noqa: E402
from src.validation import _production_scores_with_ids  # noqa: E402

log = get_logger("compare")

NEW = ["FS1", "FS2", "FS3"]
OLD = ["Low", "Medium", "High"]          # 순위 컷: 하위 70% / 70~90% / 상위 10%
OLD_CUTS = (0.70, 0.90)
NO_CHANGE_TOL = 1e-6                     # '위험 무변화' 판정 허용오차


def cp_interval(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    lo = float(stats.beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(stats.beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def build_panel(cfg: dict, cuts: list[float]) -> pd.DataFrame:
    """연도별 운영 모형 점수 + 두 체계의 등급 + 실제 라벨.

    ⚠️ 보정기는 **배포에 실제로 쓰는 것**을 그대로 쓴다. 다른 보정기로 확률을 만들면
       그 척도에서 도출된 컷이 맞지 않아 최상위 등급이 통째로 비어 버린다(실측).
    """
    import joblib
    out_dir = Path(cfg["paths"]["outputs"])
    wf = pd.read_parquet(out_dir / "walkforward_predictions.parquet")
    cal_p = out_dir / "calibrator_deploy.joblib"
    if not cal_p.exists():
        raise SystemExit("calibrator_deploy.joblib 없음 — "
                         "`python run_pipeline.py --step bands` 를 먼저 실행")
    deploy = joblib.load(cal_p)["model"]
    truth = wf.set_index(["brand_id", "year"])["y_true"]

    frames = []
    for yr in sorted({int(v) for v in wf["year"].unique()}):
        ids, raw = _production_scores_with_ids(cfg, yr)
        if raw is None:
            continue
        p = np.clip(deploy.predict(raw), 0.0, 1.0)
        d = pd.DataFrame({"brand_id": ids, "year": yr, "p": p})
        d["new_grade"] = np.array(NEW, dtype=object)[np.digitize(p, cuts)]
        # 구 체계: 그 해 코호트 안에서의 순위 백분위
        rank = pd.Series(p).rank(method="first", pct=True).to_numpy()
        d["old_grade"] = np.where(rank >= OLD_CUTS[1], "High",
                                  np.where(rank >= OLD_CUTS[0], "Medium", "Low"))
        d["y_true"] = [truth.get((b, yr), np.nan) for b in ids]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def dist_by_year(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    t = (df.groupby(["year", col]).size().unstack(fill_value=0).reindex(columns=order,
                                                                       fill_value=0))
    return t.div(t.sum(axis=1), axis=0).round(4)


def realized(df: pd.DataFrame, col: str, order: list[str], by_year: bool = False):
    lab = df.dropna(subset=["y_true"])
    keys = ["year", col] if by_year else [col]
    rows = []
    for k, g in lab.groupby(keys):
        n, e = len(g), int(g["y_true"].sum())
        lo, hi = cp_interval(e, n)
        rec = {"grade": k[-1] if isinstance(k, tuple) else k, "n": n, "events": e,
               "rate": round(e / n, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)}
        if by_year:
            rec["year"] = int(k[0])
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["_o"] = out["grade"].map({g: i for i, g in enumerate(order)})
    sort = (["year", "_o"] if by_year else ["_o"])
    return out.sort_values(sort).drop(columns="_o").reset_index(drop=True)


def monotone_ok(tab: pd.DataFrame, order: list[str]) -> bool:
    r = [tab.loc[tab["grade"] == g, "rate"].to_numpy() for g in order]
    if any(len(x) == 0 for x in r):
        return False
    v = [float(x[0]) for x in r]
    return v[0] < v[1] < v[2]


def ci_separated(tab: pd.DataFrame, order: list[str]) -> int:
    """인접 등급 신뢰구간이 겹치지 않는 쌍의 개수 (최대 2)."""
    m = {g: tab[tab["grade"] == g].iloc[0] for g in order if (tab["grade"] == g).any()}
    if len(m) < 3:
        return 0
    return int(m[order[0]]["ci_hi"] < m[order[1]]["ci_lo"]) + \
        int(m[order[1]]["ci_hi"] < m[order[2]]["ci_lo"])


def no_change_migration(df: pd.DataFrame) -> dict:
    """확률이 사실상 변하지 않았는데 등급이 바뀐 비율."""
    piv = df.pivot_table(index="brand_id", columns="year", values="p")
    gn = df.pivot_table(index="brand_id", columns="year", values="new_grade",
                        aggfunc="first")
    go = df.pivot_table(index="brand_id", columns="year", values="old_grade",
                        aggfunc="first")
    years = sorted(df["year"].unique())
    tot = new_mv = old_mv = 0
    for a, b in itertools.pairwise(years):
        if a not in piv.columns or b not in piv.columns:
            continue
        m = piv[[a, b]].notna().all(axis=1) & ((piv[b] - piv[a]).abs() <= NO_CHANGE_TOL)
        ids = piv.index[m]
        if len(ids) == 0:
            continue
        tot += len(ids)
        new_mv += int((gn.loc[ids, a] != gn.loc[ids, b]).sum())
        old_mv += int((go.loc[ids, a] != go.loc[ids, b]).sum())
    return {"n_pairs_no_change": tot, "new_moved": new_mv, "old_moved": old_mv,
            "new_rate": round(new_mv / tot, 6) if tot else None,
            "old_rate": round(old_mv / tot, 6) if tot else None}


def queue_capture(df: pd.DataFrame) -> dict:
    """최상위 등급이 실제 악화 건의 몇 %를 담는가 (그 등급이 코호트의 몇 %를 쓰면서)."""
    lab = df.dropna(subset=["y_true"])
    out = {}
    for col, top, tag in ((("new_grade"), "FS3", "new"), (("old_grade"), "High", "old")):
        m = lab[col] == top
        out[tag] = {"share_of_cohort": round(float(m.mean()), 4),
                    "capture_of_events": round(float(lab.loc[m, "y_true"].sum()
                                                     / lab["y_true"].sum()), 4),
                    "precision": round(float(lab.loc[m, "y_true"].mean()), 4) if m.any()
                                 else None,
                    "n": int(m.sum())}
    return out


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["outputs"])
    bands = json.loads((out_dir / "grade_bands.json").read_text(encoding="utf-8"))
    cuts = [float(c) for c in bands["cuts"]]
    df = build_panel(cfg, cuts)
    log.info("비교 표본: %d행 · 연도 %s · 라벨 있는 행 %d",
             len(df), sorted(df["year"].unique()), int(df["y_true"].notna().sum()))

    res: dict = {"cuts": cuts, "old_cuts": list(OLD_CUTS), "n_rows": len(df),
                 "years": [int(y) for y in sorted(df["year"].unique())]}

    # 1. 경기 대응성 — 연도별 최상위 등급 비중의 변동
    dn, do = dist_by_year(df, "new_grade", NEW), dist_by_year(df, "old_grade", OLD)
    res["distribution"] = {
        "new_top_share_by_year": {int(k): float(v) for k, v in dn["FS3"].items()},
        "old_top_share_by_year": {int(k): float(v) for k, v in do["High"].items()},
        "new_top_share_sd": round(float(dn["FS3"].std()), 6),
        "old_top_share_sd": round(float(do["High"].std()), 6)}

    # 2·3. 등급별 실현율 · 단조성 · 신뢰구간 분리
    rn, ro = realized(df, "new_grade", NEW), realized(df, "old_grade", OLD)
    rn.to_csv(out_dir / "validation" / "compare_realized_new.csv", index=False,
              encoding="utf-8-sig")
    ro.to_csv(out_dir / "validation" / "compare_realized_old.csv", index=False,
              encoding="utf-8-sig")
    rny, roy = realized(df, "new_grade", NEW, True), realized(df, "old_grade", OLD, True)
    res["realized"] = {
        "new": rn.to_dict("records"), "old": ro.to_dict("records"),
        "new_monotone": monotone_ok(rn, NEW), "old_monotone": monotone_ok(ro, OLD),
        "new_ci_separated": ci_separated(rn, NEW), "old_ci_separated": ci_separated(ro, OLD),
        "new_monotone_years": sum(monotone_ok(rny[rny["year"] == y], NEW)
                                  for y in rny["year"].unique()),
        "old_monotone_years": sum(monotone_ok(roy[roy["year"] == y], OLD)
                                  for y in roy["year"].unique()),
        "n_years": int(rny["year"].nunique())}

    # 4. 위험 무변화 시 등급 이동
    res["no_change_migration"] = no_change_migration(df)

    # 5. 큐 효율
    res["queue"] = queue_capture(df)

    (out_dir / "validation" / "grade_system_comparison.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 보고 ──────────────────────────────────────────────────────────
    print("\n=== 1. 경기 대응성 (최상위 등급 비중의 연도별 변동) ===")
    print(f"  절대 등급 FS3 : {res['distribution']['new_top_share_by_year']} "
          f"표준편차 {res['distribution']['new_top_share_sd']:.4f}")
    print(f"  순위 등급 High: {res['distribution']['old_top_share_by_year']} "
          f"표준편차 {res['distribution']['old_top_share_sd']:.4f}")

    print("\n=== 2·3. 등급별 실현 악화율 ===")
    for tag, tab in (("절대", rn), ("순위", ro)):
        s = " · ".join(f"{r['grade']} {r['rate'] * 100:.2f}%(n={r['n']})"
                       for _, r in tab.iterrows())
        print(f"  {tag}: {s}")
    print(f"  단조성(풀링) 절대 {res['realized']['new_monotone']} / "
          f"순위 {res['realized']['old_monotone']}")
    print(f"  단조성(연도별) 절대 {res['realized']['new_monotone_years']}/"
          f"{res['realized']['n_years']} · 순위 {res['realized']['old_monotone_years']}/"
          f"{res['realized']['n_years']}")
    print(f"  인접 CI 분리(2쌍 중) 절대 {res['realized']['new_ci_separated']} / "
          f"순위 {res['realized']['old_ci_separated']}")

    nc = res["no_change_migration"]
    print("\n=== 4. 위험 무변화 시 등급 이동 ===")
    print(f"  |Δp| ≤ {NO_CHANGE_TOL} 인 연속연도 쌍 {nc['n_pairs_no_change']}개")
    print(f"  절대 등급 이동 {nc['new_moved']}건 · 순위 등급 이동 {nc['old_moved']}건")

    q = res["queue"]
    print("\n=== 5. 점검 큐 효율 (최상위 등급) ===")
    for tag, key in (("절대 FS3", "new"), ("순위 High", "old")):
        v = q[key]
        print(f"  {tag}: 코호트의 {v['share_of_cohort'] * 100:.1f}% 를 쓰며 "
              f"악화 건의 {v['capture_of_events'] * 100:.1f}% 포착 "
              f"(정확도 {v['precision'] * 100:.1f}%, n={v['n']})")
    print(f"\n저장: {out_dir / 'validation' / 'grade_system_comparison.json'}")


if __name__ == "__main__":
    main()
