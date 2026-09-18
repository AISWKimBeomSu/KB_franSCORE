"""두 표본 트랙(명세 기본 / 플랜B 확장)을 모두 학습·평가·워크포워드 검증하고 비교표를 만든다.

⚠️ 체리피킹 방지 설계: 두 트랙 모두 **항상 함께** 실행·보고한다.
   좋은 쪽만 골라 보고하지 않으며, 비교표(outputs/track_comparison.csv)에 둘 다 남긴다.

사용법: python run_both_tracks.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_pipeline import _extended_cfg
from src.backtest import run_walkforward
from src.common import get_logger, load_config

log = get_logger("both_tracks")
STEPS = ["panel", "features", "labels", "model", "evaluate", "portfolio"]


def _run(scope: str) -> None:
    for step in STEPS:
        cmd = [sys.executable, "run_pipeline.py", "--step", step, "--scope", scope]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            log.error("[%s/%s] 실패:\n%s", scope, step, (r.stderr or "")[-2000:])
            raise SystemExit(1)
        log.info("[%s] step %s 완료", scope, step)


def main() -> None:
    cfg = load_config()
    rows: list[dict] = []

    for scope in ("primary", "extended"):
        log.info("=== 트랙 실행: %s ===", scope)
        _run(scope)
        c = _extended_cfg(cfg) if scope == "extended" else cfg
        wf = run_walkforward(c)                       # 워크포워드 (검정력 보강)
        out = Path(c["paths"]["outputs"])

        single = pd.read_csv(out / "metrics.csv")
        single = single[single["split"] == "test"]
        for r in single.itertuples():
            rows.append({"track": scope, "eval": "single_split_test", "model": r.model,
                         "lift_at_10": r.lift_at_10, "precision_at_10": r.precision_at_10,
                         "pr_auc": r.pr_auc, "roc_auc": r.roc_auc, "brier": r.brier,
                         "n": r.n, "base_rate": r.base_rate})
        if not wf.empty:
            # 워크포워드 집계 3종을 모두 싣는다 (주 지표 = macro_avg_folds).
            # 원점수 풀링(pooled_oos_raw)은 fold별 확률 스케일 차이로 왜곡되지만
            # 숨기지 않고 함께 공개한다 — backtest.py 상단 '집계 방식' 참조.
            for sc in ("macro_avg_folds", "pooled_oos_rank", "pooled_oos_raw"):
                for r in wf[wf["scope"] == sc].itertuples():
                    rows.append({"track": scope, "eval": f"walkforward_{sc}", "model": r.model,
                                 "lift_at_10": r.lift_at_10, "precision_at_10": r.precision_at_10,
                                 "pr_auc": r.pr_auc, "roc_auc": r.roc_auc, "brier": r.brier,
                                 "n": r.n, "base_rate": r.base_rate})
        ci_path = out / "walkforward_delta_ci.csv"
        if ci_path.exists():
            ci = pd.read_csv(ci_path)
            ci.insert(0, "track", scope)
            ci.to_csv(out / "walkforward_delta_ci.csv", index=False, encoding="utf-8-sig")

    comp = pd.DataFrame(rows)
    dest = Path(cfg["paths"]["outputs"]) / "track_comparison.csv"
    comp.to_csv(dest, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("두 트랙 비교 (기본=외식·점포30+ / 확장=전업종·점포20+) — 둘 다 공개")
    print("=" * 78)
    for ev in ("single_split_test", "walkforward_macro_avg_folds",
               "walkforward_pooled_oos_rank", "walkforward_pooled_oos_raw"):
        sub = comp[comp["eval"] == ev]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="model", columns="track", values="lift_at_10")
        print(f"\n[{ev}] Lift@10%")
        print(piv.round(3).to_string())
    print(f"\n→ 저장: {dest}")


if __name__ == "__main__":
    main()
