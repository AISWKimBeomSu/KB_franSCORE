"""표본 자격 게이트 완화 검증 — 변경관리 절차(OPERATIONS §6)에 따른 전후 비교.

무엇을 묻는가
    지금 점수가 나가는 브랜드는 2024 패널 9,333개 중 1,442개(15.5%)뿐이다.
    자격 조건이 **누적 최대 가맹점 30개 이상 · 3년 연속 관측**이기 때문이다.
    이걸 **10개 · 2년**으로 낮추면 훨씬 많은 브랜드를 볼 수 있다.
    그런데 낮춘 만큼 표본이 시끄러워진다 — 점포 12개짜리 브랜드는 2개만 닫아도
    −17% 다. 그래서 "커버리지가 늘었다"만으로는 채택 근거가 안 된다.

절차 (OPERATIONS §6 변경관리)
    ① 현행과 완화안을 **같은 방식으로** 워크포워드 재측정한다.
    ② 변경 전후 결과를 **양쪽 다** 공개한다. 좋아진 쪽만 보고하지 않는다.
    ③ 배포 승인 게이트(§3)로 판정한다:
         · Lift@10 이 persistence 기준모형 대비 +0.3 이상 (평균)
         · **그 우위가 fold t-구간에서 0 을 넘지 않을 것** ← 이게 진짜 게이트다
       통과하지 못하면 **채택하지 않고 그 사실을 기록**한다.

⚠️ 왜 t-구간까지 봐야 하는가 (이 도구가 처음엔 틀린 답을 냈다)
    처음에는 '평균 우위 ≥ 0.3' 하나만 봤고, 완화안 두 개 모두 **채택 가능**이 나왔다.
    실제로 적용해 보니 그게 아니었다. 게이트를 낮추면 fold 간 편차가 커져서
        현행   fold별 우위 +1.006 / +1.001 / +1.167  → t-CI [+0.824, +1.292]  유의
        완화B  fold별 우위 +0.280 / +0.493 / +0.591 / +1.497 → t-CI [-0.139, +1.570]  비유의
    가 된다. 즉 **이 프로젝트의 유일한 통계적 우위 주장이 사라진다.**

    README 는 연도블록 부트스트랩과 t-구간이 어긋날 때 좁은 쪽을 고르는 것은 정직하지
    않다고 이미 못박고 t-구간을 채택했다. 그 기준을 지금 바꿔 주장을 살리는 것은
    바로 그 문서가 금지한 체리피킹이다. 그래서 게이트에 t-구간을 넣는다.

⚠️ 이 도구는 운영 산출물을 건드리지 않는다. 임시 디렉터리에서만 돈다.
   (성능이 좋게 나올 때까지 조건을 바꿔 재측정하는 것을 막기 위해, 후보는
    이 파일 상단 GATES 에 **사전 선언**된 것만 돈다.)

산출
    outputs/gate_sensitivity.csv    게이트별 표본·성능
    outputs/gate_sensitivity.json   판정 결과와 근거

실행: python tools/gate_sensitivity.py
"""
from __future__ import annotations

import contextlib
import copy
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.common import get_logger, load_config, set_seed  # noqa: E402

log = get_logger("gate_sens")

# 사전 선언된 후보만 돈다 (사후 탐색 금지)
GATES: tuple[tuple[str, int, int], ...] = (
    ("현행", 30, 3),
    ("완화A", 20, 3),
    ("완화B", 10, 2),
)

LIFT_GATE = 0.3          # persistence 대비 최소 우위 (OPERATIONS §3-1)


def _run_one(base_cfg: dict, min_stores: int, min_years: int, tmp: Path) -> dict:
    """한 게이트 설정으로 라벨 재구성 → 워크포워드 재측정. 운영 경로는 안 건드린다."""
    from src import backtest
    from src.labels import build_labels
    from src.panel import apply_sample_filter

    cfg = copy.deepcopy(base_cfg)
    cfg["sample"]["min_stores"] = int(min_stores)
    cfg["sample"]["min_consecutive_years"] = int(min_years)
    proc = tmp / f"s{min_stores}_y{min_years}"
    outp = proc / "out"
    proc.mkdir(parents=True, exist_ok=True)
    outp.mkdir(parents=True, exist_ok=True)
    cfg["paths"] = {**base_cfg["paths"], "processed": proc, "outputs": outp}
    set_seed(cfg["seed"])

    src_proc = Path(base_cfg["paths"]["processed"])
    panel_full = pd.read_parquet(src_proc / "panel_full.parquet")
    panel = apply_sample_filter(panel_full, cfg)
    panel.to_parquet(proc / "panel.parquet", index=False)

    # 피처는 게이트와 무관하다(패널 전체 행으로 계산). 그대로 재사용해 비교를 깨끗이 한다.
    feats = pd.read_parquet(src_proc / "features.parquet")
    feats.to_parquet(proc / "features.parquet", index=False)

    labels = build_labels(panel, cfg)
    elig = panel.loc[panel["eligible_t"], ["brand_id", "year"]]
    labels = labels.merge(elig, on=["brand_id", "year"], how="inner")
    labels.to_parquet(proc / "labels.parquet", index=False)

    wf = backtest.run_walkforward(cfg)
    row = {
        "gate": f"점포{min_stores}+ · {min_years}년연속",
        "min_stores": min_stores, "min_consecutive_years": min_years,
        "n_eligible_rows": int(panel["eligible_t"].sum()),
        "n_labels": len(labels),
        "n_events": int(labels["label"].sum()),
        "base_rate": float(labels["label"].mean()) if len(labels) else float("nan"),
        "n_brands_labeled": int(labels["brand_id"].nunique()),
    }
    if wf.empty:
        row["note"] = "워크포워드 fold 성립 불가"
        return row
    macro = wf[wf.get("scope", "") == "macro_avg_folds"] if "scope" in wf else pd.DataFrame()
    src_tbl = macro if len(macro) else wf
    for m in ("lgbm", "persistence", "single", "logistic"):
        sub = src_tbl[src_tbl["model"] == m] if "model" in src_tbl else pd.DataFrame()
        if len(sub):
            for k in ("roc_auc", "pr_auc", "lift_at_10", "precision_at_10", "brier"):
                if k in sub:
                    row[f"{m}_{k}"] = float(sub[k].mean())
    # fold 간 변동 — 한 fold 가 끌어올린 평균인지 본다 (OPERATIONS §3-2 게이트)
    folds = wf[wf["scope"].astype(str).str.startswith("fold_test_")] if "scope" in wf \
        else pd.DataFrame()
    lg = folds[folds["model"] == "lgbm"] if len(folds) else pd.DataFrame()
    if len(lg):
        row["lgbm_lift_fold_min"] = float(lg["lift_at_10"].min())
        row["lgbm_lift_fold_max"] = float(lg["lift_at_10"].max())
        row["n_folds"] = len(lg)
    # ⚠️ 게이트를 낮추면 연속관측 요건이 짧아져 **fold 가 하나 더 생긴다**(2년연속은
    #    2020년 test 가 성립한다). 3-fold 평균과 4-fold 평균을 나란히 놓으면 그건
    #    같은 것을 비교한 게 아니다. 공통 fold 만 골라낸 값을 따로 싣는다.
    row["_folds"] = {
        str(r["scope"]).replace("fold_test_", ""): {
            "model": str(r["model"]), "lift_at_10": float(r["lift_at_10"]),
            "roc_auc": float(r["roc_auc"]), "n": int(r["n"]),
        } for _, r in folds.iterrows() if str(r["model"]) in ("lgbm", "persistence")
    } if len(folds) else {}
    row["_fold_rows"] = folds.to_dict("records") if len(folds) else []
    # 기준모형 대비 우위의 fold t-구간 — README 가 채택한 방식 그대로
    piv = folds.pivot_table(index="scope", columns="model", values="lift_at_10") \
        if len(folds) else pd.DataFrame()
    if {"lgbm", "persistence"} <= set(piv.columns):
        d = (piv["lgbm"] - piv["persistence"]).dropna().to_numpy()
        if len(d) >= 2:
            from scipy import stats
            m, s, n = float(d.mean()), float(d.std(ddof=1)), len(d)
            h = float(stats.t.ppf(0.975, n - 1)) * s / (n ** 0.5)
            row["edge_mean"] = m
            row["edge_ci_lo"] = m - h
            row["edge_ci_hi"] = m + h
            row["edge_significant"] = bool(m - h > 0)
            row["edge_folds"] = [round(float(x), 3) for x in d]
    return row


def main() -> int:
    base = load_config()
    out = Path(base["paths"]["outputs"])
    rows = []
    with tempfile.TemporaryDirectory(prefix="gate_sens_") as td:
        tmp = Path(td)
        for name, ms, my in GATES:
            log.info("게이트 %s — 점포 %d+ · %d년 연속", name, ms, my)
            r = _run_one(base, ms, my, tmp)
            r["name"] = name
            rows.append(r)

    # 공통 fold 만 남긴 apples-to-apples 비교
    fold_sets = [{str(f["scope"]).replace("fold_test_", "") for f in r["_fold_rows"]}
                 for r in rows if r.get("_fold_rows")]
    common = sorted(set.intersection(*fold_sets)) if fold_sets else []
    common_tab = []
    for r in rows:
        sub = [f for f in r.get("_fold_rows", [])
               if str(f["scope"]).replace("fold_test_", "") in common]
        d = {"name": r["name"], "gate": r["gate"], "common_folds": ",".join(common)}
        for m in ("lgbm", "persistence", "logistic"):
            v = [f for f in sub if str(f["model"]) == m]
            if v:
                d[f"{m}_lift"] = sum(x["lift_at_10"] for x in v) / len(v)
                d[f"{m}_auc"] = sum(x["roc_auc"] for x in v) / len(v)
        common_tab.append(d)
    common_df = pd.DataFrame(common_tab)

    for r in rows:                      # 표에 담지 않는 내부 필드 제거
        r.pop("_folds", None)
        r.pop("_fold_rows", None)

    tab = pd.DataFrame(rows)
    cur = tab[tab["name"] == "현행"].iloc[0] if (tab["name"] == "현행").any() else None

    verdicts = []
    for r in tab.to_dict("records"):
        if r["name"] == "현행":
            continue
        lift = r.get("lgbm_lift_at_10")
        pers = r.get("persistence_lift_at_10")
        edge = (lift - pers) if (lift is not None and pers is not None) else None
        cov_gain = (r["n_labels"] / cur["n_labels"] - 1) if cur is not None else None
        mean_ok = edge is not None and edge >= LIFT_GATE
        sig_ok = bool(r.get("edge_significant"))
        ok = mean_ok and sig_ok
        why = []
        if edge is not None:
            why.append(f"평균 우위 {edge:.3f} {'≥' if mean_ok else '<'} {LIFT_GATE}")
        if r.get("edge_ci_lo") is not None:
            why.append(f"t-구간 [{r['edge_ci_lo']:+.3f}, {r['edge_ci_hi']:+.3f}] "
                       f"{'0 초과 — 유의' if sig_ok else '0 포함 — 비유의'}")
        verdicts.append({
            "name": r["name"], "gate": r["gate"],
            "n_labels": r["n_labels"], "n_events": r["n_events"],
            "label_gain_pct": round(100 * cov_gain, 1) if cov_gain is not None else None,
            "lgbm_lift": lift, "persistence_lift": pers,
            "edge_over_persistence": edge,
            "edge_ci": [r.get("edge_ci_lo"), r.get("edge_ci_hi")],
            "edge_folds": r.get("edge_folds"),
            "gate_pass": bool(ok),
            "reason": " · ".join(why) if why else "측정 불가",
        })

    tab.to_csv(out / "gate_sensitivity.csv", index=False, encoding="utf-8-sig")
    common_df.to_csv(out / "gate_sensitivity_common_folds.csv", index=False,
                     encoding="utf-8-sig")
    (out / "gate_sensitivity.json").write_text(json.dumps({
        "candidates_predeclared": [{"name": n, "min_stores": a, "min_consecutive_years": b}
                                   for n, a, b in GATES],
        "lift_gate": LIFT_GATE,
        "rows": rows, "verdicts": verdicts,
        "common_folds": common,
        "common_fold_comparison": common_tab,
        "caveat_folds": ("연속관측 요건을 낮추면 fold 가 하나 더 생긴다. macro 평균은 "
                         "fold 수가 달라 직접 비교가 안 되므로 공통 fold 비교를 함께 싣는다."),
        "procedure": "OPERATIONS §6 변경관리 — 전후 양쪽 공개, §3 배포 게이트로 판정",
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    cols = ["name", "gate", "n_labels", "n_events", "base_rate",
            "lgbm_auc", "lgbm_lift_at_10", "persistence_lift_at_10"]
    print(tab[[c for c in cols if c in tab]].to_string(index=False))
    print()
    print(f"--- 공통 fold({', '.join(common)}) 만으로 재비교 ---")
    print(common_df.to_string(index=False))
    print()
    for v in verdicts:
        print(f"  [{'채택 가능' if v['gate_pass'] else '기각'}] {v['name']} ({v['gate']}) — "
              f"라벨 {v['n_labels']:,}건({v['label_gain_pct']:+.0f}%) · {v['reason']}")
    log.info("gate_sensitivity.csv · gate_sensitivity.json 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
