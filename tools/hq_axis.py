"""본부 축 실증 — 같은 가맹본부의 형제 브랜드가 위험을 옮기는가.

왜 이 축인가
    본부 재무 커버리지는 브랜드 11.6% / 점포가중 38.4% 에 막혀 있다(정보공개서
    정식 키가 없으면 구조적으로 더 못 늘린다). 그런데 **가맹본부관리번호(hq_mnno)**
    는 배포 코호트의 98.8% 가 갖고 있다. 재무를 못 보더라도 "이 본부의 다른
    브랜드가 지금 어떤가"는 전량 볼 수 있다.

    그리고 이 프로젝트는 이미 브랜드 간 상관 ρ_B = 0.005(거의 무관)를 측정했다.
    그렇다면 **같은 본부 안에서는 다른가**가 자연스러운 다음 질문이다. 다르다면
    관리 단위가 하나 더 생기고, 같다면 "브랜드가 위험의 단위"라는 결론이 강화된다.
    어느 쪽이든 답이 된다.

무엇을 측정하는가
    t년에 같은 본부의 **형제 브랜드**가 악화했을 때, 이 브랜드의 t+1 악화율이
    올라가는가. 표본·게이트·사건 정의는 watch_base_rates 와 완전히 동일하게 둔다
    (eligible_t · t+1 평가불가 제외 · 같은 악화 정의).

⚠️ 교란요인 둘을 반드시 분리해야 한다 — 안 하면 있지도 않은 신호를 보고하게 된다
    ① **자기 사건수(k)** — 요주의 브랜드는 k 가 클수록 재발동률이 높다
       (실측 24.1% / 45.9% / 64.8%). 형제가 악화한 본부의 브랜드는 자기 k 도
       조금 높다(1.410 → 1.536). k 를 고정하지 않으면 그 차이를 형제 효과로 오인한다.
    ② **형제 수(본부 규모)** — 형제가 많을수록 '형제 중 하나가 악화할' 확률이
       기계적으로 오른다. 게다가 형제 수 자체가 위험이다(실측: 요주의 k=1 에서
       형제 1~2 는 20.25%, 형제 3+ 는 29.20%, Fisher p=0.0399 — 형제가 악화하지
       **않은** 표본만 놓고 잰 값이다).

    그래서 귀무모형을 이렇게 짠다: **본부 배정과 형제 수는 그대로 두고, 어느
    브랜드가 악화했는지만 연도 안에서 치환**한다. 본부를 통째로 섞으면 형제 수가
    같이 바뀌어 ② 를 못 잡는다(첫 설계가 그랬고, 그 귀무는 관측을 너무 쉽게 기각했다).

산출
    outputs/hq_axis.json / hq_axis.csv

실행: python tools/hq_axis.py [--n-perm 300]
"""
from __future__ import annotations

import argparse
import contextlib
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

from src.common import get_logger, load_config, set_seed  # noqa: E402
from src.labels import brand_state  # noqa: E402

log = get_logger("hq_axis")
Z = 1.959963984540054
TARGET_K = (1, 2)          # 사전 선언: 표본이 충분한 요주의 구간만 본다


def wilson(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * ((p * (1 - p) / n + Z * Z / (4 * n * n)) ** 0.5)
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def _sib(frame: pd.DataFrame, det_col: str) -> tuple[pd.Series, pd.Series]:
    """(형제 수, 형제 악화 수) — 자기 자신은 뺀다."""
    g = frame.groupby(["hq_mnno", "year"])
    n = g["brand_id"].transform("size")
    s = g[det_col].transform("sum")
    return (n - 1).astype(int), (s - frame[det_col]).astype(int)


def build(cfg: dict, n_perm: int = 300) -> tuple[pd.DataFrame, dict]:
    proc = Path(cfg["paths"]["processed"])
    panel = pd.read_parquet(proc / "panel.parquet")
    full = pd.read_parquet(proc / "panel_full.parquet")

    st = brand_state(panel, cfg)
    st["year"] = st["year"].astype(int)
    hq = full[["brand_id", "year", "hq_mnno"]].copy()
    hq["year"] = hq["year"].astype(int)
    st = st.merge(hq, on=["brand_id", "year"], how="left").dropna(subset=["hq_mnno"])
    st["sib_n"], st["sib_det"] = _sib(st, "deteriorated_at_t")

    nxt = st[["brand_id", "year", "deteriorated_at_t", "state"]].copy()
    nxt["year"] -= 1
    nxt = nxt.rename(columns={"deteriorated_at_t": "next_det", "state": "next_state"})
    df = st.merge(nxt, on=["brand_id", "year"], how="inner")

    el = panel[["brand_id", "year", "eligible_t"]].copy()
    el["year"] = el["year"].astype(int)
    df = df.merge(el, on=["brand_id", "year"], how="left")
    df = df[df["eligible_t"].fillna(False).astype(bool)]
    df = df[df["next_state"] != "평가불가"]
    df = df[df["sib_n"] >= 1]

    rows: list[dict] = []
    for own in ("건전", "요주의"):
        sub = df[df["state"] == own]
        for k in sorted(sub["n_events"].unique()):
            s = sub[sub["n_events"] == k]
            a, b = s[s["sib_det"] == 0], s[s["sib_det"] >= 1]
            if len(a) < 30 or len(b) < 30:
                continue
            ka, na, kb, nb = int(a["next_det"].sum()), len(a), int(b["next_det"].sum()), len(b)
            la, ha = wilson(ka, na)
            lb, hb = wilson(kb, nb)
            _, p = stats.fisher_exact([[kb, nb - kb], [ka, na - ka]])
            rows.append({
                "state": own, "n_events_at_t": int(k),
                "n_sib_ok": na, "rate_sib_ok": ka / na, "ci_lo_sib_ok": la, "ci_hi_sib_ok": ha,
                "n_sib_bad": nb, "rate_sib_bad": kb / nb, "ci_lo_sib_bad": lb, "ci_hi_sib_bad": hb,
                "delta": kb / nb - ka / na, "fisher_p": float(p),
            })
    tab = pd.DataFrame(rows)

    # 교란 ② 의 크기 — 형제가 악화하지 **않은** 표본만으로 형제 수 효과를 잰다
    size_effect = []
    for k in TARGET_K:
        s = df[(df["state"] == "요주의") & (df["n_events"] == k) & (df["sib_det"] == 0)]
        small, big = s[s["sib_n"] <= 2], s[s["sib_n"] >= 3]
        if len(small) < 30 or len(big) < 30:
            continue
        _, p = stats.fisher_exact([
            [int(big["next_det"].sum()), len(big) - int(big["next_det"].sum())],
            [int(small["next_det"].sum()), len(small) - int(small["next_det"].sum())]])
        size_effect.append({"n_events_at_t": k,
                            "n_small": len(small), "rate_small": float(small["next_det"].mean()),
                            "n_big": len(big), "rate_big": float(big["next_det"].mean()),
                            "fisher_p": float(p)})

    # 귀무모형 — 본부 배정·형제 수 고정, 악화 브랜드만 연도 내 치환
    tgt = df[(df["state"] == "요주의") & (df["n_events"].isin(TARGET_K))]
    obs = float(tgt[tgt["sib_det"] >= 1]["next_det"].mean()
                - tgt[tgt["sib_det"] == 0]["next_det"].mean())
    key = tgt[["brand_id", "year", "next_det"]]
    rng = np.random.default_rng(int(cfg["seed"]))
    null: list[float] = []
    for _ in range(n_perm):
        p2 = st.copy()
        p2["_d"] = p2.groupby("year")["deteriorated_at_t"].transform(
            lambda s: rng.permutation(s.values))
        _, sd = _sib(p2, "_d")
        m = key.merge(p2.assign(_sd=sd)[["brand_id", "year", "_sd"]],
                      on=["brand_id", "year"], how="inner")
        a, b = m[m["_sd"] == 0], m[m["_sd"] >= 1]
        if len(a) < 20 or len(b) < 20:
            continue
        null.append(float(b["next_det"].mean() - a["next_det"].mean()))
    nl = np.array(null)
    p_val = float((nl >= obs).mean()) if len(nl) else float("nan")

    meta = {
        "n_rows": len(df),
        "n_brands": int(df["brand_id"].nunique()),
        "hq_coverage_in_panel": float(1 - full["hq_mnno"].isna().mean()),
        "target_k": list(TARGET_K),
        "observed_delta": obs,
        "null": {
            "design": "본부 배정·형제 수 고정, 악화 브랜드만 연도 내 치환",
            "n_perm": len(nl),
            "mean": float(nl.mean()) if len(nl) else None,
            "sd": float(nl.std()) if len(nl) else None,
            "p95": float(np.percentile(nl, 95)) if len(nl) else None,
            "p_value": p_val,
        },
        "confound_share": float(nl.mean() / obs) if len(nl) and obs else None,
        "hq_size_effect": size_effect,
        "verdict": None,      # 아래에서 채운다
        "caveat": (
            "형제 신호와 본부 규모는 완전히 분리되지 않는다. 귀무 평균이 0 이 아니라는 "
            "사실이 그 증거다 — 관측 효과의 상당 부분이 '형제가 많은 본부가 원래 더 "
            "위험하다'로 설명된다. 이 표는 모형 입력이 아니라 **측정 기록**이다."),
    }
    meta["verdict"] = (
        "귀무 기각(p<0.05) — 다만 관측 효과의 상당분이 본부 규모 교란이다"
        if p_val < 0.05 else "귀무 기각 실패 — 형제 전이 신호를 확인하지 못했다")
    return tab, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="본부 축 실증")
    ap.add_argument("--n-perm", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    tab, meta = build(cfg, args.n_perm)

    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "hq_axis.csv", index=False, encoding="utf-8-sig")
    (out / "hq_axis.json").write_text(
        json.dumps({"table": tab.to_dict("records"), **meta}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    print(f"분석 대상 {meta['n_rows']:,}행 · 브랜드 {meta['n_brands']:,} "
          f"· 패널 본부번호 보유 {meta['hq_coverage_in_panel']:.1%}\n")
    for r in tab.to_dict("records"):
        print(f"  {r['state']} k={r['n_events_at_t']}  "
              f"형제 정상 {r['rate_sib_ok']:6.2%}(n={r['n_sib_ok']:>4})  "
              f"형제 악화 {r['rate_sib_bad']:6.2%}(n={r['n_sib_bad']:>4})  "
              f"Δ{r['delta']:+.2%}  p={r['fisher_p']:.4f}")
    n = meta["null"]
    print(f"\n귀무모형({n['design']}, {n['n_perm']}회)")
    print(f"  관측 Δ {meta['observed_delta']:+.4f} · 귀무 평균 {n['mean']:+.4f} "
          f"· 95분위 {n['p95']:+.4f} · p={n['p_value']:.4f}")
    if meta["confound_share"] is not None:
        print(f"  → 관측 효과의 {meta['confound_share']:.0%} 는 본부 규모 교란으로 설명된다")
    print(f"  판정: {meta['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
