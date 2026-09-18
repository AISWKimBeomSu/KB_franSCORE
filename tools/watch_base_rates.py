"""요주의 구간 실현율표 — 모형이 없는 자리에 '측정'을 놓는다.

문제
    이 도구가 매기는 점수의 모집단은 **t년에 건전한 브랜드**다(healthy gate).
    그런데 실제로 점수가 나가는 1,442개 중 496개(34.4%)는 t년에 이미 악화사건이
    발동한 '요주의'다. 이 구간은 labels.parquet 에 행 자체가 없으므로 모형의
    판별력 근거가 없다 — 문서는 그 사실을 밝혀 왔다.

왜 모형으로 메우지 않았나 (실측으로 기각)
    요주의 전용 LightGBM 을 따로 학습해 봤다. 워크포워드 macro AUC **0.6937**.
    같은 표본에서 계약종료율 **단일 변수**가 **0.7066** 이다. 변수를 14개 쓴 모형이
    변수 1개보다 못하다. 표본이 작고(사건 재발동은 이미 흔하다) 신호가 얕다.
    모형화가 답이 아니라는 뜻이다.

무엇이 정직한 상한인가
    "이미 사건이 k건 떴다"는 사실 하나만으로도 다음 해 재발동률은 크게 갈린다.
    그렇다면 **그 실현율을 그대로 세어서 공표**하는 것이 이 구간에서 할 수 있는
    최선이다. 신용평가사가 등급별 장기 평균부도율을 공표해 등급을 앵커링하는 것과
    같은 장치이며, 모형이 아니므로 과적합도 설명불가도 없다.

    ⚠️ 이 표는 **순위를 매기지 않는다.** 같은 k 안에서는 전부 같은 확률이다.
       그것이 이 구간에 대해 우리가 아는 전부다. 더 아는 척하지 않는다.

산출
    outputs/watch_base_rates.csv    k별 실현율 + Wilson 95% 구간 + 연도별 안정성
    outputs/watch_base_rates.json   화면·문서가 읽는 요약

실행: python tools/watch_base_rates.py
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.common import get_logger, load_config  # noqa: E402
from src.labels import brand_state  # noqa: E402

log = get_logger("watch_rates")

Z = 1.959963984540054          # 95% 양측


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson 점수구간. 정규근사(Wald)를 쓰지 않는 이유: n 이 작거나 p 가 0·1 에
    가까우면 Wald 구간이 [0,1] 밖으로 나가고 피복률이 무너진다."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def build(cfg: dict) -> tuple[pd.DataFrame, dict]:
    proc = Path(cfg["paths"]["processed"])
    panel = pd.read_parquet(proc / "panel.parquet")

    st = brand_state(panel, cfg)
    st["year"] = st["year"].astype(int)

    # t 와 t+1 을 붙인다. **연속 연도만** — 갭이 있으면 t+1 이 아니다.
    nxt = st[["brand_id", "year", "deteriorated_at_t", "state"]].copy()
    nxt["year"] -= 1
    nxt = nxt.rename(columns={"deteriorated_at_t": "next_deteriorated",
                              "state": "next_state"})
    df = st.merge(nxt, on=["brand_id", "year"], how="inner")

    # 점수가 실제로 나가는 표본과 같은 자격 게이트를 건다 — 표를 적용할 대상과
    # 표를 만든 대상이 다르면 그 표는 그 대상의 실적이 아니다.
    if "eligible_t" in panel.columns:
        el = panel[["brand_id", "year", "eligible_t"]].copy()
        el["year"] = el["year"].astype(int)
        df = df.merge(el, on=["brand_id", "year"], how="left")
        df = df[df["eligible_t"].fillna(False).astype(bool)]

    # t+1 상태가 '평가불가'면 재발동 여부를 알 수 없다 — 0 으로 세면 과소보고다.
    n_unknown = int((df["next_state"] == "평가불가").sum())
    df = df[df["next_state"] != "평가불가"]

    watch = df[df["state"] == "요주의"].copy()
    healthy = df[df["state"] == "건전"].copy()

    rows = []
    # 건전 구간을 같은 표에 넣는다 — 대조군이 없으면 '높다'는 말이 성립하지 않는다.
    k_h = int(healthy["next_deteriorated"].sum())
    n_h = len(healthy)
    lo, hi = wilson(k_h, n_h)
    rows.append({"state": "건전", "n_events_at_t": 0, "n": n_h, "n_deteriorated": k_h,
                 "rate": k_h / n_h if n_h else np.nan, "ci_low": lo, "ci_high": hi})

    for k in sorted(watch["n_events_at_t"].unique()) if "n_events_at_t" in watch \
            else sorted(watch["n_events"].unique()):
        sub = watch[watch["n_events"] == k]
        if len(sub) == 0:
            continue
        kk = int(sub["next_deteriorated"].sum())
        nn = len(sub)
        lo, hi = wilson(kk, nn)
        rows.append({"state": "요주의", "n_events_at_t": int(k), "n": nn,
                     "n_deteriorated": kk, "rate": kk / nn, "ci_low": lo, "ci_high": hi})

    tab = pd.DataFrame(rows)

    # 연도별 안정성 — 한 해만 보고 만든 표는 다음 해에 무너질 수 있다
    yr_rows = []
    for (state, k, y), sub in df.assign(
            _k=np.where(df["state"] == "건전", 0, df["n_events"])
    ).groupby(["state", "_k", "year"]):
        if state == "평가불가" or len(sub) < 20:
            continue
        kk = int(sub["next_deteriorated"].sum())
        yr_rows.append({"state": state, "n_events_at_t": int(k), "year": int(y),
                        "n": len(sub), "rate": kk / len(sub)})
    yearly = pd.DataFrame(yr_rows)

    meta = {
        "n_pairs": len(df),
        "n_watch": len(watch),
        "n_healthy": len(healthy),
        "n_excluded_next_unknown": n_unknown,
        "years": sorted(int(y) for y in df["year"].unique()),
        "outcome": "t+1 년에 악화사건이 다시 발동(라벨과 동일한 사건 정의)",
        "sample_gate": "점수 산출과 동일한 eligible_t (누적 최대 점포 30+ · 3년 연속)",
        "why_not_a_model": ("요주의 전용 LightGBM 워크포워드 macro AUC 0.6937 < "
                            "계약종료율 단일변수 0.7066. 변수 14개가 1개보다 못해 "
                            "모형화를 기각하고 실현율 공표로 대체했다."),
        "caveat": ("이 표는 순위를 매기지 않는다. 같은 사건수 안에서는 모두 같은 "
                   "확률이며, 그것이 이 구간에 대해 아는 전부다."),
    }
    return tab, {"summary": meta, "yearly": yearly.to_dict("records")}


def main() -> int:
    cfg = load_config()
    out = Path(cfg["paths"]["outputs"])
    tab, extra = build(cfg)

    tab.to_csv(out / "watch_base_rates.csv", index=False, encoding="utf-8-sig")
    payload = {**extra["summary"], "table": tab.to_dict("records"),
               "yearly": extra["yearly"]}
    (out / "watch_base_rates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"쌍 {extra['summary']['n_pairs']:,}건 "
          f"(요주의 {extra['summary']['n_watch']:,} / 건전 {extra['summary']['n_healthy']:,}) · "
          f"연도 {extra['summary']['years']}")
    print(f"t+1 판정불가로 제외: {extra['summary']['n_excluded_next_unknown']:,}건")
    print()
    print(f"{'상태':<8}{'사건수':>6}{'표본':>8}{'재발동':>8}{'실현율':>9}   95% 구간")
    for r in tab.to_dict("records"):
        print(f"{r['state']:<8}{r['n_events_at_t']:>6}{r['n']:>8,}{r['n_deteriorated']:>8,}"
              f"{r['rate']*100:>8.1f}%   [{r['ci_low']*100:.1f}, {r['ci_high']*100:.1f}]")
    log.info("watch_base_rates.csv · watch_base_rates.json 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
