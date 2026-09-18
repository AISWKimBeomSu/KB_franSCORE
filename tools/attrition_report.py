"""표본 이탈 회계 — 브랜드가 패널에서 사라지는 경로를 숫자로 공개한다.

왜 필요한가
    모형은 't 년 건전 브랜드가 t+1 에 악화하는가'를 학습한다. 그러려면 t+1 행이 있어야
    하고, 없으면 표본에서 빠진다(src/labels.py:212 inner merge). 그런데 **t+1 행이
    없다는 것 자체가 나쁜 소식일 수 있다** — 브랜드가 없어졌을 수 있기 때문이다.
    이것을 조용히 빼면 생존편의(survivorship bias)가 되고, 성능이 실제보다 좋아 보인다.

    이 도구는 그 이탈을 세어 공개한다. **라벨은 바꾸지 않는다.**

왜 라벨에 넣지 않는가 (실측 근거)
    소멸을 양성으로 편입하면 양성이 335 → 384(9.22% → 10.42%)로 는다. 그런데
      · 우측절단 때문에 추가 양성률이 train 0.76% vs test 1.86% 로 2.4배 갈린다.
        train 에서 덜 보이는 사건을 test 에서 더 세면 백테스트가 무효가 된다.
      · 소멸 49건 중 14건(28.6%)은 같은 가맹본부(hq_mnno)의 다른 브랜드가 다음 해에
        살아 있다. 브랜드 정리이지 본부 도산이 아니다.
      · 등록취소 사유에 폐업·파산·회생은 소멸 브랜드의 5.4% 뿐이고 '부도'는 0건이다.
        자진취소가 76%다 — 자진취소는 사업 종료일 수도, 브랜드 통합일 수도 있다.
    즉 '사라짐 = 부실'이 아니다. 세어서 보여 주되 라벨로 삼지 않는 이유가 이것이다.

산출
    outputs/attrition_report.csv    연도별 이탈 분해
    outputs/attrition_summary.json  요약 + 등록취소 대조

실행: python tools/attrition_report.py
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.common import get_logger, load_config  # noqa: E402

log = get_logger("attrition")


def cancel_events(cfg: dict) -> pd.DataFrame:
    """브랜드 단위 등록취소 사건 (brand_id 가 아니라 정규화 명칭 기준).

    ⚠️ 패널에 연도별 플래그로 붙이지 않는다. 취소 접수연도에는 그 브랜드의 패널 행이
       이미 없는 경우가 대부분이라 대다수가 안 붙고(실측: 원본 10,441건 중 패널 241행),
       그렇다고 연도키를 빼고 붙이면 2023년 취소가 2019년 행에 얹혀 **미래 정보가
       과거로 샌다**. 그래서 별도 사건표로 둔다 — 이탈 회계와 화면 표시에만 쓴다.
    """
    from src.collect import load_snapshots
    from src.panel import normalize_name

    try:
        rows = load_snapshots(cfg, "brand_cancel")
    except Exception:
        rows = []
    if not rows:
        return pd.DataFrame(columns=["norm_brand", "cancel_year", "cancel_type", "reason"])
    cx = pd.DataFrame(rows)
    cx["norm_brand"] = cx["brandNm"].map(normalize_name)
    cx = cx[cx["norm_brand"] != ""].copy()
    yr = pd.to_numeric(cx.get("rcptDate", pd.Series(dtype=str)).astype(str).str[:4],
                       errors="coerce")
    cx["cancel_year"] = yr.fillna(pd.to_numeric(cx["_yr"], errors="coerce") - 1)
    cx = cx.dropna(subset=["cancel_year"])
    cx["cancel_year"] = cx["cancel_year"].astype(int)
    out = (cx.sort_values("cancel_year")
             .groupby("norm_brand", as_index=False)
             .agg(cancel_year=("cancel_year", "first"),
                  cancel_type=("rgsRtrcnTyNm", "first"),
                  reason=("rgsRtrcnRsnCn", "first")))
    return out


def main() -> int:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["outputs"])
    proc = Path(cfg["paths"]["processed"])

    panel = pd.read_parquet(proc / "panel.parquet")
    years = sorted(panel["year"].astype(int).unique())
    log.info("패널 %d행 · 브랜드 %d개 · 연도 %s",
             len(panel), panel["brand_id"].nunique(), years)

    last_year = max(years)
    seen = panel.groupby("brand_id")["year"].apply(lambda s: set(s.astype(int)))

    rows = []
    for y in years[:-1]:
        cur = set(panel.loc[panel["year"].astype(int) == y, "brand_id"])
        nxt = set(panel.loc[panel["year"].astype(int) == y + 1, "brand_id"])
        gone = cur - nxt
        # 사라진 뒤 **다시 돌아오는** 브랜드는 소멸이 아니라 공시 누락(갭)이다.
        # 이 둘을 섞으면 이탈률이 부풀려진다.
        real, gap = set(), set()
        for b in gone:
            (real if not any(v > y + 1 for v in seen[b]) else gap).add(b)
        # ⚠️ 마지막 관측 연도 직전 해는 '갭복귀'를 볼 수 없다 — 돌아올 연도가 아직
        #    없기 때문이다(우측절단). 그래서 그 해의 소멸 수는 **상한**이지 실측이 아니다.
        #    이 구분을 안 하면 마지막 해 이탈률이 실제보다 높게 보인다.
        censored = (y + 1 == last_year)
        rows.append({
            "year": y, "n_brands": len(cur), "n_next": len(nxt),
            "n_gone": len(gone), "n_disappeared": len(real), "n_gap_return": len(gap),
            "disappear_rate": round(len(real) / len(cur), 5) if cur else 0.0,
            "censored": censored,
            "note": "우측절단 — 갭복귀 관측 불가, 소멸 수는 상한" if censored else "",
        })
    rep = pd.DataFrame(rows)
    rep.to_csv(out_dir / "attrition_report.csv", index=False, encoding="utf-8-sig")

    # 등록취소 사건과 대조 — 사라진 브랜드 중 몇 개가 공식 취소 기록을 갖는가
    ce = cancel_events(cfg)
    from src.panel import normalize_name
    panel = panel.assign(_nb=panel["brand_name"].map(normalize_name))
    nb = panel.drop_duplicates("brand_id").set_index("brand_id")["_nb"]
    cancel_by_name = set(ce["norm_brand"])

    disappeared, with_record = 0, 0
    for y in years[:-1]:
        cur = set(panel.loc[panel["year"].astype(int) == y, "brand_id"])
        nxt = set(panel.loc[panel["year"].astype(int) == y + 1, "brand_id"])
        for b in cur - nxt:
            if any(v > y + 1 for v in seen[b]):
                continue
            disappeared += 1
            if nb.get(b) in cancel_by_name:
                with_record += 1

    types = ce["cancel_type"].value_counts().to_dict() if len(ce) else {}
    summary = {
        "panel_years": [int(y) for y in years],
        "n_brands_panel": int(panel["brand_id"].nunique()),
        "n_disappeared_total": disappeared,
        "n_disappeared_with_cancel_record": with_record,
        "cancel_events_total": len(ce),
        "cancel_types": {str(k): int(v) for k, v in types.items()},
        "note": ("소멸은 라벨에 넣지 않는다. 우측절단으로 학습·검증 기간의 관측률이 달라 "
                 "백테스트가 무효화되고, 등록취소 사유의 대부분이 자진취소라 부실과 "
                 "동일시할 수 없기 때문이다. 이 표는 생존편의의 크기를 공개하는 용도다."),
        "last_year_note": f"{int(last_year)}년은 다음 해가 없어 이탈을 관측할 수 없다",
    }
    (out_dir / "attrition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(rep.to_string(index=False))
    print()
    pct = with_record / max(1, disappeared) * 100
    print(f"소멸 브랜드 {disappeared}개 중 등록취소 공식 기록 보유 {with_record}개 ({pct:.1f}%)")
    print(f"등록취소 사건 총 {len(ce)}건 · 유형 {types}")
    log.info("attrition_report.csv · attrition_summary.json 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
