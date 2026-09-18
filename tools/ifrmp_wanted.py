"""손으로 받을 정보공개서 **우선순위 목록**을 뽑는다.

왜 이 도구가 필요한가
    정보공개서 정식 키(IFRMP_SERVICE_KEY)가 언제 나올지 알 수 없다. 그런데 이
    문서는 누구나 열람할 수 있게 공개돼 있으므로, **사람이 브라우저로 몇십 건만
    받아도** 본부 재무 공백이 크게 메워진다. 문제는 "무엇을 먼저 받는가"다.

    브랜드 개수로 세면 11,751건 중 몇십 건은 무의미해 보인다. 그러나 여신 관점의
    분모는 브랜드가 아니라 **점포(=차주) 수**다. 점포가 많은 브랜드부터 채우면
    상위 30건만으로도 점포가중 커버리지가 10%p 넘게 오른다.

    그래서 "많이 받으세요"가 아니라 **"이 순서로 받으세요"** 를 출력한다.
    사람의 시간이 가장 비싼 자원이므로, 한 건당 얻는 커버리지가 큰 것부터 준다.

    ⚠️ 여기에 구체적 수치를 적어 두지 않는다 — 한 건 받을 때마다 바뀌므로 주석이
       반드시 낡는다. 현재 값은 이 도구를 실행하면 그 자리에서 다시 계산된다.

⚠️ 캡차를 우회하지 않는다. 이 목록은 자동 수집용이 아니라 **사람이 열람할 순서**다.
   저장한 파일은 `python -m src.ifrmp_web --ingest-dir <폴더>` 로 넣는다.

실행:
    python tools/ifrmp_wanted.py                # 상위 30개 (기본)
    python tools/ifrmp_wanted.py --top 100      # 상위 100개
    python tools/ifrmp_wanted.py --csv          # CSV 로 저장
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

VIEWER = "https://franchise.ftc.go.kr/mnu/00013/program/userRqst/list.do"
OUT = _ROOT / "outputs" / "ifrmp_wanted.csv"


def wanted() -> tuple[pd.DataFrame, dict]:
    """(우선순위 프레임, 커버리지 요약). 이미 확보한 브랜드는 뺀다."""
    sc = pd.read_csv(_ROOT / "outputs" / "scores_latest.csv")
    feats = pd.read_parquet(_ROOT / "data" / "processed" / "features.parquet")
    cur = feats[feats["year"] == feats["year"].max()][["brand_id", "f_hq_has_financials"]]
    m = sc.merge(cur, on="brand_id", how="left")
    m["has_dart"] = m["f_hq_has_financials"].fillna(0) > 0

    # 웹 열람으로 이미 받아 둔 것도 보유로 친다 (같은 브랜드를 두 번 받게 하지 않는다).
    # ⚠️ 키는 **등록번호**로 맞춘다. 브랜드명으로 맞추면 공시 등록명과 표기가 달라
    #    이미 받은 건이 목록에 계속 남는다 — 실제로 4건이 그렇게 남아 있었다.
    web = _ROOT / "data" / "processed" / "ifrmp_web_financials.parquet"
    got_reg: set[str] = set()
    if web.exists():
        w = pd.read_parquet(web)
        if "reg_no" in w.columns:
            got_reg = set(w["reg_no"].astype(str))
    m["has_web"] = m["brand_id"].astype(str).str.replace("^BRD_", "", regex=True).isin(got_reg)
    m["has"] = m["has_dart"] | m["has_web"]

    total_stores = float(m["n_stores"].sum())
    base = float(m.loc[m["has"], "n_stores"].sum())
    miss = m.loc[~m["has"]].sort_values("n_stores", ascending=False).copy()
    miss["누적점포"] = miss["n_stores"].cumsum()
    miss["누적_점포가중_커버리지"] = (base + miss["누적점포"]) / total_stores
    summary = {
        "cohort_brands": len(m),
        "cohort_stores": int(total_stores),
        "have_brands": int(m["has"].sum()),
        "have_store_weighted": base / total_stores,
        "missing_brands": len(miss),
    }
    return miss, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="손으로 받을 정보공개서 우선순위")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--csv", action="store_true", help=f"{OUT.name} 로 저장")
    args = ap.parse_args()

    miss, s = wanted()
    print(f"배포 코호트 {s['cohort_brands']:,}개 · 점포 {s['cohort_stores']:,}")
    print(f"본부재무 보유 {s['have_brands']}개 · 점포가중 {s['have_store_weighted']:.1%}")
    print(f"미보유 {s['missing_brands']:,}개\n")

    print("몇 개를 받으면 얼마가 되는가:")
    for n in (10, 20, 30, 50, 100, 200):
        if n <= len(miss):
            cov = float(miss.iloc[n - 1]["누적_점포가중_커버리지"])
            print(f"  상위 {n:>3}개 → 점포가중 {cov:6.1%}"
                  f"  (+{(cov - s['have_store_weighted']) * 100:4.1f}%p)")

    top = miss.head(args.top)
    print(f"\n=== 열람 순서 (상위 {len(top)}개) ===")
    print(f"열람처: {VIEWER}")
    print("  상세 화면에서 Ctrl+S → '웹페이지, HTML만' 으로 저장\n")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        print(f"  {i:>3}. {str(r['brand_name'])[:28]:<30} 점포 {int(r['n_stores']):>5,} "
              f"· {r['grade']} · 누적 {r['누적_점포가중_커버리지']:.1%}")

    if args.csv:
        cols = ["brand_id", "brand_name", "n_stores", "grade", "누적_점포가중_커버리지"]
        OUT.parent.mkdir(parents=True, exist_ok=True)
        top[cols].to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"\n저장: {OUT}")
    print("\n받은 뒤:  python -m src.ifrmp_web --ingest-dir <저장한 폴더>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
