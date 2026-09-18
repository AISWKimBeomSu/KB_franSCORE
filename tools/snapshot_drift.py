"""커밋된 원본 스냅샷과 **지금 살아 있는 공시**를 대조한다.

왜 이 도구가 필요한가
    이 저장소는 `data/raw` 의 원본 스냅샷을 통째로 커밋한다. 이유를 "재현성 때문"이라고
    적어 두었는데, 그것은 오랫동안 **주장**이었다. 마감 당일 개인 data.go.kr 키로
    15125490 을 다시 받아 대조해 보니 나흘 사이에 **18행이 사라져 있었다** — 브랜드
    하나(나뚜루, 롯데웰푸드)의 지역별 행 전부였고, 그 브랜드는 우리 배포 코호트 안에
    FS2·요주의로 들어 있다.

    즉 공시 원천은 불변이 아니다. 그러면 두 가지가 따라온다.

      ① **판정을 재현하려면 판정 시점의 입력이 있어야 한다.** 산출물만으로는 "그때 왜
         그 등급이었나"를 증명할 수 없다. 은행 모형 문서(SR 26-2 계열)가 요구하는 것도
         결과 재현이 아니라 **입력·코드·결과의 동시 보존**이다.
      ② **"최신이 항상 낫다"가 성립하지 않는다.** 이번 변화는 행이 늘어난 것이 아니라
         줄어든 것이었다. 아무 때나 다시 받아 덮어쓰면 데이터를 잃는다.

    그래서 재수집을 하기 전에 **무엇이 얼마나 달라지는지 먼저 재는** 도구를 둔다.

⚠️ 이 도구는 `data/raw` 를 **절대 건드리지 않는다.** 받은 것은 메모리에만 두고 대조 결과만
   출력한다. 덮어쓰기는 `run_pipeline.py --step collect` 의 일이고, 그 판단은 사람이 한다.

⚠️ 활용신청이 승인되지 않은 데이터셋은 건너뛴다(오류가 아니라 '미승인'으로 보고).
   승인 상태는 `python check_keys.py --data` 가 알려 준다.

실행:
    python tools/snapshot_drift.py                 # 승인된 데이터셋 전부, 스냅샷에 있는 연도 전부
    python tools/snapshot_drift.py --year 2025     # 특정 연도만
    python tools/snapshot_drift.py --csv           # outputs/snapshot_drift.csv 로 저장
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.collect import SERVICES  # noqa: E402
from src.common import get_logger, load_config, load_secrets  # noqa: E402

log = get_logger("snapshot_drift")
OUT = _ROOT / "outputs" / "snapshot_drift.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
_PAGE = 10_000
_TIMEOUT = 60

# 행을 식별하는 자연키. 있으면 '삭제 / 신규 / 값변경'을 구분할 수 있고,
# 없으면 행 전체를 키로 삼아 '스냅샷에만 / 라이브에만' 까지만 말한다.
# ⚠️ 모르는 것을 아는 척하지 않는다 — 키가 불확실한 데이터셋은 비워 둔다.
NATURAL_KEY: dict[str, tuple[str, ...]] = {
    "brand_region_direct": ("jngBizCrtraYr", "brandMnno", "areaNm"),
    "brand_master": ("jngBizCrtraYr", "brandMnno"),
    "industry_openclose": ("jngBizCrtrYr", "tpbizLclsfNm", "tpbizMclsfNm"),
}


def _load_snapshot(name: str, year: int) -> list[dict] | None:
    for suffix in (".json", ".json.gz"):
        p = _ROOT / "data" / "raw" / f"{name}_{year}{suffix}"
        if not p.exists():
            continue
        try:
            if p.suffix == ".gz":
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    d = json.load(f)
            else:
                d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return d if isinstance(d, list) else (d.get("items") or [])
    return None


def _fetch_live(spec: dict, key: str, year: int) -> tuple[list[dict] | None, str]:
    """(행 목록, 사유). 미승인·오류면 (None, 사유)."""
    rows: list[dict] = []
    page = 1
    while True:
        try:
            r = requests.get(spec["url"], params={
                "serviceKey": key, spec["year_param"]: year, "pageNo": page,
                "numOfRows": _PAGE, "resultType": "json"},
                headers=_HEADERS, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            return None, f"통신 실패 {type(exc).__name__}"
        try:
            body = r.json()
        except ValueError:
            return None, f"JSON 아님 (HTTP {r.status_code})"
        if "OpenAPI_ServiceResponse" in body:
            hdr = body["OpenAPI_ServiceResponse"].get("cmmMsgHeader", {})
            return None, str(hdr.get("errMsg") or "게이트웨이 오류")
        if str(body.get("resultCode", "")) not in ("00", "0"):
            return None, f"resultCode={body.get('resultCode')} {body.get('resultMsg', '')}"
        items = body.get("items") or []
        rows += items
        total = int(body.get("totalCount") or 0)
        if not items or len(rows) >= total:
            return rows, "정상"
        page += 1
        if page > 200:                       # 폭주 방지 — 200페이지면 200만 행이다
            return rows, "페이지 상한 도달"


def _rowkey(r: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in r.items()))


def compare(name: str, snap: list[dict], live: list[dict]) -> dict:
    """스냅샷 ↔ 라이브 비교. 자연키가 있으면 삭제/신규/값변경까지 나눈다."""
    s_all = {_rowkey(r) for r in snap}
    l_all = {_rowkey(r) for r in live}
    res = {"snapshot_rows": len(snap), "live_rows": len(live),
           "only_snapshot": len(s_all - l_all), "only_live": len(l_all - s_all)}

    kf = NATURAL_KEY.get(name)
    if not kf or not snap or not set(kf) <= set(snap[0]):
        res.update({"deleted": None, "added": None, "modified": None, "key": ""})
        return res
    sk = {tuple(str(r.get(c)) for c in kf) for r in snap}
    lk = {tuple(str(r.get(c)) for c in kf) for r in live}
    # 키는 같은데 내용이 다른 행 = 값 변경
    same_key = sk & lk
    s_by = {tuple(str(r.get(c)) for c in kf): _rowkey(r) for r in snap}
    l_by = {tuple(str(r.get(c)) for c in kf): _rowkey(r) for r in live}
    modified = sum(1 for k in same_key if s_by[k] != l_by[k])
    res.update({"deleted": len(sk - lk), "added": len(lk - sk),
                "modified": modified, "key": "+".join(kf)})
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="커밋된 스냅샷 ↔ 살아 있는 공시 대조")
    ap.add_argument("--year", type=int, default=0, help="특정 연도만 (기본: 스냅샷 전 연도)")
    ap.add_argument("--csv", action="store_true", help=f"{OUT.name} 로 저장")
    args = ap.parse_args()

    load_secrets()
    cfg = load_config()
    import os
    key = os.environ.get(cfg["collect"]["service_key_env"], "").strip()
    if not key:
        print("DATA_GO_KR_KEY 가 없다. 본인 키를 .env 에 넣고 다시 실행한다.")
        return 2

    print("커밋된 스냅샷과 지금 살아 있는 공시를 대조한다 (data/raw 는 건드리지 않는다)\n")
    rows = []
    for name, spec in SERVICES.items():
        years = ([args.year] if args.year else
                 sorted({int(p.name.rsplit("_", 1)[1].split(".")[0])
                         for p in (_ROOT / "data" / "raw").glob(f"{name}_*.json*")}))
        if not years:
            print(f"  {name:22s} 스냅샷 없음 — 건너뜀")
            continue
        for y in years:
            snap = _load_snapshot(name, y)
            if snap is None:
                continue
            live, why = _fetch_live(spec, key, y)
            if live is None:
                print(f"  {name:22s} {y}  ⬜ {why}")
                rows.append({"dataset": name, "year": y, "status": why})
                break                        # 미승인이면 다른 연도도 같다
            c = compare(name, snap, live)
            same = c["only_snapshot"] == 0 and c["only_live"] == 0
            detail = (f"삭제 {c['deleted']} · 신규 {c['added']} · 값변경 {c['modified']}"
                      if c["key"] else
                      f"스냅샷에만 {c['only_snapshot']} · 라이브에만 {c['only_live']}")
            print(f"  {name:22s} {y}  {'✅ 동일' if same else '⚠️ 차이'}  "
                  f"{c['snapshot_rows']:>7,} → {c['live_rows']:>7,}   {detail}")
            rows.append({"dataset": name, "year": y, "status": "정상", **c})

    df = pd.DataFrame(rows)
    if len(df):
        diff = df[(df.get("only_snapshot", 0).fillna(0) > 0)
                  | (df.get("only_live", 0).fillna(0) > 0)] if "only_snapshot" in df else df.head(0)
        print(f"\n대조한 (데이터셋, 연도) {len(df[df['status'] == '정상'])}쌍 · 차이 있는 쌍 {len(diff)}")
        if len(diff):
            print("→ 차이가 있다는 것은 원천이 사후 정정됐다는 뜻이다. 재수집이 항상 이득은 아니다 —")
            print("  행이 줄어든 경우 우리가 이미 가진 관측을 잃는다(기술설명서 §42-6).")
    if args.csv and len(df):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
