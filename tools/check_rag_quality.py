"""RAG 색인 경량화가 검색 품질을 해쳤는지 실측한다.

배포 메모리 한도 때문에 색인을 줄였다(float32 + char 피처 상한). 줄이기 전후로
같은 질의를 던져 상위 결과가 얼마나 겹치는지 본다. 겹침이 크게 떨어지면 되돌린다.

사용법
    python tools/check_rag_quality.py --snapshot   # 현재 색인의 결과를 기준선으로 저장
    python -m src.rag                              # 색인 재구축
    python tools/check_rag_quality.py --compare    # 기준선과 비교
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_config
from src.rag import load_index

QUERIES = [
    "메가엠지씨커피 가맹점 계약종료",
    "인생냉면 폐점 위험",
    "교촌치킨 본부 재무 영업이익",
    "이디야커피 가맹점 수 감소",
    "컴포즈커피 신규 개점",
    "배스킨라빈스 감사보고서 자본",
    "치킨 업종 계약해지 급증",
    "가맹본부 자본잠식",
    "점포당 매출 하락 브랜드",
    "빽다방 명의변경",
    "본죽 비빔밥 가맹점",
    "굽네치킨 지역 분포",
]
K = 5
SNAP = Path("outputs/_rag_quality_baseline.json")


def top_ids(idx, q: str) -> list[str]:
    hits = idx.retrieve(q, k=K)
    return [str(x) for x in hits["doc_id"].tolist()] if len(hits) else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    idx = load_index(cfg)
    if idx is None:
        raise SystemExit("RAG 색인이 없습니다 — `python -m src.rag` 먼저 실행")

    cur = {q: top_ids(idx, q) for q in QUERIES}
    size_mb = (Path(cfg["paths"]["outputs"]) / "rag_index.joblib").stat().st_size / 1e6
    nnz = int(idx.mat_word.nnz + idx.mat_char.nnz)
    dtype = str(idx.mat_word.dtype)

    if args.snapshot:
        SNAP.write_text(json.dumps(
            {"results": cur, "size_mb": round(size_mb, 1), "nnz": nnz, "dtype": dtype},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"기준선 저장: {SNAP} · 색인 {size_mb:.1f}MB · nnz {nnz:,} · {dtype}")
        return

    if not args.compare:
        raise SystemExit("--snapshot 또는 --compare 를 지정하세요")
    if not SNAP.exists():
        raise SystemExit(f"{SNAP} 없음 — 먼저 --snapshot 을 실행하세요")

    base = json.loads(SNAP.read_text(encoding="utf-8"))
    old = base["results"]
    overlaps, top1 = [], 0
    print(f"{'질의':32s}{'상위5 겹침':>12s}{'1위 동일':>10s}")
    print("-" * 56)
    for q in QUERIES:
        a, b = old.get(q, []), cur.get(q, [])
        ov = len(set(a) & set(b)) / max(len(a), 1) if a else 1.0
        same = bool(a and b and a[0] == b[0])
        top1 += int(same)
        overlaps.append(ov)
        print(f"{q[:30]:32s}{ov:>11.0%}{'○' if same else '×':>10s}")
    mean_ov = sum(overlaps) / len(overlaps)
    print("-" * 56)
    print(f"{'평균':32s}{mean_ov:>11.0%}{f'{top1}/{len(QUERIES)}':>10s}")
    print(f"\n색인 크기 {base['size_mb']:.1f}MB → {size_mb:.1f}MB "
          f"({size_mb / base['size_mb'] - 1:+.0%})")
    print(f"비영요소  {base['nnz']:,} → {nnz:,} · dtype {base['dtype']} → {dtype}")
    verdict = "합격" if (mean_ov >= 0.80 and top1 >= len(QUERIES) * 0.75) else "불합격"
    print(f"\n판정: {verdict} (기준 — 평균 겹침 80% 이상 & 1위 일치 75% 이상)")
    if verdict == "불합격":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
