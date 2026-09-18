"""브랜드 자기 사이트의 **페이지 안**에서 로고를 건진다.

왜 필요한가 (실측 분해)
    로고 확보가 1,442개 중 548개(38.1%)에서 멈춰 있었다. 실패 원인을 세어 보니
    한 가지가 아니었다:

        도메인 공유(포털)로 제거      413  28.7%
        자기 도메인은 있는데 아이콘 없음  343  23.7%   ← 이 도구가 노리는 구간
        도메인 못 찾음                113   7.9%
        아이콘이 64px 미만             24   1.7%

    343건은 브랜드 소유가 이미 확인된 도메인인데 **파비콘만 보고 포기**한 것이다.
    파비콘을 아예 안 두거나 16~32px 만 두는 사이트가 많다. 그런데 그 사이트
    헤더에는 원본 로고가 걸려 있다.

무엇을 하지 않는가
    · **이미지 검색을 쓰지 않는다.** 실측해 보니 '{브랜드} 로고' 질의의 상위
      결과가 음식 사진·핀터레스트 이미지였고 실제 로고가 아니었다. 색 수로
      사진과 로고를 가르려 했으나 분리되지 않았다(로고 435색 vs 사진 3,558색이
      섞인다). 근거 없는 이미지를 붙이느니 글자 마크가 낫다.
    · **브랜드 소유 확인 없이 긁지 않는다.** 확인을 빼면 검색이 물어 온 엉뚱한
      사이트가 그대로 들어온다 — 실측: 일미리금계찜닭에 gimhae.go.kr(김해시청),
      생어거스틴에 booking.kakao.com(카카오 예약)이 잡혔다. 김해시 로고가 찜닭
      브랜드로 나가는 것은 로고가 없는 것보다 훨씬 나쁘다.

안전장치
    ① 페이지 본문에 브랜드명이 있어야 한다 (`site_mentions_brand`).
    ② SNS·앱스토어·지자체·협회 아이콘은 파일명 규칙으로 제외한다.
    ③ 64px 미만은 버린다 (카드에서 뭉개진다).
    ④ 끝나고 `prune_shared_logos` 를 다시 돌려, 서로 다른 브랜드에 같은 그림이
       붙은 경우를 전부 폐기한다.

실행:
    python tools/logo_from_pages.py --limit 20     # 표본으로 정밀도 확인
    python tools/logo_from_pages.py                # 전체
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import warnings
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

from src.common import get_logger, load_config  # noqa: E402
from src.naver import (  # noqa: E402
    _UA,
    LOGO_CACHE,
    LOGO_DIR,
    MIN_LOGO_PX,
    download_logo,
    logo_file_name,
    page_logo_candidates,
    prune_shared_logos,
)

log = get_logger("logo_pages")


def _measure(url: str, referer: str) -> tuple[int, int, bytes]:
    """(가로, 세로, 원본 바이트). 실패는 (0,0,b'')."""
    try:
        r = requests.get(url, headers={**_UA, "Referer": referer}, timeout=12,
                         verify=False, allow_redirects=True)
        r.raise_for_status()
        from PIL import Image
        im = Image.open(io.BytesIO(r.content))
        return im.size[0], im.size[1], r.content
    except Exception:
        return 0, 0, b""


def _targets(cache: dict, names: set[str]) -> list[tuple[str, str]]:
    """자기 도메인이 있는데 아직 로고가 없는 브랜드."""
    out = []
    for name, v in cache.items():
        if name not in names or not isinstance(v, dict):
            continue
        if v.get("file") or not v.get("domain") or v.get("pruned"):
            continue
        out.append((name, v["domain"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="브랜드 사이트 페이지에서 로고 수집")
    ap.add_argument("--limit", type=int, default=0, help="처리할 브랜드 수 (0=전체)")
    ap.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 출력")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(cfg["_root"])
    import pandas as pd
    sc = pd.read_csv(Path(cfg["paths"]["outputs"]) / "scores_latest.csv",
                     encoding="utf-8-sig")
    names = set(sc["brand_name"].astype(str))

    cache_p = root / LOGO_CACHE
    cache = json.loads(cache_p.read_text(encoding="utf-8"))
    todo = _targets(cache, names)
    # 큰 브랜드부터 — 화면 노출 빈도가 높다
    size = dict(zip(sc["brand_name"].astype(str),
                    pd.to_numeric(sc["n_stores"], errors="coerce").fillna(0),
                    strict=False))
    todo.sort(key=lambda t: -size.get(t[0], 0))
    if args.limit:
        todo = todo[:args.limit]

    log.info("대상 %d개 브랜드 (자기 도메인 보유 · 로고 미확보)", len(todo))
    ok = no_page = too_small = 0
    for i, (name, domain) in enumerate(todo, 1):
        cands = page_logo_candidates(domain, name)
        if not cands:
            # 페이지를 못 받았거나 브랜드명이 페이지에 없다 = 그 사이트는 이 브랜드 것이 아니다
            no_page += 1
            cache[name]["page_scan"] = "브랜드 미확인 또는 후보 없음"
            continue
        best = None
        for kind, u in cands:
            w, h, _ = _measure(u, f"https://{domain}")
            if max(w, h) >= MIN_LOGO_PX and (best is None or max(w, h) > best[1]):
                best = (kind, max(w, h), u, f"{w}x{h}")
        if best is None:
            too_small += 1
            cache[name]["page_scan"] = f"후보 {len(cands)}개 모두 {MIN_LOGO_PX}px 미만/실패"
            continue
        kind, px, url, dim = best
        if args.dry_run:
            print(f"  {name[:20]:22s} {dim:>11s} {kind:9s} {url[:60]}")
            ok += 1
            continue
        local = download_logo(name, url, root)
        if not local:
            cache[name]["page_scan"] = "내려받기 실패"
            continue
        cache[name].update({"url": url, "px": int(px), "file": local,
                            "source": f"page:{kind}"})
        cache[name].pop("page_scan", None)
        ok += 1
        if ok % 25 == 0:
            cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                               encoding="utf-8")
            log.info("%d/%d 처리 · 신규 확보 %d", i, len(todo), ok)
        time.sleep(0.15)

    if args.dry_run:
        print(f"\n[미적용] 신규 확보 가능 {ok}/{len(todo)} · "
              f"브랜드 미확인 {no_page} · 크기 미달 {too_small}")
        return 0

    cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    # 서로 다른 브랜드에 같은 그림이 붙었으면 전부 폐기한다 (반환값을 대입하지 말 것 —
    # 이 함수는 통계 dict 를 돌려주고 cache 는 제자리에서 고친다)
    stats = prune_shared_logos(cfg, cache)
    cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    have = sum(1 for n in names if (root / LOGO_DIR / logo_file_name(n)).exists())
    log.info("완료 — 신규 확보 %d · 브랜드 미확인 %d · 크기 미달 %d · 공유폐기 %s",
             ok, no_page, too_small, stats)
    log.info("평가 브랜드 %d개 중 로고 보유 %d개 (%.1f%%)",
             len(names), have, 100 * have / max(len(names), 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
