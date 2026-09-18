"""도메인은 찾았는데 로고를 못 가져온 브랜드를 다시 시도한다.

왜 필요한가
    로고 수집은 두 관문을 거친다. 1차는 네이버 검색으로 공식 도메인을 찾는 것,
    2차는 그 사이트에서 아이콘을 뽑는 것이다. 2차에는 "이 사이트가 정말 이
    브랜드의 것인가"를 페이지 텍스트로 확인하는 검사가 붙어 있었는데,
    1차가 이미 확인한 것을 또 확인하는 셈이라 JS 렌더링 사이트가 통째로
    탈락했다. brand_logo() 는 이 검사를 끄도록 고쳤지만 **정작 캐시를 채우는
    collect_logos() 에는 인자를 넘기지 않아 기본값 True 로 돌고 있었다.**
    그 결과 도메인이 잡힌 브랜드 중 892개가 아이콘 없이 남았다.

    다만 그 892개가 전부 되살아나지는 않는다. 재보니 71%(629개)는 창업 포털·
    언론사 도메인에 물려 있어 애초에 공식 홈페이지가 아니었다. 실제로 다시
    시도할 값어치가 있는 것은 **고유 도메인을 가진 277개**이고, 그 안에
    BHC·굽네치킨·처갓집양념치킨·네네치킨·한솥 같은 대형 브랜드가 들어 있다.

    캐시에 저장된 도메인을 재사용하므로 네이버 API 호출 한도를 쓰지 않는다.

폐를 끼치지 않기
    상대는 실제 기업 사이트다. 동시 요청을 8개로 묶고, 브랜드마다 후보를
    앞에서부터 받아 쓸 만한 것이 나오면 즉시 멈춘다.

실행:
    python tools/repair_logos.py            # 전체
    python tools/repair_logos.py --limit 50 # 맛보기
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.common import load_config  # noqa: E402
from src.naver import (  # noqa: E402
    _NOT_OFFICIAL,
    LOGO_CACHE,
    LOGO_DIR,
    NaverError,
    download_logo,
    logo_file_name,
    official_domain,
    prune_shared_logos,
    site_icon,
)

_LOCK = threading.Lock()


def _write(path: Path, cache: dict) -> None:
    """색인을 저장한다. 항목 수가 급감하면 쓰지 않고 멈춘다.

    색인 하나가 로고 1,400건의 출처 기록이다. 잘못된 객체를 넘기는 실수 한 번에
    통째로 날아가는 파일이라면, 그 실수를 파일이 스스로 막게 해 둔다.
    """
    if not isinstance(cache, dict) or len(cache) < 10:
        raise ValueError(f"색인이 비정상적으로 작다({type(cache).__name__}, "
                         f"{len(cache) if hasattr(cache, '__len__') else '?'}) — 저장 중단")
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(prev, dict) and len(cache) < len(prev) * 0.9:
            raise ValueError(f"색인 항목이 {len(prev)} -> {len(cache)} 로 급감 — 저장 중단")
        path.with_suffix(".json.bak").write_text(
            json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def _targets(cache: dict, root: Path) -> tuple[list[str], int]:
    """다시 시도할 가치가 있는 브랜드와, 공유 도메인이라 건너뛴 수.

    ⚠️ 여러 브랜드가 같은 도메인을 가리키면 그건 공식 홈페이지가 아니다.
       실측하면 fchamall.com(창업몰) 62개 · changupdo.com 45개 · weseb.com 37개 ·
       jumpoline.com(점포라인) 29개 · mt.co.kr(뉴스) 16개 처럼 창업 포털과
       언론사가 '공식 홈페이지'로 잡혀 있다. 아이콘을 받아 봐야
       prune_shared_logos() 가 곧바로 버린다 — 실제로 26건을 얻고 26건을 그대로
       잃었다. 남의 서버를 헛되이 두드리지 않도록 여기서 걸러 낸다.
    """
    dom_count: dict[str, int] = {}
    for v in cache.values():
        if isinstance(v, dict) and v.get("domain"):
            dom_count[v["domain"]] = dom_count.get(v["domain"], 0) + 1

    out, skipped = [], 0
    for name, v in cache.items():
        if not isinstance(v, dict) or not v.get("domain"):
            continue
        # 색인이 아니라 **디스크**를 본다 — 화면도 파일 존재로 판단한다
        if (root / LOGO_DIR / logo_file_name(name)).exists():
            continue
        if dom_count.get(v["domain"], 0) > 1:
            skipped += 1
            continue
        out.append(name)
    return out, skipped


def _stale(cache: dict, root: Path) -> list[str]:
    """공식 홈페이지를 **다시 찾아야** 하는 브랜드.

    차단 목록에 새 포털을 추가하면, 그 포털이 잡혀 있던 브랜드들은 이제
    도메인이 없는 상태와 같다. 다시 검색하면 진짜 브랜드 사이트가 나올 수 있다.
    도메인을 아예 못 찾았던 브랜드도 함께 다시 시도한다.
    """
    dom_count: dict[str, int] = {}
    for v in cache.values():
        if isinstance(v, dict) and v.get("domain"):
            dom_count[v["domain"]] = dom_count.get(v["domain"], 0) + 1

    out = []
    for name, v in cache.items():
        if not isinstance(v, dict):
            continue
        if (root / LOGO_DIR / logo_file_name(name)).exists():
            continue
        d = str(v.get("domain") or "")
        blocked = any(k in d.lower() for k in _NOT_OFFICIAL)
        if not d or blocked or dom_count.get(d, 0) > 1:
            out.append(name)
    return out


def _one(name: str, domain: str, root: Path) -> tuple[str, str, int]:
    try:
        url, px = site_icon(domain, name, verify=False)
    except Exception:
        return name, "", 0
    if not url:
        return name, "", 0
    try:
        local = download_logo(name, url, root)
    except Exception:
        return name, "", 0
    return name, local, int(px)


def _redomain(name: str, cfg: dict, root: Path) -> tuple[str, str, str, int]:
    """공식 도메인부터 다시 찾고 아이콘까지 간다. (브랜드, 도메인, 파일, px)"""
    try:
        domain = official_domain(name, cfg)
    except NaverError:
        raise                      # 한도 소진 등은 위에서 멈춰야 한다
    except Exception:
        return name, "", "", 0
    if not domain:
        return name, "", "", 0
    n, local, px = _one(name, domain, root)
    return n, domain, local, px


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--redomain", action="store_true",
                    help="차단 목록 갱신 후 공식 도메인부터 다시 찾는다 (네이버 API 사용)")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(cfg["_root"])
    cpath = root / LOGO_CACHE
    cache = json.loads(cpath.read_text(encoding="utf-8"))

    if args.redomain:
        todo = _stale(cache, root)
        print(f"도메인 재탐색 대상 {len(todo)}개 (동시 {args.workers}) — 네이버 검색 사용")
    else:
        todo, skipped = _targets(cache, root)
        print(f"재시도 대상 {len(todo)}개 브랜드 (동시 {args.workers}) · "
              f"공유 도메인이라 건너뜀 {skipped}개")
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        return 0

    ok = 0
    done = 0
    n_dom = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        if args.redomain:
            futs = {ex.submit(_redomain, n, cfg, root): n for n in todo}
        else:
            futs = {ex.submit(_one, n, cache[n]["domain"], root): n for n in todo}
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except NaverError as exc:
                print(f"  중단: {exc}")
                break
            if args.redomain:
                name, domain, local, px = res
                if domain:
                    n_dom += 1
                    with _LOCK:
                        cache[name]["domain"] = domain
            else:
                name, local, px = res
            done += 1
            if local:
                with _LOCK:
                    cache[name]["file"] = local
                    cache[name]["url"] = cache[name].get("url") or "(재탐색)"
                    cache[name]["px"] = px
                ok += 1
            if done % 50 == 0:
                with _LOCK:
                    _write(cpath, cache)
                print(f"  {done}/{len(todo)} 처리 · 도메인 {n_dom} · 로고 {ok}")

    _write(cpath, cache)
    print(f"재시도 완료: {done}개 중 {ok}개 신규 확보")

    # 같은 이미지가 여러 브랜드에 붙었으면 전부 버린다 (엉뚱한 로고 방어)
    # ⚠️ prune_shared_logos() 는 cache 를 **제자리에서** 고치고 통계를 돌려준다.
    #    반환값을 cache 로 받으면 색인 파일에 통계 세 줄이 덮어써진다 —
    #    실제로 1,439항목짜리 색인이 70바이트가 됐다(git 에서 복구).
    before = sum(1 for v in cache.values() if isinstance(v, dict) and v.get("file"))
    stats = prune_shared_logos(cfg, cache)
    _write(cpath, cache)
    after = sum(1 for v in cache.values() if isinstance(v, dict) and v.get("file"))
    print(f"공유 이미지 정리: {before} -> {after} "
          f"(도메인 공유 {stats['removed_domain']} · 동일 이미지 {stats['removed_image']})")
    n_files = len(list((root / LOGO_DIR).glob("*.png")))
    print(f"디스크 로고 파일 {n_files}장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
