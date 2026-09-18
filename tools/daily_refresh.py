"""매일 1회 — 새 자료를 모아 브랜드 리스크 진단을 다시 산출한다.

무엇이 매일 바뀌고 무엇이 안 바뀌는가 (이걸 섞으면 거짓말이 된다)
    · **매일 바뀐다** — 뉴스 보도, 네이버 검색수요. 여기서 나오는 진단 소견과
      그 소견을 합산한 watch_score(점검 우선순위)는 매일 갱신된다.
    · **연 1회 바뀐다** — 공정거래위원회 가맹사업 공시. 모델이 학습한 피처
      (점포 수·계약종료·매출·본부재무)가 전부 이 연간 공시에서 나오므로,
      머신러닝 확률 deterioration_1y 는 새 공시가 나오기 전에는 움직일 수 없다.
      매일 재산출하는 척하며 같은 값을 다시 쓰는 것은 실시간 갱신이 아니다.
    그래서 이 작업은 **일간 신호(뉴스·수요)로 진단과 우선순위를 갱신**하고,
    새 공시가 감지되면 전체 재학습이 필요하다고 표시한다.

호출량 (기본값 기준, 무료 한도를 넘지 않도록 여유를 크게 둔다)
    뉴스   브랜드 12개 × 질의 4개  = 약 48회/일   (네이버 검색 한도 25,000/일)
    수요   신규 브랜드 최대 40개    = 40회/일      (검색어트렌드 한도 1,000/일)
    이미 수집된 브랜드는 건너뛰므로 날이 갈수록 커버리지가 넓어진다.

실행:
    python tools/daily_refresh.py                 # 기본
    python tools/daily_refresh.py --demand-limit 0  # 수요 수집 건너뛰기
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows 기본 콘솔 코드페이지(cp949)는 '—' 같은 문자를 못 찍는다. 로그 한 줄 때문에
# UnicodeEncodeError 로 죽으면 스케줄러에는 **갱신 실패**로 남는다 — 실제로는 산출물이
# 다 만들어진 뒤였다(실측). 표준 출력을 UTF-8 로 고정해 그 오탐을 없앤다.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):  # 리다이렉트된 스트림 등
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.common import (  # noqa: E402
    capability_report,
    get_logger,
    load_config,
    load_secrets,
)

log = get_logger("daily")

NEWS_BRANDS = 12          # 하루에 뉴스를 훑을 브랜드 수
DEMAND_BRANDS = 40        # 하루에 검색수요를 새로 채울 브랜드 수
LOGO_BRANDS = 60          # 하루에 로고를 채울 브랜드 수 (LLM 판독 할당에 맞춘 값)
STATE_FILE = "refresh_state.json"


def _count_news(cfg: dict) -> int:
    d = Path(cfg["paths"].get("root", ".")) / \
        (cfg.get("news", {}) or {}).get("cache_dir", "data/raw/news")
    if not d.exists():
        d = _ROOT / "data/raw/news"
    if not d.exists():
        return 0
    return sum(len(json.loads(p.read_text(encoding="utf-8")) or [])
               for p in d.glob("*.json") if p.stat().st_size > 2)


def _count_demand(cfg: dict) -> int:
    p = Path(cfg["paths"]["outputs"]) / "demand_trends.json"
    if not p.exists():
        return 0
    try:
        return len((json.loads(p.read_text(encoding="utf-8")) or {}).get("brands") or {})
    except (OSError, ValueError):
        return 0


def _disclosure_year(cfg: dict) -> int | None:
    """패널에 들어와 있는 가장 최근 공시연도."""
    import pandas as pd
    p = Path(cfg["paths"]["processed"]) / "panel.parquet"
    if not p.exists():
        return None
    try:
        return int(pd.read_parquet(p, columns=["year"])["year"].max())
    except Exception:
        return None


def _step(name: str, fn, results: list[dict]):
    """한 단계 실행. 실패해도 다음 단계는 계속 간다 — 뉴스가 안 와도 진단은 돌아야 한다."""
    t0 = time.time()
    try:
        detail = fn()
        results.append({"name": name, "ok": True, "detail": detail,
                        "seconds": round(time.time() - t0, 1)})
        log.info("[%s] 완료 — %s (%.1fs)", name, detail, time.time() - t0)
    except Exception as exc:
        results.append({"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}",
                        "seconds": round(time.time() - t0, 1)})
        log.warning("[%s] 실패 (건너뜀): %s", name, str(exc)[:200])


def run(news_brands: int = NEWS_BRANDS, demand_limit: int = DEMAND_BRANDS,
        logo_limit: int = LOGO_BRANDS) -> dict:
    started = datetime.now()
    load_secrets()
    # 무엇이 폴백으로 돌게 되는지 **먼저** 알고 시작한다 (아래 state["capability"]).
    cap = capability_report()
    if cap["degraded"]:
        log.warning("자격증명 미비로 축소 가동 %d건: %s",
                    len(cap["degraded"]), " · ".join(cap["degraded"]))
    cfg = load_config()
    out_dir = Path(cfg["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)

    news_before, demand_before = _count_news(cfg), _count_demand(cfg)
    results: list[dict] = []

    if news_brands > 0:
        def _news():
            from src import news_llm
            names = news_llm.select_brands(cfg, n=news_brands)
            news_llm.run_news(names, cfg)
            return f"브랜드 {len(names)}개 조회"
        _step("news", _news, results)

    if demand_limit > 0:
        def _demand():
            from src import naver
            if not naver.is_enabled(cfg):
                return "네이버 자격증명 없음 — 건너뜀"
            naver.collect_demand(cfg, limit=demand_limit)
            return f"신규 최대 {demand_limit}개 브랜드"
        _step("demand", _demand, results)

    # 로고 채우기 — LLM 판독 검증을 거치므로 하루 할당만큼만 나아간다.
    # 한 번에 다 못 채우는 것이 정상이고, `img_search_done` 표시로 이어서 진행된다.
    if logo_limit > 0:
        def _logos():
            import subprocess
            r = subprocess.run(
                [sys.executable, str(_ROOT / "tools" / "logo_verified_search.py"),
                 "--limit", str(logo_limit)],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
            return tail[0][-160:] or f"종료코드 {r.returncode}"
        _step("logos", _logos, results)

    # 진단 재산출 — 위에서 들어온 뉴스·수요가 여기서 소견과 우선순위로 바뀐다
    rescored = 0
    def _diagnose():
        nonlocal rescored
        from src import diagnosis
        _, summary = diagnosis.run(cfg)
        rescored = len(summary)
        return f"{rescored:,}개 브랜드 진단 재산출"
    _step("diagnose", _diagnose, results)

    news_after, demand_after = _count_news(cfg), _count_demand(cfg)
    year = _disclosure_year(cfg)
    finished = datetime.now()

    state = {
        "started_at": started.strftime("%Y-%m-%d %H:%M"),
        "finished_at": finished.strftime("%Y-%m-%d %H:%M"),
        "elapsed_sec": round((finished - started).total_seconds(), 1),
        "news_total": news_after,
        "news_added": max(0, news_after - news_before),
        "demand_total": demand_after,
        "demand_updated": max(0, demand_after - demand_before),
        "rescored": rescored,
        "disclosure_year": year,
        # 정직성: 매일 바뀌는 것과 아닌 것을 산출물 자체에 적어 둔다.
        "daily_scope": "뉴스·검색수요 기반 진단 소견과 점검 우선순위",
        "annual_scope": f"공시 기반 리스크 확률(deterioration_1y) — {year}년 공시 기준, "
                        f"새 공시 공표 시 재학습 필요" if year else "공시 기반 리스크 확률",
        "next_due": (finished + timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
        "steps": results,
        # 축소 가동을 산출물에 남긴다. 폴백이 조용히 도는 바람에 실패가 하나도 없이
        # 성능만 줄어든 채 매일 성공으로 끝나던 일이 실제로 있었다(값은 담지 않는다).
        "capability": cap,
        "source": "daily_refresh",
    }
    (out_dir / STATE_FILE).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("갱신 완료 — 뉴스 +%d(총 %d) · 수요 +%d(총 %d) · 진단 %d개 · %.1fs",
             state["news_added"], news_after, state["demand_updated"], demand_after,
             rescored, state["elapsed_sec"])
    if cap["degraded"]:
        log.warning("축소 가동 %d건 — %s (자격증명 미설정). "
                    "폴백으로 산출했으며 화면에 그대로 표기된다.",
                    len(cap["degraded"]), " · ".join(cap["degraded"]))
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="일간 브랜드 리스크 갱신")
    ap.add_argument("--news-brands", type=int, default=NEWS_BRANDS,
                    help=f"뉴스를 훑을 브랜드 수 (0이면 건너뜀, 기본 {NEWS_BRANDS})")
    ap.add_argument("--demand-limit", type=int, default=DEMAND_BRANDS,
                    help=f"검색수요를 새로 채울 브랜드 수 (0이면 건너뜀, 기본 {DEMAND_BRANDS})")
    ap.add_argument("--logo-limit", type=int, default=LOGO_BRANDS,
                    help=f"로고를 채울 브랜드 수 (0이면 건너뜀, 기본 {LOGO_BRANDS})")
    args = ap.parse_args()
    state = run(news_brands=args.news_brands, demand_limit=args.demand_limit,
                logo_limit=args.logo_limit)
    failed = [s["name"] for s in state["steps"] if not s["ok"]]
    if failed:
        log.warning("일부 단계 실패: %s (산출물은 갱신됨)", ", ".join(failed))
    print(json.dumps({k: v for k, v in state.items() if k != "steps"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
