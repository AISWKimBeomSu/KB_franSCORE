"""이미지 검색 + **LLM 판독 검증**으로 로고를 채운다.

왜 검증이 없으면 안 되는가 (실측)
    '{브랜드} 로고' 로 이미지 검색을 하면 상위 결과가 이렇다:

        4000x3000  음식 사진 (네이버 플레이스)
         736x470   핀터레스트 이미지
         600x600   포트폴리오 사이트 썸네일

    로고가 아니다. 색 수로 사진과 로고를 가르려 했으나 분리되지 않았다
    (로고 435색 vs 사진 3,558색이 섞인다). 크기·형식·종횡비 어느 것도 신호가
    약하다. **틀린 로고는 로고가 없는 것보다 나쁘므로** 근거 없는 채택을 막을
    장치가 반드시 있어야 한다.

무엇이 검증인가
    로고에는 거의 항상 **브랜드명이 글자로 박혀 있다.** 그래서 이미지를 LLM 에
    보여 주고 ① 보이는 글자를 그대로 옮겨 적게 하고 ② 그 글자가 브랜드명과
    맞는지 판정하게 한다. 음식 사진·간판 사진·다른 회사 로고는 false 로 떨어진다.

    실측 동작 — 지코바양념치킨: 후보 4개 중 3개를 거부하고
    ('GCOVA CHICKEN' 만 있는 뉴스 이미지 등) 4번째 'GCOVA 지코바치킨' 을 채택했다.

    판독 결과(보이는 글자·판정 사유)는 캐시에 그대로 남긴다. "왜 이 그림이 이
    브랜드의 로고인가"를 나중에 사람이 확인할 수 있어야 하기 때문이다.

⚠️ 모델 선택: `gemini-flash-latest` 를 쓴다. `gemini-2.5-flash` 는 같은 이미지에
   대해 보이는 글자를 빈 문자열로 돌려주고 전부 false 로 판정했다(실측). 이미지
   판독이 되는 모델인지 확인하지 않고 붙이면 **검증이 하는 일 없이 전부 기각**된다.

실행:
    python tools/logo_verified_search.py --limit 20   # 표본
    python tools/logo_verified_search.py              # 전체 (재개 가능)
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
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

from src.common import get_logger, load_config, load_secrets  # noqa: E402
from src.naver import (  # noqa: E402
    _UA,
    LOGO_CACHE,
    LOGO_DIR,
    MIN_LOGO_PX,
    download_logo,
    logo_file_name,
    prune_shared_logos,
    search_term,
)

log = get_logger("logo_verify")

MODEL = "gemini-flash-latest"
MAX_CANDIDATES = 5          # 브랜드당 검증할 이미지 수 (앞에서 채택되면 조기 종료)
_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif",
         "WEBP": "image/webp", "BMP": "image/bmp"}

_VERDICT = {
    "type": "object",
    "properties": {
        "visible_text": {"type": "string"},
        "is_brand_logo": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["visible_text", "is_brand_logo", "reason"],
}

_PROMPT = (
    "이 이미지가 한국 프랜차이즈 브랜드 '{name}' 의 **로고(BI/CI)** 인지 판정하라.\n"
    "판정 규칙:\n"
    "1. 이미지에 보이는 글자를 그대로 옮겨 적는다(없으면 빈 문자열).\n"
    "2. 그 글자가 브랜드명과 일치하거나 브랜드명의 영문 표기여야 true 다.\n"
    "3. 음식 사진·매장 사진·간판 사진·메뉴판·인테리어·인물 사진은 로고가 아니다 → false.\n"
    "4. 다른 회사·지자체·포털·배달앱·SNS 의 로고면 false.\n"
    "5. 글자가 전혀 없고 그림만 있으면 브랜드 확인이 불가하므로 false.\n"
    "확실하지 않으면 false 로 판정하라. 틀린 로고를 붙이는 것이 로고가 없는 것보다 나쁘다."
)


class QuotaExhausted(RuntimeError):
    """모든 키가 한도 초과 — 남은 브랜드는 다음 실행에서 이어서 한다."""


class _Verifier:
    """키를 돌려 가며 이미지 판독. 429/401/403 은 그 키의 문제이므로 다음 키로 넘어간다."""

    def __init__(self) -> None:
        load_secrets()
        names = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 13)]
        self.keys = [os.environ[n].strip() for n in names if os.environ.get(n, "").strip()]
        self.i = 0
        self.calls = 0
        if not self.keys:
            raise RuntimeError("GEMINI_API_KEY 미설정 — 검증 없이는 수집하지 않는다")

    def __call__(self, name: str, img: bytes, mime: str) -> dict | None:
        body = {
            "contents": [{"parts": [
                {"text": _PROMPT.format(name=name)},
                {"inline_data": {"mime_type": mime,
                                 "data": base64.b64encode(img).decode()}}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseSchema": _VERDICT},
        }
        tried = 0
        while tried < len(self.keys):
            key = self.keys[self.i]
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{MODEL}:generateContent?key={key}")
            try:
                r = requests.post(url, json=body, timeout=90)
            except Exception:
                return None
            self.calls += 1
            if r.status_code == 200:
                try:
                    return json.loads(
                        r.json()["candidates"][0]["content"]["parts"][0]["text"])
                except Exception:
                    return None
            if r.status_code in (401, 403, 429):
                self.i = (self.i + 1) % len(self.keys)
                tried += 1
                continue
            return None                     # 400 등 — 이 이미지를 못 읽는다
        raise QuotaExhausted(f"{len(self.keys)}개 키 모두 한도 초과 (호출 {self.calls}회)")


def _image_search(query: str, nh: dict, display: int = 8) -> list[str]:
    try:
        r = requests.get("https://openapi.naver.com/v1/search/image.json",
                         params={"query": query, "display": display, "sort": "sim"},
                         headers=nh, timeout=15)
        r.raise_for_status()
        return [it.get("link", "") for it in r.json().get("items", []) if it.get("link")]
    except Exception:
        return []


def _fetch_image(url: str) -> tuple[bytes, str, int, int]:
    try:
        b = requests.get(url, headers=_UA, timeout=12, verify=False).content
        from PIL import Image
        im = Image.open(io.BytesIO(b))
        return b, _MIME.get(im.format or "", "image/png"), im.size[0], im.size[1]
    except Exception:
        return b"", "", 0, 0


def main() -> int:
    ap = argparse.ArgumentParser(description="이미지 검색 + LLM 검증으로 로고 수집")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    cfg = load_config()
    root = Path(cfg["_root"])
    load_secrets()
    nh = {"X-Naver-Client-Id": os.environ.get("NAVER_CLIENT_ID", ""),
          "X-Naver-Client-Secret": os.environ.get("NAVER_CLIENT_SECRET", "")}
    if not nh["X-Naver-Client-Id"]:
        log.error("NAVER_CLIENT_ID/SECRET 미설정 — 이미지 검색 불가")
        return 1

    import pandas as pd
    sc = pd.read_csv(Path(cfg["paths"]["outputs"]) / "scores_latest.csv",
                     encoding="utf-8-sig")
    sc["_n"] = pd.to_numeric(sc["n_stores"], errors="coerce").fillna(0)
    cache_p = root / LOGO_CACHE
    cache = json.loads(cache_p.read_text(encoding="utf-8"))
    img_dir = root / LOGO_DIR

    todo = []
    for _, r in sc.sort_values("_n", ascending=False).iterrows():
        n = str(r["brand_name"])
        if (img_dir / logo_file_name(n)).exists():
            continue
        v = cache.get(n, {})
        if isinstance(v, dict) and v.get("img_search_done"):
            continue                      # 이미 훑어서 못 찾은 브랜드 — 재시도 안 함
        todo.append(n)
    if args.limit:
        todo = todo[:args.limit]
    log.info("대상 %d개 브랜드 · 모델 %s · 브랜드당 최대 %d후보",
             len(todo), MODEL, MAX_CANDIDATES)

    verify = _Verifier()
    ok = rejected = nocand = 0
    try:
        for i, name in enumerate(todo, 1):
            urls = _image_search(f"{search_term(name)} 로고", nh)
            if not urls:
                nocand += 1
                cache.setdefault(name, {})["img_search_done"] = "검색 결과 없음"
                continue
            picked = None
            seen = 0
            for u in urls:
                if seen >= MAX_CANDIDATES:
                    break
                b, mime, w, h = _fetch_image(u)
                if not b or max(w, h) < MIN_LOGO_PX:
                    continue
                seen += 1
                d = verify(name, b, mime)
                if d and d.get("is_brand_logo"):
                    picked = (u, w, h, d)
                    break
                time.sleep(args.sleep)
            entry = cache.setdefault(name, {})
            if picked is None:
                rejected += 1
                entry["img_search_done"] = f"후보 {seen}건 전부 검증 기각"
                continue
            u, w, h, d = picked
            local = download_logo(name, u, root)
            if not local:
                entry["img_search_done"] = "내려받기 실패"
                continue
            entry.update({"url": u, "px": int(max(w, h)), "file": local,
                          "source": "image-search+llm",
                          "verified_text": d.get("visible_text", ""),
                          "verified_reason": d.get("reason", "")[:200],
                          "img_search_done": "확보"})
            entry.pop("rejected", None)
            entry.pop("pruned", None)
            ok += 1
            if ok % 10 == 0:
                cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
                log.info("%d/%d · 확보 %d · 기각 %d · LLM호출 %d",
                         i, len(todo), ok, rejected, verify.calls)
            time.sleep(args.sleep)
    except QuotaExhausted as exc:
        log.warning("%s — 여기까지 저장하고 종료한다. 다시 실행하면 이어서 한다.", exc)
    finally:
        cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                           encoding="utf-8")

    prune_shared_logos(cfg, cache)
    cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    names = set(sc["brand_name"].astype(str))
    have = sum(1 for n in names if (img_dir / logo_file_name(n)).exists())
    log.info("완료 — 신규 확보 %d · 검증 기각 %d · 후보 없음 %d · LLM 호출 %d",
             ok, rejected, nocand, verify.calls)
    log.info("평가 브랜드 %d개 중 로고 %d개 (%.1f%%)",
             len(names), have, 100 * have / max(len(names), 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
