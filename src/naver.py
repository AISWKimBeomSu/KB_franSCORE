"""네이버 오픈API — 검색어트렌드(수요)·뉴스·로고.

왜 검색어트렌드인가
    공시는 1년에 한 번, 그것도 이듬해에 나온다. 가맹점 수가 줄었다는 사실을 은행이
    아는 시점에는 이미 1~2년이 지난 뒤다. 반면 **사람들이 그 브랜드를 검색하는 양**은
    매장을 찾기 전 단계라 매출보다 먼저 움직이고, 월 단위로 즉시 관측된다.

    더 중요한 것은 **카테고리 대비 비교**다. '냉면' 카테고리 검색이 다 같이 줄었다면
    업종 축소지만, 카테고리는 보합인데 특정 브랜드만 빠졌다면 그 브랜드의 문제다.
    이 구분은 공시 데이터만으로는 절대 나오지 않는다.

API — 네이버가 개발자센터 오픈API를 **NAVER API HUB**(네이버 클라우드)로 이관했다.
    두 체계를 모두 지원하고, HUB 자격증명이 있으면 그쪽을 쓴다(아래 _PATHS 참조).
    ⚠️ 검색어트렌드는 **HUB 에서만** 된다 — 구 개발자센터 애플리케이션에는 데이터랩
       권한을 붙일 수 없어 'Scope Status Invalid' 로 막힌다(실측).
    자격증명: NCP_API_KEY_ID / NCP_API_KEY  (NCP 콘솔 → NAVER API HUB → Application)

정직성
    · 키가 없으면 **수집하지 않는다**. 추정값이나 예시값을 만들어 넣지 않는다.
      키 부재는 산출물 메타에 `enabled: false` 로 남고, 진단 규칙은 수요 소견을
      아예 생성하지 않는다(없는 근거로 위험을 말하지 않는다).
    · 데이터랩은 절대 검색량을 주지 않고 기간 내 최대값을 100으로 한 **상대지수**를
      준다. 그래서 이 모듈은 '검색량 N건'이 아니라 항상 **증감률**만 산출한다.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests

from src.common import get_logger, load_config

log = get_logger("naver")

# ── 두 가지 자격 체계 ──────────────────────────────────────────────────────
# 네이버가 개발자센터(developers.naver.com)의 오픈API를 **NAVER API HUB**(네이버
# 클라우드 플랫폼)로 이관했다. 둘은 도메인·경로·인증 헤더가 전부 다르다.
#
#              개발자센터(구)                        API HUB(신)
#   도메인     openapi.naver.com                    naverapihub.apigw.ntruss.com
#   뉴스       /v1/search/news.json                 /search/v1/news
#   웹문서     /v1/search/webkr.json                /search/v1/webkr
#   트렌드     /v1/datalab/search                   /search-trend/v1/search
#   헤더       X-Naver-Client-Id/-Secret            X-NCP-APIGW-API-KEY-ID/-KEY
#
# ⚠️ 구 체계의 검색어트렌드는 애플리케이션에 권한을 붙일 수 없어 'Scope Status
#    Invalid' 로 막힌다(실측). 트렌드를 쓰려면 API HUB 자격증명이 필요하다.
# 둘 다 지원하고 **HUB 가 있으면 HUB 를 쓴다**.
HUB_ID_ENV = "NCP_API_KEY_ID"
HUB_SECRET_ENV = "NCP_API_KEY"
ID_ENV = "NAVER_CLIENT_ID"
SECRET_ENV = "NAVER_CLIENT_SECRET"

_HUB_BASE = "https://naverapihub.apigw.ntruss.com"
_LEGACY_BASE = "https://openapi.naver.com"
# (경로 키) → (HUB 경로, 구 경로)
_PATHS = {
    "news": ("/search/v1/news", "/v1/search/news.json"),
    "webkr": ("/search/v1/webkr", "/v1/search/webkr.json"),
    "trend": ("/search-trend/v1/search", "/v1/datalab/search"),
}

# 데이터랩 제약 (공식 문서): keywordGroups 최대 5개, 그룹당 keywords 최대 20개
MAX_GROUPS = 5
_RETRY_STATUS = {429, 500, 502, 503, 504}
_TIMEOUT = 20.0
_MAX_RETRIES = 4

# 업종 중분류 → 데이터랩에 던질 카테고리 대표어.
# 공시 업종명이 그대로 검색어로는 어색한 경우(예: '기타외국식')를 사람이 쓰는 말로 옮긴다.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "치킨": ["치킨", "치킨 배달"],
    "커피": ["카페", "커피"],
    "제과제빵": ["베이커리", "빵집"],
    "피자": ["피자"],
    "햄버거": ["햄버거"],
    "한식": ["한식", "백반"],
    "분식": ["분식", "떡볶이"],
    "일식": ["일식", "초밥"],
    "중식": ["중식", "짜장면"],
    "서양식": ["파스타", "스테이크"],
    "주점": ["술집", "호프"],
    "아이스크림/빙수": ["아이스크림", "빙수"],
    "음료(커피 외)": ["음료", "주스"],
    "패스트푸드": ["패스트푸드"],
    "김밥": ["김밥"],
    "냉면": ["냉면"],
    "국수": ["국수", "칼국수"],
    "돼지고기구이": ["삼겹살", "고깃집"],
    "소고기구이": ["소고기", "한우"],
    "편의점": ["편의점"],
    "화장품": ["화장품"],
    "교육": ["학원"],
    "세탁": ["세탁소"],
    "이미용": ["미용실", "헤어샵"],
}

_PAREN = re.compile(r"[(（][^)）]*[)）]")
_NONWORD = re.compile(r"[^0-9A-Za-z가-힣 ]+")


class NaverError(RuntimeError):
    """네이버 API 호출 실패."""


# ---------------------------------------------------------------------------
# 자격증명
# ---------------------------------------------------------------------------

def credentials(cfg: dict | None = None) -> tuple[str, str, bool]:
    """(ID, SECRET, HUB 인가) — API HUB 자격증명이 있으면 그쪽을 쓴다."""
    c = (cfg or {}).get("naver", {}) if cfg else {}
    hub_id = os.environ.get(str(c.get("hub_id_env", HUB_ID_ENV)), "").strip()
    hub_sec = os.environ.get(str(c.get("hub_secret_env", HUB_SECRET_ENV)), "").strip()
    if hub_id and hub_sec:
        return hub_id, hub_sec, True
    cid = os.environ.get(str(c.get("client_id_env", ID_ENV)), "").strip()
    sec = os.environ.get(str(c.get("client_secret_env", SECRET_ENV)), "").strip()
    return cid, sec, False


def is_enabled(cfg: dict | None = None) -> bool:
    cid, sec, _ = credentials(cfg)
    return bool(cid and sec)


def has_trend(cfg: dict | None = None) -> bool:
    """검색어트렌드를 쓸 수 있는가 — API HUB 자격증명이 있어야 한다."""
    cid, sec, hub = credentials(cfg)
    return bool(cid and sec and hub)


def _endpoint(key: str, cfg: dict | None = None) -> tuple[str, dict]:
    """(전체 URL, 헤더). 어느 자격 체계인지에 따라 도메인·경로·헤더가 갈린다."""
    cid, sec, hub = credentials(cfg)
    if not (cid and sec):
        raise NaverError(
            f"네이버 자격증명 미설정 — API HUB 를 쓰려면 {HUB_ID_ENV}/{HUB_SECRET_ENV}, "
            f"구 개발자센터를 쓰려면 {ID_ENV}/{SECRET_ENV} 를 설정하세요.")
    hub_path, legacy_path = _PATHS[key]
    if hub:
        return _HUB_BASE + hub_path, {"X-NCP-APIGW-API-KEY-ID": cid,
                                      "X-NCP-APIGW-API-KEY": sec}
    if key == "trend":
        raise NaverError(
            "검색어트렌드는 API HUB 자격증명이 필요합니다 — 구 개발자센터 키로는 "
            f"권한을 붙일 수 없습니다({HUB_ID_ENV}/{HUB_SECRET_ENV} 설정).")
    return _LEGACY_BASE + legacy_path, {"X-Naver-Client-Id": cid,
                                        "X-Naver-Client-Secret": sec}


def _call(key: str, cfg: dict | None = None, *, params: dict | None = None,
          body: bytes | None = None) -> dict:
    """재시도 포함 호출. 4xx(429 제외)는 재시도해도 같으므로 즉시 실패시킨다."""
    url, headers = _endpoint(key, cfg)
    method = "POST" if body is not None else "GET"
    if body is not None:
        headers["Content-Type"] = "application/json"
    last: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            r = requests.request(method, url, headers=headers, params=params,
                                 data=body, timeout=_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            text = r.text[:300]
            if r.status_code not in _RETRY_STATUS:
                raise NaverError(f"HTTP {r.status_code} (재시도 무의미): {text}")
            last = NaverError(f"HTTP {r.status_code}: {text}")
        except NaverError:
            raise
        except Exception as exc:                      # 네트워크 계층 오류
            last = exc
        if attempt < _MAX_RETRIES:
            time.sleep(1.5 * (2 ** (attempt - 1)))
    raise NaverError(f"네이버 API 호출 실패: {last}")


# ---------------------------------------------------------------------------
# 검색어 정규화
# ---------------------------------------------------------------------------

def search_term(brand_name: str) -> str:
    """공시 등록명 → 사람이 실제로 검색하는 말.

    '메가엠지씨커피(MEGA MGC COFFEE)' 처럼 영문 병기·괄호가 붙은 등록명을 그대로
    던지면 검색량이 0으로 나온다. 괄호와 특수문자를 걷어내고 앞부분만 쓴다.
    """
    t = unicodedata.normalize("NFKC", str(brand_name or "")).strip()
    t = _PAREN.sub("", t)
    t = _NONWORD.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:40]


def category_terms(industry_mid: str, industry_major: str = "",
                   brand_name: str = "") -> list[str]:
    """비교 대상 카테고리 검색어.

    ⚠️ 브랜드명을 먼저 본다. 공시 업종 중분류는 '한식'처럼 넓게 잡혀 있어서
       '인생냉면'을 **한식 전체**와 비교하게 된다. 그러면 "냉면 수요가 줄어서인가,
       이 브랜드만 밀리는가"라는 정작 궁금한 구분이 안 된다.
       브랜드명에 '냉면'·'치킨'처럼 품목이 드러나면 그 카테고리를 쓴다.
    """
    name = str(brand_name or "")
    for key, terms in CATEGORY_KEYWORDS.items():
        if key and len(key) >= 2 and key in name:
            return terms
    for key, terms in CATEGORY_KEYWORDS.items():
        if key and key in str(industry_mid or ""):
            return terms
    mid = str(industry_mid or "").strip()
    if mid and len(mid) <= 10:
        return [mid]
    return [str(industry_major or "외식").strip()]


# ---------------------------------------------------------------------------
# 검색어트렌드
# ---------------------------------------------------------------------------

def _default_period(months: int = 24) -> tuple[str, str]:
    """데이터랩은 미래 날짜를 거절한다 → 어제까지로 끊는다."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=int(months * 30.44))
    return start.isoformat(), end.isoformat()


def datalab_trend(groups: list[tuple[str, list[str]]], cfg: dict | None = None,
                  start: str | None = None, end: str | None = None,
                  time_unit: str = "month") -> dict[str, list[dict]]:
    """검색어트렌드 조회. groups = [(그룹명, [키워드…]), …] (최대 5).

    Returns:
        {그룹명: [{"period": "2025-01-01", "ratio": 45.2}, …]}
    """
    if not groups:
        return {}
    if len(groups) > MAX_GROUPS:
        raise ValueError(f"데이터랩 keywordGroups 는 최대 {MAX_GROUPS}개입니다")
    s, e = (start, end) if (start and end) else _default_period()
    body = {
        "startDate": s, "endDate": e, "timeUnit": time_unit,
        "keywordGroups": [{"groupName": g, "keywords": [k for k in ks if k][:20]}
                          for g, ks in groups if any(ks)],
    }
    if not body["keywordGroups"]:
        return {}
    data = _call("trend", cfg, body=json.dumps(body, ensure_ascii=False).encode("utf-8"))
    return {str(r["title"]): list(r.get("data") or []) for r in (data.get("results") or [])}


def _yoy(points: list[dict]) -> float | None:
    """최근 12개월 평균 vs 그 앞 12개월 평균의 증감률.

    마지막 달 하나만 비교하면 계절성(냉면=여름, 붕어빵=겨울)에 그대로 휘둘린다.
    12개월 창끼리 비교하면 계절 성분이 양쪽에서 상쇄된다.
    """
    vals = [float(p.get("ratio", 0) or 0) for p in points]
    if len(vals) < 18:
        return None
    recent = vals[-12:]
    prior = vals[-24:-12] if len(vals) >= 24 else vals[:-12]
    if not prior:
        return None
    a, b = float(np.mean(prior)), float(np.mean(recent))
    if a <= 0:
        return None
    return b / a - 1.0


def _slope(points: list[dict]) -> float | None:
    """전체 기간 선형 추세 기울기 (평균 대비 월 변화율)."""
    vals = np.array([float(p.get("ratio", 0) or 0) for p in points], dtype=float)
    if len(vals) < 6 or vals.mean() <= 0:
        return None
    x = np.arange(len(vals), dtype=float)
    slope = float(np.polyfit(x, vals, 1)[0])
    return slope / float(vals.mean())


def brand_demand(brand_name: str, industry_mid: str, industry_major: str = "",
                 cfg: dict | None = None) -> dict | None:
    """브랜드 1개의 수요 신호 (브랜드 검색 + 카테고리 검색을 **한 번의 호출**로).

    두 그룹을 한 요청에 담아야 같은 정규화 기준(기간 내 최대=100)에서 비교된다.
    따로 호출하면 각자 100으로 재정규화돼 서로 비교할 수 없다.
    """
    term = search_term(brand_name)
    if not term:
        return None
    cats = category_terms(industry_mid, industry_major, brand_name)
    cat_label = cats[0]
    s, e = _default_period()
    res = datalab_trend([(term, [term]), (cat_label, cats)], cfg, s, e)
    bpts = res.get(term) or []
    cpts = res.get(cat_label) or []
    if not bpts:
        return None
    out = {
        "term": term, "category": cat_label, "period": f"{s}~{e}",
        "brand_yoy": _yoy(bpts), "brand_slope": _slope(bpts),
        "category_yoy": _yoy(cpts) if cpts else None,
        "series": [{"d": str(p.get("period")), "v": float(p.get("ratio", 0) or 0)}
                   for p in bpts],
        "category_series": [{"d": str(p.get("period")), "v": float(p.get("ratio", 0) or 0)}
                            for p in cpts],
    }
    return out


# ---------------------------------------------------------------------------
# 뉴스·지역
# ---------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    t = _TAG.sub("", str(s or ""))
    for a, b in (("&quot;", '"'), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&apos;", "'"), ("&nbsp;", " ")):
        t = t.replace(a, b)
    return t.strip()


def news(query: str, cfg: dict | None = None, display: int = 20,
         sort: str = "date") -> list[dict]:
    """뉴스 검색 — 제목·본문요약·링크·발행일."""
    data = _call("news", cfg, params={"query": query,
                                      "display": min(int(display), 100), "sort": sort})
    out = []
    for it in data.get("items", []):
        out.append({
            "title": _clean(it.get("title")),
            "summary": _clean(it.get("description")),
            "url": it.get("originallink") or it.get("link"),
            "published": it.get("pubDate", ""),
        })
    return out


# ---------------------------------------------------------------------------
# 브랜드 로고 — 공식 홈페이지가 스스로 내건 아이콘을 쓴다
# ---------------------------------------------------------------------------
#
# 왜 이미지 검색을 쓰지 않는가 (실측으로 폐기한 1안)
#     '브랜드 로고 CI' 로 이미지 검색을 돌려 정사각형·제목 키워드로 걸러도
#     60건 중 9건만 조건을 통과했고, 그 9건조차 로고가 아니었다:
#       · 이디야커피 → "AI페퍼스 배구단, 이디야커피 로고 찍힌 유니폼" (배구 유니폼 사진)
#       · 교촌치킨   → "치킨 브랜드 로고 CI BI 디자인 제작 사례 포트폴리오" (남의 작업물)
#       · 투다리     → "로고제작 디자인 회사 기업 CI BI : 김로고" (디자인 업체 광고)
#     배구 유니폼 사진을 로고라고 붙이면 글자 마크보다 나쁘다. 그래서 폐기했다.
#
# 지금 방식 (실측 8/8 정확)
#     웹문서 검색으로 **공식 홈페이지 도메인**을 찾고, 그 사이트가 <link rel> 로
#     선언한 아이콘(apple-touch-icon 우선)이나 og:image 를 가져온다. 출처가
#     브랜드 자신이므로 엉뚱한 이미지가 붙을 여지가 없다.

# 공식 홈페이지가 아닌 곳 — 블로그·카페·SNS·뉴스·커뮤니티·쇼핑
_NOT_OFFICIAL = (
    "blog.", "cafe.", "post.naver", "tistory", "brunch", "instagram", "facebook",
    "youtube", "twitter", "x.com", "wikipedia", "namu.wiki", "news", "dcinside",
    "fmkorea", "ppomppu", "clien", "ruliweb", "inven", "danawa", "coupang",
    "11st", "gmarket", "auction", "smartstore", "shopping", "map.naver",
    "place.naver", "search.naver", "jobkorea", "saramin", "wanted",
    # ── 디렉토리·포털 (실측으로 적발) ──────────────────────────────────────
    # 프랜차이즈 정보를 나열하는 사이트라 "○○ 공식홈페이지" 검색에 잘 걸린다.
    # 이것들을 걸러내지 않아 www.114.co.kr 하나가 **215개 브랜드**의 공식
    # 홈페이지로 잡혔고, 전화번호부 아이콘이 215개 카드에 똑같이 붙었다.
    "114.co.kr", "poolixhub", "moneypin", "diningcode", "bizno.net", "sidae.com",
    "albamon", "siksinhot", "nadoweb", "marketbz", "franchise", "changuptoday",
    "startuptoday", "mangoplate", "yogiyo", "baemin", "wingeat", "menupan",
    "creatoplus", "bizinfo", "nicebizinfo", "saramin", "catchtable", "katchup",
    "storelink", "openad", "atable", "foodbank", "thinkfood", "yeogi",
    # ── 2차 적발 (수집 결과를 역추적해 찾음) ────────────────────────────────
    # 판정 방법: 수집이 끝난 색인에서 **한 도메인에 3개 이상 브랜드**가 붙은
    # 것을 전부 뽑았다. 공식 홈페이지는 브랜드마다 다르므로 이는 정의상 오탐이다.
    # 58개 도메인이 545개 브랜드를 차지하고 있었다.
    #   창업 포털  fchamall 62 · changupdo 45 · weseb 37 · jumpoline 29 ·
    #              startuplus 27 · infosori 23 · 창업오뜨야 22 · bizk 20 ·
    #              fctier 15 · fchalab 14 · fctime 13 · yesexpo 10
    #   언론        mt.co.kr 16 · shinailbo · ksilbo · mk.co.kr · insight ·
    #              kihoilbo · dt.co.kr · widedaily · kbsm · slist · mimint
    #   카드·지자체 card.kbcard 18 · gimhae.go.kr 12 · mokpo.go.kr
    #   빌더·링크   pf.kakao · story.kakao · modoo.io · imweb.me · litt.ly
    #   예약·리뷰   tabelog · autoreserve · tabling · placeview
    # ⚠️ theborn.co.kr(더본코리아) 처럼 **진짜** 가맹본부가 여러 브랜드를 한
    #    사이트에서 운영하는 경우는 넣지 않는다. 공식 홈페이지가 맞기 때문이다.
    #    그런 경우의 로고 중복은 prune_shared_logos() 가 결과를 보고 처리한다.
    "fchamall", "changupdo", "weseb.com", "jumpoline", "startuplus",
    "infosori", "xn--hu1b83jnmp41b93b", "bizk.co.kr", "fctier", "fchalab",
    "fctime", "yesexpo", "duli.co.kr", "pervsi", "jejal.net", "iamdubu",
    "store114", "placeview", "pchanet", "jshj.net", "jusogo", "localbiz.kr",
    "ludgi.ai", "purpleo.co.kr", "kkuda.kr", "richfood", "ungteori", "suto.co.kr",
    "card.kbcard", ".go.kr", "seouldanurim",
    "mt.co.kr", "shinailbo", "ksilbo", "mk.co.kr", "insight.co.kr", "kihoilbo",
    "dt.co.kr", "widedaily", "kbsm.net", "slist.kr", "mimint", "gpkorea",
    "pf.kakao", "story.kakao", "modoo.io", "imweb.me", "litt.ly",
    "tabelog", "autoreserve", "tabling", "linkareer", "job-post", "thewiki.kr",
)
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
       "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
       "Accept-Language": "ko-KR,ko;q=0.9"}

_LINK_TAG = re.compile(r"<link\b[^>]*>", re.I)
_REL = re.compile(r'rel=["\']([^"\']+)["\']', re.I)
_HREF = re.compile(r'href=["\']([^"\']+)["\']', re.I)
_SIZES = re.compile(r'sizes=["\'](\d+)x(\d+)["\']', re.I)
_OG_IMAGE = re.compile(
    r'<meta\b[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']', re.I)


def official_domain(brand_name: str, cfg: dict | None = None) -> str:
    """브랜드의 공식 홈페이지 도메인. 못 찾으면 빈 문자열.

    상위 결과에서 블로그·뉴스·쇼핑몰을 걷어내고 **가장 자주 등장한 도메인**을 고른다.
    한 번만 등장한 도메인은 우연일 수 있으므로 빈도를 본다.
    """
    q = search_term(brand_name)
    if not q:
        return ""
    try:
        data = _call("webkr", cfg, params={"query": f"{q} 공식홈페이지", "display": 15})
    except NaverError as exc:
        if "재시도 무의미" in str(exc):
            raise
        return ""
    # ⚠️ 검색 결과의 **제목에 브랜드명이 있는가**가 가장 확실한 신호다.
    #    이 검사가 없으면 디렉토리 사이트가 통과한다 — 실측으로 www.114.co.kr
    #    하나가 215개 브랜드의 '공식 홈페이지'로 잡혔다. 그런 사이트는 브랜드를
    #    나열만 하므로 개별 결과 제목에 브랜드명이 안 들어가는 경우가 많고,
    #    들어가더라도 아래 '도메인 공유' 검사(prune_shared_logos)가 잡아낸다.
    core = _NONWORD.sub("", unicodedata.normalize("NFKC", q).lower())
    counts: dict[str, int] = {}
    for it in data.get("items", []):
        host = urlparse(str(it.get("link", ""))).netloc.lower()
        if not host or any(k in host for k in _NOT_OFFICIAL):
            continue
        title = _NONWORD.sub("", unicodedata.normalize("NFKC",
                                                       _clean(it.get("title", ""))).lower())
        named = bool(core) and (core in title or core[:6] in title)
        if not (named or domain_matches_brand(host, brand_name)):
            continue
        counts[host] = counts.get(host, 0) + 1
    if not counts:
        return ""
    # 빈도 우선, 같으면 짧은 도메인(www 없는 루트에 가까운 쪽)
    return max(counts, key=lambda h: (counts[h], -len(h)))


_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_TAGS = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)


_CHO = ("g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j",
        "jj", "ch", "k", "t", "p", "h")
_JUNG = ("a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe",
         "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i")
_JONG = ("", "k", "k", "k", "n", "n", "n", "t", "l", "l", "l", "l", "l", "l",
         "l", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t", "p", "t")
# 로마자 표기가 사람마다 갈리는 지점 — 비교 전에 한쪽으로 모은다
_ROMA_FOLD = (("eo", "o"), ("eu", "u"), ("ae", "a"), ("oe", "e"),
              ("kk", "k"), ("tt", "t"), ("pp", "p"), ("ss", "s"), ("jj", "j"),
              ("ch", "c"), ("k", "g"), ("t", "d"), ("p", "b"), ("r", "l"))


def romanize(text: str) -> str:
    """한글을 대략적인 로마자로. 도메인 비교용이라 표준 표기가 목표가 아니다.

    왜 필요한가
        공식 홈페이지 상당수가 JS 로 렌더링돼 원본 HTML 에 브랜드명이 없다.
        그래서 '페이지에 브랜드명이 있는가' 검사만 쓰면 푸라닭(puradakchicken.com)·
        명랑시대쌀핫도그(myungranghotdog.com)·우리할매떡볶이(wehalmae.co.kr) 같은
        **진짜 공식 사이트가 탈락한다**(실측 146건 중 다수). 도메인 자체가 브랜드명의
        로마자인 경우가 많으므로 그것을 보조 신호로 쓴다.
    """
    out = []
    for ch in str(text or ""):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            idx = code - 0xAC00
            out.append(_CHO[idx // 588] + _JUNG[(idx % 588) // 28] + _JONG[idx % 28])
        elif ch.isascii() and ch.isalnum():
            out.append(ch.lower())
    s = "".join(out)
    for a, b in _ROMA_FOLD:
        s = s.replace(a, b)
    return s


DOMAIN_MATCH_MIN = 0.62     # 로마자 유사도 하한


def domain_matches_brand(domain: str, brand_name: str) -> bool:
    """도메인이 브랜드명의 로마자와 충분히 닮았는가.

    접두 완전일치로는 안 된다 — 로마자 표기가 사람마다 갈린다.
    설빙 solbing↔sulbing, 명랑 myong↔myung, 우리 uli↔we 처럼 같은 브랜드도
    표기가 다르다(실측). 그래서 **부분 유사도**로 본다.

    도메인 쪽에 브랜드 로마자가 얼마나 들어와 있는지를 재므로, 도메인에 부가어가
    붙어도(puradak**chicken**, myungrang**hotdog**) 성립한다.
    """
    raw_host = re.sub(r"^www\.", "", str(domain or "").lower()).split(".")[0].replace("-", "")
    if len(raw_host) < 3:
        return False
    # 공시 등록명의 괄호 안 영문 병기가 도메인과 그대로 맞는 경우가 많다
    # ('메가엠지씨커피(MEGA MGC COFFEE)' → mega-mgccoffee.com). 한글 로마자보다
    # 훨씬 정확하므로 먼저 본다. ⚠️ 이 비교는 **로마자 변환 전** 원문끼리 해야 한다
    # (host 를 romanize 하면 p→b 폴딩 때문에 composecoffee 가 combosecoffee 가 된다).
    for m in _PAREN.finditer(str(brand_name or "")):
        eng = re.sub(r"[^a-z0-9]", "", m.group(0).lower())
        if len(eng) >= 3 and (eng in raw_host or raw_host in eng):
            return True
    host = romanize(raw_host)
    core = romanize(search_term(brand_name))
    if len(core) < 4:
        return False
    if core in host or host in core:
        return True
    # 브랜드 앞부분이 도메인 어딘가와 얼마나 겹치는가 (도메인의 부가어를 무시)
    probe = core[:max(5, min(10, len(core)))]
    m = difflib.SequenceMatcher(None, probe, host)
    match = m.find_longest_match(0, len(probe), 0, len(host))
    return match.size / len(probe) >= DOMAIN_MATCH_MIN


def site_mentions_brand(html: str, brand_name: str) -> bool:
    """그 사이트가 정말 이 브랜드의 것인지 — **페이지에 브랜드명이 있는가**로 판정한다.

    왜 차단 목록으로는 부족한가
        디렉토리·언론·링크서비스는 계속 새로 생긴다. 목록을 아무리 늘려도
        `linktr.ee`·`femiwiki.com`·`businesskorea.co.kr` 같은 것들이 계속 새어 나온다.
        반면 **공식 홈페이지라면 브랜드명이 페이지에 반드시 있다**. 이 검사는
        목록 유지보수 없이 일반적으로 동작한다.

    비교는 정규화 후에 한다 — 공시 등록명 '메가엠지씨커피(MEGA MGC COFFEE)' 와
    사이트의 '메가MGC커피' 는 구분기호·영문 병기 때문에 원문 그대로는 안 맞는다.
    """
    if not html:
        return False
    text = _TAGS.sub(" ", html)
    norm_page = _NONWORD.sub("", unicodedata.normalize("NFKC", text).lower())
    core = _NONWORD.sub("", unicodedata.normalize("NFKC", search_term(brand_name)).lower())
    if len(core) < 2:
        return False
    if core in norm_page:
        return True
    # 등록명이 길면 앞부분(브랜드 고유부)만으로도 인정한다 — '떡볶이 참 잘하는 집, 떡참'
    head = core[:6] if len(core) >= 6 else core
    return len(head) >= 3 and head in norm_page


def _fetch_html(url: str, limit: int = 250_000) -> str:
    try:
        r = requests.get(url, headers=_UA, timeout=10, allow_redirects=True)
        if r.status_code != 200:
            return ""
        return r.content[:limit].decode(r.encoding or "utf-8", "ignore")
    except Exception:
        return ""


# 선언이 없어도 관행적으로 존재하는 아이콘 경로. 파비콘만 있는 사이트에서
# 더 큰 이미지를 건지는 마지막 수단이다.
_ICON_GUESSES = (
    "/apple-touch-icon.png", "/apple-touch-icon-precomposed.png",
    "/apple-touch-icon-180x180.png",
    # 국내 프랜차이즈 사이트가 실제로 쓰는 로고 경로들. 파비콘만 뒤지면 16~32px 밖에
    # 못 건지는데(실측: 네네치킨 32px, 한솥 16px), 이 경로에는 원본 로고가 있다.
    "/images/common/logo.png", "/images/logo.png", "/img/logo.png",
    "/assets/images/logo.png", "/assets/img/logo.png", "/static/images/logo.png",
    "/images/common/logo300.jpg", "/images/logo.svg",
    "/favicon.png", "/favicon.ico")

# og:image 가 개발 호스트를 가리키는 사이트가 실제로 있다 — BHC 는 운영 페이지에서
# `http://localhost:3000/images/common/logo300.jpg` 를 내보낸다. 경로는 맞으므로
# 호스트만 실제 도메인으로 바꿔 주면 진짜 로고를 받을 수 있다.
_DEV_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "dev.", "test.", "stage.")
MIN_LOGO_PX = 64        # 이보다 작으면 카드에서 흐릿해 로고 구실을 못 한다
                        # (48px 로 뒀더니 모자이크처럼 보이는 건이 남았다)


def icon_candidates(domain: str, brand_name: str = "", *, verify: bool = True) -> list[str]:
    """도메인에서 뽑을 수 있는 아이콘 후보 URL 전부 (선언 + 관행 경로 + og:image).

    verify=True 이고 brand_name 이 있으면 **그 사이트가 정말 이 브랜드의 것인지**
    확인하고 아니면 빈 목록을 돌려준다.

    ⚠️ `official_domain()` 이 이미 검증한 도메인에는 verify=False 로 부른다.
       두 번 거르면 **같은 관문을 두 번 통과해야** 하는데, 두 번째 관문은 페이지를
       직접 받아 텍스트를 보므로 JS 렌더링 사이트에서 통째로 실패한다.
       실측: 도메인이 잡힌 1,326개 중 1,078개(81%)가 이 단계에서 버려졌다 —
       교촌·이디야·맘스터치 같은 대형 브랜드가 전부 여기서 사라졌다.
       엉뚱한 로고에 대한 방어는 `prune_shared_logos()` 가 담당한다(같은 도메인·
       같은 이미지가 여러 브랜드에 붙으면 전부 버린다).
    """
    if not domain:
        return []
    base = f"https://{domain}"
    html = _fetch_html(base)
    # 두 신호 중 **하나만** 맞아도 인정한다. 페이지 텍스트는 JS 렌더링이면 비고,
    # 도메인 로마자는 영문 브랜드명이면 안 맞는다 — 서로의 빈틈을 메운다.
    if verify and brand_name and not (site_mentions_brand(html, brand_name)
                                      or domain_matches_brand(domain, brand_name)):
        return []
    out: list[str] = []

    declared: list[tuple[int, str]] = []
    for tag in _LINK_TAG.findall(html or ""):
        rel_m, href_m = _REL.search(tag), _HREF.search(tag)
        if not (rel_m and href_m) or "icon" not in rel_m.group(1).lower():
            continue
        sz = _SIZES.search(tag)
        # apple-touch-icon 은 홈 화면용이라 대개 180px 이상이고 브랜드 심볼 원본이다.
        # 크기 선언이 없어도 파비콘보다 우선하도록 가중치를 준다.
        weight = int(sz.group(1)) if sz else (180 if "apple-touch" in rel_m.group(1).lower() else 0)
        declared.append((weight, urljoin(base, href_m.group(1))))
    out += [u for _, u in sorted(declared, reverse=True)]

    og = _OG_IMAGE.search(html or "")
    if og:
        u = urljoin(base, og.group(1))
        host = urlparse(u).netloc.lower()
        if any(h in host for h in _DEV_HOSTS):
            # 개발 호스트가 새어 나온 경우 — 경로만 살려 실제 도메인으로 돌린다
            u = urljoin(base, urlparse(u).path)
        out.append(u)
    out += [base + g for g in _ICON_GUESSES]

    seen: set[str] = set()
    return [u for u in out if not (u in seen or seen.add(u))]


def _measure(url: str) -> tuple[int, int]:
    """이미지를 실제로 받아 (짧은 변, 바이트) 를 잰다. 실패하면 (0, 0).

    선언된 sizes 속성은 거짓인 경우가 흔하다(180x180 이라 적어 놓고 32px 이미지).
    화면 품질을 좌우하는 값이므로 믿지 말고 직접 잰다.
    """
    try:
        r = requests.get(url, headers=_UA, timeout=8)
        if r.status_code != 200 or not r.headers.get("Content-Type", "").startswith("image"):
            return 0, 0
        data = r.content
        if len(data) < 300:                 # 1×1 투명 픽셀 등
            return 0, 0
        from io import BytesIO

        from PIL import Image
        return _largest_side(Image.open(BytesIO(data))), len(data)
    except Exception:
        return 0, 0


def _largest_side(im) -> int:
    """이미지의 짧은 변. **.ico 는 여러 해상도를 품으므로 가장 큰 프레임**을 본다.

    ⚠️ PIL 은 ICO 를 열 때 기본 프레임(대개 16~32px)을 준다. 그대로 재면 실제로는
       192px 프레임을 가진 파비콘이 '너무 작음'으로 버려진다 — 이디야·BHC 가
       그렇게 탈락했다. `ico.sizes()` 로 최대 해상도를 확인한다.
    """
    try:
        sizes = getattr(getattr(im, "ico", None), "sizes", None)
        if callable(sizes):
            best = max(sizes(), key=lambda wh: min(wh))
            return int(min(best))
    except Exception:
        pass
    return int(min(im.size))


def _open_best_frame(data: bytes):
    """바이트 → 가장 큰 프레임으로 연 PIL 이미지 (.ico 대응)."""
    from io import BytesIO

    from PIL import Image
    im = Image.open(BytesIO(data))
    try:
        sizes = getattr(getattr(im, "ico", None), "sizes", None)
        if callable(sizes):
            return im.ico.getimage(max(sizes(), key=lambda wh: min(wh)))
    except Exception:
        pass
    return im


def site_icon(domain: str, brand_name: str = "", *, verify: bool = True) -> tuple[str, int]:
    """도메인의 대표 아이콘 (URL, 실측 짧은 변 픽셀). 못 찾으면 ("", 0).

    후보를 앞에서부터 실제로 받아 재고, MIN_LOGO_PX 이상이면 즉시 채택한다.
    """
    best, best_px = "", 0
    for url in icon_candidates(domain, brand_name, verify=verify)[:8]:
        px, _ = _measure(url)
        if px > best_px:
            best, best_px = url, px
        if px >= MIN_LOGO_PX:
            break
    return best, best_px


_LOGO_HINT = re.compile(r"logo|_bi\b|\bci_|brand[-_]?mark|symbol|header[-_]?img", re.I)
# 로고가 아니라 그 페이지가 얹은 남의 상표인 것들. 이게 붙으면 다른 브랜드 로고가 나간다.
_NOT_LOGO_SRC = re.compile(
    r"kakao|naver|instagram|facebook|youtube|blog|band\.us|twitter|"
    r"appstore|google.?play|app_?store|qr|btn|icon_sns|sns_|share|"
    r"go\.kr|or\.kr/|kfa|kab_|footer_logo", re.I)


def page_logo_candidates(domain: str, brand_name: str) -> list[tuple[str, str]]:
    """브랜드 **자기 사이트의 페이지 안**에서 로고 이미지 후보를 찾는다.

    왜 필요한가 (실측)
        파비콘만 보면 브랜드의 34.2%(343/1,003)가 '도메인은 찾았는데 아이콘이
        없음'으로 끝난다. 파비콘을 아예 안 두거나 16~32px 만 두는 사이트가 많다.
        그런데 그 사이트의 **헤더에는 원본 로고가 걸려 있다.**

    ⚠️ 반드시 페이지 본문으로 브랜드 소유를 확인한 뒤에만 쓴다. 확인 없이 긁으면
       검색이 물어 온 엉뚱한 사이트의 로고가 붙는다 — 실측 사고: 일미리금계찜닭에
       `gimhae.go.kr`(김해시청), 생어거스틴에 `booking.kakao.com`(카카오 예약)이
       잡혔다. 그 상태로 이미지를 받으면 김해시 로고가 찜닭 브랜드에 나간다.
       **틀린 로고는 글자 마크보다 나쁘다.**

    Returns: [(종류, 절대 URL)] — 앞쪽이 더 신뢰할 만한 후보.
    """
    for scheme in ("https://", "http://"):
        html = _fetch_html(scheme + domain)
        if html:
            base = scheme + domain
            break
    else:
        return []
    if not site_mentions_brand(html, brand_name):
        return []

    out: list[tuple[str, str]] = []
    # ⚠️ og:image 는 로고가 아니라 **공유 카드용 대표 이미지**다. 그냥 받으면
    #    배너·음식 사진·SNS 버튼·카톡 캡처가 로고 자리에 들어간다. 실측 사례:
    #        고망고        main_img_01.jpg (1920px)  메인 배너
    #        할머니가래떡볶이  sns_link.png              SNS 버튼
    #        국민낙곱새      KakaoTalk_...jpg          카톡 캡처
    #        열정국밥       default.jpg               자리표시자
    #    그래서 og:image 는 **파일 경로가 로고임을 명시할 때만** 후보로 받는다.
    m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', html, re.I)
    if m and re.search(r"(^|[/_-])(logo|ci|bi|symbol|brand)([._-]|$)", m.group(1), re.I):
        out.append(("og:image", urljoin(base, m.group(1))))
    for tag in re.finditer(r"<img[^>]+>", html, re.I):
        t = tag.group(0)
        if not _LOGO_HINT.search(t):
            continue
        src = re.search(r'src=["\']([^"\']+)', t)
        if not src or _NOT_LOGO_SRC.search(src.group(1)):
            continue
        out.append(("page-logo", urljoin(base, src.group(1))))
    # 같은 URL 중복 제거 (순서 유지)
    seen, uniq = set(), []
    for kind, u in out:
        if u not in seen:
            seen.add(u)
            uniq.append((kind, u))
    return uniq[:8]


def brand_logo(brand_name: str, cfg: dict | None = None) -> str:
    """브랜드 로고 이미지 URL. 못 찾으면 빈 문자열(화면은 글자 마크로 대체)."""
    domain = official_domain(brand_name, cfg)
    if not domain:
        return ""
    # official_domain 이 이미 브랜드 소유를 검증했다 — 여기서 또 거르지 않는다.
    url, _px = site_icon(domain, brand_name, verify=False)
    return url


LOGO_CACHE = "data/raw/naver/logos.json"
LOGO_DIR = "data/raw/naver/logo_img"
# 저장 해상도. 화면 최대 84px 이고 고해상도 디스플레이는 2배로 그리므로 168px 이
# 필요하다 — 128px 로는 상세 화면에서 눈에 띄게 뭉갠다. 256px 이면 여유가 있고
# PNG 한 장이 10~30KB 라 저장소에도 부담이 없다.
LOGO_PX = 256


def logo_file_name(brand_name: str) -> str:
    """브랜드명 → 로고 파일명. 한글·특수문자가 섞이므로 해시를 쓴다.

    화면(views/common.py)도 이 함수로 경로를 **계산**한다. 색인 파일을 거치면
    수집이 색인을 덮어쓸 때 참조가 끊긴다(실측: PNG 556장이 있는데 색인은 0건).
    """
    return hashlib.sha1(str(brand_name).encode("utf-8")).hexdigest()[:16] + ".png"


def download_logo(brand_name: str, url: str, root: Path) -> str:
    """로고를 내려받아 정사각 PNG 로 저장하고 상대경로를 돌려준다.

    왜 링크를 그대로 쓰지 않는가
      · 핫링크 차단: 상당수 기업 사이트가 외부 Referer 요청을 막는다. 배포한 화면에서
        깨진 이미지가 뜨는 순간 서비스가 미완성으로 보인다.
      · 남의 대역폭: 심사역이 볼 때마다 그 회사 서버를 때리는 것은 예의가 아니다.
      · 시연 안정성: 상대 사이트가 잠깐 죽어도 우리 화면은 멀쩡해야 한다.
    """
    if not url:
        return ""
    dest_dir = root / LOGO_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / logo_file_name(brand_name)
    try:
        r = requests.get(url, headers=_UA, timeout=12)
        if r.status_code != 200 or len(r.content) < 300:
            return ""
        from PIL import Image
        with _open_best_frame(r.content) as im:
            im = im.convert("RGBA")
            # 원본이 16~32px 파비콘이면 카드 안에서 흐릿한 점으로 보인다.
            # 그런 이미지는 브랜드 색 글자 마크보다 못하므로 아예 쓰지 않는다
            # (실측: 더벤티 16px, 네네치킨 32px 이 화면에서 판독 불가였다).
            if min(im.size) < MIN_LOGO_PX:
                return ""
            # 원본이 저장 해상도보다 작으면 **키우지 않는다** — 늘리면 뭉개진다.
            if max(im.size) > LOGO_PX:
                im.thumbnail((LOGO_PX, LOGO_PX), Image.LANCZOS)
            # 정사각 캔버스 중앙 배치 — 가로로 긴 로고도 카드 안에서 잘리지 않는다.
            # ⚠️ 캔버스를 항상 LOGO_PX 로 잡으면 안 된다. 작은 원본이 큰 투명 여백
            #    안에 앉으면 CSS `object-fit: contain` 이 그 여백까지 맞추느라
            #    화면에서 **로고만 작게** 보인다 — 브랜드마다 크기가 들쭉날쭉해진다.
            #    캔버스를 이미지에 딱 맞춰 정사각으로만 만들면 모두 같은 크기로 뜬다.
            side = max(im.width, im.height)
            canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
            canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
            canvas.save(dest, "PNG", optimize=True)
    except Exception:
        return ""
    return f"{LOGO_DIR}/{dest.name}"


def _load_logo_cache(cfg: dict) -> dict:
    p = Path(cfg["_root"]) / LOGO_CACHE
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    # 구버전(브랜드→URL 문자열) 호환
    return {k: (v if isinstance(v, dict) else {"url": v or "", "domain": "", "px": 0})
            for k, v in obj.items()} if isinstance(obj, dict) else {}


def prune_shared_logos(cfg: dict, cache: dict) -> dict:
    """여러 브랜드가 나눠 쓰는 로고를 **전부 폐기**한다.

    왜 이 검사가 필수인가 (실측 사고)
        공식 홈페이지 도메인은 브랜드마다 다르다. 한 도메인이 여러 브랜드의
        '공식 홈페이지'일 수는 없다. 그런데 검색이 디렉토리 사이트를 물어 오면서
        `www.114.co.kr`(전화번호부) 하나가 **215개 브랜드**의 홈페이지로 잡혔고,
        전화번호부 아이콘이 215개 카드에 똑같이 붙었다. 알바몬·식신·다이닝코드도
        같은 방식으로 수십 개씩 오염시켰다. 전체 1,074건 중 665건(62%)이 오염이었다.

        차단 목록만으로는 막을 수 없다 — 새로운 디렉토리 사이트는 계속 생긴다.
        그래서 **결과를 보고 판정한다**: 같은 도메인 또는 같은 이미지가 둘 이상의
        브랜드에 걸리면 그것은 브랜드 로고가 아니다.

    한 회사가 여러 브랜드를 한 사이트에서 운영하는 경우(예: 한 본부의 서브 브랜드)도
    폐기 대상이다. 그 경우에도 서로 다른 브랜드에 같은 그림이 붙는 것은 마찬가지이고,
    틀린 로고보다 브랜드 색 글자 마크가 낫다.
    """
    root = Path(cfg["_root"])
    img_dir = root / LOGO_DIR

    dom_count: dict[str, int] = {}
    for v in cache.values():
        if isinstance(v, dict) and v.get("domain"):
            dom_count[v["domain"]] = dom_count.get(v["domain"], 0) + 1

    hash_owner: dict[str, list[str]] = {}
    for name in cache:
        p = img_dir / logo_file_name(name)
        if p.exists():
            try:
                h = hashlib.md5(p.read_bytes()).hexdigest()
            except OSError:
                continue
            hash_owner.setdefault(h, []).append(name)
    shared_hash = {h for h, owners in hash_owner.items() if len(owners) > 1}

    removed_dom = removed_img = 0
    for name, v in cache.items():
        if not isinstance(v, dict):
            continue
        p = img_dir / logo_file_name(name)
        reason = ""
        # ⚠️ 이미지 검색 + LLM 판독으로 얻은 로고는 **도메인에서 온 것이 아니다.**
        #    캐시에 남아 있는 domain 은 예전 사이트 탐색의 흔적일 뿐이라, 그 도메인이
        #    공유됐다는 이유로 지우면 멀쩡히 검증된 로고가 사라진다. 출처로 구분한다.
        from_domain = not str(v.get("source", "")).startswith("image-search")
        if from_domain and v.get("domain") and dom_count.get(v["domain"], 0) > 1:
            reason = f"도메인 공유({dom_count[v['domain']]}개 브랜드)"
            removed_dom += 1
        elif p.exists():
            try:
                if hashlib.md5(p.read_bytes()).hexdigest() in shared_hash:
                    reason = "동일 이미지가 다른 브랜드에도 사용됨"
                    removed_img += 1
            except OSError:
                pass
        if reason:
            p.unlink(missing_ok=True)
            v["url"], v["file"], v["rejected"] = "", "", reason

    orphan_img, orphan_idx = reconcile_logo_files(cfg, cache)

    n_left = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    log.info("공유 로고 폐기: 도메인 공유 %d건 · 동일 이미지 %d건 → 남은 로고 %d건",
             removed_dom, removed_img, n_left)
    return {"removed_domain": removed_dom, "removed_image": removed_img,
            "orphan_image": orphan_img, "orphan_index": orphan_idx,
            "remaining": n_left}


def reconcile_logo_files(cfg: dict, cache: dict) -> tuple[int, int]:
    """색인과 디스크를 일치시킨다 — **화면이 보는 것은 디스크다.**

    왜 필요한가 (실측 사고)
        화면은 색인(logos.json)을 거치지 않고 브랜드명 해시로 파일 경로를 계산해
        **파일이 있으면 띄운다**. 수집이 색인을 통째로 덮어쓰는 경합 때문에 그렇게
        만들었는데, 그 결과 색인에서 철회한 로고라도 PNG 가 남아 있으면 화면에는
        계속 뜬다. 실제로 철회된 3건이 배포 화면에 그대로 나오고 있었다.
        철회의 이유는 "이 그림은 이 브랜드 것이 아니다" 이므로, 이건 **틀린 로고를
        띄우는 것**이고 로고가 없는 것보다 나쁘다.

        반대 방향도 있다. 색인에는 url 이 있는데 PNG 가 없는 항목이 29건이었다.
        화면에는 글자 마크가 뜨므로 해롭진 않지만, 커버리지를 그만큼 부풀린다.

    그래서 한 방향의 불변식으로 못박는다:
        **디스크의 PNG 집합 == 색인에서 url 이 살아 있는 브랜드 집합**

    - 철회됐는데 PNG 가 남았다 → 파일을 지운다 (화면에서 사라진다)
    - url 은 있는데 PNG 가 없다 → 색인을 비운다 (다음 수집에서 다시 시도한다)

    반환: (지운 PNG 수, 비운 색인 수)
    """
    img_dir = Path(cfg["_root"]) / LOGO_DIR
    orphan_img = orphan_idx = 0
    for name, v in cache.items():
        if not isinstance(v, dict):
            continue
        p = img_dir / logo_file_name(name)
        exists, claimed = p.exists(), bool(v.get("url"))
        if exists and not claimed:
            p.unlink(missing_ok=True)
            v["file"] = ""
            v.setdefault("rejected", "철회된 로고의 잔존 파일")
            orphan_img += 1
        elif claimed and not exists:
            v["url"], v["file"] = "", ""
            v["rejected"] = "이미지 파일 없음 — 다음 수집에서 재시도"
            orphan_idx += 1
    if orphan_img or orphan_idx:
        log.info("색인·디스크 정합: 잔존 파일 %d건 삭제 · 실체 없는 색인 %d건 정리",
                 orphan_img, orphan_idx)
    return orphan_img, orphan_idx


def collect_logos(cfg: dict | None = None, limit: int | None = None,
                  sleep_sec: float = 0.05) -> dict:
    """코호트 브랜드의 로고를 한꺼번에 찾아 캐시에 저장한다.

    화면에서 브랜드를 볼 때마다 조회하면 느리고 호출 한도를 잡아먹는다.
    미리 채워 두면 화면은 캐시만 읽는다. 이미 조회한 브랜드는 건너뛰므로
    중단해도 이어서 돌릴 수 있다.

    캐시에는 URL 뿐 아니라 **도메인과 실측 픽셀**도 남긴다. 나중에 "이 로고가
    어디서 왔는가"를 확인할 수 있어야 하고, 화질이 낮은 건을 골라낼 수 있어야 한다.
    """
    cfg = cfg or load_config()
    dest = Path(cfg["_root"]) / LOGO_CACHE
    cache = _load_logo_cache(cfg)

    if not is_enabled(cfg):
        log.warning("네이버 자격증명이 없어 로고를 수집하지 않는다 — "
                    "developers.naver.com 에서 발급 후 %s/%s 설정", ID_ENV, SECRET_ENV)
        return {"enabled": False, "n_cached": len(cache)}

    spath = Path(cfg["paths"]["outputs"]) / "scores_latest.csv"
    if not spath.exists():
        raise FileNotFoundError(f"{spath} 없음 — 먼저 `--step score` 실행")
    scores = pd.read_csv(spath, encoding="utf-8-sig")
    scores = scores.assign(
        _n=pd.to_numeric(scores["n_stores"], errors="coerce").fillna(0))
    todo = scores.sort_values("_n", ascending=False)   # 큰 브랜드부터 (화면 노출 빈도 순)
    if limit:
        todo = todo.head(int(limit))

    def _save() -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    done = 0
    for _, r in todo.iterrows():
        name = str(r["brand_name"])
        if name in cache:
            continue
        try:
            domain = official_domain(name, cfg)
            # ⚠️ verify=False 를 반드시 넘긴다. official_domain() 이 이미 브랜드
            #    소유를 확인했고, 여기서 또 확인하면 페이지를 직접 받아 텍스트를
            #    보는 2차 관문이 걸린다 — JS 렌더링 사이트가 통째로 탈락한다.
            #    이 인자를 빼먹어서 **도메인은 찾았는데 아이콘이 없는 브랜드가
            #    867개(60.3%)** 였다. brand_logo() 만 고치고 여기를 놓쳤다.
            url, px = site_icon(domain, name, verify=False) if domain else ("", 0)
        except NaverError as exc:
            log.warning("로고 수집 중단 (%s): %s", name, exc)
            break
        except Exception as exc:                # 한 브랜드의 사이트 오류로 멈추지 않는다
            log.info("로고 조회 실패 (%s): %s", name, str(exc)[:80])
            domain, url, px = "", "", 0
        local = download_logo(name, url, Path(cfg["_root"])) if url else ""
        cache[name] = {"url": url, "domain": domain, "px": int(px), "file": local}
        done += 1
        if done % 50 == 0:
            _save()
            n_ok = sum(1 for v in cache.values() if v.get("url"))
            log.info("로고 수집 %d건 처리 · 누적 확보 %d/%d", done, n_ok, len(cache))
        time.sleep(sleep_sec)

    _save()
    prune_shared_logos(cfg, cache)
    _save()
    n_ok = sum(1 for v in cache.values() if v.get("url"))
    n_hi = sum(1 for v in cache.values() if v.get("px", 0) >= MIN_LOGO_PX)
    log.info("로고 수집 완료: 이번 %d건 · 누적 %d개 중 확보 %d (%.0f%%) · "
             "선명(%dpx 이상) %d (%.0f%%) → %s",
             done, len(cache), n_ok, 100 * n_ok / max(len(cache), 1),
             MIN_LOGO_PX, n_hi, 100 * n_hi / max(len(cache), 1), dest.name)
    return {"enabled": True, "n_processed": done, "n_cached": len(cache),
            "n_with_logo": n_ok, "n_sharp": n_hi}


# ---------------------------------------------------------------------------
# 배치 수집 (파이프라인 스텝)
# ---------------------------------------------------------------------------

def collect_demand(cfg: dict | None = None, limit: int | None = None,
                   sleep_sec: float = 0.12) -> dict:
    """점수 코호트 상위 브랜드의 검색수요를 모아 outputs/demand_trends.json 으로 저장.

    호출량: 브랜드 1개당 데이터랩 1회. 일 25,000회 한도 안에서 상위 N개를 덮는다.
    이미 수집된 브랜드는 건너뛰므로 여러 날에 나눠 넓힐 수 있다.
    """
    cfg = cfg or load_config()
    out_dir = Path(cfg["paths"]["outputs"])
    dest = out_dir / "demand_trends.json"
    ncfg = cfg.get("naver", {}) or {}
    limit = int(limit if limit is not None else ncfg.get("demand_top_n", 300))

    prev: dict = {}
    if dest.exists():
        try:
            prev = json.loads(dest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
    brands: dict = dict(prev.get("brands") or {})

    if not has_trend(cfg):
        why = ("API HUB 자격증명 미설정" if not is_enabled(cfg)
               else "구 개발자센터 키로는 검색어트렌드를 쓸 수 없음")
        meta = {"enabled": False, "reason": f"{why} ({HUB_ID_ENV}/{HUB_SECRET_ENV})",
                "n_collected": len(brands), "brands": brands}
        dest.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        log.warning("검색수요를 수집하지 않는다 — %s. NCP 콘솔의 NAVER API HUB → "
                    "Application 에서 Client ID/Secret 을 받아 %s/%s 에 설정하세요.",
                    why, HUB_ID_ENV, HUB_SECRET_ENV)
        return meta

    spath = out_dir / "scores_latest.csv"
    if not spath.exists():
        raise FileNotFoundError(f"{spath} 없음 — 먼저 `--step score` 실행")
    scores = pd.read_csv(spath, encoding="utf-8-sig")
    # 규모×위험 순으로 훑는다 — 여신 영향이 큰 브랜드부터 덮어야 한도 안에서 이득이 크다
    scores = scores.assign(_pri=(pd.to_numeric(scores["deterioration_1y"], errors="coerce").fillna(0)
                                 * pd.to_numeric(scores["n_stores"], errors="coerce").fillna(0)))
    todo = scores.sort_values("_pri", ascending=False).head(limit)

    n_new, n_fail = 0, 0
    for _, r in todo.iterrows():
        bid = str(r["brand_id"])
        if brands.get(bid):
            continue
        try:
            d = brand_demand(str(r["brand_name"]), str(r.get("industry_mid") or ""),
                             str(r.get("industry_major") or ""), cfg)
        except NaverError as exc:
            n_fail += 1
            if "Scope Status Invalid" in str(exc):
                # 키는 맞지만 애플리케이션에 '데이터랩(검색어트렌드)' API 가 추가돼
                # 있지 않을 때 나온다. 원인을 알려주지 않으면 키가 틀린 줄 안다.
                log.error(
                    "검색어트렌드 권한 없음 — 애플리케이션에 '데이터랩(검색어트렌드)' API가 "
                    "추가돼 있지 않습니다. developers.naver.com → 내 애플리케이션 → "
                    "API 설정 → '데이터랩(검색어트렌드)' 체크 후 저장하세요. "
                    "(검색 API는 정상 동작 중이므로 뉴스·로고는 영향 없습니다)")
                break
            log.warning("수요 조회 실패 (%s): %s", r["brand_name"], exc)
            if "재시도 무의미" in str(exc):
                break                       # 인증·권한 오류면 나머지도 전부 실패한다
            continue
        brands[bid] = d
        if d:
            n_new += 1
        time.sleep(sleep_sec)

    covered = {k: v for k, v in brands.items() if v}
    meta = {
        "enabled": True,
        "period": _default_period()[0] + "~" + _default_period()[1],
        "n_requested": len(todo), "n_collected": len(covered),
        "n_new_this_run": n_new, "n_failed": n_fail,
        "source": "네이버 데이터랩 검색어트렌드 (상대지수, 기간 내 최대=100)",
        "brands": brands,
    }
    dest.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("검색수요 수집: 신규 %d개 · 누적 %d개 · 실패 %d개 → %s",
             n_new, len(covered), n_fail, dest.name)
    return meta


if __name__ == "__main__":
    collect_demand()
