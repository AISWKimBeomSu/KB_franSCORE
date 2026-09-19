"""지방행정 인허가(음식점 개·폐업) → 브랜드별 월간 개·폐점 흐름과 조기경보 신호.

왜 필요한가
    공정위 정보공개서의 가맹점 수·계약종료는 연 1회, 1~2년 늦게 보인다. 반면 지자체 식품접객업
    인허가 데이터는 점포 하나하나의 인허가일자·폐업일자를 담고 매일 갱신된다(2일 전 기준). 가맹점도
    일반음식점·휴게음식점·제과점영업 인허가를 받아야 문을 열 수 있으므로, 사업장명을 브랜드에 연결하면
    '이 브랜드의 최근 3개월 폐업이 작년 같은 때보다 많은가'를 **월 단위로** 볼 수 있다.

데이터 확보 경로 (2026-09-19 실측)
    LOCALDATA 포털(www.localdata.go.kr)은 행안부가 공공데이터포털로 통합하면서(2026-01 발표) 2026-04-15
    병행운영이 끝났고 지금은 접속되지 않는다(실측: 연결 거부·시간 초과). 두 경로가 남아 있다.

    ① 파일 — 키·로그인 불필요 (download_csv)
       공공데이터포털 파일데이터 15045016(일반음식점)·15006730(휴게음식점)의 '제공데이터URL'이 가리키는 곳.
         안내   https://file.localdata.go.kr/file/{service}/info
         전국   https://file.localdata.go.kr/file/download/{service}/info
         지역   https://file.localdata.go.kr/file/download/{service}/info?orgCode={개방자치단체코드}
                (시도 '6110000_ALL' 또는 시군구 '3220000'=서울강남구 — 안내 페이지 선택상자 값)
       service: general_restaurants(일반음식점) · rest_cafes(휴게음식점) · bakeries(제과점영업)
       크기(Content-Length 실측): 전국 일반음식점 697.6MB(포털 표기 2,129,830행) · 휴게음식점 207.8MB
       (561,397행) · 제과점영업 22.7MB, 서울 전체 일반음식점 161.9MB, 구·시 단위 0.2~16MB.
       ⚠️ 함정 (모두 실측)
         · 응답 헤더는 charset=UTF-8 인데 **실제 바이트는 CP949** 다. load_records 가 판별한다.
         · WAF 가 브라우저 User-Agent **와** 같은 사이트 Referer 를 함께 요구한다. 하나라도 없으면
           302→/error.html 이나 403(HTML)이 온다. HEAD 도 막혀 있어 크기 확인도 GET 헤더로 한다.
         · 브라우저는 내려받기 전에 /file/validate/download-count 를 부른다(과다 시 429). 그대로 따른다.
         · 날짜는 파일에서 'YYYY-MM-DD', API 에서 'YYYYMMDD'. 좌표는 EPSG:5174(보정계수 없는 Bessel TM).
         · 영업상태는 01 영업/정상, 03 폐업만 보였다(휴업 미공개). 폐업일자 없는 폐업은 없었다.
    ② OpenAPI — 키 필요 (DATA_GO_KR_KEY + 데이터셋별 활용신청, 자동승인, 개발계정 10,000건/일)
         15154916 https://apis.data.go.kr/1741000/general_restaurants/{info|history}
         15154921 https://apis.data.go.kr/1741000/rest_cafes/{info|history}
         15155252 https://apis.data.go.kr/1741000/bakeries/{info|history}
       info: serviceKey·pageNo·numOfRows(최대 100)·returnType 와 조건 cond[BPLC_NM::LIKE],
             cond[LCPMT_YMD::GTE|LT], cond[SALS_STTS_CD::EQ], cond[ROAD_NM_ADDR::LIKE],
             cond[DAT_UPDT_PNT::GTE|LT], cond[OPN_ATMY_GRP_CD::EQ]
       history: cond[BASE_DATE::EQ](2026-01-01~전일, 그날 기준 상태) 와 cond[OPN_ATMY_GRP_CD::EQ] 필수
       응답 item 은 파일과 같은 39개 항목의 영문 코드(BPLC_NM, LCPMT_YMD, CLSBIZ_YMD, SALS_STTS_CD …).
       키 없이 부르면 HTTP 401 SERVICE_KEY_IS_NULL(실측). 사업장명 LIKE 로 브랜드별로 받을 수 있어,
       전국을 매달 보려면 1GB 파일 대신 이쪽이 가볍다(키가 생기면 붙일 다음 단계).
    갱신: 포털 명시 "매일 갱신, 2일 전 기준 현행화". 실측 최종수정시점 최대 2026-09-17(조회일 2026-09-19).

매칭 (match_brands) — 오탐을 먼저 막는다
    사업장명은 '메가엠지씨커피 강남역점', '교촌(정자2호점)', '가락시장역점 빽다방', '(주)공차코리아(잠실점)'처럼
    제각각이다. brand_search.normalize 로 정규화하고 지점 표기(…점·…역점·…N호점)를 떼어 비교하되,
    부분 문자열 일치는 쓰지 않는다('스무디킹영통메가박스점'에는 '메가'가, '올바른치킨'에는 '바른치킨'이 들어 있다).
      1. exact   지점 표기를 뗀 나머지가 브랜드 키와 같다 ('투다리(삼성점)', '가락시장역점 빽다방')
      2. prefix  4자 이상(한글 2자 이상) 강한 키가 이름 토막의 맨 앞에 온다. 긴 키 우선 — '본죽&비빔밥 …'은
                 '본죽'이 아니라 '본죽&비빔밥'.
      3. prefix_weak  '공차'·'한솥'·'투다리' 같은 짧은 키는 뒤가 비었거나 지점 표기여야 하고, 자기 업종어
                 (치킨 브랜드의 '치킨' 등)를 뺀 나머지에 다른 업종어가 있으면 다른 가게로 본다
                 ('한솥집'·'한솥밥 강남점'·'김가네 순대국'·'처갓집야식' 거부). 둘째 토막 이후라면 앞 토막이
                 '…점'·주소 지명·법인 표기와 맞닿은 운영법인 이름이거나, 온전한 낱말 뒤에 지점명이 따라와야 한다
                 ('라미나 더카페'·'백순대 본가새맛 인계점' 거부). 두 글자 키('달콤'·'공차')에 붙여 쓴 지점명은
                 주소 지명·역으로 시작해야 한다('공차대치역점' 인정 / '달콤바삭 마트본점' 거부).
      4. glued   토막 안쪽에 붙은 브랜드는 앞말이 '…점'·'한국'·입점처(홈플러스·백화점 …)·주소의 지명·자기
                 업종어이거나 법인 표기와 맞닿은 운영법인 이름일 때만 ('호매실굽네치킨', '(주)신맥맥도날드개포점'
                 인정 / '올바른치킨'→바른치킨, '신교동짬뽕'→교동짬뽕 거부).
      공백을 지우는 정규화 탓에 '샐러디 아주대점'이 '샐러디아…'로 읽히지 않도록, 공시명에 없던 자리의 구분자를
      건너 맞은 키는 끝도 낱말 경계여야 한다.
    키: 공시명(괄호 밖) · 괄호 안 병기 · '/' 분리 · '&'→앤/엔/앤드 · 끝 숫자 제거(역전할머니맥주1982) ·
        brand_search.ALIASES(3자 이상만 — '베라'·'파바'는 말로 부르는 약칭이지 간판 표기가 아니다) ·
        STORE_ALIASES(실측 간판 표기) · 핵심어(끝 업종어를 뗀 머리: 텐퍼센트스페셜티커피→텐퍼센트; 3자 이상,
        일반어 제외, 첫 토막·지점명 필수). 같은 키가 여러 브랜드를 가리키면 공시명 그대로인 쪽(점포 수 1위)만,
        별칭·핵심어가 모호하면 버린다.
    실측 (서울강남·송파·경기수원·부산해운대 × 3업종 194,048건, 브랜드 1,521개, 2026-09-19):
      · 매칭 15,128건(7.8%), 사업장명 11,686종, 브랜드 1,214개. 규칙별 층화 무작위 감사 105건 중 명백한 오탐 0,
        애매 4('쭈대가 제육대가' 같은 복합 간판). 남은 오탐 유형: 짧은 키 뒤에 글자가 붙어 다른 낱말이 되는 경우
        ('그리너리 강남점'→그리너) — 이것까지 막으면 '투다리선릉점' 같은 정상 표기를 더 많이 잃는다(실측 44:8).
      · 브랜드 키를 품고도 안 잡힌 사업장명은 173종(매칭의 1.5%), 표본 40건 중 명백한 누락 10건.
      · 상위 50개 브랜드 전부 1곳 이상(47개는 10곳 이상) 잡혔고, 활성 점포 합계는 공시 가맹점 수의 5.3%
        (네 지역 인구 비중 약 5.4%). 적게 잡힌 브랜드는 PC방 입점형(PC토랑·밥스토랑)이거나 해당 지역에 점포가
        거의 없는 경우(땅스부대찌개 3건 — 전부 매칭)였다.
      · tests/test_localdata.py 라벨 표본 118건(정답 75·어려운 음성 43): 정밀도 1.000, 재현율 0.973.

월간 흐름·신호
    monthly_brand_flows: 브랜드×월 개업·폐업·활성 점포. 같은 주소에서 같은 브랜드가 폐업 직후(-30~+90일)
    다시 인허가를 받은 경우는 폐점이 아니라 명의 이전으로 보고 n_transfer 로 따로 센다(실측 최근 2년 폐업의 약 3%).
    closure_signal: 최근 3개월 폐업률을 **전년 같은 3개월**과 비교한다(연말 폐업이 몰리는 계절성 때문 —
    실측 12월 폐업이 평월의 1.5배). 두 기간 폐업률 비교의 조건부 이항 정확검정으로 악화/개선/유지,
    점포 기반이 작거나 비교 기간이 없으면 판단보류.

⚠️ 지역 표본이면 신호도 그 지역의 신호다. 전국 신호는 전국 파일(①, 약 930MB) 또는 API(②)가 필요하다.

사용 (배치 → 화면)
    rec = load_records([...CSV 경로...])                 # 또는 API JSON
    brands = pd.read_csv("outputs/scores_latest.csv")
    matched = match_brands(rec, brands)
    flows = monthly_brand_flows(matched, months=24)
    signal = closure_signal(flows)                        # brand_id 로 점수표에 붙여 배지로 보여 준다
    CLI: python -m src.localdata --csv a.csv b.csv --out outputs/localdata_signal.csv
"""
from __future__ import annotations

import argparse
import codecs
import html
import json
import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from src.brand_search import ALIASES, normalize
from src.common import ROOT, get_logger

log = get_logger("localdata")

SERVICES: dict[str, str] = {
    "general_restaurants": "일반음식점",
    "rest_cafes": "휴게음식점",
    "bakeries": "제과점영업",
}
FILE_BASE = "https://file.localdata.go.kr/file"
# WAF 가 브라우저가 아닌 요청을 막는다(실측). 위장 목적이 아니라 공개 파일을 받기 위한 최소 헤더다.
_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

CANONICAL = ["service", "org_code", "mng_no", "name", "status_code", "status", "detail_status",
             "opened_on", "closed_on", "biz_type", "road_address", "jibun_address", "x", "y",
             "updated_at", "data_updated_at", "update_type", "source"]
# 원천별 열 이름 → 표준 열. 2026 파일(한글 헤더)·OpenAPI(영문 코드)는 실측. '영업상태구분코드'·'도로명전체주소'·
# '소재지전체주소'·'좌표정보(x)'·'데이터갱신일자'는 2026 이전 LOCALDATA CSV 표기(과거 스냅샷 호환용 — 실측 대상 아님).
_SOURCE_COLS: dict[str, tuple[str, ...]] = {
    "org_code": ("개방자치단체코드", "OPN_ATMY_GRP_CD"),
    "mng_no": ("관리번호", "MNG_NO"),
    "name": ("사업장명", "BPLC_NM"),
    "opened_on": ("인허가일자", "LCPMT_YMD"),
    "closed_on": ("폐업일자", "CLSBIZ_YMD"),
    "status_code": ("영업상태코드", "영업상태구분코드", "SALS_STTS_CD"),
    "status": ("영업상태명", "SALS_STTS_NM"),
    "detail_status": ("상세영업상태명", "DTL_SALS_STTS_NM"),
    "biz_type": ("업태구분명", "BZSTAT_SE_NM"),
    "road_address": ("도로명주소", "도로명전체주소", "ROAD_NM_ADDR"),
    "jibun_address": ("지번주소", "소재지전체주소", "LOTNO_ADDR"),
    "x": ("좌표정보(X)", "좌표정보(x)", "CRD_INFO_X"),
    "y": ("좌표정보(Y)", "좌표정보(y)", "CRD_INFO_Y"),
    "updated_at": ("최종수정시점", "LAST_MDFCN_PNT"),
    "data_updated_at": ("데이터갱신시점", "데이터갱신일자", "DAT_UPDT_PNT"),
    "update_type": ("데이터갱신구분", "DAT_UPDT_SE"),
    "service": ("개방서비스명",),
}
_WANTED = frozenset(c for names in _SOURCE_COLS.values() for c in names)
CLOSED = "03"


# ---------------------------------------------------------------------------
# 1) 확보·적재
# ---------------------------------------------------------------------------
def download_csv(service: str, dest_dir: str | Path, org_code: str | None = None, *,
                 max_bytes: int = 50_000_000, timeout: float = 120) -> Path:
    """인허가 CSV 한 건을 받는다(키 불필요). 전국 파일은 수백 MB 라 기본 한도(50MB)로 막는다.

    org_code: 시군구 '3220000'(서울강남구)·시도 '6110000_ALL'. None 이면 전국.
    한도를 넘으면 쓰기 전에(Content-Length) 또는 받는 도중에 중단하고 조각 파일을 지운다.
    """
    if service not in SERVICES:
        raise ValueError(f"service 는 {sorted(SERVICES)} 중 하나: {service!r}")
    landing = f"{FILE_BASE}/{service}/info"
    headers = {"User-Agent": _BROWSER_UA, "Referer": landing}
    check = requests.get(f"{FILE_BASE}/validate/download-count", headers=headers, timeout=timeout)
    if check.status_code == 429:
        raise RuntimeError("다운로드 횟수 제한(429) — 잠시 후 다시 시도하세요.")
    params = {"orgCode": org_code} if org_code else None
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{service}_{org_code or 'all'}.csv"
    part = out.with_suffix(".part")
    with requests.get(f"{FILE_BASE}/download/{service}/info", params=params, headers=headers,
                      stream=True, timeout=timeout, allow_redirects=False) as r:
        ctype = r.headers.get("Content-Type", "")
        if r.status_code != 200 or "csv" not in ctype:
            raise RuntimeError(f"다운로드 거부(HTTP {r.status_code}, {ctype or '형식 없음'}) — "
                               "WAF 차단이거나 없는 orgCode 입니다.")
        size = int(r.headers.get("Content-Length") or 0)
        if size > max_bytes:
            raise RuntimeError(f"파일이 한도보다 큽니다: {size / 1e6:.1f}MB > {max_bytes / 1e6:.0f}MB — "
                               "시군구 orgCode 로 나눠 받거나 max_bytes 를 명시적으로 올리세요.")
        got = 0
        try:
            with open(part, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    got += len(chunk)
                    if got > max_bytes:
                        raise RuntimeError(f"받는 도중 한도 초과({max_bytes / 1e6:.0f}MB) — 중단했습니다.")
                    fh.write(chunk)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    part.replace(out)
    log.info("인허가 파일 저장: %s (%s, %.1fMB)", out.name, SERVICES[service], got / 1e6)
    return out


def _sniff_encoding(path: Path) -> str:
    """헤더가 UTF-8 이라 해도 실제로는 CP949 인 파일이 온다(실측). 앞 1MB 를 UTF-8 로 풀어 보고 판별한다."""
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
    if head.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    try:
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)   # 끝에서 잘린 글자는 봐준다
        return "utf-8"
    except UnicodeDecodeError:
        return "cp949"


def _json_items(obj: object) -> list[dict]:
    """API 응답 봉투({"response":{"body":{"items":{"item":[…]}}}})·items 목록·레코드 목록을 모두 받는다."""
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if not isinstance(obj, dict):
        return []
    body = (obj.get("response") or {}).get("body") if isinstance(obj.get("response"), dict) else obj
    items = (body or {}).get("items", body)
    if isinstance(items, dict):
        items = items.get("item", items)
    if isinstance(items, dict):
        items = [items]
    return [x for x in items or [] if isinstance(x, dict)] if isinstance(items, list) else []


def _to_date(s: pd.Series) -> pd.Series:
    """'2026-09-17' · '20260917' · '2026.09.17' → Timestamp. 빈칸·이상값은 NaT."""
    t = s.astype(str).str.strip().str.replace(r"[.\-/\s]", "", regex=True).str[:8]
    return pd.to_datetime(t.where(t.str.fullmatch(r"\d{8}")), format="%Y%m%d", errors="coerce")


def _to_ts(s: pd.Series) -> pd.Series:
    """파일 '2026-09-18 22:18:00' · API '20260918221800' → Timestamp."""
    t = s.astype(str).str.strip().str.replace(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$",
                                              r"\1-\2-\3 \4:\5:\6", regex=True)
    return pd.to_datetime(t.where(t != ""), format="ISO8601", errors="coerce")


def _infer_service(path: Path) -> str:
    stem = path.stem
    for code, name in SERVICES.items():
        if code in stem or name in stem:
            return name
    return ""


def _canonicalize(raw: pd.DataFrame, source: str, service: str | None) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    for canon, names in _SOURCE_COLS.items():
        col = next((c for c in names if c in raw.columns), None)
        out[canon] = raw[col].astype(str).str.strip() if col else ""
    out["service"] = service or out["service"]
    out["opened_on"] = _to_date(out["opened_on"])
    out["closed_on"] = _to_date(out["closed_on"])
    for c in ("updated_at", "data_updated_at"):
        out[c] = _to_ts(out[c])
    code = out["status_code"]
    out["status_code"] = code.where(~code.str.fullmatch(r"\d"), "0" + code)
    out["source"] = source
    return out[CANONICAL]


def load_records(paths: str | Path | Iterable[str | Path], *, service: str | None = None) -> pd.DataFrame:
    """인허가 원천(CSV: 2026 파일 / JSON: OpenAPI 응답)을 표준 열로 읽는다.

    열: service, org_code, mng_no, name, status_code(01/03), status, detail_status, opened_on, closed_on,
        biz_type, road_address, jibun_address, x, y, updated_at, data_updated_at, update_type, source.
    service 를 주지 않으면 파일 이름(general_restaurants_… / 식품_일반음식점_…)에서 추정한다.
    같은 점포가 여러 파일(시군구·전국)에 겹치면 (업종, 자치단체, 관리번호) 기준으로 최신 1건만 남긴다.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for p in map(Path, paths):
        svc = service or _infer_service(p)
        if p.suffix.lower() == ".json":
            raw = pd.DataFrame(_json_items(json.loads(p.read_text(encoding="utf-8-sig"))), dtype=str)
        else:
            raw = pd.read_csv(p, encoding=_sniff_encoding(p), dtype=str, keep_default_na=False,
                              usecols=lambda c: c in _WANTED)
        frames.append(_canonicalize(raw.fillna(""), p.name, svc or None))
        log.info("인허가 적재: %s %d건 (%s)", p.name, len(raw), svc or "업종 미상")
    if not frames:
        return pd.DataFrame(columns=CANONICAL)
    df = pd.concat(frames, ignore_index=True)
    keyed = df["mng_no"] != ""
    newest = (df[keyed].sort_values("data_updated_at", kind="stable", na_position="first")
              .drop_duplicates(["service", "org_code", "mng_no"], keep="last"))
    return pd.concat([newest, df[~keyed]]).sort_index().reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2) 사업장명 → 브랜드
# ---------------------------------------------------------------------------
# brand_search.normalize 가 지우는 문자와 같은 집합 — 여기서는 '토막 경계'로도 쓴다.
# \x1f 는 법인 표기 자리를 표시하는 내부 구분자다(_scan 에서만 쓴다).
_SEP_CHAR = re.compile(r"[\s()（）\[\]{}·・,.\-_/&'\"\x1f]")
_CORP_MARK = "\x1f"
_CORP = re.compile(r"\(주\)|\(유\)|\(사\)|\(재\)|\(합\)|㈜|㈔|주식회사|유한회사|유한책임회사|합자회사|"
                   r"농업회사법인|영농조합법인|사단법인|재단법인")
_BRACKET = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}")
_BRANCH_END = re.compile(r"(점|\d호)$")
_BRANCH_SUFFIX = re.compile(r"(\d*호점|본점|직영점|지점|점|\d+호)$")
# '…점'으로 끝나도 지점이 아니라 업종인 말 ('교동반점'은 중국집 이름이다)
_SHOP_END = re.compile(r"(반점|주점|상점|매점|편의점|음식점|잡화점|전문점|판매점|대리점)$")
# 가게 이름 앞에 붙어도 다른 가게가 되지 않는 입점처 표기 ('수원홈플러스배스킨라빈스')
_MALL_END = re.compile(r"(홈플러스|이마트|롯데마트|롯데몰|백화점|아울렛|마트|몰|터미널|휴게소)$")

# 업종(공정위 중분류)별 '자기 말' — 브랜드 뒤에 붙어도 같은 가게로 본다 ('60계 치킨&맥주 잠실점').
CATEGORY_WORDS: dict[str, tuple[str, ...]] = {
    "치킨": ("치킨앤맥주", "치킨맥주", "치킨앤비어", "치킨호프", "숯불양념치킨", "양념치킨", "숯불치킨", "숯불", "참숯",
           "치킨", "통닭", "닭강정"),
    "커피": ("스페셜티커피", "커피", "카페", "까페", "coffee", "cafe", "디저트", "에스프레소"),
    "음료 (커피 외)": ("티", "카페", "까페", "커피", "쥬스", "주스", "tea"),
    "피자": ("피자", "파스타", "pizza"),
    "제과제빵": ("베이커리", "제과", "빵", "카페", "까페", "케이크", "케익", "도넛츠", "도너츠", "도나츠", "도넛"),
    "아이스크림/빙수": ("아이스크림", "빙수", "카페", "까페", "디저트"),
    "분식": ("분식", "떡볶이", "김밥", "튀김"),
    "패스트푸드": ("햄버거", "버거", "치킨", "피자", "앤", "엔"),
    "주점": ("호프", "맥주", "비어", "포차", "주점", "펍"),
    "일식": ("일식", "초밥", "스시", "돈까스", "돈가스", "라멘", "우동"),
    "중식": ("중식", "짬뽕", "짜장", "마라탕", "마라"),
    "서양식": ("파스타", "스테이크", "양식", "브런치"),
}
# 모든 브랜드에 허용: 본부 법인 표기 ('(주)공차코리아(잠실점)')
_COMMON_OWN: tuple[str, ...] = ("코리아",)
# 짧은 키 뒤에 오면 '다른 가게'라는 신호인 업종·상품어 (부분 문자열로 검사, 주소에 있는 말은 지명으로 보고 제외
# — 부산 해운대구 '우동'). 지명과 겹치기 쉬운 한 글자(면목·죽전·전주·회기)는 넣지 않는다.
FOOD_WORDS: tuple[str, ...] = (
    "치킨", "통닭", "닭강정", "피자", "커피", "카페", "까페", "cafe", "coffee", "베이커리", "제과", "케이크", "케익",
    "도넛", "디저트", "아이스크림", "빙수", "와플", "토스트", "햄버거", "버거", "샌드위치", "샐러드", "떡볶이",
    "분식", "김밥", "도시락", "국밥", "칼국수", "국수", "냉면", "우동", "라멘", "라면", "짬뽕", "짜장", "마라",
    "훠궈", "일식", "초밥", "스시", "돈까스", "돈가스", "횟집", "해물", "해장국", "곰탕", "설렁탕", "감자탕",
    "찌개", "찜닭", "닭갈비", "갈비", "삼겹", "고기", "한우", "정육", "곱창", "막창", "족발", "보쌈", "순대",
    "만두", "막걸리", "호프", "맥주", "주점", "포차", "이자카야", "라운지", "노래", "식당", "전문점", "뷔페",
    "파스타", "스테이크", "브런치", "비빔밥", "양꼬치", "훈제", "숯불", "바베큐", "반찬", "야식", "한식", "중식",
    "양식", "식육", "수산", "델리", "비어", "펍", "닭발", "장어", "생선", "쌀국수", "카레", "커리", "핫도그",
    "베이글", "마카롱", "젤라또", "요거트", "스무디", "쥬스", "주스", "밀크티", "버블티", "꼬치", "꼬지", "탕수육",
    "덮밥", "돈부리", "규동", "텐동", "오므라이스", "카츠", "함박",
)
_FOOD_HEADS: tuple[str, ...] = ("밥", "집", "빵", "찜")      # 나머지의 **맨 앞**에 올 때만 ('한솥밥', '한솥집')
_CORE_TAILS: tuple[str, ...] = (
    "스페셜티커피", "양념치킨", "숯불치킨", "커피", "카페", "치킨", "통닭", "떡볶이", "피자", "버거", "맥주",
    "호프", "도시락", "마라탕", "해장국", "족발", "베이커리", "강정",
)
_CORE_HEADS: tuple[str, ...] = ("카페", "까페", "커피", "cafe", "coffee")
# 업종어를 떼어 낸 머리가 이런 일반어면 핵심어로 쓰지 않는다 ('퍼스트커피'→'퍼스트', '동네커피'→'동네')
_GENERIC_CORES: frozenset[str] = frozenset({
    "퍼스트", "뉴욕", "스타", "골드", "블루", "그린", "해피", "베스트", "프리미엄", "스페셜", "로얄", "마이", "넘버원",
    "오리지널", "클래식", "모던", "시티", "하우스", "동네", "감성", "명가", "본가", "원조", "옛날", "전통", "대박",
    "행복", "사랑", "우리", "한국", "서울", "부산", "더", "뉴", "빅", "킹", "탑",
})
# 간판 표기가 공시명과 다른 경우 (2026-09 실측 사업장명). 값은 공시 브랜드명의 괄호 밖 부분.
STORE_ALIASES: dict[str, str] = {
    "메가mgc커피": "메가엠지씨커피",
    "이디야": "이디야커피",
    "교촌": "교촌치킨",
    "한솥도시락": "한솥",
    "동대문엽기떡볶이": "불닭발땡초동대문엽기떡볶이",
    "엽기떡볶이": "불닭발땡초동대문엽기떡볶이",
    "불닭발동대문엽기떡볶이": "불닭발땡초동대문엽기떡볶이",
    "뚜레주르": "뚜레쥬르",
    "한국피자헛": "피자헛",
    "전주현대옥": "현대옥",
    "텐퍼센트": "텐퍼센트스페셜티커피",
    "텐퍼센트커피": "텐퍼센트스페셜티커피",
    "유가네닭갈비": "유가네",
    "크라운호프": "크라운호프보리장인",
    "명랑핫도그": "명랑시대쌀핫도그",
    "지코바": "지코바양념치킨",
    "또봉이": "또봉이통닭",
    "봉구스": "봉구스밥버거",
    "두마리찜닭두찜": "두찜",
    "달콤커피": "달.콤",
    "멕시칸치킨": "맥시칸치킨",            # '멕시칸'만으로는 멕시코 음식점과 섞인다
    "청담동마녀김밥": "마녀김밥",
}
_KIND_RANK = {"name": 0, "variant": 1, "alias": 2, "core": 3}


@dataclass(frozen=True)
class _Key:
    text: str                     # 정규화된 키
    brand_id: str
    kind: str                     # name | variant | alias | core
    strong: bool                  # 앞부분만 맞아도 인정하는가
    own: tuple[str, ...]          # 뒤에 붙어도 같은 가게로 보는 자기 업종어
    seps: frozenset[int] = frozenset()   # 공시명에서 구분자(공백·& 등)가 있던 자리 (키 안의 글자 위치)


@dataclass
class _Index:
    by_prefix: dict[str, list[_Key]]      # 앞 2글자 → 키(긴 것부터)
    exact: dict[str, _Key]                # 지점 표기를 뗀 이름 전체 일치용 (핵심어 제외)


# 약어 점('B.H.C')·아포스트로피("Han's")는 낱말을 가르지 않는다 — 구분자 위치 비교에서 뺀다.
_SOFT_SEP = ".'"


def _prep(s: object, corp: str = " ") -> str:
    """NFKC·소문자·HTML 엔티티 해제, 법인 표기는 corp 로 바꾼다(기본은 공백)."""
    t = unicodedata.normalize("NFKC", html.unescape(str(s or ""))).lower()
    return _CORP.sub(corp, t)


def _han(s: str) -> int:
    return len(re.sub(r"[^가-힣]", "", s))


def _norm_seps(s: str) -> tuple[str, frozenset[int]]:
    """normalize(s) 와 같은 문자열 + 원문에서 구분자가 있던 자리(정규화 뒤 글자 위치)."""
    out: list[str] = []
    seps: set[int] = set()
    pending = False
    for ch in unicodedata.normalize("NFKC", s).lower():
        if _SEP_CHAR.match(ch):
            pending = pending or (ch not in _SOFT_SEP and bool(out))
            continue
        if pending:
            seps.add(len(out))
            pending = False
        out.append(ch)
    return "".join(out), frozenset(seps)


def _brand_keys(name: str) -> tuple[list[tuple[str, str, frozenset[int]]], list[str]]:
    """공시 브랜드명 1개 → [(키, 종류, 구분자 자리)], 핵심어를 만들며 떼어 낸 업종어."""
    outer = re.sub(r"\([^()]*\)|（[^（）]*）", " ", name)
    inner = re.findall(r"\(([^()]*)\)|（([^（）]*)）", name)
    keys: list[tuple[str, str, frozenset[int]]] = []
    for part in outer.split("/"):
        p, sp = _norm_seps(part)
        if not p:
            continue
        keys.append((p, "name", sp))
        if "&" in part:
            for conj in ("앤", "엔", "앤드", "and"):
                v, vs = _norm_seps(part.replace("&", conj))
                keys.append((v, "variant", vs))
        m = re.fullmatch(r"(.*?[^0-9.])[0-9.]+", p)      # '역전할머니맥주1982' → '역전할머니맥주'
        if m and _han(m.group(1)) >= 4:
            keys.append((m.group(1), "variant", frozenset(x for x in sp if x < len(m.group(1)))))
    for a, b in inner:
        v, vs = _norm_seps(a or b)
        if len(v) >= 3:
            keys.append((v, "variant", vs))
    tails: list[str] = []
    for base, _kind, bseps in list(keys):
        core = base
        for _ in range(2):
            tail = next((t for t in _CORE_TAILS if core.endswith(t) and len(core) - len(t) >= 2), None)
            if tail is None:
                break
            tails.append(tail)
            core = core[: -len(tail)]
        shift = next((len(h) for h in _CORE_HEADS if core.startswith(h) and len(core) - len(h) >= 3), 0)
        core = core[shift:]
        if core != base and len(core) >= 3 and core not in _GENERIC_CORES:
            keys.append((core, "core", frozenset(x - shift for x in bseps if shift < x < shift + len(core))))
    return keys, tails


def _build_index(brands: pd.DataFrame) -> _Index:
    """브랜드 표 → 키 색인. 같은 키가 여러 브랜드를 가리키면 공시명 그대로인 쪽(점포 수 1위)만 남긴다."""
    rows = brands.copy()
    rows["_n"] = pd.to_numeric(rows["n_stores"], errors="coerce").fillna(0) if "n_stores" in rows else 0
    rows = rows.sort_values("_n", ascending=False, kind="stable")
    hits: dict[str, list[tuple[str, str, int, frozenset[int]]]] = {}
    own: dict[str, tuple[str, ...]] = {}
    primary: dict[str, str] = {}
    for rank, r in enumerate(rows.itertuples(index=False)):
        bid, name = str(r.brand_id), html.unescape(str(r.brand_name))
        keys, tails = _brand_keys(name)
        if not keys:
            continue
        primary[bid] = keys[0][0]
        own[bid] = tuple(dict.fromkeys((*CATEGORY_WORDS.get(str(getattr(r, "industry_mid", "") or ""), ()),
                                        *tails, *_COMMON_OWN)))
        for text, kind, seps in keys:
            if len(text) >= 2:
                hits.setdefault(text, []).append((bid, kind, rank, seps))
    for alias, token in ALIASES.items():
        tok = normalize(token)
        targets = [b for b, p in primary.items() if tok and tok in p]
        if len(targets) == 1 and len(normalize(alias)) >= 3:
            hits.setdefault(normalize(alias), []).append((targets[0], "alias", len(rows), frozenset()))
    for alias, bname in STORE_ALIASES.items():
        targets = [b for b, p in primary.items() if p == normalize(bname)]
        if targets:
            # 실측 간판 표기는 띄어쓰기가 제각각이라('두마리찜닭 두찜…'·'두마리찜닭두찜…') 안쪽 어디서 띄어도 인정한다
            anywhere = frozenset(range(1, len(normalize(alias))))
            hits.setdefault(normalize(alias), []).append((targets[0], "alias", len(rows), anywhere))
    by_prefix: dict[str, list[_Key]] = {}
    exact: dict[str, _Key] = {}
    for text, hs in hits.items():
        named = [h for h in hs if h[1] == "name"]
        if named:
            bid, kind, _, seps = min(named, key=lambda h: h[2])
        elif len({h[0] for h in hs}) > 1:
            continue                                  # 별칭·핵심어가 여러 브랜드로 갈린다 → 모호, 버린다
        else:
            bid, kind, _, seps = min(hs, key=lambda h: (_KIND_RANK[h[1]], h[2]))
        strong = kind != "core" and len(text) >= 4 and (_han(text) >= 2 or len(text) >= 7)
        key = _Key(text, bid, kind, strong, own.get(bid, ()), seps)
        by_prefix.setdefault(text[:2], []).append(key)
        if kind != "core":
            exact[text] = key
    for ks in by_prefix.values():
        ks.sort(key=lambda k: -len(k.text))
    return _Index(by_prefix, exact)


def _is_branch_token(tok: str) -> bool:
    return bool(_BRANCH_END.search(tok)) and not _SHOP_END.search(tok)


def strip_branch(name: str) -> str:
    """사업장명에서 지점 표기를 떼어 브랜드 부분만 남긴다(보수적 — 확실한 것만 뗀다).

    '메가엠지씨커피 강남역점'→'메가엠지씨커피', '교촌치킨 매탄1호점'→'교촌치킨', '투다리(삼성점)'→'투다리',
    '가락시장역점 빽다방'→'빽다방', '(주)교촌 역삼점'→'교촌'. 붙여 쓴 '교촌치킨매탄1호점'은 어디서 끊을지
    브랜드 목록 없이 알 수 없으므로 그대로 둔다(그 경우는 match_brands 의 앞부분 일치가 처리한다).
    """
    t = unicodedata.normalize("NFKC", html.unescape(str(name or "")))
    toks = _BRACKET.sub(" ", _CORP.sub(" ", t)).split()
    while len(toks) >= 2 and _is_branch_token(toks[0]):
        toks = toks[1:]
    while len(toks) >= 2 and _is_branch_token(toks[-1]):
        toks = toks[:-1]
    return " ".join(toks)


def _scan(raw: str) -> tuple[str, list[int], list[tuple[int, int]], str, set[int]]:
    """(정규화 문자열 S, S→원문 위치, 토막[(시작,끝)], 전처리된 원문, 법인 표기와 맞닿은 토막 번호).

    S 는 brand_search.normalize 와 같다. 법인 표기('(주)'·'주식회사')와 맞닿은 토막은 가맹점 운영법인 이름이거나
    그 바로 뒤의 간판이다 — '(주)공영식품 기소야삼성점'에서 '기소야…'를 첫 토막처럼 대하려고 표시해 둔다.
    """
    t = _prep(raw, corp=f" {_CORP_MARK} ")
    chars: list[str] = []
    s2raw: list[int] = []
    segs: list[tuple[int, int]] = []
    corp_adj: set[int] = set()
    start: int | None = None
    after_corp = False
    for i, ch in enumerate(t):
        if _SEP_CHAR.match(ch):
            if start is not None:
                segs.append((start, len(chars)))
                start = None
            if ch == _CORP_MARK:
                if segs:
                    corp_adj.add(len(segs) - 1)
                after_corp = True
            continue
        if start is None:
            start = len(chars)
            if after_corp:
                corp_adj.add(len(segs))
                after_corp = False
        chars.append(ch)
        s2raw.append(i)
    if start is not None:
        segs.append((start, len(chars)))
    return "".join(chars), s2raw, segs, t, corp_adj


def _strip_own(r: str, own: tuple[str, ...]) -> str:
    """나머지의 앞에서 자기 업종어를 반복해 떼어 낸다 ('치킨맥주서울잠실점' → '서울잠실점')."""
    ordered = sorted(own, key=len, reverse=True)
    while r:
        w = next((w for w in ordered if w and r.startswith(w)), None)
        if w is None:
            break
        r = r[len(w):]
    return r


def _foreign_food(r: str, own: tuple[str, ...], addr: str) -> bool:
    for w in FOOD_WORDS:
        if w in r and not any(w in o for o in own) and w not in addr:
            return True
    return bool(r) and r[0] in _FOOD_HEADS and not any(o.startswith(r[0]) for o in own)


def _rest_ok(key: _Key, rest_raw: str, boundary: bool, addr: str, *, need_branch: bool = False) -> bool:
    """짧은 키·핵심어 뒤의 나머지가 '지점 표기'로 볼 만한가. 괄호를 뺀 것과 넣은 것 중 하나라도 되면 인정."""
    for r in dict.fromkeys((normalize(_BRACKET.sub(" ", rest_raw)), normalize(rest_raw))):
        r_own = _strip_own(r, key.own)
        stripped = r_own != r
        branch = need_branch
        if key.kind == "core":
            if not boundary and not stripped:         # 핵심어는 경계나 자기 업종어가 바로 이어져야 한다
                continue
            branch = True                             # 핵심어만 덩그러니('컴포즈')는 인정하지 않는다
        if len(key.text) <= 2 and not boundary:
            # 두 글자 키('달콤'·'공차')에 붙여 쓴 말은 자기 업종어 뒤의 지점이거나, 주소 지명·역으로 시작하는
            # 지점이어야 한다 — '달콤바삭 마트본점'·'달콤카페' 거부, '공차대치역점'·'본죽반송점' 인정
            head = _BRANCH_SUFFIX.sub("", r_own)
            if not r_own or not (stripped or (len(head) >= 2 and head[:2] in addr) or head.endswith("역")):
                continue
        if branch and not r_own:
            continue
        # 지점 표기는 '…점'·'…N호'. '상점'·'전문점'은 지점이 아니고, 점 앞에 이름이 없으면('…미술관점'에서
        # '점'만 남는 경우) 자기 업종어를 뗀 뒤가 아닌 한 거부한다
        if r_own and not (_BRANCH_END.search(r_own) and not _SHOP_END.search(r_own)
                          and (len(r_own) >= 2 or stripped)):
            continue
        if not _foreign_food(r_own, key.own, addr):
            return True
    return False


def _neutral_glue(glued: str, key: _Key, addr: str) -> bool:
    """토막 안에서 브랜드 앞에 붙은 말이 '다른 가게 이름'이 아니라고 볼 수 있는가."""
    if (glued.endswith("점") and not _SHOP_END.search(glued)) or glued == "한국" or _MALL_END.search(glued):
        return True
    if _han(glued) >= 2 and not re.search(r"\d", glued) and glued in addr:
        return True
    return bool(glued) and _strip_own(glued, key.own) == ""


def _match_one(name: str, index: _Index, address: str = "") -> tuple[_Key, str] | None:
    s, s2raw, segs, t, corp_adj = _scan(name)
    if not s:
        return None
    key = index.exact.get(normalize(_prep(strip_branch(name))))
    if key is not None:
        return key, "exact"
    addr = normalize(address)

    def _after(key: _Key, start: int) -> tuple[str, bool] | None:
        """키 뒤의 원문과 경계 여부. 정규화가 공백을 지우므로 '샐러디 아주대점'의 '샐러디아…'가 브랜드
        '샐러디아'로 읽힐 수 있다. 그래서 공시명에 없던 자리에서 구분자를 건너 맞은 키는 끝도 경계여야 한다
        ('짝태&노가리연무점'·'떡볶이참잘하는집 떡참…'처럼 공시명과 같은 자리의 구분자는 괜찮다). 어기면 None."""
        end = start + len(key.text)
        raw_end = s2raw[end - 1] + 1
        boundary = raw_end >= len(t) or bool(_SEP_CHAR.match(t[raw_end]))
        if not boundary:
            # 영문 키 뒤에 영문이 더 붙으면 더 긴 낱말이다 ('JuicyBros' ≠ juicy)
            if key.text[-1].isascii() and key.text[-1].isalpha() and t[raw_end].isascii() and t[raw_end].isalpha():
                return None
            for i in range(start + 1, end):
                gap = t[s2raw[i - 1] + 1:s2raw[i]]
                if gap.strip(_SOFT_SEP) and (i - start) not in key.seps:
                    return None
        return t[raw_end:], boundary

    for si, (a, _b) in enumerate(segs):
        prev = s[segs[si - 1][0]:segs[si - 1][1]] if si > 0 else ""
        for key in index.by_prefix.get(s[a:a + 2], ()):
            if not s.startswith(key.text, a) or (after := _after(key, a)) is None:
                continue
            if key.strong:
                return key, "prefix"
            if si > 0 and key.kind == "core":
                continue
            anchored = (si == 0 or si in corp_adj or (si - 1) in corp_adj or prev.endswith("점")
                        or (len(prev) >= 2 and prev in addr))
            # 앞 토막에 기대지 못하는 짧은 키는 온전한 낱말이어야 한다 ('백순대 본가새맛 인계점'의 '본가' 거부)
            if not anchored and not after[1]:
                continue
            if _rest_ok(key, *after, addr, need_branch=not anchored):
                return key, "prefix_weak"
    for si, (a, b) in enumerate(segs):
        for p in range(a + 1, b - 1):
            for key in index.by_prefix.get(s[p:p + 2], ()):
                if key.kind == "core" or not s.startswith(key.text, p):
                    continue
                glued = s[a:p]
                # 법인 표기에 맞닿은 토막의 앞말은 가맹점 운영법인 이름이다 ('(주)신맥맥도날드개포점') — 강한 키만
                corp_glue = key.strong and si in corp_adj
                if not (corp_glue or _neutral_glue(glued, key, addr)) or (after := _after(key, p)) is None:
                    continue
                reversed_ = glued.endswith("점") and not _SHOP_END.search(glued)
                if key.strong or (reversed_ and _rest_ok(key, *after, addr)):
                    return key, "glued"
    return None


def match_brands(records: pd.DataFrame, brands: pd.DataFrame, *, keep_unmatched: bool = False) -> pd.DataFrame:
    """인허가 레코드에 브랜드를 붙인다. 기본은 매칭된 행만 돌려준다.

    brands: brand_id·brand_name 필수, industry_mid(자기 업종어)·n_stores(동명 브랜드 우선순위) 있으면 사용.
    추가 열: brand_id, brand_name, match_key(맞은 키), match_rule(exact/prefix/prefix_weak/glued).
    """
    index = _build_index(brands)
    rec = records.copy()
    road = rec["road_address"] if "road_address" in rec else pd.Series("", index=rec.index)
    jibun = rec["jibun_address"] if "jibun_address" in rec else pd.Series("", index=rec.index)
    addr = road.fillna("").astype(str) + " " + jibun.fillna("").astype(str)
    pairs = pd.DataFrame({"name": rec["name"].fillna("").astype(str), "addr": addr})
    cache = {(n, a): _match_one(n, index, a) for n, a in pairs.drop_duplicates().itertuples(index=False)}
    hits = [cache[(n, a)] for n, a in pairs.itertuples(index=False)]
    rec["brand_id"] = [h[0].brand_id if h else None for h in hits]
    rec["match_key"] = [h[0].text if h else None for h in hits]
    rec["match_rule"] = [h[1] if h else None for h in hits]
    names = brands.drop_duplicates("brand_id").set_index("brand_id")["brand_name"]
    rec["brand_name"] = rec["brand_id"].map(names)
    ok = rec["brand_id"].notna()
    log.info("인허가→브랜드 매칭: %d / %d건 (%.1f%%), 브랜드 %d개, 규칙 %s", int(ok.sum()), len(rec),
             100 * ok.mean() if len(rec) else 0.0, rec.loc[ok, "brand_id"].nunique(),
             rec.loc[ok, "match_rule"].value_counts().to_dict())
    return rec if keep_unmatched else rec[ok].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3) 월간 흐름
# ---------------------------------------------------------------------------
def _as_period(v: object) -> pd.Period:
    return v if isinstance(v, pd.Period) else pd.Period(str(v), freq="M")


def _data_month_end(df: pd.DataFrame) -> pd.Period:
    """자료가 온전히 담은 마지막 달. 기준일이 월말이 아니면 그달은 덜 찼으므로 전달까지만 본다."""
    cands = [pd.to_datetime(df[c], errors="coerce").max() for c in
             ("opened_on", "closed_on", "updated_at", "data_updated_at") if c in df]
    as_of = max((c for c in cands if pd.notna(c)), default=pd.NaT)
    if pd.isna(as_of):
        raise ValueError("날짜가 있는 레코드가 없어 기준월을 정할 수 없습니다.")
    as_of = pd.Timestamp(as_of).normalize()
    month = as_of.to_period("M")
    return month if as_of == month.end_time.normalize() else month - 1


def _transfer_pairs(df: pd.DataFrame, days: int) -> tuple[set, set]:
    """같은 브랜드·같은 주소에서 폐업 직후(-30~+days일) 다시 인허가 → 명의 이전으로 본다(1:1 짝짓기)."""
    if days <= 0:
        return set(), set()
    road = df["road_address"] if "road_address" in df else pd.Series("", index=df.index)
    jibun = df["jibun_address"] if "jibun_address" in df else pd.Series("", index=df.index)
    addr = road.fillna("").astype(str).where(road.fillna("").astype(str) != "", jibun.fillna("").astype(str))
    base = pd.DataFrame({"brand_id": df["brand_id"], "addr": addr.map(normalize), "rid": df.index,
                         "opened_on": df["opened_on"], "closed_on": df["closed_on"]})
    base = base[base["addr"] != ""]
    closes = base[base["closed_on"].notna()][["brand_id", "addr", "rid", "closed_on"]]
    opens = base[base["opened_on"].notna()][["brand_id", "addr", "rid", "opened_on"]]
    pairs = closes.merge(opens, on=["brand_id", "addr"], suffixes=("_c", "_o"))
    pairs = pairs[pairs["rid_c"] != pairs["rid_o"]]
    gap = (pairs["opened_on"] - pairs["closed_on"]).dt.days
    pairs = pairs[(gap >= -30) & (gap <= days)].assign(_gap=gap.abs()).sort_values("_gap", kind="stable")
    used_c: set = set()
    used_o: set = set()
    for rc, ro in zip(pairs["rid_c"], pairs["rid_o"], strict=True):
        if rc not in used_c and ro not in used_o and rc not in used_o and ro not in used_c:
            used_c.add(rc)
            used_o.add(ro)
    return used_c, used_o


def monthly_brand_flows(matched: pd.DataFrame, months: int = 24, *, end_month: str | pd.Period | None = None,
                        transfer_days: int = 90) -> pd.DataFrame:
    """브랜드×월 개·폐점. 열: brand_id, brand_name, month(Period[M]), n_open, n_close, n_transfer, n_active_end.

    · n_open/n_close: 그달 인허가·폐업 건수에서 명의 이전(n_transfer)으로 본 짝은 뺀다.
    · n_active_end: 그달 말 영업 중 점포 수(명의 이전 여부와 무관한 실제 상태).
    · end_month 기본값: 자료가 온전히 담은 마지막 달(부분 월은 신호를 왜곡하므로 제외).
    · 폐업 상태인데 폐업일자가 없으면 시점을 알 수 없어 뺀다(실측 0건, 생기면 로그로 알린다).
    """
    df = matched[matched["brand_id"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["brand_id", "brand_name", "month", "n_open", "n_close", "n_transfer",
                                     "n_active_end"])
    for c in ("opened_on", "closed_on"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    status = df["status_code"].astype(str) if "status_code" in df else pd.Series("", index=df.index)
    undated = (status == CLOSED) & df["closed_on"].isna()
    if undated.any():
        log.warning("폐업일자 없는 폐업 %d건은 시점을 몰라 흐름에서 뺍니다.", int(undated.sum()))
        df = df[~undated]
    # 폐업일이 인허가일보다 앞서는 오기는 인허가일에 폐업한 것으로 본다(음수 기간 방지)
    df["closed_on"] = df["closed_on"].where(df["closed_on"].isna() | (df["closed_on"] >= df["opened_on"])
                                            | df["opened_on"].isna(), df["opened_on"])

    end = _as_period(end_month) if end_month is not None else _data_month_end(df)
    grid_months = pd.period_range(end=end, periods=months, freq="M")
    first = grid_months[0]
    t_close, t_open = _transfer_pairs(df, transfer_days)
    om = df["opened_on"].dt.to_period("M")
    cm = df["closed_on"].dt.to_period("M")
    is_t_open = df.index.isin(list(t_open))
    is_t_close = df.index.isin(list(t_close))

    brands = sorted(df["brand_id"].unique())
    # 창 이전 누적: 인허가일 미상은 '창 이전부터 있던 점포'로 본다
    before = (df[(om < first) | om.isna()].groupby("brand_id").size()
              .sub(df[cm < first].groupby("brand_id").size(), fill_value=0)
              .reindex(brands, fill_value=0))
    idx = pd.MultiIndex.from_product([brands, grid_months], names=["brand_id", "month"])

    def _count(mask: pd.Series | object, per: pd.Series) -> pd.Series:
        sub = df[mask & per.between(first, end)]
        return sub.groupby([sub["brand_id"], per[sub.index].rename("month")]).size().reindex(idx, fill_value=0)

    opened_all = _count(om.notna(), om)
    closed_all = _count(cm.notna(), cm)
    out = pd.DataFrame({
        "n_open": _count(om.notna() & ~is_t_open, om),
        "n_close": _count(cm.notna() & ~is_t_close, cm),
        "n_transfer": _count(cm.notna() & is_t_close, cm),
    })
    out["n_active_end"] = ((opened_all - closed_all).groupby(level="brand_id").cumsum()
                           + before.reindex(idx.get_level_values("brand_id")).to_numpy())
    out = out.astype(int).reset_index()
    names = df.drop_duplicates("brand_id").set_index("brand_id")["brand_name"] if "brand_name" in df else None
    out.insert(1, "brand_name", out["brand_id"].map(names) if names is not None else "")
    log.info("월간 흐름: 브랜드 %d개 × %d개월 (%s~%s), 명의 이전 짝 전체 이력 %d쌍·창 안 %d쌍 분리",
             len(brands), months, first, end, len(t_close), int(out["n_transfer"].sum()))
    return out


# ---------------------------------------------------------------------------
# 4) 조기경보 신호
# ---------------------------------------------------------------------------
def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X ≤ k), X~Binomial(n, p). scipy 없이 로그공간 합(n 은 두 기간 폐업 합 — 많아야 수천)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lc = math.lgamma(n + 1)
    return min(1.0, sum(math.exp(lc - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
                        for i in range(k + 1)))


def closure_signal(flows: pd.DataFrame, *, window: int = 3, min_base: int = 10, min_events: int = 3,
                   alpha: float = 0.05) -> pd.DataFrame:
    """최근 window 개월 폐업을 전년 같은 기간과 비교한 브랜드별 조기경보.

    열: brand_id, close_3m, open_3m, net_3m, close_rate_3m, trend(악화/개선/유지/판단보류)
        + brand_name, as_of_month, n_active_end, base_active, close_3m_prev_year, close_rate_prev_year,
          yoy_ratio, p_value, transfer_3m, reason.
    close_rate_3m = 최근 폐업 ÷ 창 직전 달 말 활성 점포(명의 이전은 폐업에서 뺐다).

    검정: 두 기간의 폐업률이 같다면, 두 기간 폐업 합 n 중 올해 몫은 Binomial(n, 올해 점포/(올해+전년 점포))을
    따른다(두 포아송 비율 비교의 조건부 정확검정). 전년 건수를 '참값'으로 두는 검정은 전년 0건이면 올해 3건만
    나와도 경보를 울린다(실측: 점포 18곳 브랜드, 전년 0건→올해 3건, 이 검정으로는 p=0.12) — 전년 건수의
    표본오차도 함께 반영하려고 이 방식을 쓴다.
    올해 쪽 단측 p<alpha 이고 폐업 min_events 건 이상이면 악화, 반대쪽이면 개선. 점포 기반이 min_base 미만이거나
    비교 기간(window+13개월)이 flows 에 없으면 판단보류 — 없는 근거로 경보를 만들지 않는다.
    """
    f = flows.copy()
    f["month"] = f["month"].map(_as_period)
    end = f["month"].max()
    cur = [end - i for i in range(window)]
    prev = [m - 12 for m in cur]
    base_m, prev_base_m = end - window, end - window - 12
    have = set(f["month"])
    if not all(m in have for m in cur):
        raise ValueError(f"flows 에 최근 {window}개월이 모두 있어야 합니다(monthly_brand_flows 의 months 확인).")

    def _piv(col: str) -> pd.DataFrame:
        return f.pivot_table(index="brand_id", columns="month", values=col, aggfunc="sum", fill_value=0)

    close, opened, trans, active = _piv("n_close"), _piv("n_open"), _piv("n_transfer"), _piv("n_active_end")
    res = pd.DataFrame(index=close.index)
    res["close_3m"] = close[cur].sum(axis=1).astype(int)
    res["open_3m"] = opened[cur].sum(axis=1).astype(int)
    res["net_3m"] = res["open_3m"] - res["close_3m"]
    res["base_active"] = active[base_m] if base_m in have else float("nan")
    res["close_rate_3m"] = res["close_3m"] / res["base_active"].where(res["base_active"] > 0)
    history = all(m in have for m in (*prev, prev_base_m))
    res["close_3m_prev_year"] = close[prev].sum(axis=1) if history else float("nan")
    prev_base = active[prev_base_m] if history else pd.Series(float("nan"), index=res.index)
    res["close_rate_prev_year"] = res["close_3m_prev_year"] / prev_base.where(prev_base > 0)
    res["yoy_ratio"] = res["close_rate_3m"] / res["close_rate_prev_year"].where(res["close_rate_prev_year"] > 0)
    res["transfer_3m"] = trans[cur].sum(axis=1).astype(int)
    res["n_active_end"] = active[end].astype(int)

    trends, pvals, reasons = [], [], []
    for bid, r in res.iterrows():
        pb = prev_base.get(bid, float("nan"))
        if not history:
            trends.append("판단보류")
            pvals.append(float("nan"))
            reasons.append(f"전년 동기 비교에 {window + 13}개월 흐름이 필요합니다.")
            continue
        if not (r["base_active"] >= min_base and pb >= min_base):
            trends.append("판단보류")
            pvals.append(float("nan"))
            reasons.append(f"관측 점포 {min_base}곳 미만(현재 {r['base_active']:.0f}·전년 {pb:.0f}).")
            continue
        k, k_prev = int(r["close_3m"]), int(r["close_3m_prev_year"])
        share = r["base_active"] / (r["base_active"] + pb)
        p_up = 1.0 - _binom_cdf(k - 1, k + k_prev, share)
        p_down = _binom_cdf(k, k + k_prev, share)
        expected = r["close_rate_prev_year"] * r["base_active"]
        head = f"폐업 {k}건(전년 동기 {k_prev}건, 점포 {r['base_active']:.0f}·{pb:.0f}곳 기준 기대 {expected:.1f}건)"
        if k >= min_events and p_up < alpha and r["close_rate_3m"] > r["close_rate_prev_year"]:
            trends.append("악화")
            pvals.append(p_up)
            reasons.append(f"{head} — 전년보다 유의하게 많음.")
        elif k_prev >= min_events and p_down < alpha and r["close_rate_3m"] < r["close_rate_prev_year"]:
            trends.append("개선")
            pvals.append(p_down)
            reasons.append(f"{head} — 전년보다 유의하게 적음.")
        else:
            trends.append("유지")
            pvals.append(min(p_up, p_down, 1.0))
            reasons.append(f"{head} — 통계적으로 다르지 않음.")
    res["trend"], res["p_value"], res["reason"] = trends, pvals, reasons
    res["as_of_month"] = end
    names = f.drop_duplicates("brand_id").set_index("brand_id")["brand_name"] if "brand_name" in f else None
    res["brand_name"] = res.index.map(names) if names is not None else ""
    res = res.reset_index()
    cols = ["brand_id", "close_3m", "open_3m", "net_3m", "close_rate_3m", "trend", "brand_name", "as_of_month",
            "n_active_end", "base_active", "close_3m_prev_year", "close_rate_prev_year", "yoy_ratio", "p_value",
            "transfer_3m", "reason"]
    log.info("폐점 신호 (%s 기준, 최근 %d개월): %s", end, window, res["trend"].value_counts().to_dict())
    return res[cols]


# ---------------------------------------------------------------------------
# CLI — 배치에서 신호표를 만들어 두고 화면은 그 파일을 읽는다(화면에서 내려받지 않는다)
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="인허가 CSV/JSON → 브랜드별 폐점 신호")
    ap.add_argument("--csv", nargs="+", required=True, help="load_records 가 읽을 파일들")
    ap.add_argument("--brands", default=str(ROOT / "outputs" / "scores_latest.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "localdata_signal.csv"))
    ap.add_argument("--flows-out", default=None, help="월간 흐름도 저장하려면 경로")
    ap.add_argument("--months", type=int, default=24)
    args = ap.parse_args(argv)
    matched = match_brands(load_records(args.csv), pd.read_csv(args.brands))
    flows = monthly_brand_flows(matched, months=args.months)
    if args.flows_out:
        flows.assign(month=flows["month"].astype(str)).to_csv(args.flows_out, index=False, encoding="utf-8-sig")
    sig = closure_signal(flows)
    sig.assign(as_of_month=sig["as_of_month"].astype(str)).to_csv(args.out, index=False, encoding="utf-8-sig")
    log.info("저장: %s (%d개 브랜드)", args.out, len(sig))


if __name__ == "__main__":
    main()
