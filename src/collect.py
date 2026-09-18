"""M1 수집: 공정위 가맹정보 오픈API → data/raw/ 원본 스냅샷.

데이터 소스 (조사로 확정, 2026-07-30):
- brand_frcs_stats  : 15110241 FftcBrandFrcsStatsService/getBrandFrcsStats
                      브랜드×연도 가맹점수·신규등록·계약종료/해지·명의변경·평균매출 (패널 본체)
- brand_overview    : 15109828 FftcBrandBrandStatsService/getBrandBrandStats
                      가맹사업개시일·임원수·직원수 (라벨 차원 밖 보조 피처)
- industry_openclose: 15157660 FftcIndutyFrcsCntOpclStatsService/getIndutyFrcsCntOpclStats
                      업종×연도 개폐점률 (업종 기준선 전용)

커버리지: yr 2017~2025 (yr=정보공개서 기준연도, 수치는 통상 직전 회계연도 실적).
스냅샷 원칙: 원본 JSON을 data/raw/에 그대로 보존 (재현성·정직성), 존재 시 재호출 생략.
            단, 개인 성명 필드(PERSONAL_FIELDS)만은 저장 전에 버린다 (개인정보 최소수집).
"""
from __future__ import annotations

import gzip
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from src.common import get_logger, load_config

log = get_logger("collect")

SERVICES: dict[str, dict] = {
    "brand_frcs_stats": {
        "url": "https://apis.data.go.kr/1130000/FftcBrandFrcsStatsService/getBrandFrcsStats",
        "year_param": "yr",
        "page_size": 3000,
    },
    "brand_overview": {
        "url": "https://apis.data.go.kr/1130000/FftcBrandBrandStatsService/getBrandBrandStats",
        "year_param": "yr",
        "page_size": 3000,
    },
    "industry_openclose": {
        "url": "https://apis.data.go.kr/1130000/FftcIndutyFrcsCntOpclStatsService/getIndutyFrcsCntOpclStats",
        "year_param": "jngBizCrtrYr",
        "page_size": 1000,
    },
    # 15125467 브랜드 등록 정보 — 브랜드관리번호(brandMnno)·가맹본부관리번호·사업자등록번호.
    # 명세 §2.2 "연결 키 = 브랜드 관리번호(우선)" 를 충족하는 정합 마스터.
    # 응답의 jnghdqrtrsRprsvNm(대표자 성명)은 저장하지 않는다 — PERSONAL_FIELDS 참조.
    "brand_master": {
        "url": "https://apis.data.go.kr/1130000/FftcBrandRlsInfo2_Service/getBrandinfo",
        "year_param": "jngBizCrtraYr",
        "page_size": 5000,
    },
    # 15125490 브랜드 가맹점·직영점 정보 — 브랜드×연도×지역 그레인.
    # areaNm=='전체' 행 = 브랜드-연도 합계(직영비중 산출), 17개 시도 행 = 지역분산 산출.
    # ⚠️ 전체(18행/브랜드)를 단순 합산하면 정확히 2배 중복이므로 '전체'와 지역 행을 구분해 사용.
    "brand_region_direct": {
        "url": "https://apis.data.go.kr/1130000/FftcBrandFrcsDropInfo3_Service/getbrandFrcsDmsstus2",
        "year_param": "jngBizCrtraYr",
        "page_size": 10000,
        "gzip": True,          # 연 18~21만행 → 원본 스냅샷 gzip 보존
    },
    # 15110265 브랜드별 창업 금액 현황 — 가맹금·교육비·보증금·기타(인테리어 등) 실측 공시.
    # **여신 익스포저를 합성 난수가 아니라 실제 공시 금액에서 유도**하기 위한 원천.
    # 단위 천원. smtnAmt = 4개 항목 합계 = 가맹점 1곳 개설에 드는 총 투자액.
    "brand_startup_cost": {
        "url": "https://apis.data.go.kr/1130000/FftcBrandFntnStatsService/getBrandFntnStats",
        "year_param": "yr",
        "page_size": 3000,
    },
    # 15125518 브랜드 등록취소 목록 — 자진/직권 취소(하드 실패 신호). 관리번호 없음(명칭 조인).
    "brand_cancel": {
        "url": "https://apis.data.go.kr/1130000/FftcBrandRtrcnInfo2_Service/getbrandRtrcnListinfo",
        "year_param": "jngBizCrtraYr",
        "page_size": 3000,
    },
}

YEARS = list(range(2017, 2026))

# 개인정보 최소수집 — 응답에 섞여 오는 개인 성명 필드는 스냅샷에 쓰기 전에 버린다.
#   jnghdqrtrsRprsvNm : 15125467(brand_master)의 가맹본부 대표자 성명. 공정위 공개 자료라도
#                       개인 식별정보를 원본째 재배포하지 않는다 (data/raw 는 통째로 커밋된다).
#   코드 참조 0건이라 산출물 영향이 없다 — 패널·피처·점수 어디에도 들어가지 않는다.
#   (tests/test_privacy.py 가 data/raw 에 이 필드가 다시 들어오는 것을 막는다.)
PERSONAL_FIELDS: frozenset[str] = frozenset({"jnghdqrtrsRprsvNm"})

# 공정위 가맹사업정보제공시스템(franchise.ftc.go.kr)의 공개 미리보기 페이지
# (https://franchise.ftc.go.kr/openApi.do?service=FftcBrandFrcsStatsService)에
# 내장·공개되어 있는 데모 인증키. 비밀키가 아니라 공정위가 공개 배포한 값이지만
# 회전될 수 있으므로, 제출/운영 파이프라인은 반드시 본인 data.go.kr 키(DATA_GO_KR_KEY)를 사용할 것.
_PUBLIC_PREVIEW_KEY = (
    ""
)


def get_service_key(cfg: dict) -> str:
    key = os.environ.get(cfg["collect"]["service_key_env"], "").strip()
    if key:
        log.info("serviceKey: 환경변수 %s 사용", cfg["collect"]["service_key_env"])
        return key
    log.warning(
        "serviceKey 미설정 — 공정위 공개 미리보기 데모키로 대체합니다. "
        "제출 전 data.go.kr에서 본인 키를 발급받아 %s 에 설정하세요 (개발단계 자동승인).",
        cfg["collect"]["service_key_env"],
    )
    return _PUBLIC_PREVIEW_KEY


def _fetch_page(url: str, params: dict, cfg: dict) -> dict:
    """단일 페이지 호출 (재시도·백오프 포함). 응답 body(dict)를 반환."""
    c = cfg["collect"]
    last_err: Exception | None = None
    for attempt in range(1, c["max_retries"] + 1):
        try:
            r = requests.get(url, params=params, timeout=c["timeout_sec"])
            if r.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests")
            r.raise_for_status()
            body = r.json()
            # 응답 형태: 평면형 {resultCode,...,items:[...]} 또는 중첩형 {response:{header,body}}
            if "response" in body:
                header = body["response"].get("header", {})
                inner = body["response"].get("body", {})
                code = header.get("resultCode", inner.get("resultCode"))
                merged = {**header, **inner}
            else:
                merged = body
                code = body.get("resultCode")
            if str(code) not in ("00", "0"):
                raise RuntimeError(f"API resultCode={code} msg={merged.get('resultMsg')}")
            return merged
        except Exception as e:
            last_err = e
            wait = c["retry_backoff_sec"] * (2 ** (attempt - 1))
            # 예외 문자열에 요청 URL(serviceKey 포함)이 들어갈 수 있어 키를 마스킹 후 로깅
            msg = str(e)
            sk = str(params.get("serviceKey", ""))
            if sk:
                import urllib.parse
                msg = msg.replace(sk, "***KEY***").replace(urllib.parse.quote(sk, safe=""), "***KEY***")
            log.warning("호출 실패 (%d/%d): %s → %.1fs 후 재시도", attempt, c["max_retries"], msg, wait)
            time.sleep(wait)
    yr_param = params.get("yr") or params.get("jngBizCrtrYr")
    raise RuntimeError(f"API 호출 최종 실패: {url} params(연도)={yr_param}") from last_err


def _extract_items(body: dict) -> list[dict]:
    items = body.get("items", [])
    if isinstance(items, dict):  # XML 스타일 {items: {item: [...]}}
        items = items.get("item", [])
    if isinstance(items, dict):  # 단건
        items = [items]
    return items or []


def drop_personal_fields(item: dict) -> dict:
    """응답 item 에서 PERSONAL_FIELDS 를 뺀 사본. 나머지 키·값·순서는 그대로 둔다."""
    return {k: v for k, v in item.items() if k not in PERSONAL_FIELDS}


def fetch_service_year(service: str, year: int, key: str, cfg: dict) -> dict:
    """서비스×연도 전체 페이지 수집 → {"totalCount", "items"} 반환."""
    spec = SERVICES[service]
    page_size = int(spec.get("page_size", 3000))
    all_items: list[dict] = []
    page = 1
    total = None
    while True:
        params = {
            "serviceKey": key,
            "pageNo": page,
            "numOfRows": page_size,
            "resultType": "json",
            spec["year_param"]: year,
        }
        body = _fetch_page(spec["url"], params, cfg)
        total = int(body.get("totalCount", 0))
        # 개인 성명은 메모리에 쌓기 전에 버린다 → 어떤 스냅샷에도 기록되지 않는다
        items = [drop_personal_fields(it) for it in _extract_items(body)]
        all_items.extend(items)
        log.info("%s yr=%d page=%d: +%d건 (누적 %d / 총 %d)", service, year, page, len(items), len(all_items), total)
        if len(all_items) >= total or not items:
            break
        page += 1
        time.sleep(0.3)
    return {"totalCount": total, "items": all_items}


def collect_all(cfg: dict, services: list[str] | None = None, years: list[int] | None = None) -> dict[str, list[Path]]:
    """전 서비스×연도 수집. 스냅샷 존재 시 생략(강제 재수집은 파일 삭제 후 실행)."""
    key = get_service_key(cfg)
    raw_dir: Path = cfg["paths"]["raw"]
    out: dict[str, list[Path]] = {}
    for service in services or list(SERVICES):
        paths: list[Path] = []
        use_gzip = bool(SERVICES[service].get("gzip"))
        for year in years or YEARS:
            snap = raw_dir / (f"{service}_{year}.json.gz" if use_gzip
                              else f"{service}_{year}.json")
            if snap.exists():
                log.info("스냅샷 존재 → 생략: %s", snap.name)
                paths.append(snap)
                continue
            try:
                data = fetch_service_year(service, year, key, cfg)
            except RuntimeError as e:
                log.warning("%s yr=%d 수집 실패 → 건너뜀: %s", service, year, e)
                continue
            snapshot = {
                "service": service,
                "dataset_url": SERVICES[service]["url"],  # 키는 저장하지 않음
                "year_param": SERVICES[service]["year_param"],
                "year": year,
                "fetched_at": datetime.now(UTC).isoformat(),
                "totalCount": data["totalCount"],
                "items": data["items"],
            }
            payload = json.dumps(snapshot, ensure_ascii=False)
            if use_gzip:
                with gzip.open(snap, "wt", encoding="utf-8") as fh:
                    fh.write(payload)
            else:
                snap.write_text(payload, encoding="utf-8")
            log.info("저장: %s (%d건, %.1fMB)", snap.name, len(data["items"]),
                     snap.stat().st_size / 1e6)
            paths.append(snap)
        out[service] = paths
    return out


def load_snapshots(cfg: dict, service: str, keep: callable | None = None) -> list[dict]:
    """저장된 스냅샷 로드 → item dict 리스트 (연도 오름차순, 각 item에 _yr 부여).

    keep: 선택적 행 필터 (대용량 서비스에서 메모리 절약용). keep(item) -> bool.
    .json / .json.gz 모두 지원.
    """
    raw_dir: Path = cfg["paths"]["raw"]
    rows: list[dict] = []
    snaps = sorted(list(raw_dir.glob(f"{service}_*.json"))
                   + list(raw_dir.glob(f"{service}_*.json.gz")),
                   key=lambda p: p.name)
    for snap in snaps:
        if snap.suffix == ".gz":
            with gzip.open(snap, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = json.loads(snap.read_text(encoding="utf-8"))
        for it in data["items"]:
            if keep is not None and not keep(it):
                continue
            it["_yr"] = data["year"]
            rows.append(it)
    return rows


if __name__ == "__main__":
    cfg = load_config()
    collect_all(cfg)
