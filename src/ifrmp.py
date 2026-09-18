"""M1.6 공정위 정보공개서 원문 (franchise.ftc.go.kr) — 가맹본부가 법으로 제출한 1차 문서.

왜 원문인가:
    지금까지 쓰던 공정위 오픈API(data.go.kr)는 **집계 통계**다 — 가맹점 수, 평균매출 같은
    숫자 몇 개. 반면 정보공개서 원문에는 그 숫자가 나온 **근거 문서 자체**가 들어 있다.
    실측한 문서 1건에서 110개의 표준 코드 섹션이 확인됐고, 그중 여신 심사에 직접 쓰이는 것:

      RB_TTYR_FNNR_STUS              본부 3년치 재무상황 (자산·부채·자본·매출·영업이익·순이익)
      RB_TTYR_FRCS_CNT               3년치 가맹점 수 (연초·신규·계약종료·해지·명의변경·연말)
      RB_BIZ_YR_FYER_AVRG_SLS_AMT    지역별 평균매출 (상·하한, 3.3㎡당)
      COREC_ACTN_INFO                공정위·시도지사 시정조치 이력
      CVLST_INFO                     민사소송·민사상 화해
      PETY_ADJU_INFO                 형(刑)의 선고
      CTRT_UPDT_REJECT_RSN           계약 갱신 거절 사유
      CTRT_CNCLTN_RSN_PROCSS         계약 해지 사유 및 절차

    앞의 둘은 **집계 API 에 없는 값**이고(본부 재무·연도별 점포 유출입 내역), 뒤의 셋은
    RAG 심사메모가 "출처를 인용"할 때 인용할 **진짜 원문**이다. 지금까지 RAG 코퍼스는
    패널 수치를 문장으로 다시 쓴 것이라 '원문 인용'이라 부르기 민망했다(자체 감사 지적).

⛔ 두 가지 함정 — 이 모듈이 명시적으로 방어한다:

  ① WAF (실측 확정)
     franchise.ftc.go.kr 은 User-Agent 헤더가 없으면 **HTTP 406 Access denied** 를 준다.
     키·파라미터 문제로 오인하기 쉽다(실제로 그렇게 오진했다). UA 를 붙이면 200 이 온다.

  ② 데모키의 고정 응답 — **여기가 핵심**
     문서에 공개된 `serviceKey=sampleKey` 는 진짜로 동작하지만 범위가 제한된다:
       - type=list    : 실제 데이터. 단 페이지당 1건, totalCount 는 1 로 고정 표기.
       - type=content : jngIfrmpSn 을 무엇으로 주든 **항상 같은 데모 문서 1건**을 준다
                        (실측: sn 149646·150572·150958·152502·151420 → SHA 전부 동일,
                         3,722,374자, 놀부 흥부찜닭).
     즉 sampleKey 로는 "브랜드별 정보공개서를 파싱했다"고 말할 수 없다. 이 모듈은 응답
     해시를 대조해 **데모 고정응답을 스스로 탐지**하고, 그 경우 브랜드별 산출물을 만들지
     않는다(is_demo=True 로 표시하고 중단). 파서가 옳아도 데이터가 1건이면 1건이라고
     말해야 하기 때문이다.
     정식 키(공정위 가맹사업정보제공시스템 발급)를 IFRMP_SERVICE_KEY 에 넣으면 같은
     코드가 전량 수집으로 전환된다 — 파서는 이미 실제 문서로 검증돼 있다.

산출:
    data/raw/ifrmp/            원본 스냅샷 (list/toc/content)
    data/processed/ifrmp_sections.parquet   sn × 섹션코드 × 본문
    outputs/ifrmp_status.json  수집 가능 범위 진단 (데모 여부·건수)

실행: `python -m src.ifrmp`
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.common import get_logger, load_config

log = get_logger("ifrmp")

BASE = "https://franchise.ftc.go.kr/api/search.do"
VIEWER = "https://franchise.ftc.go.kr/api/viewer.do"
KEY_ENV = "IFRMP_SERVICE_KEY"
DEMO_KEY = "sampleKey"          # 공정위가 API 문서에 공개한 데모키

# ⚠️ 이 헤더가 없으면 전부 406 이다. 브라우저 위장이 아니라 WAF 가 UA 없는 요청을 막는 것.
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://franchise.ftc.go.kr/",
}
_PACE_SEC = 0.35
_TIMEOUT = 120

# 여신 심사에 쓰는 섹션 (코드는 문서에 attr= 로 박혀 있어 제목 변경에 영향받지 않는다)
FINANCE_SECTION = "RB_TTYR_FNNR_STUS"
STORE_SECTION = "RB_TTYR_FRCS_CNT"
RISK_SECTIONS = ("COREC_ACTN_INFO", "CVLST_INFO", "PETY_ADJU_INFO")
CONTRACT_SECTIONS = ("CTRT_UPDT_REJECT_RSN", "CTRT_CNCLTN_RSN_PROCSS",
                     "JNGHDQRTRS_CTRT_END_ACTN_CN")
RAG_SECTIONS = RISK_SECTIONS + CONTRACT_SECTIONS + (FINANCE_SECTION, STORE_SECTION)


_warned = False


def get_key() -> tuple[str, bool]:
    """(키, 데모여부). 정식 키가 없으면 공개 데모키로 내려가되 데모임을 알린다."""
    global _warned
    k = os.environ.get(KEY_ENV, "").strip()
    if k:
        return k, False
    if not _warned:      # 호출마다 같은 경고를 찍으면 정작 중요한 로그가 묻힌다
        _warned = True
        log.warning("%s 미설정 — 공정위 공개 데모키(sampleKey)로 동작합니다. "
                    "본문(content)은 **고정 데모 문서 1건**만 반환되므로 브랜드별 수집은 "
                    "불가합니다. 정식 키는 franchise.ftc.go.kr 에서 신청하세요.", KEY_ENV)
    return DEMO_KEY, True


def _fetch(params: dict, *, key: str) -> str:
    last = None
    for attempt in range(1, 4):
        try:
            r = requests.get(BASE, params={"serviceKey": key, **params},
                             headers=_HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
            if "<Error>" in r.text[:300]:
                err = re.search(r"<errorCn>(.*?)</errorCn>", r.text)
                raise RuntimeError(f"API 오류: {err.group(1) if err else r.text[:120]}")
            return r.text
        except Exception as e:
            last = e
            log.warning("정보공개서 호출 실패 (%d/3): %s", attempt, str(e)[:140])
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"정보공개서 호출 실패: {last}")


# ---------------------------------------------------------------------------
# 목록
# ---------------------------------------------------------------------------
_LIST_ITEM = re.compile(
    r"<jngIfrmpSn>(\d+)</jngIfrmpSn>.*?<corpNm>(.*?)</corpNm>.*?"
    r"<brandNm>(.*?)</brandNm>.*?<brno>(.*?)</brno>", re.S)


def fetch_list(cfg: dict, year: int, *, max_pages: int = 50) -> pd.DataFrame:
    """연도별 정보공개서 목록. 데모키에서는 페이지당 1건씩만 나온다(그대로 기록한다)."""
    key, is_demo_key = get_key()
    # 목록도 키 종류에 따라 범위가 다르다(데모키는 페이지당 1건) — 캐시를 섞지 않는다
    sub = "demo" if is_demo_key else "full"
    snap = Path(cfg["paths"]["raw"]) / "ifrmp" / f"list_{year}_{sub}.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    if snap.exists():
        return pd.DataFrame(json.loads(snap.read_text(encoding="utf-8")))

    rows, seen = [], set()
    page = 0
    for page in range(1, max_pages + 1):
        try:
            xml = _fetch({"type": "list", "yr": year, "pageNo": page, "numOfRows": 200},
                         key=key)
        except Exception as e:
            # 중간 실패를 부분 결과로 확정 저장하면 이후 실행이 영원히 그 조각만 본다.
            log.error("목록 %d년 %d페이지에서 실패 — 부분 결과를 캐시하지 않는다: %s",
                      year, page, str(e)[:140])
            return pd.DataFrame(rows)
        hits = _LIST_ITEM.findall(xml)
        if not hits:
            break
        fresh = [h for h in hits if h[0] not in seen]
        if not fresh:
            break                       # 같은 페이지가 반복되면 종료
        for sn, corp, brand, brno in fresh:
            seen.add(sn)
            rows.append({"jngIfrmpSn": sn, "corpNm": corp, "brandNm": brand,
                         "brno": re.sub(r"\D", "", brno), "yr": year})
        time.sleep(_PACE_SEC)
    snap.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    log.info("정보공개서 목록 %d년: %d건 (%d페이지 순회)", year, len(rows), page)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 본문 · 섹션 파싱
# ---------------------------------------------------------------------------
_HEAD = re.compile(r'<h[1-6][^>]*attrb_sn="([^"]+)"[^>]*attr="([^"]+)"[^>]*'
                   r'title="([^"]*)"[^>]*>')


def parse_sections(xml: str) -> dict[str, dict]:
    """정보공개서 XML → {섹션코드: {title, text}}.

    헤더의 attr= 코드는 공정위 표준 항목코드라 제목 문구가 바뀌어도 안정적이다.
    각 섹션 본문은 다음 헤더 직전까지로 자른다.
    """
    heads = list(_HEAD.finditer(xml))
    out: dict[str, dict] = {}
    for i, h in enumerate(heads):
        s = h.end()
        e = heads[i + 1].start() if i + 1 < len(heads) else len(xml)
        body = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml[s:e])).strip()
        code, title = h.group(2), h.group(3)
        if code not in out:            # 같은 코드가 반복되면 첫 번째만
            out[code] = {"title": title, "text": body}
    return out


_YEAR_ROW = re.compile(r"(\d{4})\s*년?\s+((?:-?[\d,]+\s+){5}-?[\d,]+)")


def parse_financial_summary(sections: dict) -> list[dict]:
    """RB_TTYR_FNNR_STUS → 연도별 재무 (단위: 천원, 공시 표기 그대로).

    표 형식: 연도 자산총계 부채총계 자본총계 매출액 영업이익 당기순이익
    """
    sec = sections.get(FINANCE_SECTION)
    if not sec:
        return []
    cols = ["assets", "liabilities", "equity", "revenue",
            "operating_income", "net_income"]
    rows = []
    for m in _YEAR_ROW.finditer(sec["text"]):
        nums = [n for n in m.group(2).split() if n]
        if len(nums) != 6:
            continue
        rec = {"fiscal_year": int(m.group(1))}
        for c, n in zip(cols, nums, strict=True):
            try:
                rec[c] = float(n.replace(",", "")) * 1000.0    # 천원 → 원
            except ValueError:
                rec[c] = None
        rows.append(rec)
    return rows


_STORE_COLS = ("n_begin", "n_new", "n_contract_end", "n_contract_cancel",
               "n_name_change", "n_end")
# 표는 '연도 연초 신규 계약종료 계약해지 명의변경 연말' = 연도 뒤 **6열**이다.
# 값이 없는 칸은 '-' 로 채워져 있어 자릿수는 유지된다 (실측: "2019 - 114 4 - 7 110").
# ⚠️ 행 끝에 후방부정탐색을 걸면 안 된다 — 표가 "… 110 2020 110 56 …" 처럼 이어붙어
#    있어 다음 행의 연도(숫자)에 걸려 마지막 행 하나만 잡힌다(실제 겪은 버그).
#    finditer 가 매칭 뒤부터 이어서 훑으므로 행 경계는 저절로 맞는다.
_STORE_ROW = re.compile(r"\b(19\d{2}|20\d{2})\s+((?:[\d,]+|-)(?:\s+(?:[\d,]+|-)){5})")


def parse_store_counts(sections: dict) -> list[dict]:
    """RB_TTYR_FRCS_CNT → 연도별 점포 유출입 (연초·신규·계약종료·해지·명의변경·연말).

    이 표는 집계 API 에 없는 **연도별 유출입 내역**이라, 우리 패널의 n_new·n_contract_end
    를 원문으로 교차검증할 수 있는 유일한 공개 원천이다.
    """
    sec = sections.get(STORE_SECTION)
    if not sec:
        return []
    rows = []
    for m in _STORE_ROW.finditer(sec["text"]):
        toks = m.group(2).split()
        if len(toks) != len(_STORE_COLS):
            continue
        rec: dict = {"year": int(m.group(1))}
        for c, t in zip(_STORE_COLS, toks, strict=True):
            rec[c] = None if t == "-" else float(t.replace(",", ""))
        rows.append(rec)
    return rows


_NO_ITEM = re.compile(r"해당사항\s*없|사실이\s*없습니다|없음")


def parse_risk_sections(sections: dict) -> dict:
    """법 위반·소송·형벌 섹션에서 '실제로 사건이 있는가'를 가른다.

    이 섹션들은 사건이 없어도 표 틀과 법조문 설명이 그대로 들어가 길이가 길다.
    길이로 판단하면 전 브랜드가 위험으로 뜨므로, **'해당사항 없음' 문구의 부재**와
    본문 길이를 함께 본다. 애매하면 None(판정보류) — 억지로 0/1 을 만들지 않는다.
    """
    out: dict = {}
    for code in RISK_SECTIONS:
        sec = sections.get(code)
        if not sec:
            out[code] = None
            continue
        txt = sec["text"]
        if _NO_ITEM.search(txt):
            out[code] = 0
        elif len(txt) > 400:
            out[code] = 1
        else:
            out[code] = None
    return out


def fetch_content(cfg: dict, sn: str) -> tuple[str, str]:
    """본문 XML 을 받아 스냅샷 보관. (xml, sha256) 반환."""
    key, is_demo_key = get_key()
    # ⚠️ 캐시를 키 종류별로 분리한다. 분리하지 않으면 데모키로 받은 **고정 응답**이
    #    캐시에 남아, 나중에 정식 키를 넣어도 그 문서가 재사용되어 데모 판정이 계속 참이
    #    되고 수집이 영영 열리지 않는다(적대적 리뷰 지적).
    sub = "demo" if is_demo_key else "full"
    snap = Path(cfg["paths"]["raw"]) / "ifrmp" / "content" / sub / f"{sn}.xml"
    snap.parent.mkdir(parents=True, exist_ok=True)
    if snap.exists():
        xml = snap.read_text(encoding="utf-8")
    else:
        xml = _fetch({"type": "content", "jngIfrmpSn": sn}, key=key)
        snap.write_text(xml, encoding="utf-8")
        time.sleep(_PACE_SEC)
    return xml, hashlib.sha256(xml.encode("utf-8")).hexdigest()


def detect_demo_mode(cfg: dict, sns: list[str]) -> dict:
    """서로 다른 sn 이 **같은 문서**를 주는지 확인한다 (데모 고정응답 탐지).

    이 검사가 없으면 "브랜드 N개의 정보공개서를 파싱했다"는 거짓 주장이 만들어진다.
    """
    probe = sns[:3]
    hashes = {}
    for sn in probe:
        try:
            _, h = fetch_content(cfg, sn)
            hashes[sn] = h
        except Exception as e:
            log.warning("본문 조회 실패 sn=%s: %s", sn, str(e)[:120])
    uniq = set(hashes.values())
    # ⛔ fail-open 금지. 프로브가 실패해 표본이 1건 이하면 "데모가 아니다"라고 **결론낼 수
    #    없다.** 그런데도 is_demo=False 로 떨어뜨리면 고정 데모 문서를 브랜드별 정보공개서
    #    원문으로 확정 저장하게 된다 — 판정 불가는 안전한 쪽(데모로 간주)으로 닫는다.
    if len(hashes) < 2:
        log.error("⛔ 본문 프로브 %d건만 성공 — 데모 여부를 판정할 수 없다. "
                  "안전을 위해 데모로 간주하고 브랜드별 수집을 중단한다.", len(hashes))
        return {"probed": hashes, "distinct_documents": len(uniq), "is_demo": True,
                "undetermined": True}
    is_demo = len(uniq) == 1
    if is_demo:
        log.error("⛔ 서로 다른 정보공개서 %d건이 **동일 문서**를 반환했다 (sha=%s…). "
                  "데모 고정응답이므로 브랜드별 산출물을 생성하지 않는다.",
                  len(hashes), next(iter(uniq))[:12])
    return {"probed": hashes, "distinct_documents": len(uniq), "is_demo": is_demo}


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------
def build(cfg: dict, year: int | None = None) -> dict:
    """목록 → 데모 판정 → (정식 키일 때만) 전량 섹션 파싱."""
    if year is None:
        year = int((cfg.get("ifrmp") or {}).get("year", 2023))
    _, using_demo = get_key()
    out_dir, proc = Path(cfg["paths"]["outputs"]), Path(cfg["paths"]["processed"])
    lst = fetch_list(cfg, year)
    status: dict = {"year": year, "list_rows": len(lst),
                    "key_source": "demo(sampleKey)" if using_demo else KEY_ENV}

    if lst.empty:
        status["error"] = "목록이 비어 있음"
        (out_dir / "ifrmp_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    status["demo_check"] = detect_demo_mode(cfg, lst["jngIfrmpSn"].tolist())
    is_demo = status["demo_check"]["is_demo"]

    # 파서 자체는 실제 문서로 검증한다 — 데모여도 파싱 결과를 보여 '작동함'을 증명하되,
    # 브랜드별 데이터로는 절대 쓰지 않는다.
    sn0 = lst["jngIfrmpSn"].iloc[0]
    xml, _ = fetch_content(cfg, sn0)
    secs = parse_sections(xml)
    status["parser_validation"] = {
        "sections_found": len(secs),
        "financial_rows": len(parse_financial_summary(secs)),
        "store_rows": len(parse_store_counts(secs)),
        "risk_flags": parse_risk_sections(secs),
        "note": "파서 검증용 1건. 데모 고정응답이면 브랜드 데이터로 쓰지 않는다.",
    }

    if is_demo:
        status["collected_brands"] = 0
        status["blocked_reason"] = (
            "sampleKey 는 본문을 고정 데모 1건으로만 준다. 브랜드별 수집에는 공정위 "
            f"정식 인증키가 필요하다 (환경변수 {KEY_ENV}). 파서는 이미 실제 문서로 "
            "검증되어 있어 키만 들어오면 즉시 전량 수집된다.")
        log.warning("브랜드별 정보공개서 수집을 **건너뛴다** — %s", status["blocked_reason"])
    else:
        rows = []
        for r in lst.itertuples(index=False):
            try:
                x, _ = fetch_content(cfg, r.jngIfrmpSn)
            except Exception as e:
                log.warning("sn=%s 본문 실패: %s", r.jngIfrmpSn, str(e)[:100])
                continue
            s = parse_sections(x)
            for code in RAG_SECTIONS:
                if code in s:
                    rows.append({"jngIfrmpSn": r.jngIfrmpSn, "brandNm": r.brandNm,
                                 "corpNm": r.corpNm, "brno": r.brno, "yr": r.yr,
                                 "section": code, "title": s[code]["title"],
                                 "text": s[code]["text"]})
        if rows:
            pd.DataFrame(rows).to_parquet(proc / "ifrmp_sections.parquet", index=False)
        status["collected_brands"] = int(lst["jngIfrmpSn"].nunique())
        status["section_rows"] = len(rows)
        log.info("정보공개서 섹션 %d행 (브랜드 %d개) → ifrmp_sections.parquet",
                 len(rows), status["collected_brands"])

    (out_dir / "ifrmp_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


if __name__ == "__main__":
    print(json.dumps(build(load_config()), ensure_ascii=False, indent=2))
