"""국세청 사업자등록 상태조회 — 가맹본부·가맹점의 휴·폐업 여부를 '지금' 기준으로 확인한다.

왜 필요한가
    이 도구의 핵심 원천인 공정위 정보공개서는 연 1회 공시이고 수치가 1~2년 늦다. 본부가 이미
    폐업했거나 차주(가맹점)가 휴업 중이어도 공시에는 한참 뒤에야 나타난다. 국세청 상태조회는
    사업자등록번호만으로 계속·휴업·폐업과 폐업일을 알려 주고 30분 주기로 갱신되므로, 연간 공시가
    놓치는 '이미 문을 닫았다'는 사실을 심사 시점에 바로 확인할 수 있다.
    (본부 사업자번호는 공정위 브랜드 등록정보 15125467 의 brno 로 이미 수집돼 있다.)

검증한 명세 (2026-09-19 실측·원문 대조)
    · 데이터셋: 공공데이터포털 15081808 「국세청_사업자등록정보 진위확인 및 상태조회 서비스」
      https://www.data.go.kr/data/15081808/openapi.do  (등록 2021-05-27, 수정 2026-05-13,
      개발·운영 모두 자동승인, 개발계정 트래픽 1,000,000, 호출허용 1회 100건·1일 100만건,
      "국세청 사업자등록정보와 30분 주기로 업데이트(신규 개업자는 1~2일 소요)")
    · Swagger 원문: https://infuser.odcloud.kr/api/stages/28493/api-docs
      host api.odcloud.kr, basePath /api/nts-businessman/v1, 경로 /status·/validate, **POST 전용**
    · 요청: POST https://api.odcloud.kr/api/nts-businessman/v1/status  본문 {"b_no": ["0000000000", ...]}
      b_no 는 '-' 없는 숫자 10자리, 1회 최대 100개(초과 시 413 TOO_LARGE_REQUEST).
    · 인증: 쿼리 serviceKey 또는 헤더 `Authorization: Infuser {키}` (명세 securityDefinitions 두 가지).
      헤더 방식은 실측으로 확인했다 — 접두어 'Infuser' 가 없으면 키 누락(-401)으로 처리된다.
      이 모듈은 **헤더 방식**을 쓴다. 쿼리에 키를 넣으면 requests 예외 문자열(URL 포함)에 키가 섞여
      로그로 새기 쉽기 때문이다.
    · 응답: {"status_code":"OK","request_cnt":n,"match_cnt":m,"data":[{b_no, b_stt, b_stt_cd,
      tax_type, tax_type_cd, end_dt, utcc_yn, tax_type_change_dt, invoice_apply_dt, rbf_tax_type,
      rbf_tax_type_cd}]}. b_stt_cd 01=계속사업자, 02=휴업자, 03=폐업자, end_dt=폐업일(YYYYMMDD).
      미등록·삭제 번호는 b_stt_cd 가 비고 tax_type 에 "국세청에 등록되지 않은 사업자등록번호입니다".
    · 오류: 400 BAD_JSON_REQUEST, 411 REQUEST_DATA_MALFORMED, 413 TOO_LARGE_REQUEST,
      500 INTERNAL_ERROR (명세). 게이트웨이 인증 오류는 실측상 HTTP 401 + JSON:
      키 없음 {"code":-401,"msg":"인증키는 필수 항목 입니다."},
      미등록 키 {"code":-4,"msg":"등록되지 않은 인증키 입니다."}.

⚠️ 활용신청이 따로 필요하다
    키는 기존 공공데이터포털 키(DATA_GO_KR_KEY)를 그대로 쓰지만, **이 데이터셋(15081808)에 대한
    활용신청**을 사용자가 별도로 해야 한다(자동승인, 게이트웨이 반영에 수십 분 걸릴 수 있다).
    신청 전이거나 반영 전이면 -4 '등록되지 않은 인증키'가 온다 — 키를 재발급하지 말고 기다린다.

검증번호 (normalize_bno)
    사업자등록번호 = 청코드 3 + 구분코드 2 + 일련번호 4 + 검증번호 1. 검증번호 산식(앞 9자리에
    1,3,7,1,3,7,1,3,5 를 곱해 더하고, 9번째 자리×5 의 10의 몫을 더한 뒤, 10에서 합의 끝자리를 뺀 값)은
    국세청이 공식 산식을 공개하지 않아 **실데이터로 검증**했다: 공정위 공시 가맹본부 사업자번호
    19,691건 중 99.31%가 통과한다(무작위 번호라면 10%). 불일치 0.69%(135건)는 공시 오기로 보이나
    실제 유효 번호일 가능성을 배제할 수 없다. 그래서
      · normalize_bno()   — 입력 검증용. 기본은 엄격(검증번호 불일치 → None).
      · lookup_status()   — 10자리면 검증번호가 틀려도 국세청에 묻는다(권위 있는 답은 국세청이 준다).
                            결과의 checksum_ok=False 로 '번호 오기 의심'을 화면에 알릴 수 있다.

개인정보·비밀
    키 값은 어디에도 출력·기록하지 않는다(오류 문자열도 마스킹). 사업자번호는 개인사업자에게는
    준식별정보이므로 로그에는 가운데를 가린 형태(124-**-***98)와 건수만 남긴다.

사용 (UI·배치)
    df = lookup_status(["124-81-00998", "2208162517"])   # 키는 DATA_GO_KR_KEY 에서
    df[["b_no", "status", "closed_on", "error"]]
    → 국세청 원천이 30분 주기로 갱신되므로 화면에서는 st.cache_data(ttl=1800) 정도로 감싸면 충분하다.
"""
from __future__ import annotations

import numbers
import os
import re
import time
import unicodedata
import urllib.parse
from collections.abc import Iterable

import pandas as pd
import requests

from src.common import get_logger, load_secrets

log = get_logger("nts")

KEY_ENV = "DATA_GO_KR_KEY"
DATASET_ID = "15081808"
STATUS_URL = "https://api.odcloud.kr/api/nts-businessman/v1/status"
BATCH_SIZE = 100                 # 명세: 1회 최대 100개 (초과 시 413)

# 재시도: 429(한도)·5xx·네트워크 오류만. 4xx 는 다시 보내도 같은 답이라 즉시 실패로 처리한다.
MAX_ATTEMPTS = 3
BACKOFF_SEC = 1.0
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_sleep = time.sleep              # 테스트에서 대기 없이 돌리기 위해 모듈 변수로 둔다

STATUS_BY_CODE = {"01": "계속", "02": "휴업", "03": "폐업"}
UNREGISTERED = "미등록"
UNKNOWN = "확인불가"
_UNREGISTERED_MSG = "등록되지 않은"

COLUMNS = ["input", "b_no", "status", "status_code", "closed_on", "tax_type", "error", "checksum_ok"]

_WEIGHTS = (1, 3, 7, 1, 3, 7, 1, 3, 5)
_NON_DIGIT_SEP = re.compile(r"[\s\-‐‑‒–—―_.]")


# ---------------------------------------------------------------------------
# 번호 정규화·검증
# ---------------------------------------------------------------------------
def check_digit_ok(bno: str) -> bool:
    """10자리 숫자 문자열의 검증번호가 맞는지 (산식 검증 근거는 모듈 docstring)."""
    if not (len(bno) == 10 and bno.isascii() and bno.isdigit()):
        return False
    d = [int(c) for c in bno]
    s = sum(x * w for x, w in zip(d[:9], _WEIGHTS, strict=True)) + (d[8] * 5) // 10
    return (10 - s % 10) % 10 == d[9]


def normalize_bno(s: object, *, check_digit: bool = True) -> str | None:
    """사업자등록번호 → 숫자 10자리. 형식이 틀리거나(기본값) 검증번호가 맞지 않으면 None.

    하이픈·공백·전각 숫자('１２４－８１…')를 허용한다 — 엑셀·스캔 문서에서 복사해 오는 값이 이렇다.
    check_digit=False 면 10자리 형식만 본다(국세청에 직접 물어볼 때 쓴다).
    """
    if s is None or isinstance(s, bool):
        return None
    if isinstance(s, float) and s.is_integer():
        s = int(s)                   # 엑셀 숫자 셀을 pandas 로 읽으면 1248100998.0 이 된다
    if isinstance(s, numbers.Integral):
        # 엑셀 숫자 셀은 앞자리 0 이 사라진다(예: 0123456789 → 123456789). 되살린다.
        s = f"{int(s):010d}"
    t = unicodedata.normalize("NFKC", str(s)).strip()
    t = _NON_DIGIT_SEP.sub("", t)
    if not (len(t) == 10 and t.isascii() and t.isdigit()):
        return None
    if check_digit and not check_digit_ok(t):
        return None
    return t


def mask_bno(bno: str | None) -> str:
    """로그·화면용 마스킹: '1248100998' → '124-**-***98'. 청코드와 끝 두 자리만 남긴다."""
    t = str(bno or "")
    if len(t) != 10:
        return "***"
    return f"{t[:3]}-**-***{t[-2:]}"


# ---------------------------------------------------------------------------
# 키
# ---------------------------------------------------------------------------
def _resolve_key(key: str | None) -> str:
    """명시 인자 > 환경변수(DATA_GO_KR_KEY, .env·st.secrets 포함). 값은 반환만 하고 기록하지 않는다."""
    if key is None:
        load_secrets()
        key = os.environ.get(KEY_ENV, "")
    key = str(key).strip()
    # 포털의 'Encoding' 키(%2B 등)를 넣은 경우: 헤더로 보낼 때는 원문(Decoding) 키가 필요하다.
    if "%" in key:
        key = urllib.parse.unquote(key)
    return key


def _mask_key(text: str, key: str) -> str:
    """오류 문자열에 키(원문·URL 인코딩형)가 섞여 들어가는 것을 막는다."""
    if not key:
        return text
    for form in {key, urllib.parse.quote(key, safe=""), urllib.parse.quote_plus(key)}:
        if form:
            text = text.replace(form, "***KEY***")
    return text


# ---------------------------------------------------------------------------
# 호출
# ---------------------------------------------------------------------------
class _BatchError(RuntimeError):
    """한 묶음(최대 100건) 전체가 실패한 경우 — 그 묶음의 모든 행이 '확인불가'가 된다."""


def _describe_error(resp: requests.Response) -> str:
    """HTTP 오류 응답 → 사람이 읽을 원인. 명세 오류코드와 게이트웨이 코드(-4, -401)를 구분한다."""
    code = msg = ""
    try:
        body = resp.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        code = str(body.get("status_code") or body.get("code") or "")
        msg = str(body.get("msg") or body.get("message") or "")
    status = resp.status_code
    if status == 401 or code in ("-4", "-401"):
        if code == "-401":
            return "인증키 누락(-401) — DATA_GO_KR_KEY 가 전달되지 않았습니다."
        return (f"인증키 거절(HTTP {status}, code={code or '?'}) — data.go.kr 에서 데이터셋 {DATASET_ID} "
                "활용신청 여부를 확인하세요. 신청 직후라면 게이트웨이 반영(수십 분)을 기다리세요(키 재발급 금지).")
    if status == 429:
        return "호출 한도 초과(HTTP 429) — 잠시 후 다시 시도하세요."
    if status == 413 or code == "TOO_LARGE_REQUEST":
        return "요청 번호 100개 초과(413 TOO_LARGE_REQUEST)."
    if status >= 500:
        return f"국세청/게이트웨이 서버 오류(HTTP {status}{', ' + code if code else ''})."
    detail = ", ".join(x for x in (code, msg) if x)
    return f"요청 오류(HTTP {status}{', ' + detail if detail else ''})."


def _post_batch(bnos: list[str], key: str, timeout: float) -> list[dict]:
    """최대 100건 한 묶음 호출. 429·5xx·네트워크 오류만 짧게 재시도한다."""
    headers = {"Authorization": f"Infuser {key}", "Content-Type": "application/json",
               "Accept": "application/json"}
    last = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(STATUS_URL, json={"b_no": bnos}, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last = f"네트워크 오류({type(e).__name__}): {_mask_key(str(e), key)[:160]}"
        else:
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except ValueError as e:
                    raise _BatchError("응답 해석 실패(JSON 아님).") from e
                if not isinstance(body, dict) or not isinstance(body.get("data"), list):
                    code = (body.get("status_code") or body.get("code")) if isinstance(body, dict) else ""
                    raise _BatchError(f"예상 밖 응답 형식(status_code={code or '?'}).")
                return body["data"]
            last = _mask_key(_describe_error(resp), key)
            if resp.status_code not in _RETRY_STATUS:
                raise _BatchError(last)
        if attempt < MAX_ATTEMPTS:
            wait = BACKOFF_SEC * (2 ** (attempt - 1))
            log.warning("국세청 상태조회 재시도 (%d/%d, %d건, 예: %s): %s → %.1fs 후",
                        attempt, MAX_ATTEMPTS, len(bnos), mask_bno(bnos[0]), last, wait)
            _sleep(wait)
    raise _BatchError(last or "알 수 없는 오류")


def _row_from_item(item: dict) -> dict:
    """국세청 응답 1건 → 결과 행의 상태 필드."""
    code = str(item.get("b_stt_cd") or "").strip()
    tax_type = str(item.get("tax_type") or "").strip()
    if code in STATUS_BY_CODE:
        end = str(item.get("end_dt") or "").strip()
        # end_dt 는 폐업일이다. 계속·휴업 사업자에게 값이 와도(과거 폐업 후 재개 등) 폐업일로 쓰지 않는다.
        closed = pd.to_datetime(end, format="%Y%m%d", errors="coerce") if (end and code == "03") else pd.NaT
        return {"status": STATUS_BY_CODE[code], "status_code": code, "tax_type": tax_type,
                "closed_on": closed, "error": None}
    if _UNREGISTERED_MSG in tax_type:
        return {"status": UNREGISTERED, "status_code": "", "tax_type": "", "closed_on": pd.NaT, "error": None}
    return {"status": UNKNOWN, "status_code": code, "tax_type": tax_type, "closed_on": pd.NaT,
            "error": f"알 수 없는 응답(b_stt_cd={code or '없음'})."}


def lookup_status(bnos: Iterable[object], key: str | None = None, *, timeout: float = 15) -> pd.DataFrame:
    """사업자등록번호 목록의 현재 상태. 입력 1건당 결과 1행(입력 순서 유지).

    열: input(원문), b_no(숫자 10자리 또는 None), status(계속/휴업/폐업/미등록/확인불가),
        status_code(01/02/03), closed_on(폐업일), tax_type(과세유형), error(확인불가 사유), checksum_ok.

    · 키가 없으면 네트워크를 쓰지 않고 전 행을 '확인불가'로 돌려준다(사유에 설정 방법).
    · 형식이 틀린 번호(10자리 아님)는 보내지 않고 그 행만 '확인불가'.
    · 중복 번호는 한 번만 묻는다. 100건씩 나눠 보낸다(명세 상한).
    · 한 묶음이 실패하면 그 묶음의 행만 '확인불가' — 다른 묶음 결과는 살린다.
    """
    inputs = list(bnos)
    rows: list[dict] = []
    for raw in inputs:
        b = normalize_bno(raw, check_digit=False)
        rows.append({"input": "" if raw is None else str(raw), "b_no": b, "status": UNKNOWN, "status_code": "",
                     "closed_on": pd.NaT, "tax_type": "",
                     "error": None if b else "형식 오류 — 숫자 10자리가 아닙니다.",
                     "checksum_ok": bool(b) and check_digit_ok(b)})
    unique = list(dict.fromkeys(r["b_no"] for r in rows if r["b_no"]))

    k = _resolve_key(key)
    results: dict[str, dict] = {}
    if unique and not k:
        reason = (f"{KEY_ENV} 미설정 — data.go.kr 「국세청_사업자등록정보 진위확인 및 상태조회 서비스」"
                  f"({DATASET_ID}) 활용신청 후 같은 키를 환경변수 {KEY_ENV} 에 넣으세요.")
        log.warning("국세청 상태조회 생략: %s 미설정 (%d건 확인불가)", KEY_ENV, len(unique))
        results = {b: {"error": reason} for b in unique}
    elif unique:
        for i in range(0, len(unique), BATCH_SIZE):
            batch = unique[i:i + BATCH_SIZE]
            try:
                data = _post_batch(batch, k, timeout)
            except _BatchError as e:
                reason = _mask_key(str(e), k)
                log.warning("국세청 상태조회 실패 (%d건, 예: %s): %s", len(batch), mask_bno(batch[0]), reason)
                results.update({b: {"error": reason} for b in batch})
                continue
            got = {normalize_bno(it.get("b_no"), check_digit=False): it for it in data if isinstance(it, dict)}
            for b in batch:
                results[b] = (_row_from_item(got[b]) if b in got
                              else {"error": "응답에 해당 번호가 없습니다."})
            log.info("국세청 상태조회 %d건 (예: %s): %s", len(batch), mask_bno(batch[0]),
                     pd.Series([results[b].get("status", UNKNOWN) for b in batch]).value_counts().to_dict())

    for r in rows:
        if r["b_no"] and r["b_no"] in results:
            r.update(results[r["b_no"]])
    return pd.DataFrame(rows, columns=COLUMNS)
