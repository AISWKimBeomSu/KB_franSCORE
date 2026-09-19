"""국세청 사업자등록 상태조회(src/nts.py) — 네트워크 없이 모의 응답으로 검증한다.

무엇을 지키는가
    A. 번호 정규화: 하이픈·공백·전각 숫자·엑셀 숫자 셀(앞자리 0 소실)을 받아 10자리로, 검증번호가
       틀리면 None (산식은 공정위 공시 본부 사업자번호 19,691건으로 실측 검증 — src/nts.py docstring)
    B. 키가 없으면 네트워크를 쓰지 않고 '확인불가' + 설정 방법을 돌려준다
    C. 100건 단위 분할·중복 제거·입력 순서 보존
    D. 응답 코드 → 계속/휴업/폐업/미등록/확인불가, 인증 오류는 재시도하지 않고 429·5xx 만 재시도
    E. 키와 사업자번호 전체가 오류 문자열·로그에 남지 않는다

실행: python -m pytest -q tests/test_nts.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import nts

KEY = "TEST-SECRET-KEY-123"
# 공개된 대기업 사업자등록번호(홈페이지 하단 표기 의무) — 산식이 실제 번호를 통과시키는지 본다.
PUBLIC_BNOS = ("124-81-00998", "220-81-62517", "120-81-47521")


def _valid(n9: str) -> str:
    """앞 9자리에 검증번호를 붙인 합성 번호 (모의 호출 전용)."""
    for d in "0123456789":
        if nts.check_digit_ok(n9 + d):
            return n9 + d
    raise AssertionError("unreachable")


def _item(b: str, code: str = "01", end_dt: str = "") -> dict:
    """국세청 status 응답 data[] 1건 (명세 BusinessStatus 모양)."""
    names = {"01": "계속사업자", "02": "휴업자", "03": "폐업자"}
    if code == "none":
        return {"b_no": b, "b_stt": "", "b_stt_cd": "", "tax_type": "국세청에 등록되지 않은 사업자등록번호입니다",
                "tax_type_cd": "", "end_dt": "", "utcc_yn": "", "tax_type_change_dt": "", "invoice_apply_dt": "",
                "rbf_tax_type": "", "rbf_tax_type_cd": ""}
    return {"b_no": b, "b_stt": names.get(code, ""), "b_stt_cd": code, "tax_type": "부가가치세 일반과세자",
            "tax_type_cd": "01", "end_dt": end_dt, "utcc_yn": "N", "tax_type_change_dt": "",
            "invoice_apply_dt": "", "rbf_tax_type": "해당없음", "rbf_tax_type_cd": "99"}


class _Resp:
    def __init__(self, status: int = 200, body: object = None):
        self.status_code = status
        self._body = body

    def json(self) -> object:
        if self._body is None:
            raise ValueError("not json")
        return self._body


class _Post:
    """requests.post 대역. 호출을 기록하고, 준비된 응답을 차례로 돌려준다(함수면 요청 본문으로 만든다)."""

    def __init__(self, *responses):
        self.calls: list[dict] = []
        self._responses = list(responses)

    def __call__(self, url, json=None, headers=None, timeout=None, **kw):
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {}), "timeout": timeout})
        r = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(r, Exception):
            raise r
        return r(json) if callable(r) else r


def _ok_all(code: str = "01"):
    return lambda body: _Resp(200, {"status_code": "OK", "request_cnt": len(body["b_no"]),
                                    "match_cnt": len(body["b_no"]), "data": [_item(b, code) for b in body["b_no"]]})


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, caplog):
    """로그는 pipeline.log 대신 caplog 로, 대기는 0초로, 키는 테스트가 준 것만 쓰게 한다."""
    logger = logging.getLogger("tests.nts")
    monkeypatch.setattr(nts, "log", logger)
    caplog.set_level(logging.INFO, logger="tests.nts")
    monkeypatch.setattr(nts, "_sleep", lambda s: None)
    monkeypatch.setattr(nts, "load_secrets", lambda: [])
    monkeypatch.delenv(nts.KEY_ENV, raising=False)


# ---------------------------------------------------------------------------
# A. 번호 정규화
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["124-81-00998", "124 81 00998", " 1248100998 ", "１２４－８１－００９９８",
                                 "124.81.00998", 1248100998])
def test_normalize_accepts_common_formats(raw):
    assert nts.normalize_bno(raw) == "1248100998"


def test_normalize_rejects_bad_length_letters_and_checksum():
    for bad in ("12481009", "12481009981", "12a8100998", "", None, "124-81-0099"):
        assert nts.normalize_bno(bad) is None, bad
    assert nts.normalize_bno("124-81-00997") is None                      # 끝자리만 틀림
    assert nts.normalize_bno("124-81-00997", check_digit=False) == "1248100997"
    # 엑셀 숫자 셀에서 사라진 앞자리 0 을 되살린다 (pandas 가 읽으면 float·numpy 정수로 온다)
    assert nts.normalize_bno(123456789, check_digit=False) == "0123456789"
    assert nts.normalize_bno(1248100998.0) == "1248100998"
    assert nts.normalize_bno(pd.Series([1248100998]).iloc[0]) == "1248100998"
    assert nts.normalize_bno(float("nan")) is None and nts.normalize_bno(True) is None


def test_check_digit_accepts_real_numbers_and_rejects_single_digit_errors():
    for b in PUBLIC_BNOS:
        digits = b.replace("-", "")
        assert nts.check_digit_ok(digits), b
        wrong = digits[:9] + str((int(digits[9]) + 1) % 10)
        assert not nts.check_digit_ok(wrong), wrong


def test_mask_hides_middle_digits():
    assert nts.mask_bno("1248100998") == "124-**-***98"
    assert nts.mask_bno(None) == "***"


# ---------------------------------------------------------------------------
# B. 키 없음
# ---------------------------------------------------------------------------
def test_no_key_returns_unknown_without_network(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("키가 없는데 네트워크를 호출했다")
    monkeypatch.setattr(nts.requests, "post", _boom)
    df = nts.lookup_status(["124-81-00998", "abc"])
    assert list(df.columns) == nts.COLUMNS
    assert df["status"].tolist() == ["확인불가", "확인불가"]
    assert nts.KEY_ENV in df.loc[0, "error"] and nts.DATASET_ID in df.loc[0, "error"]
    assert "형식" in df.loc[1, "error"] and df.loc[1, "b_no"] is None


def test_env_key_is_used_via_header_not_url(monkeypatch):
    monkeypatch.setenv(nts.KEY_ENV, KEY)
    post = _Post(_ok_all())
    monkeypatch.setattr(nts.requests, "post", post)
    nts.lookup_status([PUBLIC_BNOS[0]])
    call = post.calls[0]
    assert call["url"] == nts.STATUS_URL and "serviceKey" not in call["url"] and KEY not in call["url"]
    assert call["headers"]["Authorization"] == f"Infuser {KEY}"
    assert call["json"] == {"b_no": ["1248100998"]}


def test_url_encoded_portal_key_is_decoded_for_header(monkeypatch):
    post = _Post(_ok_all())
    monkeypatch.setattr(nts.requests, "post", post)
    nts.lookup_status([PUBLIC_BNOS[0]], key="abc%2Bdef%3D%3D")
    assert post.calls[0]["headers"]["Authorization"] == "Infuser abc+def=="


# ---------------------------------------------------------------------------
# C. 분할·중복·순서
# ---------------------------------------------------------------------------
def test_batches_of_100_dedup_and_order(monkeypatch):
    uniq = [_valid(f"{100000000 + i:09d}") for i in range(250)]
    inputs = [*uniq, *uniq[:5]]                      # 뒤에 중복 5건
    post = _Post(_ok_all())
    monkeypatch.setattr(nts.requests, "post", post)
    df = nts.lookup_status(inputs, key=KEY)
    assert [len(c["json"]["b_no"]) for c in post.calls] == [100, 100, 50]
    sent = [b for c in post.calls for b in c["json"]["b_no"]]
    assert sent == uniq                              # 중복 없이, 입력 순서대로
    assert len(df) == len(inputs) and df["b_no"].tolist() == inputs
    assert (df["status"] == "계속").all() and df["error"].isna().all()


def test_invalid_rows_are_not_sent(monkeypatch):
    post = _Post(_ok_all())
    monkeypatch.setattr(nts.requests, "post", post)
    df = nts.lookup_status(["12-34", PUBLIC_BNOS[1]], key=KEY)
    assert post.calls[0]["json"]["b_no"] == ["2208162517"]
    assert df["status"].tolist() == ["확인불가", "계속"]


def test_checksum_mismatch_is_still_asked_and_flagged(monkeypatch):
    post = _Post(_ok_all("none"))
    monkeypatch.setattr(nts.requests, "post", post)
    df = nts.lookup_status(["124-81-00997"], key=KEY)
    assert post.calls[0]["json"]["b_no"] == ["1248100997"]
    assert df.loc[0, "status"] == "미등록" and not df.loc[0, "checksum_ok"]


# ---------------------------------------------------------------------------
# D. 응답·오류 매핑
# ---------------------------------------------------------------------------
def test_status_code_mapping(monkeypatch):
    b = [_valid(f"{200000000 + i:09d}") for i in range(6)]
    body = {"status_code": "OK", "request_cnt": 6, "match_cnt": 5,
            "data": [_item(b[0], "01"), _item(b[1], "02"), _item(b[2], "03", "20260812"),
                     _item(b[3], "none"), _item(b[4], "09")]}     # b[5] 은 응답에서 빠짐
    monkeypatch.setattr(nts.requests, "post", _Post(_Resp(200, body)))
    df = nts.lookup_status(b, key=KEY).set_index("b_no")
    assert df["status"].tolist() == ["계속", "휴업", "폐업", "미등록", "확인불가", "확인불가"]
    assert df.loc[b[2], "closed_on"] == pd.Timestamp("2026-08-12")
    assert pd.isna(df.loc[b[0], "closed_on"])
    assert df.loc[b[2], "status_code"] == "03" and df.loc[b[0], "tax_type"] == "부가가치세 일반과세자"
    assert df.loc[b[3], "tax_type"] == "" and df.loc[b[3], "error"] is None
    assert "알 수 없는 응답" in df.loc[b[4], "error"]
    assert "응답에 해당 번호가 없습니다" in df.loc[b[5], "error"]


def test_auth_error_is_explained_and_not_retried(monkeypatch):
    post = _Post(_Resp(401, {"code": -4, "msg": "등록되지 않은 인증키 입니다."}))
    monkeypatch.setattr(nts.requests, "post", post)
    df = nts.lookup_status([PUBLIC_BNOS[0]], key=KEY)
    assert len(post.calls) == 1
    assert df.loc[0, "status"] == "확인불가"
    assert "활용신청" in df.loc[0, "error"] and nts.DATASET_ID in df.loc[0, "error"]


def test_too_large_request_is_not_retried(monkeypatch):
    post = _Post(_Resp(413, {"status_code": "TOO_LARGE_REQUEST"}))
    monkeypatch.setattr(nts.requests, "post", post)
    df = nts.lookup_status([PUBLIC_BNOS[0]], key=KEY)
    assert len(post.calls) == 1 and "100개 초과" in df.loc[0, "error"]


def test_retry_on_429_then_success(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(nts, "_sleep", waits.append)
    post = _Post(_Resp(429, {"code": -10, "msg": "too many"}), _ok_all("03"))
    monkeypatch.setattr(nts.requests, "post", post)
    df = nts.lookup_status([PUBLIC_BNOS[0]], key=KEY)
    assert len(post.calls) == 2 and waits == [nts.BACKOFF_SEC]
    assert df.loc[0, "status"] == "폐업"


def test_server_errors_exhaust_retries(monkeypatch):
    post = _Post(_Resp(500, {"status_code": "INTERNAL_ERROR"}))
    monkeypatch.setattr(nts.requests, "post", post)
    df = nts.lookup_status([PUBLIC_BNOS[0]], key=KEY)
    assert len(post.calls) == nts.MAX_ATTEMPTS
    assert df.loc[0, "status"] == "확인불가" and "서버 오류" in df.loc[0, "error"]


def test_failed_batch_does_not_poison_the_next(monkeypatch):
    uniq = [_valid(f"{300000000 + i:09d}") for i in range(150)]
    fail = _Resp(503, None)
    post = _Post(*([fail] * nts.MAX_ATTEMPTS), _ok_all())
    monkeypatch.setattr(nts.requests, "post", post)
    df = nts.lookup_status(uniq, key=KEY)
    assert (df["status"].iloc[:100] == "확인불가").all()
    assert (df["status"].iloc[100:] == "계속").all()


def test_non_json_success_body_is_reported(monkeypatch):
    monkeypatch.setattr(nts.requests, "post", _Post(_Resp(200, None)))
    df = nts.lookup_status([PUBLIC_BNOS[0]], key=KEY)
    assert "응답 해석 실패" in df.loc[0, "error"]


# ---------------------------------------------------------------------------
# E. 비밀·개인정보가 새지 않는다
# ---------------------------------------------------------------------------
def test_key_and_full_numbers_never_leak(monkeypatch, caplog):
    leak = requests.ConnectionError(f"HTTPSConnectionPool: Authorization=Infuser {KEY} refused")
    post = _Post(leak)
    monkeypatch.setattr(nts.requests, "post", post)
    df = nts.lookup_status([PUBLIC_BNOS[0]], key=KEY)
    assert len(post.calls) == nts.MAX_ATTEMPTS
    assert "네트워크 오류" in df.loc[0, "error"]
    assert KEY not in df.loc[0, "error"] and "***KEY***" in df.loc[0, "error"]
    logged = caplog.text
    assert logged, "재시도·실패는 로그에 남아야 한다"
    assert KEY not in logged
    assert "1248100998" not in logged and "124-81-00998" not in logged
    assert "124-**-***98" in logged
