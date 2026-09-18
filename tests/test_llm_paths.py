"""LLM 층 실동작 검증 — 모의 HTTP 전송으로 **실제 LLM 코드 경로**를 통과시킨다.

왜 필요한가 (정직성):
    개발 환경에 LLM 키가 없으면 news_llm/memo_llm은 규칙기반 폴백으로만 동작한다.
    폴백만 테스트하면 "LLM을 붙였다"는 주장이 검증되지 않는다(허수아비).
    본 테스트는 `src.llm._http_post` (HTTP 경계) 하나만 모의로 치환해 **키 없이도
    LLM 분기 전체** — 요청 본문 구성 → responseSchema 변환 → 사고(thought) 파트 제거
    → 응답 검증 → 안전차단/절단/네트워크오류 폴백 → 재시도 — 를 실행·검증한다.

    HTTP 경계에서만 치환하므로 우리가 작성한 코드는 한 줄도 우회되지 않는다.

실행: python -m tests.test_llm_paths      (종료코드 0 = 전부 통과)
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 기본 콘솔(cp949)에서 한글·기호 출력이 깨지지 않도록
with contextlib.suppress(AttributeError, OSError):  # 리다이렉트된 스트림 등
    sys.stdout.reconfigure(encoding="utf-8")

from src import llm as src_llm
from src.common import load_config

PASS, FAIL = "PASS", "FAIL"

_REAL_POST = src_llm._http_post


# ---------------------------------------------------------------------------
# 모의 HTTP 전송 (Gemini generateContent 응답 형태)
# ---------------------------------------------------------------------------

class _MockResp:
    """requests.Response 최소 대체 (status_code / text / json())."""

    def __init__(self, payload: dict | None, status_code: int = 200, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {}, ensure_ascii=False)

    def json(self):
        if self._payload is None:
            raise ValueError("응답이 JSON이 아님")
        return self._payload


def _ok(text: str, *, thought: str | None = None, finish: str = "STOP") -> _MockResp:
    """정상 응답. thought 를 주면 사고(thinking) 파트를 함께 실어 보낸다."""
    parts: list[dict] = []
    if thought is not None:
        parts.append({"text": thought, "thought": True})
    parts.append({"text": text, "thoughtSignature": "mock-sig"})
    return _MockResp({
        "candidates": [{"content": {"parts": parts}, "finishReason": finish, "index": 0}],
        "modelVersion": "mock-gemini",
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 22,
                          "thoughtsTokenCount": 33},
    })


def _blocked(reason: str = "SAFETY") -> _MockResp:
    """안전정책 차단 응답 (응답 단계 차단)."""
    return _MockResp({
        "candidates": [{"content": {"parts": []}, "finishReason": reason, "index": 0}],
        "modelVersion": "mock-gemini",
    })


class MockTransport:
    """호출 순서대로 스크립트된 응답을 돌려주는 모의 전송.

    항목이 Exception 이면 raise, _MockResp 면 반환. 스크립트가 소진되면 마지막 항목 반복.
    """

    calls: ClassVar[list[dict]] = []
    script: ClassVar[list] = []

    @classmethod
    def install(cls, *responses) -> None:
        cls.calls = []
        cls.script = list(responses)
        src_llm._http_post = cls._post

    @classmethod
    def _post(cls, url: str, headers: dict, body: dict, timeout: float):
        cls.calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        item = cls.script[min(len(cls.calls) - 1, len(cls.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    @classmethod
    def last_body(cls) -> dict:
        assert cls.calls, "LLM 호출이 일어나지 않음 (LLM 경로 미실행)"
        return cls.calls[-1]["body"]


def _has_key_anywhere(obj, key: str) -> bool:
    """중첩 dict/list 어디에든 특정 키가 있는지 (스키마 정제 검증용)."""
    if isinstance(obj, dict):
        return key in obj or any(_has_key_anywhere(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_key_anywhere(v, key) for v in obj)
    return False


def _tmp_cfg(tmp: Path) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["paths"] = {"raw": tmp / "raw", "processed": tmp / "processed",
                    "outputs": tmp / "outputs", "demo_outputs": tmp / "outputs" / "_smoke"}
    for p in cfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["raw"]) / "news").mkdir(parents=True, exist_ok=True)
    cfg["llm"]["retry_backoff_sec"] = 0.0   # 재시도 경로를 지연 없이 검증
    return cfg


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------


# ⚠️ 예비 키(GEMINI_API_KEY_2 … _5)가 도입되면서 테스트가 환경에 오염되기 시작했다.
#    주 키만 지우고 예비 키를 남겨 두면 "키 없음" 검사가 통과해 버리고, 재시도 횟수도
#    키 개수만큼 곱해진다(실측: 4회 기대 → 12회). 테스트는 **모든 키 슬롯**을 격리한다.
class _IsolatedKeys:
    """테스트 동안 Gemini 키 환경변수를 전부 치우고, 끝나면 원래대로 돌린다."""

    def __init__(self, cfg: dict, value: str | None = None):
        self.envs = src_llm.api_key_envs(cfg)
        self.value = value
        self.saved: dict[str, str] = {}

    def __enter__(self):
        for e in self.envs:
            if e in os.environ:
                self.saved[e] = os.environ.pop(e)
        if self.value is not None:
            os.environ[self.envs[0]] = self.value
        return self

    def __exit__(self, *exc):
        for e in self.envs:
            os.environ.pop(e, None)
        os.environ.update(self.saved)
        return False

def test_llm_client(cfg: dict) -> None:
    """src.llm 클라이언트 자체: 스키마 정제·사고파트 제거·차단/절단/재시도 분기."""
    env = src_llm.api_key_env(cfg)
    with _IsolatedKeys(cfg, "mock-key-for-test"):
        return _test_llm_client_body(cfg, env)


def _test_llm_client_body(cfg: dict, env: str) -> str:
    # (0) JSON Schema → Gemini responseSchema 정제
    gschema = src_llm.to_gemini_schema({
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    })
    assert "additionalProperties" not in gschema, "미지원 키워드가 제거되지 않음 (400 유발)"
    assert gschema["propertyOrdering"] == ["a", "b"], "필드 순서 고정(propertyOrdering) 누락"
    print("    (0) 스키마 정제: additionalProperties 제거 + propertyOrdering 부여 확인")

    # (1) 사고(thought) 파트가 결과 텍스트에 새어나오지 않아야 한다
    MockTransport.install(_ok("최종답변입니다", thought="이건 내부추론이라 노출되면 안 됨"))
    text, meta = src_llm.generate(cfg, system="sys", user="usr")
    assert text == "최종답변입니다", f"사고 파트가 섞여 들어옴: {text!r}"
    assert "내부추론" not in text, "모델 내부 추론이 사용자 텍스트로 노출됨"
    assert meta["model"] == "mock-gemini" and meta["usage"]["candidatesTokenCount"] == 22
    body = MockTransport.last_body()
    assert body["systemInstruction"]["parts"][0]["text"] == "sys", "systemInstruction 미전달"
    assert body["generationConfig"]["temperature"] == 0.0, "temperature 미전달"
    assert body["safetySettings"], "safetySettings 미전달 (부정 기업뉴스 오탐 차단 방지)"
    assert MockTransport.calls[-1]["headers"]["X-goog-api-key"], "API 키 헤더 미전달"
    assert MockTransport.calls[-1]["url"].endswith(
        f"/models/{cfg['llm']['model']}:generateContent"), "모델 엔드포인트 불일치"
    print("    (1) 사고파트 제거 + 요청 본문(시스템·temperature·안전설정·키헤더) 확인")

    # (2) 안전정책 차단 → LLMRefused (재시도하지 않음)
    MockTransport.install(_blocked("SAFETY"))
    try:
        src_llm.generate(cfg, system="s", user="u")
        raise AssertionError("차단 응답인데 예외가 발생하지 않음")
    except src_llm.LLMRefused:
        pass
    assert len(MockTransport.calls) == 1, "차단은 재시도 무의미한데 재시도가 일어남"
    print("    (2) 안전차단(SAFETY): LLMRefused 즉시 발생, 재시도 없음 확인")

    # (3) 토큰 한도 절단 → LLMError (잘린 응답을 그대로 쓰지 않음)
    MockTransport.install(_ok('{"signals": [{"artic', finish="MAX_TOKENS"))
    try:
        src_llm.generate(cfg, system="s", user="u")
        raise AssertionError("절단 응답인데 예외가 발생하지 않음")
    except src_llm.LLMFatal as exc:
        assert "절단" in str(exc), f"절단 사유가 메시지에 없음: {exc}"
    assert len(MockTransport.calls) == 1, \
        f"같은 예산이면 또 잘리는데 재시도가 일어남 ({len(MockTransport.calls)}회 — 쿼터 낭비)"
    print("    (3) 토큰 절단(MAX_TOKENS): 잘린 응답 거부 + 무의미한 재시도 없음 확인")

    # (4) 일시 오류 → 지수 백오프 재시도 후 성공
    MockTransport.install(requests.RequestException("일시 네트워크 오류"), _ok("복구됨"))
    text4, meta4 = src_llm.generate(cfg, system="s", user="u")
    assert text4 == "복구됨" and meta4["attempts"] == 2, f"재시도 경로 실패: {meta4}"
    print("    (4) 일시 오류: 재시도 후 성공 확인 (attempts=2)")

    # (5) 키 미설정 → 즉시 LLMError (호출 시도 자체를 하지 않음)
    _iso_none = _IsolatedKeys(cfg, None)
    _iso_none.__enter__()
    try:
        MockTransport.install(_ok("불려선 안 됨"))
        try:
            src_llm.generate(cfg, system="s", user="u")
            raise AssertionError("키가 없는데 예외가 발생하지 않음")
        except src_llm.LLMError:
            pass
        assert not MockTransport.calls, "키가 없는데 HTTP 호출을 시도함"
    finally:
        _iso_none.__exit__(None, None, None)
    print("    (5) 키 미설정: 호출 시도 없이 즉시 예외 확인")


def test_news_llm_path(cfg: dict) -> None:
    """news_llm의 LLM 분기: 구조화 요청 구성·파싱·검증·차단/예외 폴백."""
    from src import news_llm

    articles = {"테스트브랜드": [
        {"title": "테스트브랜드 가맹점주協 본사와 분쟁 격화", "link": "http://a",
         "published": "Mon, 01 Jul 2024 00:00:00 GMT", "source": "테스트일보"},
        {"title": "테스트브랜드 운영사 자금난 심화", "link": "http://b",
         "published": "Tue, 02 Jul 2024 00:00:00 GMT", "source": "테스트경제"},
    ]}
    _iso = _IsolatedKeys(cfg, "mock-key-for-test")
    _iso.__enter__()
    try:
        # (1) 정상 응답 — LLM 경로가 실제로 사용되고 신호가 파싱되는지
        payload = json.dumps({"signals": [
            {"article_index": 0, "event_type": "본부분쟁",
             "evidence_sentence": "가맹점주協 본사와 분쟁 격화", "confidence": "상"},
            {"article_index": 1, "event_type": "재무이슈",
             "evidence_sentence": "운영사 자금난 심화", "confidence": "중"},
        ]}, ensure_ascii=False)
        MockTransport.install(_ok(payload, thought="분류 근거를 따져보는 중"))
        sigs = news_llm.extract_signals(articles, cfg)
        assert sigs, "LLM 경로에서 신호가 반환되지 않음"
        body = MockTransport.last_body()
        gcfg = body["generationConfig"]
        assert gcfg.get("responseMimeType") == "application/json", "구조화 출력 미요청"
        assert not _has_key_anywhere(gcfg["responseSchema"], "additionalProperties"), \
            "responseSchema 에 미지원 키워드가 남아 있음 (실호출 시 400)"
        assert "지시문" in body["systemInstruction"]["parts"][0]["text"], \
            "프롬프트 인젝션 방어 지시가 시스템 프롬프트에 없음"
        assert any(s.get("llm_used") for s in sigs), "llm_used 플래그가 True로 표기되지 않음"
        types_found = {s.get("event_type") for s in sigs}
        assert "본부분쟁" in types_found, f"사건유형 파싱 실패: {types_found}"
        print(f"    (1) 정상 응답: 신호 {len(sigs)}건, 유형 {sorted(types_found)}, llm_used=True")

        # (2) 범위 밖 인덱스·미지 enum → 코드가 다시 검증해 걸러내는지
        bad = json.dumps({"signals": [
            {"article_index": 99, "event_type": "본부분쟁", "evidence_sentence": "x", "confidence": "상"},
            {"article_index": 0, "event_type": "존재하지않는유형",
             "evidence_sentence": "가맹점주協 본사와 분쟁 격화", "confidence": "최상"},
        ]}, ensure_ascii=False)
        MockTransport.install(_ok(bad))
        sigs2 = news_llm.extract_signals(articles, cfg)
        assert len(sigs2) == 1, f"범위 밖 인덱스가 걸러지지 않음: {len(sigs2)}건"
        assert sigs2[0]["event_type"] == "기타" and sigs2[0]["confidence"] == "하", \
            f"미지 enum 값이 안전값으로 치환되지 않음: {sigs2[0]}"
        print("    (2) 모델 출력 재검증: 범위밖 인덱스 제거 + 미지 enum 안전값 치환 확인")

        # (3) 안전차단 → 폴백으로 안전 전환
        MockTransport.install(_blocked("SAFETY"))
        sigs_r = news_llm.extract_signals(articles, cfg)
        assert sigs_r, "차단 시 폴백이 신호를 반환하지 않음"
        assert all(not s.get("llm_used") for s in sigs_r), "차단인데 llm_used=True로 표기됨"
        print("    (3) 안전차단 처리: 규칙 폴백 전환 확인 (llm_used=False)")

        # (4) 네트워크 예외(재시도 소진) → 폴백
        MockTransport.install(requests.RequestException("모의 네트워크 장애"))
        sigs_e = news_llm.extract_signals(articles, cfg)
        assert sigs_e, "예외 발생 시 폴백이 신호를 반환하지 않음"
        assert all(not s.get("llm_used") for s in sigs_e), "예외인데 llm_used=True"
        assert len(MockTransport.calls) == int(cfg["llm"]["max_retries"]), \
            f"재시도 횟수 불일치: {len(MockTransport.calls)}"
        print(f"    (4) 네트워크 예외: {len(MockTransport.calls)}회 재시도 후 폴백 전환 확인")

        # (5) 잘못된 JSON → 폴백 (파서 견고성)
        MockTransport.install(_ok("이것은 JSON이 아님"))
        sigs_b = news_llm.extract_signals(articles, cfg)
        assert sigs_b, "JSON 파싱 실패 시 폴백이 동작하지 않음"
        assert all(not s.get("llm_used") for s in sigs_b), "파싱 실패인데 llm_used=True"
        print("    (5) 비JSON 응답 처리: 폴백 전환 확인")
    finally:
        _iso.__exit__(None, None, None)


def test_memo_llm_path(cfg: dict) -> None:
    """memo_llm의 LLM 분기: 프롬프트 구성·필수 고지 강제·차단/예외 폴백."""
    from src import memo_llm

    context = {
        "brand_name": "테스트브랜드", "grade": "High", "prob": 0.31,
        "shap_top": [{"feature": "f_chg_contract_end_rate", "feature_kr": "계약종료율",
                      "shap_value": 0.05, "feature_value": 0.2}],
        "panel_metrics": {"가맹점 수": 57, "평균매출(천원)": 115552},
        "news": [{"event_type": "본부분쟁", "evidence_sentence": "분쟁 격화",
                  "source_url": "http://a", "published": "2024-07-01"}],
        "portfolio": {"exposure_ekw": 120.5},
    }
    _iso = _IsolatedKeys(cfg, "mock-key-for-test")
    _iso.__enter__()
    try:
        # (1) 정상 — LLM 텍스트가 사용되고 고지문이 보장되는지
        MockTransport.install(_ok("## 심사메모\n브랜드 위험도가 높습니다.",
                                  thought="내부추론: 등급 High이므로..."))
        memo = memo_llm.generate_memo(context, cfg)
        body = MockTransport.last_body()
        system_text = body["systemInstruction"]["parts"][0]["text"]
        assert system_text, "시스템 프롬프트 미전달(근거 제한 불가)"
        assert "responseSchema" not in body["generationConfig"], \
            "메모는 자유 마크다운인데 구조화 스키마가 걸림"
        assert memo_llm.DISCLAIMER in memo, "필수 고지문이 메모에 없음"
        assert "브랜드 위험도가 높습니다" in memo, "LLM 응답 텍스트가 반영되지 않음"
        assert "내부추론" not in memo, "모델 내부 추론이 메모에 노출됨"
        assert "LLM 생성" in memo and "mock-gemini" in memo, "LLM 사용·모델 각주가 없음"
        assert "지시문" in system_text or "명령" in system_text, \
            "뉴스 제목 프롬프트 인젝션 방어 지시가 시스템 프롬프트에 없음"
        print(f"    (1) 정상 응답: LLM 텍스트 반영 + 사고파트 차단 + 고지문 강제 "
              f"+ 인젝션 방어 지시 확인 ({len(memo)}자)")

        # (2) 고지문 누락 응답 → 강제 부착
        MockTransport.install(_ok("고지문 없는 응답"))
        memo2 = memo_llm.generate_memo(context, cfg)
        assert memo_llm.DISCLAIMER in memo2, "고지문 누락 시 강제 부착이 동작하지 않음"
        print("    (2) 고지문 누락 응답: 안전장치로 강제 부착 확인")

        # (3) 안전차단 → 결정적 템플릿 폴백
        MockTransport.install(_blocked("PROHIBITED_CONTENT"))
        memo3 = memo_llm.generate_memo(context, cfg)
        assert "LLM 미사용 폴백" in memo3, "차단 시 폴백 템플릿이 사용되지 않음"
        print("    (3) 안전차단 처리: 결정적 템플릿 폴백 확인")

        # (4) HTTP 400(설정 오류) → 재시도 없이 폴백
        MockTransport.install(_MockResp(None, status_code=400, text="Invalid JSON payload"))
        memo4 = memo_llm.generate_memo(context, cfg)
        assert "LLM 미사용 폴백" in memo4, "HTTP 400 시 폴백이 동작하지 않음"
        assert len(MockTransport.calls) == 1, "재시도 무의미한 400에서 재시도가 일어남"
        print("    (4) HTTP 400 처리: 재시도 없이 즉시 폴백 확인")

        # (5) 폴백 결정성 — 같은 입력 → 같은 출력
        a = memo_llm.generate_memo(context, cfg, force_fallback=True)
        b = memo_llm.generate_memo(context, cfg, force_fallback=True)
        assert a == b, "폴백 메모가 결정적이지 않음 (재현성 위반)"
        print("    (5) 폴백 결정성: 동일 입력 → 동일 출력 확인")

        # (6) **환각 검증** — 입력에 없는 수치를 지어내면 잡아내는가
        #     프롬프트로 지시만 하고 검증하지 않으면 보증이 아니다.
        clean = memo_llm.check_faithfulness(
            "가맹점 수 57개, 평균매출 115,552천원, 확률 31.0%", context)
        assert clean["ok"], f"입력에 있는 수치인데 환각으로 오탐: {clean['unsupported']}"
        halluc = memo_llm.check_faithfulness(
            "가맹점 수 57개이며 부채비율은 412.7%, 영업이익률 -18.3%로 악화됐다", context)
        assert not halluc["ok"], "입력에 없는 수치(412.7·18.3)를 잡아내지 못함"
        assert "412.7" in halluc["unsupported"], f"미검출: {halluc['unsupported']}"
        MockTransport.install(_ok("부채비율 412.7%로 급등했습니다."))
        memo6 = memo_llm.generate_memo(context, cfg)
        assert "자동 근거 검증 경고" in memo6, "환각 수치가 경고 없이 메모에 실림"
        print(f"    (6) 환각 검증: 정상 통과 + 지어낸 수치 {halluc['unsupported']} 검출·경고 부착 확인")
    finally:
        _iso.__exit__(None, None, None)


def test_rag_retrieval(cfg: dict) -> None:
    """RAG: 코퍼스 구축 → 색인 → 검색이 실제로 관련 문서를 회수하는지."""
    import pandas as pd

    from src import rag

    # 최소 코퍼스 (뉴스 스냅샷 형식)
    news_dir = Path(cfg["paths"]["raw"]) / "news"
    (news_dir / "테스트브랜드.json").write_text(json.dumps({
        "brand": "테스트브랜드",
        "articles": [
            {"title": "테스트브랜드 가맹점 계약종료 급증", "link": "http://x",
             "published": "2024-07-01", "source": "테스트일보"},
            {"title": "테스트브랜드 본사 자금난", "link": "http://y",
             "published": "2024-07-02", "source": "테스트경제"},
        ]}, ensure_ascii=False), encoding="utf-8")
    # 최소 패널 (공시 문서 원천)
    pd.DataFrame([{
        "brand_id": "B1", "brand_name": "테스트브랜드", "industry_major": "외식",
        "industry_mid": "한식", "year": 2023, "n_stores": 57.0, "n_direct": 1.0,
        "avg_sales": 115552.0, "n_contract_end": 5.0, "n_contract_cancel": 2.0,
        "n_new": 3.0, "n_regions": 8.0, "region_hhi": 0.21, "cancel_flag": 0,
        "cancel_type": None, "eligible_t": True,
    }]).to_parquet(Path(cfg["paths"]["processed"]) / "panel.parquet", index=False)

    corpus = rag.build_corpus(cfg)
    assert not corpus.empty, "RAG 코퍼스가 비어 있음"
    assert set(corpus["source_type"]) >= {"news", "disclosure"}, \
        f"코퍼스에 두 문서 유형이 모두 없음: {set(corpus['source_type'])}"
    index = rag.build_index(cfg, corpus)

    hits = index.retrieve("테스트브랜드 계약종료 급증", k=3)
    assert not hits.empty, "검색 결과가 비어 있음"
    assert any("계약종료" in str(t) for t in hits["text"]), \
        f"관련 문서를 회수하지 못함: {hits['text'].tolist()}"

    ev = rag.retrieve_evidence(cfg, "테스트브랜드", ["계약종료율"], index=index)
    assert ev, "retrieve_evidence 결과 없음"
    assert all({"doc_id", "url", "published", "score"} <= set(e) for e in ev), \
        "근거 항목에 출처 추적 필드가 없음"
    print(f"    코퍼스 {len(corpus)}문서 (뉴스+공시), 검색 상위 {len(hits)}건, "
          f"근거 {len(ev)}건 (출처·발행일·유사도 포함)")

    # 개체 매칭 필터: 무관 기사가 코퍼스에 들어오지 않는지
    (news_dir / "속속브랜드.json").write_text(json.dumps({
        "brand": "속속브랜드",
        "articles": [{"title": "세계 곳곳서 사찰 속속 재건", "link": "http://z",
                      "published": "2023-06-29", "source": "무관일보"}]},
        ensure_ascii=False), encoding="utf-8")
    corpus2 = rag.build_corpus(cfg)
    bad = corpus2[(corpus2["source_type"] == "news") & (corpus2["text"].str.contains("사찰"))]
    assert bad.empty, "개체 매칭 필터가 무관 기사를 걸러내지 못함"
    print("    개체 매칭 필터: 무관 기사 코퍼스 유입 차단 확인")

    # 색인 파일이 **진입점에 독립적**인지 (실측 결함 회귀 방지):
    # 인스턴스를 그대로 피클하면 클래스 경로가 박히고, `python -m src.rag` 로 만든 색인은
    # 경로가 `__main__.RagIndex` 가 되어 대시보드·다른 모듈에서 로드가 조용히 실패했다.
    idx_path = Path(cfg["paths"]["outputs"]) / rag.INDEX_FILE
    assert idx_path.exists(), "RAG 색인 파일이 저장되지 않음"
    raw = idx_path.read_bytes()
    assert b"__main__" not in raw, (
        "색인 파일에 '__main__' 참조가 있음 — 만든 진입점에서만 로드된다(대시보드에서 실패). "
        "객체가 아니라 state()를 저장해야 한다.")
    assert b"RagIndex" not in raw, "색인 파일에 우리 클래스 참조가 박혀 있음 (state 저장 아님)"
    loaded = rag.load_index(cfg)
    assert loaded is not None, "저장된 색인을 다시 로드하지 못함"
    hits2 = loaded.retrieve("테스트브랜드 계약종료 급증", k=3)
    assert not hits2.empty, "재로드한 색인으로 검색이 되지 않음"
    print("    색인 이식성: 클래스 참조 없이 저장 → 재로드·검색 정상 확인")


def main() -> int:
    print("=== FranSCORE LLM/RAG 경로 검증 (모의 HTTP — 실키 불필요) ===")
    tests = [("llm_client", test_llm_client),
             ("news_llm_path", test_news_llm_path),
             ("memo_llm_path", test_memo_llm_path),
             ("rag_retrieval", test_rag_retrieval)]
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="franscore_llm_") as td:
            for i, (name, fn) in enumerate(tests, 1):
                cfg = _tmp_cfg(Path(td) / name)
                print(f"[{i}/{len(tests)}] {name} ...")
                try:
                    fn(cfg)
                    results.append((name, PASS))
                    print(f"[{i}/{len(tests)}] {name} ... {PASS}")
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    results.append((name, f"{FAIL} ({exc})"))
                    print(f"[{i}/{len(tests)}] {name} ... {FAIL}")
    finally:
        src_llm._http_post = _REAL_POST  # 실제 전송 함수 복원

    ok = sum(1 for _, r in results if r == PASS)
    print(f"\n=== SUMMARY: {ok}/{len(results)} passed ===")
    for n, r in results:
        print(f"  {PASS:5s} {n}" if r == PASS else f"  FAIL  {n}: {r}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
