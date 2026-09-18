"""화면 스모크 — 모든 메뉴가 **API 키 없이** 예외 없이 뜨는지.

배포 직후 한 화면이 통째로 죽는 사고(모듈 버전 불일치, 없는 컬럼 참조)는 단위 테스트로는
안 잡힌다. Streamlit 의 AppTest 로 실제 진입 스크립트를 돌려 화면마다 예외가 없는지 본다.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "src" / "app.py")
MENUS = ["FRANSCORE", "일괄 조회", "점검 큐", "여신 포트폴리오", "AI 상담", "서비스 소개"]


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("GEMINI_API_KEY", "NAVER_", "NCP_", "DART_API_KEY")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FRANSCORE_QUEUE_STORE", "session")     # 테스트가 공유 파일을 쓰지 않게


@pytest.mark.parametrize("menu", MENUS)
def test_every_screen_renders_without_exception(menu):
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["nav_view"] = menu
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_brand_deep_link_opens_detail():
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["brand"] = "BRD_20090100502"
    at.run()
    assert not at.exception
    assert any("빽다방" in m.value for m in at.markdown)
