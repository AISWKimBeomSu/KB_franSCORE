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
MENUS = ["FRANSCORE", "브랜드 탐색·비교", "일괄 조회", "점검 큐", "여신 포트폴리오", "등급 검증", "AI 상담",
         "서비스 소개"]


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


def test_validation_demo_runs_end_to_end():
    """등급 검증 — 가상 자료 시연이 판정·표·반출까지 예외 없이 돌고, 가상임을 밝힌다."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["nav_view"] = "등급 검증"
    at.run()
    next(b for b in at.button if b.label == "가상 자료로 시연").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("가상 자료입니다" in w.value for w in at.warning)
    assert any("판정" in m.value for m in at.markdown)


def test_changed_code_purges_stale_modules_in_a_fresh_process():
    """배포 뒤 옛 모듈이 남는 문제 — 코드 서명이 바뀌면 진입 스크립트가 src.* 를 비운다.

    테스트 프로세스에서는 이 동작을 끄므로(conftest) 별도 프로세스에서 확인한다.
    """
    import subprocess
    import sys
    import textwrap
    script = textwrap.dedent(f"""
        import importlib.util, sys
        sys.path.insert(0, {str(ROOT)!r})
        import src.grading as old
        sys._franscore_code_sig = -1.0                      # 예전 코드 서명
        spec = importlib.util.spec_from_file_location("franscore_app", {APP!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                        # 진입 스크립트 최상단 실행 (main 은 안 부름)
        assert sys.modules.get("src.grading") is not old, "옛 모듈이 그대로 남았다"
        assert sys._franscore_code_sig != -1.0
        import src.grading as new
        assert new is not old
        print("purged")
    """)
    env = {k: v for k, v in os.environ.items() if k != "FRANSCORE_NO_MODULE_PURGE"}
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and "purged" in r.stdout, r.stderr[-2000:]


@pytest.mark.skipif(not (ROOT / "outputs" / "localdata_signal.csv").exists(), reason="월별 신호표 없음")
def test_queue_collects_monthly_signal_brands_regardless_of_grade():
    """점검 큐 — 월별 폐점 신호가 '악화'·'확인 필요'인 브랜드는 등급이 '안정'이어도 큐에 오르고,
    한 번에 모아 볼 수 있다 (공시를 1~2년 기다리지 않는 조기경보)."""
    import pandas as pd
    sig = pd.read_csv(ROOT / "outputs" / "localdata_signal.csv", encoding="utf-8-sig")
    scored = set(pd.read_csv(ROOT / "outputs" / "scores_latest.csv", encoding="utf-8-sig")["brand_id"].astype(str))
    flagged = sig[sig["trend"].isin(["악화", "확인 필요"]) & sig["brand_id"].astype(str).isin(scored)]
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["nav_view"] = "점검 큐"
    at.run()
    box = next(c for c in at.checkbox if c.label.startswith("월별 폐점 신호만"))
    assert f"({len(flagged):,}건" in box.label
    box.check().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any(f"조건에 맞는 **{len(flagged):,}건**" in c.value for c in at.caption)
    assert any("월별 인허가 신호" in m.value for m in at.markdown)
