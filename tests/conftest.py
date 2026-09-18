"""pytest 진입 지원 — `pytest tests` 와 `python -m tests.test_sanity` 를 동일하게 만든다.

왜 필요한가
    `tests/test_sanity.py` 는 표준 assert 기반 독립 스크립트로 설계돼, 각 검증
    함수가 공유 컨텍스트 `ctx`(config + 격리용 임시 디렉터리)를 **인자로** 받는다.
    그런데 함수 이름이 `test_*` 라 pytest 가 이를 테스트로 수집하고, `ctx` 를
    fixture 로 해석해 찾지 못한 채 6건이 ERROR 로 떨어진다.

    실제로는 아무것도 깨지지 않았는데 `pytest tests` 는 "10 errors" 를 찍는다.
    **검사자가 관례대로 pytest 를 돌렸을 때 거짓 실패를 보는 것**은 그 자체로
    결함이므로, 스크립트를 고치는 대신 여기서 같은 컨텍스트를 fixture 로 제공한다.

격리 원칙은 스크립트와 동일하다
    ctx["tmp"] 아래에서만 쓰고, 실데이터(data/processed)가 변하지 않았음을 세션
    종료 시 지문(파일명·크기·mtime)으로 재확인한다. 스크립트의 main() 이 하던
    가드를 pytest 경로에서도 잃지 않기 위함이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common import load_config, set_seed
from tests.test_llm_paths import _REAL_POST, _tmp_cfg
from tests.test_sanity import ROOT, _fingerprint_dir


@pytest.fixture(scope="session")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """test_sanity 의 공유 컨텍스트. 세션 스코프 — 파이프라인 1회 실행을 재사용한다."""
    cfg = load_config()
    set_seed(cfg["seed"])
    return {"cfg": cfg, "tmp": tmp_path_factory.mktemp("franscore_sanity")}


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    """test_llm_paths 의 격리 config. 검증마다 새 임시 경로를 준다(스크립트와 동일)."""
    return _tmp_cfg(tmp_path / "llm")


@pytest.fixture(scope="session", autouse=True)
def _restore_http_post() -> object:
    """모의 전송으로 치환된 `src.llm._http_post` 를 세션 종료 시 실제 함수로 되돌린다."""
    yield
    from src import llm as src_llm
    src_llm._http_post = _REAL_POST


@pytest.fixture(scope="session", autouse=True)
def _processed_dir_guard() -> object:
    """테스트가 실데이터를 건드리지 않았음을 세션 종료 시 확인한다."""
    processed = ROOT / "data" / "processed"
    before = _fingerprint_dir(processed)
    yield
    after = _fingerprint_dir(processed)
    if before != after:
        b, a = {f[0]: f for f in before}, {f[0]: f for f in after}
        changed = sorted(set(b) ^ set(a) | {k for k in set(b) & set(a) if b[k] != a[k]})
        pytest.fail(f"data/processed 가 테스트 중 변경됐다: {changed}")
