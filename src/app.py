"""FranSCORE — 프랜차이즈 여신 리스크 관리 서비스.

실행: streamlit run src/app.py  (프로젝트 루트에서)

이 파일은 **셸**이다. 테마를 깔고, 좌측 내비게이션을 그리고, 선택된 화면 모듈을
호출한다. 화면 내용은 src/views/ 아래에 있다.

화면 구성 원칙 — 이것은 심사역이 매일 쓰는 서비스 화면이지 개발 결과 보고서가 아니다.
    · 산출물 파일 존재 여부 체크리스트, 하이퍼파라미터 JSON, 백테스트 지표표,
      "--step 무엇을 실행하세요" 안내문은 이 화면에 두지 않는다.
      그 근거들은 개발 명세(docs/)와 저장소 산출물에 있고, 심사 제출은 그쪽으로 한다.
    · 대신 데이터가 없으면 사용자 언어로 "아직 평가 결과가 없습니다"라고만 말한다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _fresh_modules() -> None:
    """배포 뒤 옛 모듈이 섞여 도는 것을 막는다 — 코드가 바뀌었으면 src.* 모듈을 비운다.

    실측(2026-09-19): 푸시 뒤 Streamlit Cloud 는 진입 스크립트(이 파일)는 새로 읽었지만
    이미 import 된 모듈은 **일부만** 새로 읽었다. 목록 머리글(franscore.py)은 새 판인데 같은
    화면이 쓰는 공통 모듈(common.py)은 옛 판이라 '공시 공백 보정' 안내가 빠졌고, 일괄 조회는
    사업자번호 열이 없는 옛 화면이었다. 화면마다 버전이 섞이면 수치가 서로 어긋난다.

    그래서 src/**/*.py 의 최종 수정 시각을 코드 서명으로 삼아, 서명이 바뀌면 src.* 를
    sys.modules 에서 지워 아래 import 가 새 코드를 읽게 한다. 프로세스가 처음 이 코드를
    실행할 때도 한 번 비운다 — 이 코드가 배포되기 전부터 떠 있던 프로세스를 바로잡기 위해서다.
    이미 실행 중인 다른 세션은 쥐고 있던 옛 모듈 객체로 그 실행을 마치고, 다음 실행부터 새 코드를 쓴다.

    FRANSCORE_NO_MODULE_PURGE 가 켜져 있으면 건너뛴다 — 테스트는 모듈 객체에 가짜 전송·
    monkeypatch 를 걸어 두는데, 화면 스모크가 모듈을 비우면 그 설정이 사라진다(tests/conftest.py).
    """
    if os.environ.get("FRANSCORE_NO_MODULE_PURGE"):
        return
    try:
        sig = max(p.stat().st_mtime for p in (_ROOT / "src").rglob("*.py"))
    except (OSError, ValueError):
        return
    if getattr(sys, "_franscore_code_sig", None) == sig:
        return
    for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
        del sys.modules[name]
    sys._franscore_code_sig = sig


_fresh_modules()

import streamlit as st

from src import theme
from src.common import load_config, set_seed

MENU = {
    # '한눈에 보기'와 '브랜드 조회'는 같은 일의 두 단계여서 하나로 합쳤다 —
    # 목록에서 고르고, 고른 것을 자세히 본다. 화면을 오가며 이름을 다시 칠 이유가 없다.
    "FRANSCORE": ("franscore", "브랜드 리스크 현황과 상세 진단"),
    # 조건으로 거르고 후보를 나란히 놓는 일 — 신규 협약·신규 취급 검토의 출발점이다.
    "브랜드 탐색·비교": ("explore", "조건 검색과 후보 비교"),
    # 심사역의 하루는 브랜드 하나가 아니라 신청 목록으로 시작한다 — 목록째 조회한다.
    "일괄 조회": ("batch", "신청 목록 한 번에 진단"),
    "점검 큐": ("queue", "담당자 배정과 확인 결과 기록"),
    "여신 포트폴리오": ("portfolio", "쏠림·예상손실 실시간 점검"),
    # 등급의 라벨은 연체가 아니다 — 은행 연체 자료로 검증을 통과해야 심사 보조지표가 된다.
    "등급 검증": ("validate", "은행 연체 자료로 등급 검증"),
    "AI 상담": ("assistant", "자연어 질의와 근거 인용"),
    "서비스 소개": ("about", "무엇을 어떻게 평가하는가"),
}

DISCLAIMER = ("본 서비스의 지표는 2선 리스크 관리 참고용이며 자동 여신 결정에 "
              "사용되지 않습니다.")
# 공개 배포(누구나 접속)에서만 덧붙이는 고지. 모형 사용 명세(MODEL_USE_SPEC §3)는 실명 브랜드
# 등급의 대외 공표를 금지한다 — 공개 데모는 방법론 시연이지 브랜드 평가의 공표가 아니라는 점을
# 화면에서 분명히 한다.
PUBLIC_NOTE = ("공개 데모 — 모형 사용 명세 §3(실명 브랜드 등급의 대외 공표 금지)에 따라 브랜드를 "
               "가명(예: 치킨 017)으로 표시합니다. 수치는 실제 공시 그대로입니다. KB국민은행의 공식 "
               "서비스가 아닌 개인 연구 프로젝트이며, 등급을 인용·배포하지 마십시오.")


def _is_public_demo() -> bool:
    from src.public import is_public
    return is_public()


_NAV = "nav_view"


def _year_text(year) -> str:
    """'2024년 실적 (2025년 공시)' — 패널 연도는 실적연도다. src.grading 의 새 함수를 부르지 않고
    여기서 만든다: 진입 스크립트가 옛 버전 모듈과 섞여 떠도 사이드바가 죽지 않게 한다."""
    try:
        y = int(float(year))
    except (TypeError, ValueError):
        return "-"
    return f"{y}년 실적 ({y + 1}년 공시)"


def _sidebar() -> str:
    # 공개 배포에서는 은행 로고처럼 읽히는 표식을 쓰지 않는다 — 공식 서비스로 오인되지 않게
    theme.sidebar_brand("FranSCORE", "프랜차이즈 여신 리스크", mark="FS" if _is_public_demo() else "KB")
    theme.sidebar_label("메뉴")

    # ⚠️ 라디오가 아니라 버튼을 쓴다. Streamlit 라디오는 **이미 선택된 항목을 다시
    #    눌러도 값이 안 바뀌어 아무 일도 일어나지 않는다** — 브랜드 상세를 보다가
    #    사이드바의 FRANSCORE 를 눌러도 목록으로 못 돌아갔다(실측 결함).
    #    버튼은 누를 때마다 실행되므로 '같은 메뉴 다시 누르기 = 그 화면 처음으로'가 된다.
    current = st.session_state.setdefault(_NAV, next(iter(MENU)))
    for label in MENU:
        if st.sidebar.button(label, key=f"nav_{MENU[label][0]}",
                             width="stretch",
                             type="primary" if label == current else "secondary"):
            st.session_state[_NAV] = label
            st.session_state["fs_selected"] = None
            st.rerun()
    view = st.session_state[_NAV]

    from src.views import common as C
    _, meta = C.load_scores()
    # ⚠️ 여기서 theme 의 **색·크기 상수를 직접 꺼내 쓰지 않는다.** 클래스 이름만 적고
    #    모양은 theme 의 CSS 가 정한다.
    #    이유: 이 파일은 진입 스크립트라 배포 때마다 항상 새로 실행되는데, import 된
    #    src.theme 은 sys.modules 에 **옛 버전이 남을 수 있다**. 그러면 새 app.py 가
    #    옛 theme 의 새 상수를 찾다가 AttributeError 로 앱 전체가 죽는다
    #    (실측: 배포 직후 theme.FS_XS 로 화면이 통째로 뜨지 않았다).
    #    클래스 이름은 그냥 문자열이라, 설령 CSS 가 옛 버전이어도 모양만 밋밋해질 뿐
    #    화면은 뜬다. 서비스 화면은 못생겨질지언정 죽으면 안 된다.
    if meta:
        st.sidebar.markdown(
            f"<div class='kb-navmeta'>"
            f"기준 <b>{_year_text(meta.get('scored_year'))}</b><br>"
            f"평가 대상 <b>{meta.get('n_scored', 0):,}개</b> 브랜드</div>",
            unsafe_allow_html=True)
    # 고지문은 법적 성격이라 '작게 흘려두는' 문구가 아니다. 예전엔 11.2px·
    # 명도대비 2.78 로 사실상 안 읽히게 두었는데, 안 읽히는 고지는 고지가 아니다.
    st.sidebar.markdown(f"<div class='kb-navnote'>{DISCLAIMER}</div>",
                        unsafe_allow_html=True)
    if _is_public_demo():
        st.sidebar.markdown(f"<div class='kb-navnote'>{PUBLIC_NOTE}</div>", unsafe_allow_html=True)
    return view


def main() -> None:
    theme.setup(page_title="FranSCORE · KB")
    cfg = load_config()
    set_seed(cfg.get("seed", 42))

    view = _sidebar()
    module_name = MENU[view][0]

    # 다른 화면에 다녀오면 브랜드 상세 선택을 푼다. 안 그러면 점검 큐에 갔다가
    # FRANSCORE 로 돌아왔을 때 목록이 아니라 아까 보던 브랜드 상세가 뜬다.
    if st.session_state.get("_last_view") != module_name:
        if module_name != "franscore":
            st.session_state["fs_selected"] = None
        st.session_state["_last_view"] = module_name

    from src.views import about, assistant, batch, explore, franscore, portfolio, queue, validate
    modules = {"franscore": franscore, "explore": explore, "batch": batch, "queue": queue,
               "portfolio": portfolio, "validate": validate, "assistant": assistant, "about": about}
    modules[module_name].render()


if __name__ == "__main__":
    main()
