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

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src import theme
from src.common import load_config, set_seed

MENU = {
    # '한눈에 보기'와 '브랜드 조회'는 같은 일의 두 단계여서 하나로 합쳤다 —
    # 목록에서 고르고, 고른 것을 자세히 본다. 화면을 오가며 이름을 다시 칠 이유가 없다.
    "FRANSCORE": ("franscore", "브랜드 리스크 현황과 상세 진단"),
    "점검 큐": ("queue", "담당자 배정과 확인 결과 기록"),
    "여신 포트폴리오": ("portfolio", "쏠림·예상손실 실시간 점검"),
    "AI 상담": ("assistant", "자연어 질의와 근거 인용"),
    "서비스 소개": ("about", "무엇을 어떻게 평가하는가"),
}

DISCLAIMER = ("본 서비스의 지표는 2선 리스크 관리 참고용이며 자동 여신 결정에 "
              "사용되지 않습니다.")


_NAV = "nav_view"


def _sidebar() -> str:
    theme.sidebar_brand("FranSCORE", "프랜차이즈 여신 리스크")
    theme.sidebar_label("메뉴")

    # ⚠️ 라디오가 아니라 버튼을 쓴다. Streamlit 라디오는 **이미 선택된 항목을 다시
    #    눌러도 값이 안 바뀌어 아무 일도 일어나지 않는다** — 브랜드 상세를 보다가
    #    사이드바의 FRANSCORE 를 눌러도 목록으로 못 돌아갔다(실측 결함).
    #    버튼은 누를 때마다 실행되므로 '같은 메뉴 다시 누르기 = 그 화면 처음으로'가 된다.
    current = st.session_state.setdefault(_NAV, next(iter(MENU)))
    for label in MENU:
        if st.sidebar.button(label, key=f"nav_{MENU[label][0]}",
                             use_container_width=True,
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
            f"기준 공시연도 <b>{meta.get('scored_year', '-')}년</b><br>"
            f"평가 대상 <b>{meta.get('n_scored', 0):,}개</b> 브랜드</div>",
            unsafe_allow_html=True)
    # 고지문은 법적 성격이라 '작게 흘려두는' 문구가 아니다. 예전엔 11.2px·
    # 명도대비 2.78 로 사실상 안 읽히게 두었는데, 안 읽히는 고지는 고지가 아니다.
    st.sidebar.markdown(f"<div class='kb-navnote'>{DISCLAIMER}</div>",
                        unsafe_allow_html=True)
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

    from src.views import about, assistant, franscore, portfolio, queue
    modules = {"franscore": franscore, "queue": queue, "portfolio": portfolio,
               "assistant": assistant, "about": about}
    modules[module_name].render()


if __name__ == "__main__":
    main()
