"""KB 디자인 시스템 — 색·타이포·컴포넌트를 한 곳에 정의한다.

색값의 출처
    KB국민은행 운영 사이트(kbstar.com)의 계산된 스타일에서 직접 추출했다.
    시그니처 옐로 #FFCC00, 보조 옐로 #FFE85A, 텍스트 계열 #4E473F·#5A5A5A,
    웜 그레이 보더 #EAE5DF. 추정이 아니라 실제 서비스가 쓰는 값이다.

왜 별도 모듈인가
    화면마다 색을 직접 적으면 한 곳을 고쳐도 다른 화면이 따라오지 않는다.
    차트(plotly)·배지·카드·표가 모두 같은 팔레트를 참조하도록 토큰을 여기 한 곳에 둔다.

사용법
    import streamlit as st
    from src import theme
    theme.setup(page_title="...")      # set_page_config + CSS 주입 (앱 최상단 1회)
    theme.page_header("제목", "설명")
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# 색 토큰
# ---------------------------------------------------------------------------

YELLOW = "#FFCC00"          # KB 시그니처 (kbstar.com 실측)
YELLOW_DEEP = "#F0B400"     # 눌림·강조
YELLOW_SOFT = "#FFF6D6"     # 배경 틴트
YELLOW_LINE = "#FFE85A"     # 보조 라인 (실측)

INK = "#26221E"             # 최상위 제목 — BG 대비 14.6
TEXT = "#3D3833"            # 본문 — BG 대비 10.5
TEXT_SUB = "#6B635A"        # 보조 설명 — BG 대비 5.37
TEXT_MUTED = "#766E65"      # 캡션·출처 — BG 대비 4.56
#   ⚠️ TEXT_MUTED 는 예전에 #9A9289 였고 배경 위 대비가 **2.79** 였다.
#      캡션·출처·근거표기가 전부 이 색이라 화면에서 가장 많이 쓰이는데
#      가장 안 읽혔다. '흐리게'는 대비를 깨라는 뜻이 아니라 위계를 낮추라는
#      뜻이다 — 위계는 크기와 굵기로 주고 대비는 지킨다.

BG = "#F5F4F1"              # 페이지 배경 (웜 오프화이트)
SURFACE = "#FFFFFF"
BORDER = "#E8E4DE"          # 웜 그레이 보더 (실측 #EAE5DF 계열)
BORDER_STRONG = "#D6D0C7"

NAV_BG = "#332F2A"          # 사이드바 (KB 그레이 계열 딥톤)
NAV_TEXT = "#CFC8BE"        # NAV_BG 대비 8.01
NAV_SUB = "#A69D92"         # 사이드바 보조 — NAV_BG 대비 4.97
#   ⚠️ 예전엔 라벨 #8C8378(3.56)·고지문 #7A7168(2.78)로 각각 달랐고 둘 다 미달.
NAV_ACTIVE_BG = YELLOW
NAV_ACTIVE_TEXT = "#26221E"  # YELLOW 대비 10.44
NAV_HOVER_BG = "#423D37"

# 의미색 — 은행 화면이므로 채도를 낮춰 경고가 과장되지 않게 한다.
#
# ⚠️ 글자용과 도형용을 **반드시 나눈다**. 접근성 기준이 서로 다르기 때문이다.
#    본문 글자 4.5:1 (WCAG 1.4.3) · 도형/그래프 3:1 (WCAG 1.4.11).
#    하나로 합치면 둘 중 하나가 반드시 깨진다 — 실제로 그랬다. 예전 값을
#    페이지 배경 #F5F4F1 위에서 실측하면
#        관찰 #D9860F 2.59 · 안정 #1F8A5B 3.94 · 주의 #C8433B 4.42
#        보조텍스트 #9A9289 2.79
#    로 **네 개 모두 글자 기준 미달**이었다. 등급색과 캡션은 이 서비스에서
#    가장 자주 읽히는 정보인데 가장 안 읽히고 있었다.
#
#    아래 값은 색조(hue)와 채도를 원래 값 근처로 묶어 두고 명도만 최소한
#    낮춰서 찾았다. 채도를 최대로 올리면 대비는 쉽게 오르지만 #DB0C00 같은
#    원색이 되어 "경고를 과장하지 않는다"는 이 화면의 전제를 깬다.
#    검증은 tests/test_theme_contrast.py 가 매번 다시 계산한다.
DANGER = "#B44C46"          # 글자 — BG 4.70 · 흰 5.17 · soft 4.53
DANGER_FILL = "#C8433B"     # 도형 — BG 4.42 (3:1 충족)
DANGER_SOFT = "#FBEDEB"
WARN = "#986216"            # 글자 — BG 4.66 · 흰 5.13 · soft 4.70
WARN_FILL = "#C07C1B"       # 도형 — BG 3.11
WARN_SOFT = "#FDF4E3"
SAFE = "#287B57"            # 글자 — BG 4.70 · 흰 5.17 · soft 4.63
SAFE_FILL = "#1F8A5B"       # 도형 — BG 3.94
SAFE_SOFT = "#EAF5EF"
INFO = "#3D6EA4"            # 글자 — BG 4.82 · 흰 5.30 · soft 4.74
INFO_FILL = "#2F6FB5"       # 도형 — BG 4.70
INFO_SOFT = "#EDF3FA"

GRADE_COLOR = {"High": DANGER, "Medium": WARN, "Low": SAFE}          # 글자용
GRADE_FILL = {"High": DANGER_FILL, "Medium": WARN_FILL, "Low": SAFE_FILL}  # 도형용
GRADE_SOFT = {"High": DANGER_SOFT, "Medium": WARN_SOFT, "Low": SAFE_SOFT}
GRADE_KR = {"High": "주의", "Medium": "관찰", "Low": "안정"}

# 차트 계열색 — 옐로를 1순위로 두되 인접색이 서로 구분되게 배열.
# 계열색은 '도형'이므로 채움용 값을 쓴다.
SERIES = [YELLOW_DEEP, "#4E473F", INFO_FILL, SAFE_FILL, "#B07CC6", WARN_FILL,
          "#5FA8A0", DANGER_FILL]

FONT_STACK = ("Pretendard, 'Pretendard Variable', -apple-system, BlinkMacSystemFont, "
              "'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif")

# ---------------------------------------------------------------------------
# 타이포 스케일
# ---------------------------------------------------------------------------
# 왜 필요한가
#     화면마다 필요할 때마다 크기를 적으면 .76 .78 .8 .82 .84 .85 .86 .87 .88
#     .9 .92 .95 .96 .97 .98 .99 1 1.02 1.04 1.05 1.06 1.08 … 처럼 늘어난다.
#     실제로 이 저장소에 **29종**이 있었고 브라우저에서 렌더된 크기는 28종이었다.
#     0.02rem 차이는 아무도 의도한 적 없고 아무도 구분하지 못한다 — 그저 화면이
#     정돈돼 보이지 않을 뿐이다. 여덟 단계로 못박고 여기 없는 값은 쓰지 않는다.
#
# 왜 13px 이 바닥인가
#     예전 화면엔 10px(게이지 눈금)·11px(등급 라벨)·11.2px(고지문)이 있었다.
#     매일 몇 시간씩 이 화면을 보는 심사역에게 11px 은 읽으라고 준 크기가 아니다.
FS_XS = ".8125rem"      # 13px — 배지·범례·표 안 보조
FS_SM = ".875rem"       # 14px — 캡션·출처·근거표기
FS_MD = ".9375rem"      # 15px — 조밀한 목록
FS_BASE = "1rem"        # 16px — 본문 기본
FS_LG = "1.125rem"      # 18px — 카드 제목·강조 본문
FS_XL = "1.375rem"      # 22px — 섹션 제목 (h3)
FS_2XL = "1.75rem"      # 28px — 페이지 부제 (h2)
FS_3XL = "2.125rem"     # 34px — 페이지 제목 (h1)·지표 숫자

# 모서리 — 예전엔 2·4·7·8·9·10·12·16·50·999 열 종류가 섞여 있었다.
RADIUS_SM = "6px"       # 배지 안쪽·작은 태그
RADIUS_MD = "9px"       # 버튼·입력·알림
RADIUS_LG = "12px"      # 카드·지표·표
RADIUS_PILL = "999px"   # 알약형 칩


# ---------------------------------------------------------------------------
# Plotly 템플릿
# ---------------------------------------------------------------------------

def _register_plotly() -> None:
    """KB 팔레트 plotly 템플릿을 등록하고 기본값으로 지정한다.

    ⚠️ 이미 등록됐다고 조기 반환하면 안 된다. pio.templates 는 **프로세스 전역**이라
       한 번 등록되면 이 파일을 고쳐도 그대로 남는다. Streamlit 이 모듈을 다시 읽어도
       템플릿은 옛 값이라, 화면 글자는 커졌는데 차트 눈금만 예전 12px 로 남는 일이
       실제로 났다(실측). 매번 덮어써야 파일이 곧 화면이 된다.
    """
    tpl = go.layout.Template()
    # 차트 글자는 CSS 밖이라 rem 토큰이 닿지 않는다. px 로 같은 계단을 다시 적는다.
    #   눈금 13px = FS_XS · 범례/본문 14px = FS_SM · 제목 16px = FS_BASE
    #   (예전엔 눈금 12·본문 13·제목 15 로 화면 글자보다 한 단계씩 작아
    #    차트만 유독 잘아 보였다.)
    tpl.layout = go.Layout(
        font={"family": FONT_STACK, "size": 14, "color": TEXT},
        title={"font": {"size": 16, "color": INK}, "x": 0, "xanchor": "left"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES,
        margin={"l": 8, "r": 8, "t": 34, "b": 8},
        xaxis={"gridcolor": BORDER, "linecolor": BORDER_STRONG, "zerolinecolor": BORDER,
               "tickfont": {"size": 13, "color": TEXT_SUB}, "automargin": True},
        yaxis={"gridcolor": BORDER, "linecolor": BORDER_STRONG, "zerolinecolor": BORDER,
               "tickfont": {"size": 13, "color": TEXT_SUB}, "automargin": True},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0,
                "font": {"size": 13}, "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"font": {"family": FONT_STACK, "size": 13}, "bgcolor": SURFACE,
                    "bordercolor": BORDER_STRONG},
        separators=".,",
    )
    pio.templates["kb"] = tpl
    pio.templates.default = "plotly_white+kb"


_register_plotly()


def plot(fig: go.Figure, *, height: int | None = None, key: str | None = None) -> None:
    """차트 공통 렌더 — 모드바를 숨기고 컨테이너 폭에 맞춘다.

    ⚠️ theme=None 이 핵심이다. st.plotly_chart 는 기본값이 theme="streamlit" 이고,
       그 경우 Streamlit 이 **자기 템플릿으로 우리 템플릿을 덮어쓴다**. 위에서
       colorway·글꼴·격자색을 아무리 정해도 화면에는 Streamlit 기본값이 나온다.
       실측: 눈금을 13px 로 지정했는데 브라우저에서는 12px 로 렌더됐다.
       trace 에 직접 준 marker 색만 살아남아서, 색이 맞으니 테마가 먹은 줄 알았다.
       theme=None 이라야 pio.templates.default 인 'plotly_white+kb' 가 적용된다.
    """
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, key=key, theme=None,
                    config={"displayModeBar": False, "responsive": True})


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

# 카드(테두리 있는 컨테이너) 선택자.
# ⚠️ Streamlit 1.57 에서 st.container(border=True) 는 **stLayoutWrapper 의 자식
#    stVerticalBlock** 으로 렌더된다. 구버전의 stVerticalBlockBorderWrapper 는 이
#    버전 DOM 에 아예 없다(실측 0개) — 그것만 적어 두면 카드 스타일이 통째로 죽는다.
_CARD = '[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]'

_BLOCK = '[data-testid="stMainBlockContainer"] [data-testid="stLayoutWrapper"]'
# 카드마다 등장 지연을 조금씩 늘려 한 덩어리가 아니라 차례로 놓이게 한다
# ⚠️ 이 문자열은 아래 f-string 에 **값으로** 꽂힌다. 값은 다시 파싱되지 않으므로
#    중괄호를 이중으로 쓰면 안 된다(그러면 CSS 에 `{{` 가 그대로 남는다).
_STAGGER = "\n".join(
    f"{_BLOCK}:nth-of-type({n}) {{ animation-delay: {d:.2f}s; }}"
    for n, d in ((1, .02), (2, .06), (3, .10), (4, .14), (5, .18))
) + f"\n{_BLOCK}:nth-of-type(n+6) {{ animation-delay: .21s; }}"

_CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

html, body, [class*="st-"], button, input, textarea, select {{
    font-family: {FONT_STACK};
    -webkit-font-smoothing: antialiased;
}}
/* ⚠️ 위 규칙의 `[class*="st-"]` 는 Streamlit 의 **아이콘 span 까지** 잡는다
   (클래스가 st-emotion-cache-… 라서). 그러면 Material Symbols 글리프가 폰트를
   못 찾아 'keyboard_double_arrow_right' 라는 **글자 이름 그대로** 화면에 찍힌다.
   실제로 사이드바 접기 버튼이 그렇게 보였다. 아이콘 요소만 되돌린다. */
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined,
.material-icons, span[class*="material-symbols"], span[class*="material-icons"] {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}}

/* ── 타이포 스케일 ───────────────────────────────────────────
   Streamlit 기본값(본문 14px 상당)은 1440px 이상 화면에서 너무 작아 가독성이
   떨어진다. 루트를 17px 로 올리고 전 요소를 rem 으로 잡아 한 곳에서 조절한다. */
/* Streamlit 이 `[data-testid=stMarkdownContainer] p {{font-size:.9rem}}` 처럼 자체
   규칙을 갖고 있어 같은 선택자로는 밀린다(실측: 1rem 지정 → 15.3px 렌더).
   테마 계층이므로 !important 로 확정한다. */
/* ⚠️ 루트 17px + 본문 1.06rem 이면 본문이 **18.02px** 로 렌더된다(실측). 대시보드
   본문으로는 과하게 커서 카드 안 여백이 먹히고 비율이 어색해진다. 루트를 브라우저
   기본값 16px 로 되돌리고 본문을 16.3px 로 잡는다 — Streamlit 기본(14px 상당)보다는
   확실히 크면서 화면 밀도를 해치지 않는 지점이다. */
html {{ font-size: 16px; }}

/* 화면 코드(views/*.py 인라인 스타일)가 참조할 토큰.
   ⚠️ 여기 없는 크기를 인라인으로 적지 않는다 — 그렇게 29종이 생겼다. */
:root {{
    --fs-xs: {FS_XS};   --fs-sm: {FS_SM};   --fs-md: {FS_MD};  --fs-base: {FS_BASE};
    --fs-lg: {FS_LG};   --fs-xl: {FS_XL};   --fs-2xl: {FS_2XL}; --fs-3xl: {FS_3XL};
    --r-sm: {RADIUS_SM}; --r-md: {RADIUS_MD}; --r-lg: {RADIUS_LG}; --r-pill: {RADIUS_PILL};
    --ink: {INK}; --text: {TEXT}; --text-sub: {TEXT_SUB}; --text-muted: {TEXT_MUTED};
    --danger: {DANGER}; --warn: {WARN}; --safe: {SAFE}; --info: {INFO};
    --border: {BORDER}; --surface: {SURFACE}; --bg: {BG};
}}

body, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
    font-size: {FS_BASE} !important; line-height: 1.68; color: {TEXT};
}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] div, .stCaption, small {{
    font-size: {FS_SM} !important; line-height: 1.62; color: {TEXT_SUB} !important;
}}
label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
    font-size: {FS_MD} !important; font-weight: 600; color: {TEXT};
}}
/* 표·선택지 안의 글씨도 함께 (기본값이 12~13px 로 작다) */
[data-testid="stDataFrame"], [data-baseweb="select"], [role="option"],
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{
    font-size: {FS_MD} !important;
}}

/* ── 페이지 셸 ───────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {{ background: {BG}; }}
[data-testid="stHeader"] {{
    background: transparent; height: 0; pointer-events: none;
}}
/* Streamlit 기본 크롬(Deploy 버튼·메뉴·상태 위젯)은 서비스 화면의 것이 아니다.
   ⚠️ 단, stToolbar 를 통째로 숨기면 **사이드바 펼치기 버튼까지 사라진다** —
   그 버튼이 툴바의 자식이라 사이드바를 접으면 다시 열 방법이 없어진다(실측 결함).
   개별 요소만 숨기고 툴바 자체는 살려 둔다. */
[data-testid="stStatusWidget"], [data-testid="stDecoration"],
[data-testid="stToolbarActions"], .stDeployButton,
[data-testid="stAppDeployButton"], #MainMenu {{
    display: none !important;
}}
[data-testid="stToolbar"] {{ pointer-events: auto; }}
[data-testid="stExpandSidebarButton"] {{
    pointer-events: auto;
    width: 40px !important; height: 40px !important;
    display: flex !important; align-items: center; justify-content: center;
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {RADIUS_LG};
    box-shadow: 0 2px 8px rgba(38,34,30,.10);
    transition: background .16s ease, box-shadow .16s ease, transform .16s ease;
}}
[data-testid="stExpandSidebarButton"]:hover {{
    background: {YELLOW_SOFT}; border-color: {YELLOW_DEEP};
    box-shadow: 0 4px 14px rgba(38,34,30,.14); transform: translateY(-1px);
}}
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stExpandSidebarButton"] span {{ color: {INK} !important; }}
[data-testid="stMainBlockContainer"] {{
    padding: 1.8rem 2.6rem 4rem 2.6rem; max-width: 1560px;
}}
#MainMenu, footer {{ visibility: hidden; }}

/* Streamlit 자체 규칙이 `[data-testid=stMarkdownContainer] h1` 같은 형태라
   맨 h1 선택자는 밀린다(실측: 지정 1.72rem 인데 44px 로 렌더). 컨테이너를 함께 적어
   특이도를 맞춘다. */
h1, h2, h3, h4,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {{
    color: {INK}; letter-spacing: -0.02em; font-weight: 700; padding: 0;
}}
h1, [data-testid="stMarkdownContainer"] h1, [data-testid="stHeading"] h1 {{ font-size: {FS_3XL}; }}
h2, [data-testid="stMarkdownContainer"] h2, [data-testid="stHeading"] h2 {{ font-size: {FS_2XL}; }}
h3, [data-testid="stMarkdownContainer"] h3, [data-testid="stHeading"] h3 {{ font-size: {FS_XL}; }}
h4, [data-testid="stMarkdownContainer"] h4 {{ font-size: {FS_LG}; }}
h5, [data-testid="stMarkdownContainer"] h5 {{
    font-size: {FS_MD}; font-weight: 700; color: {TEXT_SUB};
    letter-spacing: .01em; margin-bottom: .5rem;
}}
p, li, label, span, div {{ color: {TEXT}; }}
/* 본문 안의 `코드` — 기본이 10.5px 이라 본문 16px 사이에서 혼자 잘아 보인다 */
code, [data-testid="stMarkdownContainer"] code {{
    font-size: {FS_SM} !important; background: {BG}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: {RADIUS_SM}; padding: 1px 5px;
}}
a {{ color: {INFO}; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
a:focus-visible {{ outline: 2px solid {YELLOW_DEEP}; outline-offset: 2px; border-radius: 3px; }}
hr {{ border-color: {BORDER}; margin: 1.4rem 0; }}

/* ── 사이드바 ────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {NAV_BG};
    border-right: 1px solid #241F1B;
    width: 286px !important;
}}
/* ⚠️ 여기가 무너지면 사이드바 전체가 안 보인다 — 실제로 그랬다.
   `[data-testid="stSidebar"] *` 는 특이도가 (0,1,0) 인데, 본문 규칙
   `[data-testid="stMarkdownContainer"] p` 는 (0,1,1) 이라 **본문 규칙이 이긴다**.
   Streamlit 은 버튼·캡션 라벨을 전부 stMarkdownContainer > p 로 렌더하므로
   사이드바 메뉴 글자가 본문색 #3D3833 이 되어 딥톤 배경 #332F2A 위에 얹혔다.
   실측 명도대비 **1.14** — 글자가 있다는 것조차 안 보이는 값이다.
   그래서 * 하나로 퉁치지 않고, 이기는 특이도로 다시 못박는다. */
[data-testid="stSidebar"] * {{ color: {NAV_TEXT}; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] .stMarkdown p {{ color: {NAV_TEXT} !important; }}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding-top: 0.6rem; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: #FFFFFF;
}}
/* 접기 버튼은 기본이 visibility:hidden 이라 사이드바에 마우스를 올려야 보인다.
   숨은 컨트롤은 있는 줄도 모르니 항상 보이게 두고, 대신 은은하게 처리한다. */
[data-testid="stSidebarCollapseButton"] {{ visibility: visible !important; }}
[data-testid="stSidebarCollapseButton"] button {{
    color: {NAV_TEXT} !important; opacity: .55;
    transition: opacity .18s var(--ease), background .18s var(--ease);
    border-radius: {RADIUS_MD};
}}
[data-testid="stSidebarCollapseButton"] button:hover {{
    opacity: 1; background: #423D37 !important;
}}

/* 사이드바 내비게이션 — 라디오가 아니라 버튼이다.
   Streamlit 라디오는 이미 선택된 항목을 다시 눌러도 값이 안 바뀌어 아무 일도
   일어나지 않는다(브랜드 상세에 갇히는 실측 결함). 버튼은 누를 때마다 실행된다.
   버튼 기본 모양(흰 배경·테두리)이 어두운 사이드바에서 튀므로 항목처럼 다시 그린다.

   ⚠️ 글자색은 button 이 아니라 **button 안의 p** 에 걸어야 한다. Streamlit 이
      라벨을 stMarkdownContainer > p 로 렌더하므로 button 에만 주면 본문 규칙에 진다. */
[data-testid="stSidebar"] [data-testid="stButton"] > button {{
    width: 100%; justify-content: flex-start; text-align: left;
    padding: 11px 14px; border-radius: {RADIUS_MD}; border: none;
    background: transparent; box-shadow: none;
    transition: background .12s ease;
}}
[data-testid="stSidebar"] [data-testid="stButton"] > button p {{
    font-size: {FS_BASE} !important; font-weight: 500; color: {NAV_TEXT} !important;
    transition: color .12s ease;
}}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {{
    background: {NAV_HOVER_BG}; border: none;
}}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover p {{ color: #FFFFFF !important; }}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {{
    background: {NAV_ACTIVE_BG} !important;
}}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] p {{
    color: {NAV_ACTIVE_TEXT} !important; font-weight: 700;
}}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:hover {{
    background: {YELLOW_DEEP} !important;
}}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:hover p {{
    color: {NAV_ACTIVE_TEXT} !important;
}}
/* 활성 항목 왼쪽 강조선 — 어느 화면에 있는지 옐로 하나로 끝내지 않는다 */
[data-testid="stSidebar"] [data-testid="stButton"]:has(button[kind="primary"]) {{
    position: relative;
}}
[data-testid="stSidebar"] [data-testid="stButton"]:has(button[kind="primary"])::before {{
    content: ""; position: absolute; left: -10px; top: 50%; transform: translateY(-50%);
    width: 4px; height: 60%; border-radius: 0 3px 3px 0; background: {YELLOW}; z-index: 1;
}}
[data-testid="stSidebar"] [data-testid="stButton"] > button:focus:not(:active) {{
    border: none; box-shadow: none;
}}
[data-testid="stSidebar"] [data-testid="stButton"] > button:focus-visible {{
    outline: 2px solid {YELLOW}; outline-offset: 2px;
}}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {{
    margin-bottom: 2px;
}}

/* ── 브랜드 헤더 (사이드바 상단) ─────────────────────────── */
.kb-brand {{
    display: flex; align-items: center; gap: 10px;
    padding: 14px 12px 16px 12px; margin-bottom: 6px;
    border-bottom: 1px solid #453F39;
}}
.kb-brand .mark {{
    width: 38px; height: 38px; border-radius: {RADIUS_MD}; background: {YELLOW};
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1rem; color: #26221E; letter-spacing: -0.04em;
    flex: 0 0 38px;
}}
.kb-brand .txt {{ line-height: 1.25; }}
.kb-brand .txt .t1 {{ font-size: {FS_LG}; font-weight: 700; color: #FFFFFF; }}
.kb-brand .txt .t2 {{ font-size: {FS_XS}; color: {NAV_SUB}; }}
.kb-navlabel {{
    padding: 16px 14px 7px 14px; font-size: {FS_XS}; font-weight: 700;
    letter-spacing: .08em; color: {NAV_SUB};
}}
/* 사이드바 하단 — 기준연도·평가규모와 사용범위 고지.
   app.py 는 클래스 이름만 적고 모양은 여기서 정한다(모듈 세대 어긋남 방어). */
.kb-navmeta {{
    padding: 14px 12px 4px 12px; margin-top: 14px; font-size: {FS_XS};
    line-height: 1.7; color: {NAV_SUB}; border-top: 1px solid #453F39;
}}
.kb-navmeta b {{ color: {NAV_TEXT}; font-weight: 700; }}
.kb-navnote {{
    padding: 16px 12px; font-size: {FS_XS}; line-height: 1.6; color: {NAV_SUB};
}}
/* ⚠️ 사이드바 안의 색은 `[data-testid="stSidebar"] *` 로도 걸리므로 특이도를
   맞춰 다시 못박는다 — 안 그러면 위 두 규칙이 그 규칙에 밀린다. */
[data-testid="stSidebar"] .kb-navmeta,
[data-testid="stSidebar"] .kb-navnote {{ color: {NAV_SUB}; }}
[data-testid="stSidebar"] .kb-navmeta b {{ color: {NAV_TEXT}; }}

/* ── 페이지 헤더 ─────────────────────────────────────────── */
.kb-page {{ margin-bottom: 1.1rem; }}
.kb-page .eyebrow {{
    display: inline-block; font-size: {FS_XS}; font-weight: 700; letter-spacing: .07em;
    color: {TEXT_SUB}; background: {SURFACE}; border: 1px solid {BORDER};
    padding: 4px 11px; border-radius: {RADIUS_PILL}; margin-bottom: 10px;
}}
.kb-page h1 {{ margin: 0 0 4px 0; }}
.kb-page .sub {{ color: {TEXT_SUB}; font-size: {FS_BASE}; margin: 0; line-height: 1.6; }}
.kb-rule {{ height: 3px; width: 42px; background: {YELLOW}; border-radius: 2px; margin: 10px 0 0 0; }}

/* ── 카드 / 컨테이너 ─────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
    border-radius: {RADIUS_LG};
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE}; border: 1px solid {BORDER} !important;
    border-radius: {RADIUS_LG}; box-shadow: 0 1px 2px rgba(38,34,30,.04);
}}
.kb-card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {RADIUS_LG};
    padding: 16px 18px; box-shadow: 0 1px 2px rgba(38,34,30,.04);
}}

/* ── 지표(metric) ────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {RADIUS_LG};
    padding: 17px 19px; box-shadow: 0 1px 2px rgba(38,34,30,.04);
}}
[data-testid="stMetricLabel"] p {{
    font-size: {FS_SM} !important; color: {TEXT_SUB} !important; font-weight: 600;
}}
/* 지표 이름은 잘라내지 말고 줄바꿈한다.
   기본값은 한 줄 + 말줄임이라 좁은 화면에서 '주의 · 100점포 이상' 이
   '주의 · 100점포 이…' 로 잘렸다(1024px 실측: 필요 115px / 가용 109px).
   지표 이름이 잘리면 그 숫자가 무엇의 숫자인지 알 수 없다 — 지울 수 없는 정보다.
   높이를 맞춰 두어 한 줄짜리와 두 줄짜리가 나란히 있어도 숫자 baseline 이 어긋나지
   않게 한다. */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] > div {{
    white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; line-height: 1.35;
}}
[data-testid="stMetricLabel"] {{ min-height: 2.7em; align-items: flex-start; }}
[data-testid="stMetricValue"] {{
    font-size: {FS_3XL} !important; font-weight: 700; color: {INK};
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
}}
[data-testid="stMetricDelta"] {{ font-size: {FS_XS}; }}
/* 도움말 아이콘 — 16px 기본이라 겨냥하기 어렵다. 보이는 크기는 두되 눌리는 면을 넓힌다 */
[data-testid="stTooltipHoverTarget"] {{
    padding: 6px; margin: -6px; border-radius: {RADIUS_SM};
}}

/* ── 배지 ────────────────────────────────────────────────── */
.kb-chip {{
    display: inline-flex; align-items: center; gap: 5px; font-size: {FS_XS};
    font-weight: 700; padding: 4px 12px; border-radius: {RADIUS_PILL}; line-height: 1.5;
    border: 1px solid transparent; white-space: nowrap;
}}
.kb-chip.g-High   {{ background: {DANGER_SOFT}; color: {DANGER}; border-color: #F0CFCB; }}
.kb-chip.g-Medium {{ background: {WARN_SOFT};   color: {WARN};   border-color: #F3DFBB; }}
.kb-chip.g-Low    {{ background: {SAFE_SOFT};   color: {SAFE};   border-color: #C6E4D6; }}
.kb-chip.g-Info   {{ background: {INFO_SOFT};   color: {INFO};   border-color: #CBDDF0; }}
.kb-chip.g-Neutral{{ background: #F2F0EC;       color: {TEXT_SUB}; border-color: {BORDER}; }}

/* ── 소견 항목 ───────────────────────────────────────────── */
.kb-finding {{
    display: flex; gap: 13px; padding: 14px 0;
    border-bottom: 1px dashed {BORDER};
}}
.kb-finding:last-child {{ border-bottom: none; }}
.kb-finding .bar {{ width: 3px; border-radius: 2px; flex: 0 0 3px; }}
.kb-finding .body {{ flex: 1; }}
.kb-finding .head {{ font-weight: 700; font-size: {FS_LG}; color: {INK}; margin-bottom: 4px; }}
.kb-finding .desc {{ font-size: {FS_BASE}; color: {TEXT}; line-height: 1.68; }}
.kb-finding .src  {{ font-size: {FS_SM}; color: {TEXT_MUTED}; margin-top: 5px; }}

/* ── 버튼 ────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    border-radius: {RADIUS_MD}; border: 1px solid {BORDER_STRONG}; background: {SURFACE};
    color: {TEXT}; font-weight: 600; font-size: {FS_MD}; padding: .5rem 1.15rem;
    transition: all .12s ease;
}}
/* 본문 버튼 라벨도 p 로 렌더된다 — 본문 규칙이 크기를 덮어쓰지 않게 맞춰 둔다 */
[data-testid="stMainBlockContainer"] .stButton > button p,
[data-testid="stMainBlockContainer"] .stDownloadButton > button p {{
    font-size: {FS_MD} !important; font-weight: 600;
}}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
    outline: 2px solid {YELLOW_DEEP}; outline-offset: 2px;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: {YELLOW_DEEP}; color: {INK}; background: {YELLOW_SOFT};
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
    background: {YELLOW}; border-color: {YELLOW}; color: {INK}; font-weight: 700;
}}
.stButton > button[kind="primary"]:hover {{ background: {YELLOW_DEEP}; border-color: {YELLOW_DEEP}; }}

/* ── 입력 ────────────────────────────────────────────────── */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {{
    border-radius: {RADIUS_MD} !important; border-color: {BORDER_STRONG} !important;
    background: {SURFACE} !important;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: {YELLOW_DEEP} !important; box-shadow: 0 0 0 3px rgba(255,204,0,.22) !important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{ background: {YELLOW_DEEP}; }}

/* 안내문(placeholder) — 기본값 rgba(49,51,63,.6) 은 흰 배경 위 대비 3.70 이라
   "무엇을 입력해야 하는지"가 가장 안 보인다. 빈 입력칸에서 유일한 단서다.
   ⚠️ BaseWeb 셀렉트의 안내문에는 붙잡을 속성이 없다 — 클래스가 st-ck 같은
      난수라 버전마다 바뀐다. 그래서 값 영역 전체를 TEXT_SUB(5.37)로 올린다.
      선택된 값도 같은 색이 되지만 5.37 은 충분히 읽히고, 빈 칸과 채운 칸은
      글자 내용으로 구분된다. 난수 클래스에 기대는 것보다 이 쪽이 안 깨진다.
      멀티셀렉트 태그 규칙보다 **앞에** 둔다(특이도가 같아 뒤가 이긴다). */
input::placeholder, textarea::placeholder {{
    color: {TEXT_SUB} !important; opacity: 1 !important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div {{
    color: {TEXT_SUB};
}}

/* 멀티셀렉트 태그 — 손대지 않으면 Streamlit 기본 빨강 #FF4B4B 이 그대로 나온다.
   화면 전체가 웜 옐로 계열인데 여기만 형광 빨강이라 눈에 먼저 걸린다(실측 확인). */
[data-baseweb="tag"] {{
    background: {YELLOW_SOFT} !important; border: 1px solid {YELLOW_DEEP} !important;
    border-radius: {RADIUS_SM} !important;
}}
[data-baseweb="tag"] span, [data-baseweb="tag"] div {{ color: {INK} !important; }}
[data-baseweb="tag"] svg {{ fill: {TEXT_SUB} !important; }}

/* ── 탭 ──────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {BORDER}; }}
[data-baseweb="tab"] {{
    padding: 11px 19px; font-weight: 600; color: {TEXT_SUB};
    border-radius: {RADIUS_MD} {RADIUS_MD} 0 0;
}}
[data-baseweb="tab"] p {{ font-size: {FS_BASE} !important; font-weight: 600; }}
[data-baseweb="tab"][aria-selected="true"] {{ color: {INK}; background: {SURFACE}; }}
[data-baseweb="tab"][aria-selected="true"] p {{ color: {INK} !important; font-weight: 700; }}
[data-baseweb="tab-highlight"] {{ background: {YELLOW} !important; height: 3px; }}
[data-baseweb="tab-border"] {{ display: none; }}

/* ── 표 ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER}; border-radius: {RADIUS_LG}; overflow: hidden;
}}

/* ── 알림 ────────────────────────────────────────────────── */
[data-testid="stAlert"] {{ border-radius: {RADIUS_LG}; border-width: 1px; }}

/* ── 지표 증감 배지 ──────────────────────────────────────── */
/* 기본은 회청색 pill 이라 웜 팔레트에서 혼자 차갑다 */
[data-testid="stMetricDelta"] {{
    background: {BG} !important; border: 1px solid {BORDER};
    border-radius: {RADIUS_SM}; padding: 1px 7px;
}}
[data-testid="stMetricDelta"] * {{ color: {TEXT_SUB} !important; }}

/* ── expander ────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER}; border-radius: {RADIUS_LG}; background: {SURFACE};
}}
[data-testid="stExpander"] summary {{ font-weight: 600; color: {TEXT}; }}
[data-testid="stExpander"] summary p {{ font-size: {FS_BASE} !important; font-weight: 600; }}

/* ── 채팅 ────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {RADIUS_LG};
    padding: 12px 14px;
}}
[data-testid="stChatInput"] {{ border-radius: {RADIUS_MD}; border-color: {BORDER_STRONG}; }}

/* ── 스크롤바 ────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: {BORDER_STRONG}; border-radius: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb:hover {{ background: {TEXT_MUTED}; }}

/* ═══════════════════════════════════════════════════════════════════════
   깊이·움직임 — 요소가 배경에 얹힌 그림이 아니라 화면의 일부로 보이게 한다
   ═══════════════════════════════════════════════════════════════════════
   원칙 세 가지
     1. 그림자는 한 겹이 아니라 **두 겹**이다. 가까운 곳의 또렷한 경계선과
        멀리 퍼지는 부드러운 그늘이 함께 있어야 종이처럼 보인다.
     2. 움직임은 **감속**한다. 시작이 빠르고 끝이 느린 곡선이라야 물리적으로 읽힌다.
     3. 등장은 **아래에서 위로 살짝**. 뚝 나타나면 붙여넣은 이미지처럼 보인다.
   ─────────────────────────────────────────────────────────────────────── */

:root {{
    --ease: cubic-bezier(.22, 1, .36, 1);
    --shadow-1: 0 1px 2px rgba(38,34,30,.05), 0 1px 3px rgba(38,34,30,.03);
    --shadow-2: 0 2px 6px rgba(38,34,30,.06), 0 8px 24px rgba(38,34,30,.07);
    --shadow-3: 0 4px 12px rgba(38,34,30,.08), 0 16px 40px rgba(38,34,30,.10);
}}

@keyframes kb-rise {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes kb-fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

/* 본문 요소가 떠오르며 들어온다.
   ⚠️ 선택자 주의: Streamlit 버전에 따라 카드 컨테이너의 식별자가 다르다.
      구버전은 stVerticalBlockBorderWrapper, 현재는 **stLayoutWrapper 의 자식인
      stVerticalBlock** 이다(실측: 전자는 DOM 에 0개). 둘 다 적어 어느 쪽이든 걸리게 한다. */
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"],
[data-testid="stMainBlockContainer"] [data-testid="stLayoutWrapper"] {{
    animation: kb-rise .42s var(--ease) both;
}}
{_STAGGER}

/* 카드 — 정지 상태는 얕게, 손이 닿으면 떠오른다 */
{_CARD}, div[data-testid="stVerticalBlockBorderWrapper"] {{
    box-shadow: var(--shadow-1);
    transition: box-shadow .28s var(--ease), transform .28s var(--ease),
                border-color .28s var(--ease);
    will-change: transform;
}}
{_CARD}:hover, div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow: var(--shadow-2);
    transform: translateY(-2px);
    border-color: {BORDER_STRONG} !important;
}}
[data-testid="stMetric"] {{
    box-shadow: var(--shadow-1);
    transition: box-shadow .28s var(--ease), transform .28s var(--ease);
}}
[data-testid="stMetric"]:hover {{ box-shadow: var(--shadow-2); transform: translateY(-2px); }}

/* 브랜드 마크 — 로고가 카드 안에서 '얹힌 그림'이 되지 않도록 자리를 만든다 */
.kb-mark {{
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(160deg, #FFFFFF 0%, #FBFAF8 100%);
    border: 1px solid {BORDER};
    box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 1px 3px rgba(38,34,30,.06);
    overflow: hidden; flex-shrink: 0;
    transition: transform .3s var(--ease), box-shadow .3s var(--ease);
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover .kb-mark {{
    transform: scale(1.04);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 4px 12px rgba(38,34,30,.12);
}}
.kb-mark img {{
    width: 100%; height: 100%; object-fit: contain;
    animation: kb-fade .5s var(--ease) both;
}}
.kb-mark .letter {{
    width: 100%; height: 100%; display: flex; align-items: center;
    justify-content: center; font-weight: 800; letter-spacing: -.04em;
}}

/* 소견 — 왼쪽 색 막대가 자라 들어온다 */
.kb-finding {{ animation: kb-rise .38s var(--ease) both; }}
.kb-finding .bar {{
    transition: width .3s var(--ease);
    box-shadow: 0 0 0 1px rgba(255,255,255,.4) inset;
}}
.kb-finding:hover .bar {{ width: 5px; }}

/* 버튼·탭·입력 — 손이 닿는 것에는 전부 반응을 준다 */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    transition: all .2s var(--ease); box-shadow: var(--shadow-1);
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-1px); box-shadow: var(--shadow-2);
}}
.stButton > button:active {{ transform: translateY(0); box-shadow: var(--shadow-1); }}
[data-baseweb="tab"] {{ transition: color .2s var(--ease), background .2s var(--ease); }}
[data-testid="stTextInput"] input, [data-baseweb="select"] > div {{
    transition: border-color .2s var(--ease), box-shadow .2s var(--ease);
}}
[data-testid="stExpander"] {{ transition: box-shadow .25s var(--ease); }}
[data-testid="stExpander"]:hover {{ box-shadow: var(--shadow-1); }}

/* 차트 — 나타날 때 부드럽게 */
.js-plotly-plot {{ animation: kb-fade .55s var(--ease) both; }}

/* 채팅 말풍선 */
[data-testid="stChatMessage"] {{
    animation: kb-rise .34s var(--ease) both; box-shadow: var(--shadow-1);
}}

/* 움직임을 원치 않는 사용자 설정을 존중한다 (접근성) */
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation-duration: .01ms !important; animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }}
}}
</style>
"""


def setup(page_title: str = "FranSCORE", page_icon: str = "◆") -> None:
    """페이지 설정 + CSS 주입. 앱 최상단에서 딱 한 번 호출한다."""
    st.set_page_config(page_title=page_title, page_icon=page_icon,
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 컴포넌트
# ---------------------------------------------------------------------------

def sidebar_brand(title: str, subtitle: str) -> None:
    st.sidebar.markdown(
        f"<div class='kb-brand'><div class='mark'>KB</div>"
        f"<div class='txt'><div class='t1'>{title}</div>"
        f"<div class='t2'>{subtitle}</div></div></div>",
        unsafe_allow_html=True)


def sidebar_label(text: str) -> None:
    st.sidebar.markdown(f"<div class='kb-navlabel'>{text}</div>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    bits = ["<div class='kb-page'>"]
    if eyebrow:
        bits.append(f"<span class='eyebrow'>{eyebrow}</span>")
    bits.append(f"<h1>{title}</h1>")
    if subtitle:
        bits.append(f"<p class='sub'>{subtitle}</p>")
    bits.append("<div class='kb-rule'></div></div>")
    st.markdown("".join(bits), unsafe_allow_html=True)


def chip(text: str, kind: str = "Neutral") -> str:
    """인라인 배지 HTML. kind: High/Medium/Low/Info/Neutral."""
    return f"<span class='kb-chip g-{kind}'>{text}</span>"


def grade_chip(grade: str) -> str:
    g = str(grade)
    return chip(GRADE_KR.get(g, g), g if g in GRADE_COLOR else "Neutral")


def finding_html(title: str, desc: str, severity: str, source: str = "") -> str:
    """진단 소견 한 줄. severity: High/Medium/Low."""
    color = GRADE_COLOR.get(severity, TEXT_MUTED)
    src = f"<div class='src'>{source}</div>" if source else ""
    return (f"<div class='kb-finding'><div class='bar' style='background:{color}'></div>"
            f"<div class='body'><div class='head'>{title}</div>"
            f"<div class='desc'>{desc}</div>{src}</div></div>")
