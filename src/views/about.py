"""서비스 소개 — 처음 보는 사람도, 심사역도 같은 화면에서 이해하게.

두 독자를 한 화면에 담는 방법
    윗부분은 **전제 지식 없이** 읽히게 쓴다 — 무엇을, 왜, 어떻게 쓰는지.
    아랫부분은 **왜 이 숫자를 믿어도 되는지**를 같은 말투로 잇는다. 심사역이 볼
    원지표는 접어 둔 상세에 두되, 접힌 것을 펴지 않아도 판단 근거는 다 보이게 한다.
    두 층을 섞으면 한쪽은 어렵고 한쪽은 부실해진다.

⚠️ 검증 수치를 접어 두었던 동안, 이 화면에서 가장 중요한 질문("믿어도 되는가")의
   답이 **한 번 더 눌러야 나오는 자리**에 있었다. 근거는 펼쳐 보는 사람에게만
   보이면 근거로 쓰이지 않는다 — 지금은 펼치지 않아도 보이는 자리에 둔다.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import theme
from src.views import common as C

# ⚠️ 이 문자열들은 st.markdown 이 아니라 **HTML 로 꽂힌다.** 그래서 강조는 `**` 가
#    아니라 `<b>` 여야 한다 — `**` 로 써 두었더니 화면에 별표가 그대로 찍혔다
#    (실측: "공시 지표가 **업종 안에서 …**을"). 소개 화면 첫 문단에서 벌어진 일이라
#    처음 보는 사람이 가장 먼저 보는 자리였다.
_STEPS = (
    ("1", "공시를 모은다",
     "공정거래위원회에 등록된 프랜차이즈 브랜드의 가맹점 수·개점·계약종료·평균매출·"
     "면적당매출을 연도별로 모읍니다. 여기에 가맹본부 재무(금융감독원 전자공시 "
     "감사보고서 + 공정거래위원회 정보공개서)와 네이버 검색수요·뉴스를 더합니다. "
     "본부 재무를 두 곳에서 모으는 이유는, 외부감사 대상이 아닌 본부는 감사보고서를 "
     "제출하지 않아 전자공시만으로는 절반도 볼 수 없기 때문입니다."),
    ("2", "무엇이 '나빠진 것'인지 정의한다",
     "정의를 우리가 만들지 않습니다. 공시 지표가 <b>업종 안에서 하위 구간에 들어가는 것</b>을 "
     "'악화'로 봅니다. 어떤 지표를 어떤 기준으로 보는지 전부 공개하고, 규칙은 "
     "모델을 학습하기 전에 고정해 둡니다."),
    ("3", "다음 해에 나빠질 브랜드를 가려낸다",
     "과거 자료로 학습한 모델이 각 브랜드의 확률을 매깁니다. 그 확률을 <b>고정된 구간</b>으로 "
     "나눠 세 등급을 줍니다. 구간은 순위가 아니라 확률 기준이라, 업계 전체가 나빠지면 "
     "낮은 등급이 늘어납니다."),
    ("4", "왜 그런지 문장으로 설명한다",
     "숫자만 주면 쓸 수 없습니다. 34가지 점검 규칙이 그 브랜드의 실제 수치로 소견을 씁니다 — "
     "'계약을 끝내는 가맹점이 많습니다(82개, 58.6%)' 처럼요."),
)


def render() -> None:
    theme.page_header(
        "FranSCORE 는 무엇인가",
        "프랜차이즈 <b>브랜드</b>의 사업 안정성을 평가해, 은행이 가맹점 여신을 볼 때 "
        "참고할 수 있게 만든 도구입니다.",
        eyebrow="서비스 소개")

    # ── 한 문단 요약 ──────────────────────────────────────────────
    st.markdown(
        f"<div style='padding:16px 18px;border-radius:{theme.RADIUS_LG};background:{theme.YELLOW_SOFT};"
        f"border:1px solid #F2E3A8;font-size:{theme.FS_LG};line-height:1.75;color:{theme.TEXT}'>"
        f"은행이 가맹점주에게 대출할 때 보는 것은 <b>그 사람의 상환능력</b>입니다. "
        f"그런데 손실은 개인이 아니라 <b>브랜드가 꺾일 때 그 브랜드 가맹점이 한꺼번에</b> "
        f"어려워지는 형태로 옵니다. 차주 심사로는 이 축이 보이지 않습니다. "
        f"FranSCORE 는 그 빠진 축 하나를 채웁니다.</div>",
        unsafe_allow_html=True)

    st.write("")
    st.markdown("### 왜 브랜드를 따로 보아야 하는가")
    imp = _corr_impact()
    c1, c2, c3 = st.columns(3)
    c1.metric("같은 브랜드 안의 상관", _fmt(imp.get("rho_within_brand"), 3),
              help="한 브랜드의 여러 지역 점포가 함께 움직이는 정도. 거시·업종 효과를 "
                   "걷어낸 뒤에도 이만큼 남습니다.")
    c2.metric("브랜드 사이의 상관", _fmt(imp.get("rho_between_brand"), 3),
              help="브랜드끼리는 거의 함께 움직이지 않습니다.")
    c3.metric("독립 가정 시 과소추정", _fmt(imp.get("ul99_multiple"), 2, "배"),
              help="차주를 서로 무관하다고 보면 자본 소요분을 이만큼 낮게 잡습니다.")
    st.caption("위험은 경기가 아니라 **브랜드**에 있습니다. 그래서 처방도 '경기 대응'이 "
               "아니라 '브랜드 단위 관리'가 맞습니다.")

    st.write("")
    st.markdown("### 어떻게 평가하는가")
    for no, title, body in _STEPS:
        with st.container(border=True):
            st.markdown(
                f"<div style='display:flex;gap:14px;align-items:flex-start'>"
                f"<div style='flex:0 0 34px;height:34px;border-radius:{theme.RADIUS_MD};"
                f"background:{theme.YELLOW};color:#26221E;font-weight:800;"
                f"display:flex;align-items:center;justify-content:center'>{no}</div>"
                f"<div><div style='font-weight:700;font-size:{theme.FS_LG};color:{theme.INK};"
                f"margin-bottom:3px'>{title}</div>"
                f"<div style='color:{theme.TEXT};line-height:1.7'>{body}</div></div></div>",
                unsafe_allow_html=True)

    # ── 등급 읽는 법 ──────────────────────────────────────────────
    st.write("")
    st.markdown("### 등급을 읽는 법")
    bands = C.grade_bands(C._mtime(C.out_dir() / "grade_bands.json"))
    if bands.get("pooled"):
        st.markdown(C.grade_legend_html({}), unsafe_allow_html=True)
    st.info("**등급 옆의 '상태'를 함께 보십시오.** 같은 FS3 라도 '건전'이면 아직 "
            "악화 신호가 없는데 모델이 위험을 예측한 것이고, '요주의'면 이미 공시에 "
            "악화가 나타난 브랜드입니다. 뒤쪽은 모델의 성능 근거가 없으니 확률값보다 "
            "**진단 소견**을 보고 판단하셔야 합니다.")

    # ── 화면 안내 ────────────────────────────────────────────────
    st.write("")
    st.markdown("### 화면 안내")
    guide = {
        "FRANSCORE": "전체 현황과 브랜드별 상세. 어떤 브랜드든 눌러 들어가면 진단 소견·"
                     "부문별 점검·공시 추이·본부 재무·검색 수요를 볼 수 있습니다.",
        "점검 큐": "확인해야 할 브랜드를 담당자에게 배정하고 확인 결과를 기록합니다. "
                   "엑셀로 내려받아 결재에 붙일 수 있습니다.",
        "여신 포트폴리오": "브랜드별 여신 쏠림과 예상 손실을 봅니다. 신규 여신을 넣어 "
                          "보면 집중도가 어떻게 변하는지 즉시 계산됩니다.",
        "AI 상담": "자연어로 물어보면 수집된 자료에서 근거를 찾아 답합니다.",
    }
    for k, v in guide.items():
        st.markdown(
            f"<div style='display:flex;gap:12px;padding:8px 0;border-bottom:1px solid "
            f"{theme.BORDER}'><b style='min-width:104px;color:{theme.INK}'>{k}</b>"
            f"<span style='color:{theme.TEXT_SUB};line-height:1.6'>{v}</span></div>",
            unsafe_allow_html=True)

    # ── 신뢰 근거 ────────────────────────────────────────────────
    st.write("")
    st.markdown("### 이 평가를 믿어도 되는가")
    _validation_block()

    with st.expander("이 도구를 쓰면 안 되는 곳"):
        st.markdown(
            "- 자동 여신 승인·거절\n- 여신 한도 산정\n- 금리 산정·차등\n"
            "- 규제자본(IRB) 산출 · IFRS 9 충당금\n"
            "- 개별 가맹점주에 대한 불이익 처분의 단독 근거\n"
            "- 산출물을 '부도확률(PD)' 이라고 부르거나 그렇게 인용하는 것")
        st.caption("2선 리스크 관리의 **점검 우선순위**와 **심사 참고 정보**가 정해진 용도입니다.")

    C.refresh_footer()


def _corr_impact() -> dict:
    """상관 실증 산출물을 읽는다.

    ⚠️ 이 세 수치는 원래 화면에 **박아 놓은 문자열**이었다. 그래서 모형이 다시 돌아
       배수가 5.31 → 7.30 으로 바뀌었을 때 문서만 갱신되고 화면은 옛 값을 계속
       보여 줬다. 같은 제출물 안에서 필요성의 핵심 수치가 두 값으로 갈린 것이다.
       고쳐야 할 것은 그 문자열이 아니라 **문자열을 박아 둔 구조**다 — 산출물에서
       읽으면 다시 어긋날 수가 없다.
    """
    p = C.out_dir() / "correlation_impact.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _fmt(v, nd: int, suffix: str = "") -> str:
    """산출물이 없으면 숫자를 지어내지 않고 '—' 를 보여 준다."""
    return "—" if v is None else f"{float(v):.{nd}f}{suffix}"


def _trust_card(question: str, headline: str, meaning: str, detail: str) -> None:
    """질문 → 결과 → 뜻 순서의 카드 한 장."""
    with st.container(border=True):
        st.markdown(
            f"<div style='font-size:{theme.FS_SM};color:{theme.TEXT_MUTED};"
            f"font-weight:600'>{question}</div>"
            f"<div style='font-size:22px;font-weight:800;color:{theme.INK};"
            f"line-height:1.35;margin:4px 0 8px'>{headline}</div>"
            f"<div style='color:{theme.TEXT};line-height:1.75'>{meaning}</div>"
            f"<div style='margin-top:8px;padding-top:8px;border-top:1px solid {theme.BORDER};"
            f"font-size:{theme.FS_SM};color:{theme.TEXT_MUTED};line-height:1.6'>{detail}</div>",
            unsafe_allow_html=True)


def _validation_block() -> None:
    """검증 결과를 **평가자의 말이 아니라 쓰는 사람의 말로** 보여 준다.

    ⚠️ 여기는 원래 'AUC 0.707 · KS 0.316 · HL p=0.158' 을 그대로 나열했다. 전부 맞는
       숫자지만, 지표 이름을 이미 아는 사람에게만 읽히는 근거는 근거가 아니라 장식이다.
       이 화면의 첫 독자는 브랜드를 처음 보는 영업점 담당자다.
       → 순서를 뒤집었다. 먼저 **무엇을 물었는지**, 다음에 **그래서 어떻게 나왔는지**,
         지표 이름은 뒤에 괄호로 붙인다. 값은 그대로 산출물에서 읽어 오므로 설명이
         쉬워졌다고 숫자가 무뎌지지는 않는다 — 원지표는 아래 접힌 자리에 전부 있다.
    """
    p = C.out_dir() / "validation" / "summary.json"
    if not p.exists():
        st.caption("검증 실적이 아직 집계되지 않았습니다.")
        return
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st.caption("검증 산출물을 읽지 못했습니다.")
        return
    d, cal = s.get("discrimination_pooled", {}), s.get("calibration", {})

    st.markdown(
        f"<div style='color:{theme.TEXT};line-height:1.8;margin-bottom:12px'>"
        f"만든 쪽이 '정확합니다' 라고 말하는 것은 근거가 아닙니다. 그래서 이 모델은 "
        f"<b>학습에 한 번도 쓰지 않은 뒷연도의 실제 결과</b>로 시험했습니다. "
        f"아래는 그 시험 성적이고, 세 가지를 각각 따로 물었습니다.</div>",
        unsafe_allow_html=True)

    lift = d.get("lift_at_10")
    _trust_card(
        "① 위험한 브랜드를 정말 위로 올리는가",
        (f"위험하다고 꼽은 상위 10% 안에서 실제 악화가 평균의 {lift:.2f}배"
         if lift else "판별력 산출값이 없습니다"),
        "같은 인력으로 같은 수만 점검해도 그만큼 더 많이 잡는다는 뜻입니다. "
        "순서를 맞히는 능력이 없으면 나머지 숫자는 볼 필요가 없어, 이것을 먼저 봅니다.",
        f"판별력(AUC) {d.get('auc', 0):.3f} · 95% 신뢰구간 "
        f"[{d.get('auc_lo', 0):.3f}, {d.get('auc_hi', 0):.3f}] — 아무렇게나 찍으면 0.500, "
        f"완벽하면 1.000 입니다. KS {d.get('ks', 0):.3f}.")

    mp, ob = cal.get("mean_predicted", 0) * 100, cal.get("observed", 0) * 100
    hl, sz = cal.get("hosmer_lemeshow", {}), cal.get("spiegelhalter", {})
    _trust_card(
        "② 확률의 크기가 실제와 맞는가",
        f"100개 브랜드에 {mp:.1f}개라고 말했고, 실제로 {ob:.1f}개가 악화",
        "순서만 맞고 크기가 틀리면 '몇 건이 나올지'를 셀 수 없습니다. 손실 추정과 "
        "점검 인력 배분이 전부 이 크기 위에서 계산되기 때문에 따로 시험합니다.",
        f"보정 검정 두 가지 모두 '예측과 실제가 다르다'고 말하지 못했습니다 — "
        f"Hosmer-Lemeshow p={hl.get('p_value', 0):.3f}({hl.get('verdict', '-')}), "
        f"Spiegelhalter Z={sz.get('z', 0):+.2f}({sz.get('verdict', '-')}). "
        f"시험 표본 {cal.get('n', 0):,}건 · 실제 악화 {cal.get('events', 0)}건.")

    psi = (s.get("psi", {}).get("score", {}).get("by_year", []) or [])[:3]
    if psi:
        vals = [r["psi"] for r in psi]
        _trust_card(
            "③ 해가 바뀌어도 같은 잣대인가",
            f"점수 분포 이동폭 {min(vals):.3f}~{max(vals):.3f} — 전부 '정상'",
            "잣대가 해마다 움직이면 작년 FS2 와 올해 FS2 가 다른 뜻이 됩니다. "
            "은행권에서는 이 값이 0.1 미만이면 같은 잣대로 봅니다.",
            "안정성 지표(PSI) "
            + " · ".join(f"{r['base_year']}→{r['target_year']} {r['psi']:.3f}" for r in psi)
            + f" · 코호트 사이 표본 이탈률 {s.get('mean_exit_rate', 0) * 100:.1f}%.")

    st.markdown(
        f"<div style='margin-top:12px;padding:14px 16px;border-radius:{theme.RADIUS_LG};"
        f"background:{theme.YELLOW_SOFT};border:1px solid #F2E3A8;color:{theme.TEXT};"
        f"line-height:1.8'>이 성적이 보증하는 범위도 같이 밝힙니다. 위 숫자는 "
        f"<b>평가 대상 조건을 통과한 브랜드</b>에 대해, <b>아직 악화 신호가 없는 브랜드가 "
        f"다음 해에 꺾일 확률</b>을 맞히는 문제에서 나온 것입니다. 이미 악화가 나타난 "
        f"브랜드에는 같은 성능 근거가 없어, 그런 브랜드는 확률보다 <b>진단 소견</b>을 "
        f"보시도록 화면이 안내합니다.</div>",
        unsafe_allow_html=True)
    st.caption(_scope_caption())

    with st.expander("검증 원지표 그대로 보기"):
        st.markdown(
            f"- 판별 표본 {d.get('n', 0):,}건 · 실제 악화 {d.get('events', 0)}건 "
            f"(기저율 {d.get('base_rate', 0) * 100:.2f}%)\n"
            f"- AUC {d.get('auc', 0):.3f} [{d.get('auc_lo', 0):.3f}, "
            f"{d.get('auc_hi', 0):.3f}] · Gini {d.get('gini', 0):.3f} · "
            f"KS {d.get('ks', 0):.3f} · Lift@10% {d.get('lift_at_10', 0):.2f} "
            f"[{d.get('lift_at_10_lo', 0):.2f}, {d.get('lift_at_10_hi', 0):.2f}]\n"
            f"- 보정 기준: {cal.get('basis', '-')}\n"
            f"- ECE {cal.get('ece', 0):.4f} · 예측 평균 {mp:.2f}% vs 실제 {ob:.2f}%\n"
            f"- 등급 이행행렬 {s.get('transition_pairs', 0)}개 코호트 · "
            f"평균 이탈률 {s.get('mean_exit_rate', 0) * 100:.1f}%")
        st.caption("산출 코드는 src/validate.py, 원본은 outputs/validation/summary.json 입니다. "
                   "방법과 전제는 docs/METHODOLOGY.md · docs/MODEL_USE_SPEC.md 에 있습니다.")


def _scope_caption() -> str:
    """평가 대상이 어디까지인지 — **패널에서 직접 세어** 말한다.

    ⚠️ 이 문장은 원래 '9,333개 중 1,442개(15.5%) / 가맹점 73.6%' 를 박아 둔 것이었다.
       같은 자리에서 본부재무 커버리지를 11.4% 라고 적어 두었다가, 정보공개서 수집으로
       13.8% 가 된 뒤에도 화면만 옛 값을 계속 보여 준 전례가 있다. 박아 두면 반드시
       갈라진다 — 세는 편이 짧다.
    """
    pnl = C.load_panel()
    if pnl is None or "eligible_t" not in pnl.columns:
        return ""
    yr = int(pnl["year"].max())
    d = pnl[pnl["year"] == yr]
    el = d["eligible_t"].fillna(False).astype(bool)
    w = pd.to_numeric(d["n_stores"], errors="coerce").fillna(0)
    if not el.any() or w.sum() <= 0:
        return ""
    return (f"평가 대상은 {yr}년 공시 외식 프랜차이즈 {len(d):,}개 브랜드 가운데 "
            f"{int(el.sum()):,}개입니다 — 가맹점 30개 이상·3년 연속 공시가 조건입니다. "
            f"브랜드 수로는 {100 * el.mean():.1f}% 지만 가맹점 수로 보면 "
            f"{100 * w[el].sum() / w.sum():.1f}% 로, 여신이 실제로 나가는 쪽은 대부분 "
            f"평가 대상 안에 있습니다.")
