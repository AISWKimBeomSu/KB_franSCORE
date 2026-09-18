"""AI 상담 — 자연어로 묻고 근거와 함께 답을 받는다."""
from __future__ import annotations

import streamlit as st

from src import chat, theme
from src.views import common as C

_HISTORY = "chat_history"
_PENDING = "chat_pending"
EXAMPLES = [
    "인생냉면 창업을 고민 중인데 전반적으로 분석해줘",
    "달콤왕가탕후루 가맹점 수 추이를 가져와줘",
    "메가커피와 컴포즈커피 중 어디가 더 안정적이야?",
    "치킨 업종에서 지금 가장 위험한 브랜드는?",
]


def render() -> None:
    theme.page_header(
        "AI 상담",
        "프랜차이즈에 대해 자유롭게 물어보십시오. 공정거래위원회 공시·금융감독원 "
        "감사보고서·뉴스에서 근거를 찾아 답합니다.",
        eyebrow="상담")

    if _HISTORY not in st.session_state:
        st.session_state[_HISTORY] = []
    history = st.session_state[_HISTORY]

    if not history:
        st.markdown("##### 이렇게 물어보실 수 있습니다")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                _enqueue(ex)
                st.rerun()
        st.write("")

    # ⚠️ chat_message 의 avatar 는 이모지·이미지 경로·URL 만 받는다. '◆'(U+25C6) 같은
    #    기호 문자를 넣으면 StreamlitAPIException 으로 화면 전체가 죽는다(실측).
    for turn in history:
        with st.chat_message("user" if turn["role"] == "user" else "assistant",
                             avatar="🙋" if turn["role"] == "user" else "🟡"):
            st.markdown(turn["content"])
            if turn.get("evidence"):
                _evidence_block(turn["evidence"], turn.get("idx", 0))
            # 실패 원인을 정확히 알린다. 키 미설정과 무료 한도 초과는 사용자가 할
            # 일이 전혀 다르다 — 전자는 키 등록, 후자는 잠시 기다리기다.
            # 대체 모델이 답했다면 **그 사실을 답 옆에** 적는다. 성능을 정량 평가한
            # 모델은 config 의 llm.model 하나뿐이고, 대체 모델에는 그 근거가 없다.
            # 조용히 갈아타면 "평가한 모델로 답한다"는 문서가 거짓이 된다.
            if turn["role"] == "assistant" and turn.get("llm_used") and turn.get("model"):
                pinned = str(C.cfg()["llm"].get("model", ""))
                used = str(turn["model"])
                if pinned and pinned.split("-preview")[0] not in used:
                    st.caption(f"이 답변은 **{used}** 가 작성했습니다 — 등록된 키가 "
                               f"평가 모델({pinned})을 쓸 수 없어 대체 모델로 답했습니다. "
                               f"성능 수치는 평가 모델 기준입니다.")
            if turn["role"] == "assistant" and not turn.get("llm_used", True):
                st.caption({
                    "rate_limit": "등록된 키가 모두 분당 호출 한도에 걸렸습니다. "
                                  "1~2분 뒤 다시 물어보세요.",
                    "rate_limit_day": "등록된 키가 모두 무료 등급 일일 한도를 다 썼습니다. "
                                      "한국시간 오후 4시경 초기화됩니다.",
                    "model_unavailable": "등록된 키가 지금 설정된 모델을 쓸 수 없습니다.",
                    "no_key": "답변 생성 모델이 설정되지 않아 수집된 사실만 정리했습니다.",
                    "bad_key": "등록된 키가 Gemini API 키 형식(AIza…)이 아닙니다.",
                    "auth": "등록된 키가 인증을 통과하지 못했습니다.",
                }.get(turn.get("reason", ""), "답변 생성에 실패해 수집된 사실만 정리했습니다."))

    # ⚠️ 답을 만든 뒤에 질문까지 한꺼번에 그리면, 사용자가 엔터를 친 뒤 수십 초 동안
    #    **자기가 뭘 물었는지도 화면에 안 보인다**. 질문을 먼저 세션에 넣고 즉시 rerun 해
    #    말풍선을 띄운 다음, 다음 실행에서 답을 만든다.
    pending = st.session_state.get(_PENDING)
    if pending:
        with st.chat_message("assistant", avatar="🟡"), st.spinner(
                "공시·재무·뉴스에서 근거를 찾는 중…"):
            _answer(pending)
        st.rerun()

    q = st.chat_input("무엇이든 물어보십시오")
    if q:
        _enqueue(q)
        st.rerun()

    if history:
        c1, c2 = st.columns([1, 5])
        if c1.button("대화 지우기"):
            st.session_state[_HISTORY] = []
            st.rerun()
        c2.download_button(
            "대화 내려받기 (.md)",
            "\n\n".join(f"**{'질문' if t['role'] == 'user' else '답변'}**\n\n{t['content']}"
                        for t in history).encode("utf-8-sig"),
            file_name="franscore_상담.md", mime="text/markdown")


def _enqueue(question: str) -> None:
    """질문만 먼저 대화에 올린다. 답변 생성은 다음 실행에서."""
    st.session_state[_HISTORY].append({"role": "user", "content": question})
    st.session_state[_PENDING] = question


def _answer(question: str) -> None:
    history = st.session_state[_HISTORY]
    st.session_state[_PENDING] = None
    try:
        res = chat.answer(C.cfg(), question, history[:-1])
    except Exception as exc:                          # 화면이 죽으면 안 된다
        history.append({"role": "assistant",
                        "content": f"답변 중 문제가 발생했습니다: {exc}",
                        "llm_used": False, "reason": "error"})
        return
    history.append({"role": "assistant", "content": res["text"],
                    "evidence": res.get("evidence") or [],
                    "llm_used": bool(res.get("llm_used")),
                    "reason": res.get("reason", ""),
                    "intent": res.get("intent", ""),
                    "model": res.get("model") or "",
                    "idx": len(history)})


def _evidence_block(evidence: list[dict], idx: int) -> None:
    if not evidence:
        return
    with st.expander(f"근거로 쓴 자료 {len(evidence)}건"):
        for i, e in enumerate(evidence):
            kind = str(e.get("출처유형", ""))
            chip = theme.chip(kind or "문서",
                              "Info" if kind == "실시간뉴스" else "Neutral")
            url = str(e.get("url") or "")
            head = f"{chip} <b>{e.get('출처', '')}</b>"
            if e.get("발행"):
                head += (f"<span style='color:{theme.TEXT_MUTED};font-size:{theme.FS_SM}'>"
                         f" · {e['발행']}</span>")
            st.markdown(head, unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:{theme.FS_MD};color:{theme.TEXT_SUB};line-height:1.55;"
                f"margin:2px 0 10px 0'>{str(e.get('내용', ''))[:400]}…</div>",
                unsafe_allow_html=True)
            if url and url.startswith("http"):
                st.markdown(f"[원문 보기]({url})")
            del i
    del idx
