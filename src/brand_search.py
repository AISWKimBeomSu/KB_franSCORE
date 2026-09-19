"""브랜드 검색 — 사람이 부르는 이름으로 공시 등록명을 찾는다.

왜 필요한가 (실사용에서 드러난 문제):
    사용자가 "메가커피"를 찾지 못했다. 데이터에는 있다 — 다만 공시 등록명이
    **"메가엠지씨커피(MEGA MGC COFFEE)"** 라서 단순 문자열 일치로는 걸리지 않는다.
    공시명은 법인이 등록한 정식 명칭이라 괄호·영문 병기·법인 접두가 섞여 있고,
    사람이 쓰는 통칭과 자주 다르다. 검색이 이 간극을 메우지 못하면
    "데이터에 그 브랜드가 없다"는 잘못된 결론이 난다.

전략 (앞에서부터 순서대로 시도):
    1. 정규화 부분일치 — 공백·괄호·영문·특수문자를 제거해 비교
       ("메가커피" → "메가커피", "메가엠지씨커피(MEGA MGC COFFEE)" → "메가엠지씨커피")
    2. 통칭 별칭 사전 — 위 규칙으로도 안 걸리는 널리 쓰이는 이름을 명시적으로 연결
    3. 초성/부분 토큰 — 앞 2글자 + 뒤 2글자가 모두 포함되면 후보로 (news_llm 과 같은 규칙)
    4. 유사도 순위 — 그래도 없으면 가장 가까운 이름을 제안한다("혹시 이것을 찾으셨나요")

화면의 자동완성은 이 모듈이 아니라 Streamlit selectbox 의 네이티브 타입어헤드가 맡는다
(src/views/franscore.py). 여기서는 **확정된 질의**를 등록명으로 해석하는 일만 한다.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

import pandas as pd

# 사람이 부르는 통칭 → 공시 등록명에 포함된 핵심 토큰.
# 규칙 기반 정규화로 못 잡는 것만 최소한으로 둔다(사전이 커지면 유지보수가 어렵다).
ALIASES: dict[str, str] = {
    "메가커피": "메가엠지씨",
    "메가엠지씨커피": "메가엠지씨",
    "mgc커피": "메가엠지씨",
    "빽다방커피": "빽다방",
    "베스킨라빈스": "배스킨라빈스",
    "베라": "배스킨라빈스",
    "던킨도너츠": "던킨",
    "비비큐": "bbq",
    "비에이치씨": "bhc",
    "씨유": "cu",
    "지에스25": "gs25",
    "쥐에스25": "gs25",
    "파바": "파리바게뜨",
    "뚜쥬": "뚜레쥬르",
    "스벅": "스타벅스",
    "맥날": "맥도날드",
    "롯데리아": "롯데리아",
    "버거킹": "burger king",
}

_STRIP = re.compile(r"[\s()（）\[\]{}·・,.\-_/&'\"]+")


def normalize(s: str) -> str:
    """검색 비교용 정규화: NFKC → 소문자 → 구분기호 제거."""
    t = unicodedata.normalize("NFKC", str(s or "")).lower()
    return _STRIP.sub("", t)


def _korean_only(s: str) -> str:
    """한글만 남긴다 — 영문 병기가 붙은 공시명과 통칭을 맞추기 위한 보조 키."""
    return re.sub(r"[^가-힣]", "", str(s or ""))


def search(df: pd.DataFrame, query: str, col: str = "brand_name",
           limit: int = 50) -> tuple[pd.DataFrame, list[str]]:
    """브랜드 검색.

    반환: (매칭된 행, 제안 목록)
    - 매칭이 있으면 제안은 빈 리스트. 행은 **일치 정도 → 가맹점 수** 순으로 정렬된다.
    - 정규화 이름이 질의와 **정확히 같은** 브랜드가 있으면 그 브랜드만 돌려준다.
    - 매칭이 없으면 가장 가까운 이름 몇 개를 제안한다(오타·통칭 대응).

    ⚠️ 예전에는 부분일치 결과를 **정렬 없이 앞 50건에서 잘랐다.** 두 가지가 깨졌다.
       ① 자동완성에서 "DDC치킨"을 골라도 한글만 떼어 본 '치킨'이 치킨 브랜드 전부와
          맞아, 정작 DDC치킨은 카드 6장 밖으로 밀려났다(자기 이름으로 한 건에 딱 맞지
          않는 브랜드가 47개).
       ② "커피"로 찾으면 파일 순서(위험순) 앞 50건만 남아 컴포즈·이디야 같은 대형
          브랜드가 빠졌다 — 화면은 "가맹점 수가 많은 순"이라고 적고 있었는데도.
       그래서 일치 정도를 등급으로 매겨 먼저 줄세우고, 같은 등급 안에서는 가맹점 수로,
       그다음에 자른다.
    """
    q = normalize(query)
    if not q:
        return df.head(0), []

    names = df[col].astype(str)
    norm = names.map(normalize)
    kor = names.map(_korean_only)

    # 별칭 → 핵심 토큰으로 치환해서도 시도.
    # '정확 일치'는 질의 자체와 그 질의의 **직접** 별칭만 인정한다. 부분 문자열로 걸린
    # 별칭까지 정확 일치로 치면 "커피" 한 단어가 별칭 '빽다방커피'를 거쳐 빽다방 한 건으로
    # 좁혀진다(실측). 느슨한 별칭은 부분일치 후보를 넓히는 데만 쓴다.
    direct = {q}
    if q in ALIASES:
        direct.add(normalize(ALIASES[q]))
    loose = set(direct)
    for alias, token in ALIASES.items():
        if q in alias or alias in q:
            loose.add(normalize(token))

    # 일치 등급 (작을수록 가깝다). 한 행은 가장 가까운 등급 하나만 갖는다.
    #   0 정확 일치 · 1 부분일치 · 3 한글만 비교 · 4 앞2·뒤2 토큰
    # 앞부분 일치를 따로 올리지 않는다 — "치킨" 같은 업종어로 찾으면 이름이 '치킨…'으로
    # 시작하는 소형 브랜드가 대형 브랜드를 전부 밀어낸다. 같은 등급 안에서는 가맹점 수 순.
    tier = pd.Series(9, index=df.index, dtype=int)
    exact = norm.isin(direct)
    tier[exact] = 0
    tier[(tier > 1) & pd.concat([norm.str.contains(re.escape(k), na=False) for k in loose],
                                axis=1).any(axis=1)] = 1
    # 한글만 떼어 비교하는 보조 키는 **질의 자체가 한글뿐일 때만** 쓴다.
    # 'DDC치킨' 처럼 영문이 섞인 질의에서 '치킨'만 남기면 업종명 검색이 돼 버린다.
    kq = _korean_only(q)
    if kq and kq == q and len(kq) >= 2:
        tier[(tier > 3) & kor.str.contains(re.escape(kq), na=False)] = 3

    # 앞2·뒤2 토큰 규칙 (예: '메가커피' vs '메가엠지씨커피') — 앞의 규칙이 모두 실패했을 때만
    if not (tier < 9).any() and len(q) >= 4:
        head, tail = re.escape(q[:2]), re.escape(q[-2:])
        tier[norm.str.contains(head, na=False) & norm.str.contains(tail, na=False)] = 4

    if (tier < 9).any():
        if exact.any():
            tier = tier.where(exact, 9)          # 정확히 그 이름이 있으면 그것만
        hit = df[tier < 9].assign(_tier=tier[tier < 9])
        size = (pd.to_numeric(hit["n_stores"], errors="coerce").fillna(0)
                if "n_stores" in hit.columns else pd.Series(0, index=hit.index))
        hit = (hit.assign(_size=size)
                  .sort_values(["_tier", "_size"], ascending=[True, False], kind="stable")
                  .drop(columns=["_tier", "_size"]))
        return hit.head(limit), []

    # 근접 제안
    sugg = difflib.get_close_matches(q, norm.dropna().unique().tolist(), n=5, cutoff=0.5)
    proposals = names[norm.isin(sugg)].drop_duplicates().tolist()[:5]
    return df.head(0), proposals
