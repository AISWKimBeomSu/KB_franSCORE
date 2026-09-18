"""정보공개서 API 키가 **어느 체계의 키인지** 10초 안에 판정한다.

왜 이 도구가 필요한가
    본부 재무 커버리지 11.4% 를 여는 유일한 열쇠가 정보공개서 정식 키인데, "어디서
    받는 키인가"가 헷갈린다. 공공데이터포털(data.go.kr)과 공정위 가맹사업정보제공시스템
    (franchise.ftc.go.kr)은 **서로 다른 키 체계**이고, 우리가 쓰는 엔드포인트는 후자다.

    실측(2026-08-03, `python tools/probe_ifrmp_key.py --selftest`):

        sampleKey(공정위 공개 데모키)  → 인증 통과 (REQUIRE [yr] PARAMETER)
        data.go.kr 계열 키             → INVALID_KEY
        아무 문자열                    → INVALID_KEY      ← 위와 **완전히 동일**

    즉 이 엔드포인트는 data.go.kr 키를 쓰레기 문자열과 구분하지 않는다. 그러므로
    공공데이터포털에서 키를 받아도 이 API 는 열리지 않는다.

    ✅ 남아 있던 미확인 사항을 닫았다 (2026-08-03, 개인 data.go.kr 키 확보 후).
       "data.go.kr 에 같은 자료를 프록시하는 별도 엔드포인트가 있지 않을까"가 유일한
       미확인 가능성이었다. **없다.** 그리고 왜 없는지까지 확인했다.

       ① data.go.kr 의 정보공개서 데이터셋(15125569 목록 / 15125571 본문)은
          **API 유형이 「LINK」** 다. LINK 형은 data.go.kr 이 자료를 호스팅하지 않고
          제공기관 시스템으로 연결만 한다. 즉 실제 엔드포인트가 처음부터
          franchise.ftc.go.kr 이고, data.go.kr 은 안내판 역할이다.
       ② 그 결과 `apis.data.go.kr/1130000/` 아래에는 정보공개서 엔드포인트가 **존재하지
          않는다.** 실재 여부는 응답코드로 갈린다 — 실재하는 주소는 키가 없어도 403
          (SERVICE_KEY_IS_NOT_REGISTERED_ERROR), 없는 주소는 400 이다. 이 판별로
          목록·본문 후보 9개를 찔러 전부 400 임을 확인했다(대조군 2개로 판별력 확인).
       ③ 개인 data.go.kr 정식 키(88자 Base64)를 franchise.ftc.go.kr 에 그대로 대면
          여전히 INVALID_KEY 다.

       세 갈래가 같은 곳을 가리킨다: **공공데이터포털 키로는 정보공개서를 열 수 없고,
       그것은 우리 설정 문제가 아니라 등록 유형(LINK)에서 오는 구조다.** 필요한 것은
       공정위 가맹사업정보제공시스템이 직접 발급하는 키다.

사용:
    IFRMP_SERVICE_KEY=<받은 키> python tools/probe_ifrmp_key.py
    python tools/probe_ifrmp_key.py --selftest      # 키 없이 세 가지 대조군만

판정:
    [정식] 인증 통과 + 서로 다른 sn 이 서로 다른 문서를 준다 → 전량 수집 가능
    [데모] 인증은 통과하나 sn 을 무엇으로 줘도 같은 문서 1건 → 브랜드별 수집 불가
    [무효] INVALID_KEY → 이 체계의 키가 아니다
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import sys
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.common import load_secrets  # noqa: E402
from src.ifrmp import _HEADERS, BASE, DEMO_KEY, KEY_ENV  # noqa: E402

# 데모 고정응답 탐지용 — 서로 다른 문서여야 정상이다.
_PROBE_SN = ("149646", "150572", "150958")
_TIMEOUT = 25


def _call(key: str, params: dict) -> tuple[int, str]:
    r = requests.get(BASE, params={"serviceKey": key, **params},
                     headers=_HEADERS, timeout=_TIMEOUT)
    return r.status_code, r.text


def _err(body: str) -> str | None:
    m = re.search(r"<errorCn>(.*?)</errorCn>", body)
    return m.group(1) if m else None


def classify(key: str, label: str) -> str:
    """키 하나를 판정해 [정식]/[데모]/[무효]/[오류] 중 하나를 돌려준다."""
    try:
        _, body = _call(key, {"type": "list", "pageNo": 1, "numOfRows": 1, "yr": 2023})
    except Exception as exc:
        print(f"  {label:34s} 통신 실패 — {type(exc).__name__}: {exc}")
        return "오류"

    e = _err(body)
    if e == "INVALID_KEY":
        print(f"  {label:34s} [무효]  INVALID_KEY — 이 체계의 키가 아니다")
        return "무효"
    if e:
        # 인증은 통과했으나 파라미터 문제 — 키 자체는 유효하다는 뜻
        print(f"  {label:34s} [통과]  인증 OK (파라미터 오류: {e})")

    # 본문(content)을 서로 다른 sn 으로 받아 고정응답인지 본다
    hashes = {}
    for sn in _PROBE_SN:
        try:
            _, doc = _call(key, {"type": "content", "jngIfrmpSn": sn})
        except Exception:
            continue
        if _err(doc) == "INVALID_KEY":
            print(f"  {label:34s} [무효]  content 단계에서 INVALID_KEY")
            return "무효"
        hashes[sn] = hashlib.sha256(doc.encode("utf-8", "replace")).hexdigest()

    n = len(set(hashes.values()))
    if not hashes:
        print(f"  {label:34s} [오류]  본문 응답 없음")
        return "오류"
    if n == 1:
        print(f"  {label:34s} [데모]  sn {len(hashes)}건이 **같은 문서** — 브랜드별 수집 불가")
        return "데모"
    print(f"  {label:34s} [정식]  sn {len(hashes)}건이 서로 다른 문서 {n}종 — 전량 수집 가능")
    return "정식"


def main() -> int:
    ap = argparse.ArgumentParser(description="정보공개서 API 키 체계 판정")
    ap.add_argument("--selftest", action="store_true",
                    help="키 없이 대조군 3종만 확인 (공개 데모키 · data.go.kr 계열 · 무효)")
    args = ap.parse_args()

    load_secrets()
    print(f"엔드포인트: {BASE}\n")

    if args.selftest:
        from src.collect import _PUBLIC_PREVIEW_KEY
        print("대조군 (키 체계가 다르다는 것을 보이는 검사):")
        classify(DEMO_KEY, "공정위 공개 데모키(sampleKey)")
        classify(_PUBLIC_PREVIEW_KEY, "data.go.kr 계열 키")
        classify("THIS_IS_NOT_A_KEY", "아무 문자열")
        print("\n→ data.go.kr 키가 '아무 문자열'과 같은 판정을 받으면 별도 체계다.")
        return 0

    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        print(f"{KEY_ENV} 가 비어 있다. 키를 넣고 다시 실행하거나 --selftest 를 쓴다.")
        print(f"  예:  IFRMP_SERVICE_KEY=<받은 키> python {Path(__file__).name}")
        return 2

    print(f"판정 대상: {KEY_ENV} (길이 {len(key)}자)")     # 키 값 자체는 절대 찍지 않는다
    verdict = classify(key, "받은 키")
    print()
    if verdict == "정식":
        print("→ 다음 명령으로 전량 수집을 시작한다:  python -m src.ifrmp")
        print("  ⚠️ 11,751건 규모다. 마감이 가까우면 착수 전에 소요 시간을 먼저 재라.")
        return 0
    if verdict == "데모":
        print("→ 인증은 되지만 본문이 고정 1건이다. 정식 키가 아니다.")
        return 1
    print("→ 이 엔드포인트의 키가 아니다. 공정위 가맹사업정보제공시스템 발급분이 필요하다.")
    print("  문의: 웹사이트 전산 070-4190-8609 · 가맹사업거래 상담 1670-0007 (평일 9~18시)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
