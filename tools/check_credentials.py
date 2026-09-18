"""자격증명 점검 — 지금 이 환경이 **실제로** 무엇을 할 수 있는지 찍는다.

왜 만들었나 (실측)
    일간 배치가 GitHub Actions 에서 매일 `success` 로 끝나고 있었다. 실행 로그를
    열어 보니 이랬다:

        NCP_API_KEY_ID:                       ← 빈 값
        뉴스 원천: Google News RSS(제목만)
        LLM 미사용 (env GEMINI_API_KEY 없음) — 규칙기반 키워드 폴백으로 추출

    시크릿이 **하나도** 등록돼 있지 않았다(`gh secret list` 가 공백). 폴백이 제대로
    동작한 덕분에 어떤 단계도 실패하지 않았고, 그래서 성능이 줄어든 채로 매일
    초록불이 켜졌다. 폴백은 옳은 설계지만, **줄어든 사실이 안 보이는 것**은 결함이다.

    이 스크립트는 배치보다 먼저 돌아서 그 사실을 로그 맨 앞에 세운다.

정직성
    · 키 **값**은 절대 출력하지 않는다. 있음/없음만 본다.
    · 키가 없다고 실패로 끝내지 않는다(0 을 반환). 폴백으로 돌아가는 것이 정상 운영
      경로이기 때문이다. 대신 무엇이 축소 가동인지 명시한다.
    · `--strict` 를 주면 축소 가동을 실패로 취급한다 — 전 기능이 필요한 배포 전
      점검용이다.

실행:
    python tools/check_credentials.py
    python tools/check_credentials.py --strict
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.common import capability_report  # noqa: E402

BAR = "=" * 74


def main() -> int:
    ap = argparse.ArgumentParser(description="자격증명·가동능력 점검")
    ap.add_argument("--strict", action="store_true",
                    help="축소 가동이 하나라도 있으면 실패(1)로 끝낸다")
    args = ap.parse_args()

    rep = capability_report()

    print(BAR)
    print(f"가동 능력  {rep['n_full']}/{rep['n_total']} 기능 정상")
    print(BAR)
    for f in rep["features"]:
        mark = "정상" if f["full"] else "축소"
        print(f"  [{mark}] {f['name']}")
        if not f["full"]:
            print(f"         필요: {'  또는  '.join(f['needs'])}")
            print(f"         대체: {f['fallback']}")
            print(f"         영향: {f['impact']}")

    missing = sorted(k for k, v in rep["keys"].items() if not v)
    if missing:
        print()
        print(f"미설정 키 {len(missing)}개: {', '.join(missing)}")

    if rep["degraded"]:
        print()
        print(f"⚠️ 축소 가동 {len(rep['degraded'])}건 — {' · '.join(rep['degraded'])}")
        print("   폴백으로 계속 진행한다. 이 사실은 outputs/refresh_state.json 의")
        print("   capability 블록과 서비스 화면에 그대로 표기된다.")
        if args.strict:
            print("   --strict 지정 — 실패로 처리한다.")
            return 1
    else:
        print()
        print("전 기능 정상 가동.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
