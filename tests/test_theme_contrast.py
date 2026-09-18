"""디자인 토큰 회귀 방지 — 명도대비와 타이포 스케일을 매번 다시 계산한다.

왜 테스트로 두는가
    색은 눈으로 고치면 눈으로 다시 무너진다. 실제로 이 저장소는 팔레트에
    "은행 화면이므로 채도를 낮춘다"는 의도를 주석으로 적어 두고도, 그 결과가
    본문 배경 위에서 몇 대 몇인지는 한 번도 재지 않았다. 그래서
        관찰 #D9860F 2.59 · 안정 #1F8A5B 3.94 · 주의 #C8433B 4.42
        보조텍스트 #9A9289 2.79 · 사이드바 메뉴 1.14
    가 오래 살아 있었다. 숫자를 지키는 일은 사람이 아니라 테스트가 한다.

기준 (WCAG 2.1)
    1.4.3  본문 글자        4.5:1
           굵은 18.66px+ / 24px+ 글자   3:1
    1.4.11 도형·그래프·경계  3:1

실행: python tests/test_theme_contrast.py
"""
from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows 기본 콘솔(cp949)에서 한글이 깨지면 결과를 읽을 수 없다
for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src import theme

AA_TEXT = 4.5
AA_SHAPE = 3.0


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    ch = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in ch]
    return 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2]


def contrast(fg: str, bg: str) -> float:
    a, b = _lum(fg), _lum(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _check(results: list, ok: bool, label: str, got: float, need: float) -> None:
    results.append((ok, label, got, need))


def run() -> int:
    T = theme
    R: list = []

    # 표준값 자체 검증 — 계산식이 틀리면 나머지 결과가 전부 무의미하다
    for label, got, want in (("검정/흰", contrast("#000000", "#FFFFFF"), 21.00),
                             ("#767676/흰", contrast("#767676", "#FFFFFF"), 4.54)):
        _check(R, abs(got - want) < 0.01, f"[자체검증] {label} = {want}", got, want)

    # 본문 계열 — 페이지 배경과 카드 배경 두 곳 모두에서 성립해야 한다
    for name in ("INK", "TEXT", "TEXT_SUB", "TEXT_MUTED"):
        for bg_name in ("BG", "SURFACE"):
            c = contrast(getattr(T, name), getattr(T, bg_name))
            _check(R, c >= AA_TEXT, f"{name} on {bg_name}", c, AA_TEXT)

    # 의미색 글자 — 배경·카드·자기 소프트 틴트 위 전부
    for name, soft in (("DANGER", "DANGER_SOFT"), ("WARN", "WARN_SOFT"),
                       ("SAFE", "SAFE_SOFT"), ("INFO", "INFO_SOFT")):
        for bg_name in ("BG", "SURFACE", soft):
            c = contrast(getattr(T, name), getattr(T, bg_name))
            _check(R, c >= AA_TEXT, f"{name} on {bg_name}", c, AA_TEXT)

    # 의미색 도형 — 차트 막대·신호등 점. 글자보다 낮은 기준이지만 지켜야 한다
    for name in ("DANGER_FILL", "WARN_FILL", "SAFE_FILL", "INFO_FILL"):
        for bg_name in ("BG", "SURFACE"):
            c = contrast(getattr(T, name), getattr(T, bg_name))
            _check(R, c >= AA_SHAPE, f"{name} on {bg_name} (도형)", c, AA_SHAPE)

    # 사이드바 — 여기가 무너지면 메뉴가 안 보인다
    for name in ("NAV_TEXT", "NAV_SUB"):
        c = contrast(getattr(T, name), T.NAV_BG)
        _check(R, c >= AA_TEXT, f"{name} on NAV_BG", c, AA_TEXT)
    c = contrast(T.NAV_ACTIVE_TEXT, T.NAV_ACTIVE_BG)
    _check(R, c >= AA_TEXT, "NAV_ACTIVE_TEXT on NAV_ACTIVE_BG", c, AA_TEXT)
    c = contrast(T.NAV_TEXT, T.NAV_HOVER_BG)
    _check(R, c >= AA_TEXT, "NAV_TEXT on NAV_HOVER_BG", c, AA_TEXT)

    # 링크
    c = contrast(T.INFO, T.SURFACE)
    _check(R, c >= AA_TEXT, "링크(INFO) on SURFACE", c, AA_TEXT)

    fails = [r for r in R if not r[0]]
    for ok, label, got, need in R:
        if not ok:
            print(f"  FAIL  {label:44s} {got:5.2f}  (필요 {need})")
    print(f"명도대비 {len(R) - len(fails)}/{len(R)} 통과")

    # ── 타이포 스케일 ────────────────────────────────────────────────
    scale = [T.FS_XS, T.FS_SM, T.FS_MD, T.FS_BASE, T.FS_LG, T.FS_XL, T.FS_2XL, T.FS_3XL]
    px = [float(s.replace("rem", "")) * 16 for s in scale]
    t_fail = 0
    if px != sorted(px):
        print(f"  FAIL  타이포 스케일이 오름차순이 아님: {px}")
        t_fail += 1
    if len(set(px)) != len(px):
        print(f"  FAIL  타이포 스케일에 중복: {px}")
        t_fail += 1
    if px[0] < 13:
        print(f"  FAIL  최소 글자 {px[0]}px < 13px")
        t_fail += 1
    print(f"타이포 스케일 {len(px)}단계 {px[0]:.0f}~{px[-1]:.0f}px - "
          f"{'통과' if not t_fail else '실패'}")

    # ── 화면 코드가 스케일 밖 크기를 쓰지 않는지 ──────────────────────
    stray = []
    for f in sorted((_ROOT / "src" / "views").glob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(r"font-size:\s*([\d.]+)(rem|px)", line):
                if m.group(2) == "px" and "{" in line[:m.start()]:
                    continue        # 타일 크기에 비례하는 글자마크는 예외
                stray.append(f"{f.name}:{i} {m.group(0)}")
    for s in stray:
        print(f"  FAIL  스케일 밖 크기: {s}")
    print(f"화면 코드 인라인 크기 - 스케일 이탈 {len(stray)}건")

    # ── 진입 스크립트는 theme 의 '상수'를 쓰지 않는다 ──────────────────
    # 배포 환경에서 src/app.py 는 매번 새로 실행되지만 import 된 src.theme 은
    # 옛 버전이 sys.modules 에 남을 수 있다. 그 상태에서 새 app.py 가 새 상수를
    # 참조하면 AttributeError 로 **앱 전체가 뜨지 않는다**(실측: theme.FS_XS).
    # 화면 모듈(views/*.py)은 theme 과 같은 세대로 함께 낡으므로 문제가 없지만,
    # 진입 스크립트만은 예외다. 여기서는 클래스 이름만 적고 모양은 CSS 가 정한다.
    app_src = (_ROOT / "src" / "app.py").read_text(encoding="utf-8")
    app_code = "\n".join(ln for ln in app_src.splitlines()
                         if not ln.lstrip().startswith("#"))
    consts = sorted(set(re.findall(r"theme\.([A-Z][A-Z0-9_]*)", app_code)))
    for c in consts:
        print(f"  FAIL  app.py 가 theme.{c} 를 직접 참조 — CSS 클래스로 옮길 것")
    print(f"진입 스크립트 theme 상수 참조 {len(consts)}건")

    bad = len(fails) + t_fail + len(stray) + len(consts)
    print("=" * 58)
    print("전체 통과" if not bad else f"실패 {bad}건")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(run())
