"""보정기 회귀 방지 — 조용히 틀리는 종류의 결함을 잡는다.

왜 이 테스트가 있는가 (실제로 난 사고)
    ShrunkIsotonic.predict() 가 `self.mapping.get(계단값, self.base)` 로 조회했다.
    그런데 IsotonicRegression.predict() 는 학습 knot **사이**의 입력에 대해 선형
    보간값을 돌려준다 — 그 값은 mapping 의 키가 아니다. 조회가 실패하면 조용히
    기저율(base)로 대체됐다.

    실측(2024 코호트 1,442행): 14행이 이 경로를 탔고, 그 결과 **12건의 등급이
    틀렸다 — 11건이 FS2 로 표시되었으나 실제로는 FS3**, 즉 위험을 낮게 보여줬다.
    더 나쁜 것은 단조성이 깨졌다는 점이다: 보간 구간의 브랜드만 평균값을 받으므로
    원점수가 더 나쁜데 등급이 더 좋은 구간(FS3→FS2→FS3)이 실재했다.

    예외도 로그도 없었다. 확률은 그럴듯한 값이었고 아무것도 실패하지 않았다.
    이런 결함은 사람이 아니라 테스트가 잡아야 한다.

실행: python tests/test_calibration.py
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    RESULTS.append((ok, name, detail))


def _deploy():
    import joblib
    p = _ROOT / "outputs" / "calibrator_deploy.joblib"
    return joblib.load(p)["model"] if p.exists() else None


def test_no_silent_base_fallback() -> None:
    """계단 사이 입력이 기저율로 조용히 대체되지 않는가.

    이것이 실제로 났던 사고다. 보정기 출력은 **반드시** 계단값 중 하나여야 한다.
    """
    cal = _deploy()
    if cal is None:
        check(True, "계단 대체 없음", "보정기 없음 — 건너뜀")
        return
    levels = np.array(sorted(cal.mapping.values()), dtype=float)
    # 계단 사이를 일부러 겨냥한 입력을 만든다
    knots = np.array(sorted(cal.mapping), dtype=float)
    probe = np.concatenate([knots, (knots[:-1] + knots[1:]) / 2.0,
                            np.linspace(0.0, 1.0, 501)])
    out = np.asarray(cal.predict(probe), dtype=float)
    off = ~np.isin(np.round(out, 10), np.round(levels, 10))
    check(off.sum() == 0, "보정기 출력이 전부 계단값",
          f"계단 밖 출력 {int(off.sum())}/{len(probe)}건")

    # 기저율이 계단값이 아니라면, 기저율이 출력에 등장하는 것 자체가 사고의 흔적이다
    base_is_level = bool(np.isin(round(float(cal.base), 10), np.round(levels, 10)))
    if not base_is_level:
        n_base = int(np.isclose(out, cal.base).sum())
        check(n_base == 0, "기저율 대체가 출력에 없음", f"기저율 출력 {n_base}건")
    else:
        check(True, "기저율 대체가 출력에 없음", "기저율이 계단값과 일치 — 판별 불가")


def test_monotone() -> None:
    """원점수가 나빠지면 보정확률도 나빠져야 한다 (등온회귀의 정의)."""
    cal = _deploy()
    if cal is None:
        check(True, "단조성", "보정기 없음 — 건너뜀")
        return
    x = np.linspace(0.0, 1.0, 2001)
    p = np.asarray(cal.predict(x), dtype=float)
    bad = int((np.diff(p) < -1e-12).sum())
    check(bad == 0, "보정확률 단조 증가", f"역행 {bad}곳")


def test_grades_monotone_in_raw() -> None:
    """산출물에서 원점수 순으로 등급이 역행하지 않는가."""
    import pandas as pd
    p = _ROOT / "outputs" / "scores_latest.csv"
    if not p.exists():
        check(True, "등급 단조성", "점수표 없음 — 건너뜀")
        return
    sc = pd.read_csv(p, encoding="utf-8-sig").sort_values("score_raw")
    rank = {"FS1": 0, "FS2": 1, "FS3": 2}
    g = sc["grade"].map(rank).to_numpy()
    bad = int((np.diff(g) < 0).sum())
    check(bad == 0, "등급이 원점수에 대해 단조", f"역행 {bad}곳")


def test_grade_matches_validation_scale() -> None:
    """화면 등급이 **검증이 쓰는 척도**로 매겨졌는가.

    컷은 tools/derive_grade_bands.py 가 보정기 계단값 위에서 도출했고 공표 실현율도
    그 척도의 실적이다. 표시용 보간값에 컷을 적용하면 등급 범례가 설명하는 대상과
    화면의 등급이 달라진다(실측: 48개 브랜드).
    """
    import json

    import pandas as pd
    sp = _ROOT / "outputs" / "scores_latest.csv"
    bp = _ROOT / "outputs" / "grade_bands.json"
    if not (sp.exists() and bp.exists()):
        check(True, "등급 척도 일치", "산출물 없음 — 건너뜀")
        return
    sc = pd.read_csv(sp, encoding="utf-8-sig")
    bands = json.loads(bp.read_text(encoding="utf-8"))
    cuts, grades = bands["cuts"], np.array(bands["grades"], dtype=object)
    expect = grades[np.digitize(sc["deterioration_step"].to_numpy(), cuts)]
    bad = int((sc["grade"].to_numpy() != expect).sum())
    check(bad == 0, "화면 등급 == 검증 척도 등급", f"불일치 {bad}건")

    # 표시값이 자기 등급 구간 밖으로 나가면 "16.4% 인데 관찰" 같은 모순이 보인다
    edges = [0.0, *cuts, 1.0]
    idx = np.digitize(sc["deterioration_step"].to_numpy(), cuts)
    lo = np.array(edges[:-1], dtype=float)[idx]
    hi = np.array(edges[1:], dtype=float)[idx]
    out = int(((sc["deterioration_1y"] < lo) | (sc["deterioration_1y"] >= hi)).sum())
    check(out == 0, "표시 확률이 등급 구간 안", f"구간 밖 {out}건")


def main() -> int:
    for fn in (test_no_silent_base_fallback, test_monotone,
               test_grades_monotone_in_raw, test_grade_matches_validation_scale):
        try:
            fn()
        except Exception as exc:
            check(False, fn.__name__, f"예외: {exc}")
    fails = [r for r in RESULTS if not r[0]]
    for ok, name, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}")
    print(f"=== {len(RESULTS) - len(fails)}/{len(RESULTS)} passed ===")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
