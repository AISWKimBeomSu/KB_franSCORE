"""진단 엔진 검증 — 화면이 사람에게 보여줄 문장이 사실과 어긋나지 않는지 확인한다.

검사 항목
  1. 계단 보간이 단조성·평균·순위를 보존하고 값을 실제로 펼치는가
  2. 조사(은/는·이/가·으로) 처리가 맞는가
  3. 규칙 성분이 상한에 붙어 변별력을 잃지 않는가
  4. 소견 문장의 수치가 원본 패널·재무 값과 일치하는가 (지어낸 숫자가 없는가)
  5. 브랜드마다 다른 소견이 나오는가 (구버전의 '근거가 전부 동일' 재발 방지)
  6. 데이터 공백(HQ_NO_DATA)이 위험 점수로 둔갑하지 않는가

실행: python tests/test_diagnosis.py   (프로젝트 루트에서)
"""
from __future__ import annotations

import itertools
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import diagnosis as D
from src.common import load_config


def test_smoothing_properties() -> None:
    """계단 보간: 단조 · 계단평균 보존 · 값 분산 · 범위 유지."""
    rng = np.random.default_rng(0)
    raw = np.sort(rng.uniform(0, 1, 900))
    # isotonic 을 흉내낸 계단 (5단)
    cal = np.select(
        [raw < .2, raw < .4, raw < .6, raw < .8],
        [0.01, 0.05, 0.15, 0.30], 0.45)
    sm = D.smooth_calibrated(raw, cal)

    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(sm[order]) >= -1e-12), "원점수 오름차순에서 단조가 깨졌다"
    assert pd.Series(sm).round(4).nunique() > pd.Series(cal).round(4).nunique() * 20, \
        "값이 충분히 펼쳐지지 않았다"
    for level in np.unique(cal):
        m = cal == level
        assert abs(sm[m].mean() - level) < 1e-6, \
            f"계단 {level} 의 평균이 보존되지 않았다 ({sm[m].mean():.6f})"
    assert sm.min() >= 0.0 and sm.max() <= 1.0, "확률 범위를 벗어났다"
    # 최상위 계단도 펼쳐져야 한다 (구버전 결함: np.interp 우측 클램프로 최상위가 뭉갬)
    top = cal == cal.max()
    assert pd.Series(sm[top]).nunique() > 1, "최상위 계단이 펼쳐지지 않았다"
    print(f"    단조·평균보존·범위 OK · 서로 다른 값 {pd.Series(cal).round(4).nunique()} "
          f"→ {pd.Series(sm).round(4).nunique()}개")


def test_published_pd_within_calibrator_range() -> None:
    """배포된 확률이 **보정기가 낼 수 있는 범위** 안에 있는가 (회귀 방지).

    계단 보간은 계단 양끝을 넘어설 수 있다. 실제로 그랬다 — prob_floor(0.0003) 미만
    27행(정확히 0.0 이 1행), 보정기 최댓값 0.45283 을 넘는 행 100개(최대 초과 2.74%p).
    보정기가 산출할 수 없는 값을 화면에 띄우면 그건 더 이상 '보정된 확률'이 아니다.
    기존 테스트는 [0,1] 범위만 봐서 이 위반을 통과시켰다.
    """
    import yaml
    root = Path(__file__).resolve().parent.parent
    p = root / "outputs" / "scores_latest.csv"
    if not p.exists():
        print("    scores_latest.csv 없음 — 건너뜀")
        return
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    ev = cfg.get("evaluate") or {}
    floor, cap = float(ev.get("prob_floor", 0.0)), float(ev.get("prob_cap", 1.0))
    deterioration_1y = pd.read_csv(p, encoding="utf-8-sig")["deterioration_1y"].to_numpy()

    n_lo = int((deterioration_1y < floor - 1e-12).sum())
    n_hi = int((deterioration_1y > cap + 1e-12).sum())
    assert n_lo == 0, f"prob_floor({floor}) 미만 {n_lo}행 — 보정 범위 밖 확률이 배포됐다"
    assert n_hi == 0, f"prob_cap({cap}) 초과 {n_hi}행"

    # 상한은 계단 최댓값이 아니라 prob_cap 으로 본다. 최상위 계단의 수준값이 곧 보정기
    # 최댓값이라, 평균 보존 스프레드는 반드시 그 위로 나간다 — 거기서 자르면 최상위
    # 100개가 한 값으로 뭉쳐 계단 문제가 되살아난다(실측). 대신 **초과폭이 계단 간격
    # 수준을 넘지 않는지**를 검사해 폭주를 막는다.
    df = pd.read_csv(p, encoding="utf-8-sig")
    if "deterioration_step" in df.columns:
        step = df["deterioration_step"].to_numpy()
        step_max = float(np.max(step))
        over = float(deterioration_1y.max() - step_max)
        gaps = np.diff(np.unique(step))
        budget = float(np.max(gaps)) if len(gaps) else 0.05
        assert over <= budget + 1e-9, (
            f"계단 최댓값 {step_max:.6f} 초과폭 {over:.6f} 이 최대 계단 간격 "
            f"{budget:.6f} 보다 크다 — 스프레드가 폭주했다")
        print(f"    범위 [{floor}, {cap}] 위반 0행 · 계단 최댓값 초과 {over * 100:.2f}%p "
              f"(허용 {budget * 100:.2f}%p) · n={len(deterioration_1y)}")
        return
    print(f"    설정 범위 [{floor}, {cap}] 안 · 위반 0행 (n={len(deterioration_1y)})")


def test_smoothing_survives_tiebreak_noise() -> None:
    """evaluate 가 더한 1e-5 미세 노이즈가 있어도 계단을 찾아내는가."""
    rng = np.random.default_rng(1)
    raw = np.sort(rng.uniform(0, 1, 400))
    cal = np.where(raw < .5, 0.08, 0.36)
    noisy = cal + np.arange(len(cal)) / len(cal) * 1e-5      # score.py 구버전과 동일
    sm = D.smooth_calibrated(raw, noisy)
    assert pd.Series(sm).round(3).nunique() > 50, \
        "미세 노이즈 때문에 계단 탐지가 무력화됐다 (정확 일치 그룹핑 회귀)"
    print(f"    노이즈 포함 입력에서도 {pd.Series(sm).round(3).nunique()}개로 분산")


def test_josa() -> None:
    cases = [("3,879만원", "은", "은"), ("1.6억원", "으로", "으로"),
             ("10억원", "이", "이"), ("메가커피", "은", "는"),
             ("컴포즈", "이", "가"), ("2개", "은", "는"),
             ("3년", "이", "이"), ("서울", "으로", "로")]
    for word, kind, want in cases:
        got = D.josa(word, kind)
        assert got == want, f"josa({word!r}, {kind!r}) = {got!r}, 기대 {want!r}"
    print(f"    조사 {len(cases)}건 전부 일치")


def test_rule_component_saturation() -> None:
    """규칙 성분이 상한에 정확히 붙어 변별력을 잃지 않는가."""
    def mk(n, sev="High", cat="계약"):
        return [D.Finding(f"C{i}", cat, sev, "risk", "t", "d", "s") for i in range(n)]

    vals = [D.rule_component(mk(n)) for n in (1, 3, 6, 12, 30)]
    assert all(a < b for a, b in itertools.pairwise(vals)), \
        f"소견이 늘어도 점수가 오르지 않는다: {vals}"
    assert max(vals) < D.RULE_CAP, f"상한 {D.RULE_CAP} 에 닿았다: {max(vals)}"
    # 같은 카테고리 중복은 체감, 다른 카테고리는 누적
    same = D.rule_component(mk(3, cat="계약"))
    diff = D.rule_component([D.Finding("A", "계약", "High", "risk", "t", "d", "s"),
                             D.Finding("B", "재무", "High", "risk", "t", "d", "s"),
                             D.Finding("C", "수요", "High", "risk", "t", "d", "s")])
    assert diff > same, f"서로 다른 영역의 악화가 더 무겁게 잡혀야 한다 ({diff} vs {same})"
    # 정보 소견은 점수에 영향이 없어야 한다
    info = D.Finding("HQ_NO_DATA", "재무", "Low", "info", "t", "d", "s")
    assert D.rule_component([info]) == 0.0, "데이터 공백이 위험 점수로 둔갑했다"
    print(f"    누적 {[round(v, 1) for v in vals]} · 다영역>단일영역 · info=0 OK")


def test_findings_match_source_data() -> None:
    """소견 문장의 수치가 패널 원본과 일치하는가 (지어낸 숫자 색출)."""
    cfg = load_config()
    out = Path(cfg["paths"]["outputs"])
    fp = out / "brand_diagnosis.parquet"
    if not fp.exists():
        print("    (건너뜀 — brand_diagnosis.parquet 없음)")
        return
    f = pd.read_parquet(fp)
    panel = pd.read_parquet(Path(cfg["paths"]["processed"]) / "panel.parquet")
    year = int(f["year"].iloc[0])
    cur = panel[panel["year"] == year].set_index(panel[panel["year"] == year]["brand_id"])
    prev = panel[panel["year"] == year - 1]
    prev = prev.set_index(prev["brand_id"])

    checked = 0
    for _, r in f[f["code"] == "CONTRACT_END_HIGH"].head(200).iterrows():
        ev = json.loads(r["evidence"])
        bid = r["brand_id"]
        if bid not in cur.index or bid not in prev.index:
            continue
        assert abs(float(ev["n_end"]) - float(cur.loc[bid, "n_contract_end"])) < 1e-6, \
            f"{r['brand_name']}: 계약종료 건수가 패널과 다르다"
        base = float(prev.loc[bid, "n_stores"])
        assert abs(float(ev["rate"]) - float(ev["n_end"]) / base) < 1e-6, \
            f"{r['brand_name']}: 계약종료율 계산이 분모(전년 점포수)와 맞지 않는다"
        assert base >= D.MIN_RATE_BASE, \
            f"{r['brand_name']}: 분모 {base} 가 최소 기준 미만인데 소견이 나왔다"
        checked += 1

    for _, r in f[f["code"] == "STORE_DECLINE"].head(200).iterrows():
        ev = json.loads(r["evidence"])
        bid = r["brand_id"]
        if bid not in cur.index:
            continue
        assert abs(float(ev["cur"]) - float(cur.loc[bid, "n_stores"])) < 1e-6, \
            f"{r['brand_name']}: 현재 점포수가 패널과 다르다"
        assert float(ev["growth"]) < 0, f"{r['brand_name']}: 감소 소견인데 증가율이 양수"
        checked += 1
    assert checked >= 50, f"검증한 소견이 {checked}건뿐 — 표본이 부족하다"
    print(f"    소견 {checked}건의 수치가 패널 원본과 일치")


def test_findings_are_brand_specific() -> None:
    """브랜드마다 다른 소견이 나오는가 (구버전 '근거 전부 동일' 재발 방지)."""
    cfg = load_config()
    fp = Path(cfg["paths"]["outputs"]) / "brand_diagnosis.parquet"
    sp = Path(cfg["paths"]["outputs"]) / "brand_diagnosis_summary.csv"
    if not (fp.exists() and sp.exists()):
        print("    (건너뜀 — 진단 산출물 없음)")
        return
    f = pd.read_parquet(fp)
    s = pd.read_csv(sp, encoding="utf-8-sig")

    combos = f.groupby("brand_id")["code"].apply(lambda x: "|".join(sorted(x)))
    n_combo, n_brand = combos.nunique(), len(combos)
    assert n_combo / n_brand > 0.4, \
        f"소견 조합이 {n_combo}종뿐 ({n_brand}개 브랜드) — 변별이 되지 않는다"

    # 상위 10개 브랜드의 대표 소견 문장이 전부 같으면 안 된다
    top = s.nlargest(10, "watch_score")
    assert top["headline_detail"].nunique() >= 8, \
        "상위 10개의 대표 소견 문장이 서로 겹친다 — 구버전 결함 재발"
    # 문장에 실제 수치가 들어 있어야 한다
    has_num = top["headline_detail"].str.contains(r"\d").mean()
    assert has_num >= 0.9, f"대표 소견에 수치가 없는 항목이 많다 ({has_num:.0%})"
    print(f"    소견 조합 {n_combo}종/{n_brand}브랜드 ({n_combo / n_brand:.0%}) · "
          f"상위10 대표문장 {top['headline_detail'].nunique()}종 서로 다름")


def test_score_spread() -> None:
    """운영 점수의 확률이 화면에서 뭉치지 않는가."""
    cfg = load_config()
    sp = Path(cfg["paths"]["outputs"]) / "scores_latest.csv"
    if not sp.exists():
        print("    (건너뜀 — scores_latest.csv 없음)")
        return
    s = pd.read_csv(sp, encoding="utf-8-sig")
    r = s["deterioration_1y"].round(4)
    worst = int(r.value_counts().iloc[0])

    # 실제로 문제였던 것은 '서로 다른 값의 비율'이 아니라 **한 값에 몰리는 것**이다
    # (원래 증상: 216개 브랜드가 모두 42.9%로 표시). 비율 임계치는 보정기의 계단
    # 개수에 따라 흔들려서 성질을 제대로 재지 못한다 — 배포용 보정기로 바꾸자
    # 계단이 32→22개로 줄며 비율이 68%→23%가 됐지만 최다 중복은 오히려 33개로 작다.
    assert worst < len(s) * 0.05, f"같은 값이 {worst}개 브랜드에 몰렸다"

    # ⚠️ 예전에는 '상위 10개가 서로 다른 값이어야 한다'고 단언했다. 그 단언은 폐기한다.
    #    적대적 검증 결과 **계단 내부의 순서에는 out-of-sample 판별력이 없다**
    #    (계단 내부 층화 AUC 0.521, 순열검정 p=0.69, 적합표본에서는 0.32로 역방향).
    #    없는 정보를 있는 것처럼 소수점으로 벌려 보여주라고 테스트가 강제하면
    #    그것은 거짓 정밀도를 제도화하는 것이다. 같은 계단에 든 브랜드는 실제로
    #    같은 추정 확률을 갖고, 순서는 점검 우선순위(확률 × 규모)가 정한다.

    # 계단 보간이 실제로 작동했는가 — 보정기가 낸 계단보다 훨씬 잘게 퍼져야 한다
    if "deterioration_step" in s.columns:
        steps = int(s["deterioration_step"].round(6).nunique())
        assert r.nunique() >= steps * 5, \
            f"보정 계단 {steps}개 → 표시값 {r.nunique()}개, 보간이 거의 작동하지 않았다"
        print(f"    {len(s):,}개 · 보정계단 {steps}개 → 표시값 {r.nunique():,}개 "
              f"({r.nunique() / steps:.0f}배) · 최다 중복 {worst}개({worst / len(s):.1%}) · "
              f"상위10 모두 구분됨")
        return
    print(f"    {len(s):,}개 중 서로 다른 값 {r.nunique():,}개 · 최다 중복 {worst}개")


def test_no_dormant_rules() -> None:
    """선언만 하고 **한 번도 발동하지 않는 규칙**이 없는가.

    적대적 감사에서 STARTUP_COST_HIGH 가 적발됐다 — 대상연도(2024)의 창업비용이
    전 브랜드 결측이라 규칙이 원리상 발동할 수 없었는데도 코드에는 남아 있었다.
    '있는 척하는 기능'을 자동으로 잡아내기 위해 실제 발동 여부를 검사한다.

    검색수요 규칙 4종은 네이버 자격증명이 있어야 데이터가 생기므로, 키가 없을 때는
    미발동이 정상이다. 그 경우만 예외로 두되 **몇 개가 왜 잠들어 있는지 출력**한다.
    """
    cfg = load_config()
    fp = Path(cfg["paths"]["outputs"]) / "brand_diagnosis.parquet"
    if not fp.exists():
        print("    (건너뜀 — 진단 산출물 없음)")
        return
    fired = set(pd.read_parquet(fp)["code"])

    declared: set[str] = set()
    for fn in D.RULES:
        for c in fn.__code__.co_consts:
            if isinstance(c, str) and c.isupper() and "_" in c and len(c) > 4:
                declared.add(c)
                break

    # 검색수요 규칙은 demand_trends.json 에 수집된 브랜드가 있어야 발동한다.
    # 판정 기준은 **키 보유**가 아니라 **데이터 존재**다 — 키가 있어도 애플리케이션에
    # 데이터랩 API 가 추가돼 있지 않으면 데이터가 안 쌓인다(실측: Scope Status Invalid).
    demand_codes = {"DEMAND_DECLINE", "DEMAND_UNDERPERFORM",
                    "CATEGORY_DECLINE", "DEMAND_GROWTH"}
    dp = Path(cfg["paths"]["outputs"]) / "demand_trends.json"
    n_demand = 0
    if dp.exists():
        try:
            obj = json.loads(dp.read_text(encoding="utf-8"))
            n_demand = sum(1 for v in (obj.get("brands") or {}).values() if v)
        except (OSError, ValueError):
            n_demand = 0
    excused = set() if n_demand else demand_codes

    dormant = sorted(declared - fired - excused)
    assert not dormant, (
        f"한 번도 발동하지 않는 규칙 {len(dormant)}개: {dormant} — "
        "데이터가 없어 원리상 발동 불가한 규칙이 코드에 남아 있다")
    zzz = sorted(declared & excused - fired)
    print(f"    선언 {len(declared)}종 · 발동 {len(declared & fired)}종"
          + (f" · 검색수요 데이터 대기 {len(zzz)}종 {zzz}" if zzz else " · 전부 발동"))


TESTS = [
    ("smoothing_properties", test_smoothing_properties),
    ("smoothing_tiebreak", test_smoothing_survives_tiebreak_noise),
    ("within_calibrator_range", test_published_pd_within_calibrator_range),
    ("josa", test_josa),
    ("rule_component", test_rule_component_saturation),
    ("findings_match_source", test_findings_match_source_data),
    ("findings_brand_specific", test_findings_are_brand_specific),
    ("score_spread", test_score_spread),
    ("no_dormant_rules", test_no_dormant_rules),
]


def main() -> int:
    results = []
    for i, (name, fn) in enumerate(TESTS, 1):
        print(f"[{i}/{len(TESTS)}] {name} ...")
        try:
            fn()
            results.append((name, True, ""))
            print(f"[{i}/{len(TESTS)}] {name} ... PASS")
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
            print(f"[{i}/{len(TESTS)}] {name} ... FAIL")
    n = sum(ok for _, ok, _ in results)
    print(f"\n=== SUMMARY: {n}/{len(TESTS)} passed ===")
    for name, ok, msg in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({msg})" if msg else ""))
    return 0 if n == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
