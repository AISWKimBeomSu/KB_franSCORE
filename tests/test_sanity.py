"""FranSCORE 안전성(sanity) 테스트 — 계약 docs/INTERFACES.md §7 (pytest 불요, 표준 assert).

실행: 프로젝트 루트에서  python -m tests.test_sanity

검증 항목:
  1. 라벨 규칙  : 수제 미니 패널 → 사건 발동(2개 이상)·극단 규칙(1개 극단)·
                  |Δ점포| 노이즈 게이트·최소점포 게이트·healthy 게이트의 기대 라벨 일치
  2. 시점 누출  : features의 (brand, t) 행이 t+1 이후 패널값 교란에 완전 불변
  3. 시간 분할  : train/valid/test 연도 교집합 없음, test = 최대 연도, valid = test-1
  4. 예측 스키마: predictions.parquet 계약 컬럼·확률 [0,1] 범위 (합성 임시 실행으로 생성)
  5. 포트폴리오 : EL ≤ exposure, LGD 단조성(LGD↑ → EL↑), 스트레스 EL ≥ 기본 EL

설계 원칙 (정직성·격리):
- 실데이터(data/processed)·제출 산출물(outputs)은 절대 쓰지 않는다. 모든 실행은
  tempfile 임시 디렉터리로 cfg['paths'] "사본"을 교체해 격리한다 (원본 cfg 불변).
- 종료 시 data/processed 디렉터리가 불변(파일명·크기·mtime)임을 가드로 재확인한다.
  (outputs/pipeline.log 는 모듈 로거가 append 하므로 가드 대상에서 제외.)
- 미니 패널의 기대 라벨은 pandas 분위수(linear interpolation)·엄격 부등호·게이트
  규칙(labels.py 문서화된 해석)을 손으로 계산해 고정했다. config.yaml 의 라벨
  파라미터가 전제이므로, 전제가 깨지면 명확한 메시지로 실패하도록 사전검사를 둔다.
"""
from __future__ import annotations

import contextlib
import copy
import itertools
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# `python tests/test_sanity.py` 로 직접 실행하면 sys.path[0] 이 tests/ 가 되어 src 를 못 찾는다.
# README 가 그 형태로 안내하므로 두 실행 방식이 모두 되도록 루트를 먼저 올린다.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import model as model_mod
from src import portfolio as portfolio_mod
from src.common import (
    get_logger,
    load_config,
    make_synthetic_panel,
    set_seed,
)
from src.dart import make_synthetic_hq_financials
from src.features import HQ_VALUE_COLS, build_features
from src.labels import build_labels
from src.panel import PANEL_VALUE_COLS

# Windows 기본 콘솔(cp949)에서 한글·기호 출력이 깨지지 않도록
for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):  # 리다이렉트된 스트림 등
        _s.reconfigure(encoding="utf-8", errors="replace")

log = get_logger("test_sanity")

ROOT = Path(__file__).resolve().parent.parent

# 계약 §3 predictions.parquet 컬럼 (순서 포함)
PREDICTION_COLUMNS = [
    "brand_id", "year", "split", "y_true",
    "p_lgbm", "p_persistence", "p_single", "p_logistic", "is_new_brand",
]
# 계약 §2 build_labels 반환 컬럼 (순서 포함)
LABEL_COLUMNS = [
    "brand_id", "year", "label", "healthy_at_t",
    "ev_store", "ev_sales", "ev_contract", "n_events", "extreme_fired",
]


# ---------------------------------------------------------------------------
# 격리 유틸
# ---------------------------------------------------------------------------

def _make_test_cfg(base_cfg: dict, tmp: Path) -> dict:
    """cfg 딥카피 후 paths 를 임시 디렉터리로 교체 (실데이터·제출 산출물 격리)."""
    tcfg = copy.deepcopy(base_cfg)
    tcfg["paths"] = {
        "raw": tmp / "raw",
        "processed": tmp / "processed",
        "outputs": tmp / "outputs",
        "demo_outputs": tmp / "outputs" / "_smoke",
    }
    for p in tcfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return tcfg


def _fingerprint_dir(d: Path) -> list[tuple[str, int, int]]:
    """디렉터리 내 모든 파일의 (상대경로, 크기, mtime_ns) 스냅샷 — 불변 가드용."""
    if not d.exists():
        return []
    out = []
    for p in sorted(d.rglob("*")):
        if p.is_file():
            st = p.stat()
            out.append((str(p.relative_to(d)), st.st_size, st.st_mtime_ns))
    return out


# ---------------------------------------------------------------------------
# 테스트 1 — 라벨 규칙 (수제 미니 패널)
# ---------------------------------------------------------------------------
# 설계: 20개 브랜드 × 3개년(2019~2021), 단일 업종셀(외식/치킨 — 그룹크기 20 < 30
# 이므로 업종그룹 = major). t=2020 행의 라벨은 2021년 사건으로 결정된다.
#
# 2021년 사건 지표 풀(20개 값)을 정확히 "low 4개 + high 16개"로 설계:
#   pandas quantile(0.20) = sorted[3] + 0.8*(sorted[4]-sorted[3]) 는 low/high 사이
#   → 엄격 부등호 기준 low 4개만 발동. extreme quantile(0.05) 는 sorted[0]과
#   sorted[1] 사이 → 유일 최솟값만 극단 (최솟값을 중복시키면 극단 없음).
#
# 브랜드 역할:
#   N01..N08          중립 (사건 없음 → label 0)
#   B_storeext        점포 -60% (극단, |Δ|=63) → 사건 1개지만 극단 → label 1
#   B_2ev             점포 -27.6% + 매출 -50%  → 사건 2개(극단 아님) → label 1
#   B_smallgate       점포 -20% 이지만 |Δ|=2 < 3 → 게이트 차단 → label 0
#   B_gate            t 점포수 8 < min_stores_at_t=10 → 표본 자체에서 제외
#   B_unh             2020년에 점포 -30%(극단) → t=2019 label 1,
#                     t=2020 은 healthy 게이트로 표본 제외
#   B_salesonly       매출 -50% (중복 최솟값 → 극단 아님) 사건 1개 → label 0
#   B_saleslow2/3     매출 -10% / -8% 사건 1개 → label 0
#   B_contractonly    계약종료율 0.40 (중복 최댓값 → 극단 아님) 사건 1개 → label 0
#   B_contracthigh2~4 계약종료율 0.40/0.30/0.28 사건 1개 → label 0
# ---------------------------------------------------------------------------

_NEUTRAL_STORES = (100.0, 105.0, 111.0)  # 성장 +5%, +5.71% (중립 high)


def _brand_specs() -> dict[str, dict]:
    sp: dict[str, dict] = {f"N{i:02d}": {} for i in range(1, 9)}
    sp["B_storeext"] = {"stores": (100.0, 105.0, 42.0)}
    sp["B_2ev"] = {"stores": (100.0, 105.0, 76.0), "sales_g21": -0.50}
    sp["B_smallgate"] = {"stores": (10.0, 10.0, 8.0)}
    sp["B_gate"] = {"stores": (8.0, 8.0, 6.0)}
    sp["B_unh"] = {"stores": (100.0, 70.0, 74.0)}
    sp["B_salesonly"] = {"sales_g21": -0.50}
    sp["B_saleslow2"] = {"sales_g21": -0.10}
    sp["B_saleslow3"] = {"sales_g21": -0.08}
    sp["B_contractonly"] = {"c21": 0.40}
    sp["B_contracthigh2"] = {"c21": 0.40}
    sp["B_contracthigh3"] = {"c21": 0.30}
    sp["B_contracthigh4"] = {"c21": 0.28}
    return sp


def _mini_panel() -> pd.DataFrame:
    """계약 §1 스키마의 수제 미니 패널 (위 설계 주석 참조)."""
    rows = []
    for name, spec in _brand_specs().items():
        stores = spec.get("stores", _NEUTRAL_STORES)
        g21_sales = spec.get("sales_g21", 0.04)   # 중립 매출성장 +4%
        c21 = spec.get("c21", 0.05)               # 중립 계약종료율 5%
        st = dict(zip((2019, 2020, 2021), stores, strict=False))
        sales = {2019: 300_000.0}
        sales[2020] = sales[2019] * 1.03          # 전 브랜드 동일 +3% → 사건 없음
        sales[2021] = sales[2020] * (1.0 + g21_sales)
        crate = {2020: 0.05, 2021: c21}           # 2020 전 브랜드 동일 → 사건 없음
        for yr in (2019, 2020, 2021):
            prev = st.get(yr - 1)
            # (종료+해지)/전년점포 = 정확히 crate. 2019는 전년이 없어 rate가 NaN이므로 미사용.
            total_end = crate[yr] * prev if (yr in crate and prev) else 1.0
            n_end = total_end * 0.6
            n_cancel = total_end - n_end
            n_new = max(0.0, st[yr] - (prev or st[yr]) + total_end)
            rows.append({
                "brand_id": name,
                "brand_name": name,
                "company_name": f"HQ_{name}",
                "industry_major": "외식",
                "industry_mid": "치킨",
                "year": yr,
                "n_stores": st[yr],
                "n_direct": 2.0,
                "n_new": n_new,
                "n_contract_end": n_end,
                "n_contract_cancel": n_cancel,
                "n_name_change": 1.0,
                "avg_sales": sales[yr],
                "avg_sales_per_area": sales[yr] / 10.0,
            })
    return pd.DataFrame(rows)


def _expected_labels() -> dict[tuple[str, int], int]:
    """리뷰 수정 반영: 2019(첫 관측연도)는 t지표 전부 결측 → '양호 판정 불가'로 표본 제외.
    (구버전은 판정 불가를 양호로 간주 — 적대적 리뷰 확정 결함이었음)"""
    exp: dict[tuple[str, int], int] = {}
    for nm in _brand_specs():
        if nm == "B_gate":
            continue                      # 최소점포 게이트: 전 행 표본 제외
        if nm == "B_unh":
            continue                      # healthy 게이트: t=2020 행 표본 제외 (t=2019는 판정불가 제외)
        exp[(nm, 2020)] = 1 if nm in ("B_storeext", "B_2ev") else 0
    return exp


def _check_label_preconditions(cfg: dict) -> None:
    """미니 패널 기대값이 전제하는 config 라벨 파라미터 검사 (변경 시 명확히 실패)."""
    lab = cfg["label"]
    pre = {
        "min_events": (int(lab["min_events"]), 2),
        "extreme_quantile": (float(lab["extreme_quantile"]), 0.05),
        "min_abs_store_change": (float(lab["min_abs_store_change"]), 3.0),
        "min_stores_at_t": (float(lab["min_stores_at_t"]), 10.0),
        "healthy_gate_at_t": (bool(lab["healthy_gate_at_t"]), True),
    }
    for k, (got, want) in pre.items():
        assert got == want, (
            f"config label.{k}={got} 가 테스트 설계 전제({want})와 다릅니다 — "
            "미니 패널 기대 라벨을 재도출해야 합니다.")
    ev = lab["events"]
    assert float(ev["store_net_decline"]["quantile"]) == 0.20 and \
        ev["store_net_decline"]["side"] == "lower", "store 사건 분위수/방향 전제 변경됨"
    assert float(ev["sales_decline"]["quantile"]) == 0.20 and \
        ev["sales_decline"]["side"] == "lower", "sales 사건 분위수/방향 전제 변경됨"
    assert float(ev["contract_end_spike"]["quantile"]) == 0.80 and \
        ev["contract_end_spike"]["side"] == "upper", "contract 사건 분위수/방향 전제 변경됨"


def test_label_rules(ctx: dict) -> None:
    tcfg = _make_test_cfg(ctx["cfg"], ctx["tmp"] / "t1_labels")
    _check_label_preconditions(tcfg)
    panel = _mini_panel()
    out = build_labels(panel, tcfg)

    assert list(out.columns) == LABEL_COLUMNS, \
        f"라벨 반환 컬럼이 계약과 다름: {list(out.columns)}"

    got = {(str(r.brand_id), int(r.year)): int(r.label) for r in out.itertuples()}
    exp = _expected_labels()
    missing = sorted(set(exp) - set(got))
    extra = sorted(set(got) - set(exp))
    assert not missing and not extra, (
        f"표본 행 불일치 — 게이트 동작 오류 의심. missing={missing} extra={extra}")
    diff = {k: (got[k], exp[k]) for k in exp if got[k] != exp[k]}
    assert not diff, f"라벨 불일치 (got, expected): {diff}"

    # 게이트 케이스 명시 확인
    assert all(k[0] != "B_gate" for k in got), "min_stores_at_t 게이트 미작동 (B_gate 포함됨)"
    assert ("B_unh", 2020) not in got, "healthy 게이트 미작동 (B_unh t=2020 포함됨)"
    assert all(k[1] != 2019 for k in got), \
        "판정불가 게이트 미작동 — 첫 관측연도(t지표 전부 결측) 행은 표본에서 제외돼야 함"

    idx = out.set_index(["brand_id", "year"])
    r = idx.loc[("B_storeext", 2020)]
    assert bool(r["ev_store"]) and int(r["n_events"]) == 1 and bool(r["extreme_fired"]), \
        "극단 규칙: B_storeext 는 사건 1개 + 극단이어야 함"
    r = idx.loc[("B_2ev", 2020)]
    assert bool(r["ev_store"]) and bool(r["ev_sales"]) and int(r["n_events"]) == 2 \
        and not bool(r["extreme_fired"]), "B_2ev 는 사건 2개(극단 아님)이어야 함"
    r = idx.loc[("B_smallgate", 2020)]
    assert not bool(r["ev_store"]) and int(r["label"]) == 0, \
        "|Δ점포| 게이트: B_smallgate 의 점포 사건은 차단돼야 함"
    r = idx.loc[("B_salesonly", 2020)]
    assert bool(r["ev_sales"]) and int(r["n_events"]) == 1 and int(r["label"]) == 0, \
        "단일 사건(비극단)은 label=0 이어야 함 (B_salesonly)"
    r = idx.loc[("B_contractonly", 2020)]
    assert bool(r["ev_contract"]) and int(r["label"]) == 0, \
        "단일 계약종료 사건(비극단)은 label=0 이어야 함 (B_contractonly)"

    # healthy 게이트 OFF 변형: 판정불가(2019) 행 19개 + B_unh t=2020 행이 추가되어야 함
    tcfg_off = copy.deepcopy(tcfg)
    tcfg_off["label"]["healthy_gate_at_t"] = False
    out_off = build_labels(panel, tcfg_off)
    idx_off = out_off.set_index(["brand_id", "year"])
    assert ("B_unh", 2020) in idx_off.index, "게이트 OFF 시 B_unh t=2020 행이 있어야 함"
    r = idx_off.loc[("B_unh", 2020)]
    assert not bool(r["healthy_at_t"]) and int(r["label"]) == 0, \
        "B_unh t=2020: healthy_at_t=False, 2021년은 중립이므로 label=0 이어야 함"
    # 극단 규칙은 게이트 OFF 변형의 t=2019 행에서 계속 검증 (2020년 극단 점포감소)
    r = idx_off.loc[("B_unh", 2019)]
    assert bool(r["ev_store"]) and bool(r["extreme_fired"]) and int(r["label"]) == 1, \
        "B_unh 의 2020년 극단 점포감소가 t=2019 라벨 1 이어야 함 (게이트 OFF)"
    assert not bool(r["healthy_at_t"]), \
        "t=2019(첫 관측연도)는 판정불가 → healthy_at_t=False 여야 함"
    # 추가 행 = B_unh 2020 1행 + 2019 행 19개 (B_gate 제외 전 브랜드)
    assert len(out_off) == len(out) + 20, \
        f"게이트 OFF 는 정확히 20행 추가여야 함 (got {len(out_off) - len(out)})"

    print(f"    labels: {len(out)} sample rows, positives="
          f"{int(out['label'].sum())} (expected 3), gate cases OK")


# ---------------------------------------------------------------------------
# 테스트 2 — 시점 누출 (t+1 교란 불변성)
# ---------------------------------------------------------------------------

def test_time_leakage(ctx: dict) -> None:
    cfg = ctx["cfg"]
    tcfg = _make_test_cfg(cfg, ctx["tmp"] / "t2_leak")
    panel = make_synthetic_panel(cfg)
    t0 = 2022  # 합성 패널 연도 2019~2024 의 중간 기준연도

    # 커버리지 가드 ①: 합성 패널이 실패널의 값 컬럼을 하나라도 빠뜨리면, build_features 의
    # `if "<col>" in df.columns` 분기가 통째로 건너뛰어져 그 피처는 아래 검사 범위 밖이 된다.
    # (적대적 리뷰에서 실제로 발견된 사각지대 — 신규 피처 9종이 검증되지 않고 있었다.)
    missing = [c for c in PANEL_VALUE_COLS if c not in panel.columns]
    assert not missing, (
        f"합성 패널에 실패널 컬럼 누락 → 해당 피처가 누출 검증에서 조용히 제외됨: {missing}. "
        f"src/common.py make_synthetic_panel 에 추가하세요.")

    # 본부 재무(f_hq_*)는 패널이 아니라 **별도 표**에서 온다. 합성 재무를 명시적으로
    # 넘기지 않으면 그 피처들이 아래 검증 범위에서 통째로 빠진다 — 과거에 신규 피처가
    # 조용히 검증 밖으로 나갔던 것과 같은 사고다.
    hq = make_synthetic_hq_financials(cfg, panel)
    base = build_features(panel, tcfg, hq_fin=hq)

    # 커버리지 가드 ③: f_hq_* 가 전부 결측/상수여도 "교란 전후 동일"은 자동으로 성립한다.
    # 즉 피처가 죽어 있으면 이 테스트는 통과해도 아무것도 검증하지 못한다 —
    # 실제로 값이 살아 있는지 먼저 확인한다.
    hq_cols = [c for c in base.columns if c.startswith("f_hq_")]
    assert hq_cols, "f_hq_* 피처가 생성되지 않음 — 합성 본부재무 결합 실패"
    dead = [c for c in hq_cols if base[c].nunique(dropna=True) <= 1]
    assert not dead, (
        f"f_hq_* 피처가 상수/전결측이라 누출 검증이 공회전한다: {dead}")

    pert = panel.copy()
    fut = pert["year"] > t0
    assert fut.any() and (~fut).any(), "교란 대상/비대상 연도가 모두 있어야 함"
    # t0 이후 연도의 값을 전면 교란 (행 추가/삭제·업종 변경은 없음).
    # 컬럼 하나라도 교란하지 않으면 그 컬럼을 쓰는 피처는 '통과한 것처럼' 보이므로,
    # 아래 커버리지 가드 ②로 실패널 값 컬럼 전부가 교란 대상인지 강제한다.
    perturb = {
        "n_stores":          lambda s: s * 2.0 + 37.0,
        "n_direct":          lambda s: s + 7.0,
        "n_new":             lambda s: s + 29.0,
        "n_contract_end":    lambda s: s + 41.0,
        "n_contract_cancel": lambda s: s + 13.0,
        "n_name_change":     lambda s: s + 11.0,
        "avg_sales":         lambda s: s * 0.3 + 999.0,
        "avg_sales_per_area": lambda s: s * 1.7,
        "biz_start_year":    lambda s: s - 5.0,
        "emp_cnt":           lambda s: s * 3.0 + 17.0,
        "exec_cnt":          lambda s: s + 5.0,
        "n_regions":         lambda s: (s + 4.0).clip(upper=17.0),
        "region_hhi":        lambda s: s * 0.5,
        "top_region_share":  lambda s: s * 0.5,
        "region_max_stores": lambda s: s + 23.0,
        "startup_fee":       lambda s: s * 1.5 + 900.0,
        "startup_edu":       lambda s: s + 700.0,
        "startup_deposit":   lambda s: s * 2.0 + 300.0,
        "startup_etc":       lambda s: s * 0.7 + 1_100.0,
        "startup_total":     lambda s: s * 1.3 + 2_500.0,
        "cancel_flag":       lambda s: 1.0 - s,
        "cancel_type":       lambda s: "직권취소",
    }
    uncovered = [c for c in PANEL_VALUE_COLS if c not in perturb]
    assert not uncovered, (
        f"교란 대상에서 빠진 패널 컬럼 → 누출 검증 사각지대: {uncovered}")
    for col, fn in perturb.items():
        pert.loc[fut, col] = fn(pert.loc[fut, col])

    # 본부 재무도 같은 기준으로 교란한다. 피처는 회계연도 ≤ t-1 만 쓰므로
    # 회계연도 > t0-1 (= 미래 결산)을 흔들어도 year<=t0 피처는 불변이어야 한다.
    hq_pert = hq.copy()
    hq_fut = hq_pert["fiscal_year"] > t0 - 1
    assert hq_fut.any() and (~hq_fut).any(), "본부 재무 교란 대상/비대상이 모두 있어야 함"
    hq_perturb = {
        "assets":            lambda s: s * 3.0 + 1e10,
        "liabilities":       lambda s: s * 0.2 + 5e9,
        "equity":            lambda s: -s - 7e9,
        "capital_stock":     lambda s: s * 4.0 + 1e8,
        "revenue":           lambda s: s * 0.1 + 2e10,
        "operating_income":  lambda s: -s - 3e9,
        "net_income":        lambda s: -s - 9e9,
        "going_concern_flag": lambda s: 1.0 - s,
    }
    uncovered_hq = [c for c in HQ_VALUE_COLS if c not in hq_perturb]
    assert not uncovered_hq, (
        f"교란 대상에서 빠진 본부 재무 컬럼 → 누출 검증 사각지대: {uncovered_hq}. "
        f"src/features.py HQ_VALUE_COLS 와 이 표를 함께 유지하세요.")
    for col, fn in hq_perturb.items():
        hq_pert.loc[hq_fut, col] = fn(hq_pert.loc[hq_fut, col])

    after = build_features(pert, tcfg, hq_fin=hq_pert)

    b = base[base["year"] <= t0].reset_index(drop=True)
    a = after[after["year"] <= t0].reset_index(drop=True)
    assert list(b.columns) == list(a.columns), "교란 전후 피처 컬럼 불일치"
    assert len(b) == len(a) and len(b) > 0, "교란 전후 행 수 불일치"
    try:
        pd.testing.assert_frame_equal(b, a, check_exact=True)
    except AssertionError as e:
        # 어떤 피처가 미래값에 의존하는지 진단 정보 첨부
        bad = [c for c in b.columns
               if not (b[c].isna() == a[c].isna()).all()
               or not (b[c].dropna().to_numpy() == a[c].dropna().to_numpy()).all()]
        raise AssertionError(
            f"시점 누출: year<={t0} 피처가 t+1 이후 교란에 변함. 변한 컬럼={bad}") from e
    print(f"    leakage: {len(b)} rows (year<={t0}) x {len(b.columns)} cols "
          f"bit-identical after perturbing years>{t0}")


# ---------------------------------------------------------------------------
# 테스트 3~5 공용 — 합성 소규모 파이프라인 (임시 디렉터리 격리)
# ---------------------------------------------------------------------------

def _pipeline(ctx: dict) -> dict:
    """합성 패널로 features→labels→model→portfolio 를 임시 경로에서 1회 실행·캐시."""
    if "pipe_error" in ctx:
        raise ctx["pipe_error"]
    if "pipe" not in ctx:
        try:
            cfg = ctx["cfg"]
            tcfg = _make_test_cfg(cfg, ctx["tmp"] / "t345_pipe")
            set_seed(cfg["seed"])
            panel = make_synthetic_panel(cfg)
            feats = build_features(panel, tcfg)
            labs = build_labels(panel, tcfg)
            res = model_mod.train_all(feats, labs, tcfg)
            panel.to_parquet(Path(tcfg["paths"]["processed"]) / "panel.parquet",
                             index=False)
            portfolio_mod.build_portfolio(tcfg)
            ctx["pipe"] = {"tcfg": tcfg, "model": res}
        except Exception as e:  # 이후 테스트가 같은 원인을 즉시 보고하도록 캐시
            ctx["pipe_error"] = e
            raise
    return ctx["pipe"]


def test_time_split(ctx: dict) -> None:
    pipe = _pipeline(ctx)
    preds = pipe["model"]["predictions"]
    years = {s: set(preds.loc[preds["split"] == s, "year"].astype(int))
             for s in ("train", "valid", "test")}
    for s in ("train", "valid", "test"):
        assert years[s], f"{s} split 이 비어 있음"
    assert not (years["train"] & years["valid"]), \
        f"train/valid 연도 겹침: {years['train'] & years['valid']}"
    assert not (years["train"] & years["test"]), \
        f"train/test 연도 겹침: {years['train'] & years['test']}"
    assert not (years["valid"] & years["test"]), \
        f"valid/test 연도 겹침: {years['valid'] & years['test']}"
    all_years = years["train"] | years["valid"] | years["test"]
    t_max = max(all_years)
    assert years["test"] == {t_max}, f"test 연도 {years['test']} 는 최대 {t_max} 단일이어야 함"
    assert years["valid"] == {t_max - 1}, f"valid 연도는 test-1({t_max - 1})이어야 함"
    assert max(years["train"]) <= t_max - 2, "train 연도는 test-2 이하여야 함"

    # split_years.json 기록과 일치 확인
    sy_path = Path(pipe["tcfg"]["paths"]["outputs"]) / "split_years.json"
    assert sy_path.exists(), f"split_years.json 미기록: {sy_path}"
    with open(sy_path, encoding="utf-8") as f:
        sy = json.load(f)
    assert int(sy["test_year"]) == t_max and int(sy["valid_year"]) == t_max - 1, \
        f"split_years.json({sy['test_year']},{sy['valid_year']}) 과 실제 분할 불일치"
    print(f"    split: train={sorted(years['train'])} valid={t_max - 1} test={t_max}")


def test_predictions_schema(ctx: dict) -> None:
    pipe = _pipeline(ctx)
    pred_path = Path(pipe["tcfg"]["paths"]["processed"]) / "predictions.parquet"
    assert pred_path.exists(), f"predictions.parquet 미생성: {pred_path}"
    df = pd.read_parquet(pred_path)

    assert list(df.columns) == PREDICTION_COLUMNS, (
        f"predictions 컬럼이 계약 §3 과 다름:\n got={list(df.columns)}\n"
        f" want={PREDICTION_COLUMNS}")
    assert len(df) > 0, "predictions 가 비어 있음"
    assert not df.duplicated(subset=["brand_id", "year"]).any(), \
        "(brand_id, year) 중복 존재"
    assert set(df["split"].unique()) <= {"train", "valid", "test"}, \
        f"split 값 이상: {set(df['split'].unique())}"
    assert set(df["y_true"].unique()) <= {0, 1}, f"y_true 이상: {set(df['y_true'].unique())}"
    assert df["is_new_brand"].dtype == bool, "is_new_brand 는 bool 이어야 함"
    for col in ("p_lgbm", "p_persistence", "p_single", "p_logistic"):
        v = df[col].to_numpy(dtype=float)
        assert np.isfinite(v).all(), f"{col} 에 NaN/inf 존재"
        assert (v >= 0.0).all() and (v <= 1.0).all(), (
            f"{col} 확률 범위 위반: min={v.min():.6f} max={v.max():.6f}")
    print(f"    predictions: {len(df)} rows, columns OK, all 4 prob cols in [0,1]")


def test_portfolio(ctx: dict) -> None:
    pipe = _pipeline(ctx)
    tcfg = pipe["tcfg"]
    out_dir = Path(tcfg["paths"]["outputs"])
    csv_path = out_dir / "portfolio.csv"
    json_path = out_dir / "portfolio_summary.json"
    assert csv_path.exists(), f"portfolio.csv 미생성: {csv_path}"
    assert json_path.exists(), f"portfolio_summary.json 미생성: {json_path}"

    port = pd.read_csv(csv_path, encoding="utf-8")
    assert len(port) > 0, "portfolio.csv 가 비어 있음"
    assert (port["exposure_mkrw"] > 0).all(), "exposure 는 양수여야 함"

    lgds = sorted(float(x) for x in tcfg["portfolio"]["lgd_scenarios"])
    assert all(0.0 < lgd <= 1.0 for lgd in lgds), f"LGD 시나리오가 (0,1] 밖: {lgds}"
    el_cols = [f"el_lgd{round(lgd * 100):02d}_mkrw" for lgd in lgds]
    sel_cols = [f"stress_el_lgd{round(lgd * 100):02d}_mkrw" for lgd in lgds]
    for c in el_cols + sel_cols:
        assert c in port.columns, f"portfolio.csv 에 {c} 컬럼 없음"

    tol = 1e-6
    exp_v = port["exposure_mkrw"].to_numpy(dtype=float)
    for c in el_cols + sel_cols:
        v = port[c].to_numpy(dtype=float)
        assert (v >= -tol).all(), f"{c} 에 음수 EL 존재"
        assert (v <= exp_v + tol).all(), (
            f"EL > exposure 위반 ({c}): max EL={v.max():.3f}, "
            f"해당 exposure={exp_v[v.argmax()]:.3f}")

    # LGD 단조성 (브랜드별): LGD ↑ → EL ↑ (기본·스트레스 모두)
    for cols in (el_cols, sel_cols):
        for lo, hi in itertools.pairwise(cols):
            bad = (port[hi].to_numpy(dtype=float)
                   < port[lo].to_numpy(dtype=float) - tol).sum()
            assert bad == 0, f"LGD 단조성 위반: {hi} < {lo} 인 브랜드 {bad}개"

    # 스트레스 EL ≥ 기본 EL (multiplier ≥ 1 전제 — config 확인)
    mult = float(tcfg["portfolio"]["stress_pd_multiplier"])
    assert mult >= 1.0, f"stress_pd_multiplier={mult} < 1 (테스트 전제 위반)"
    for base_c, st_c in zip(el_cols, sel_cols, strict=False):
        bad = (port[st_c].to_numpy(dtype=float)
               < port[base_c].to_numpy(dtype=float) - tol).sum()
        assert bad == 0, f"스트레스 EL < 기본 EL 인 브랜드 {bad}개 ({st_c})"

    with open(json_path, encoding="utf-8") as f:
        summary = json.load(f)
    # 익스포저 출처 표기가 **실제 경로와 일치**해야 한다.
    # 예전에는 synthetic 이 하드코딩 True 라 공시 창업비용 실측을 쓰면서도 "합성"이라고
    # 알렸다. 과대 주장뿐 아니라 이런 과소 주장도 부정확한 표기다 — 양방향으로 검증한다.
    expo = (summary.get("assumptions") or {}).get("exposure") or {}
    basis = expo.get("basis")
    assert basis in ("actual_loan_book", "disclosed_startup_cost", "synthetic_lognormal"), (
        f"익스포저 산출 근거(basis) 표기 없음/미상: {basis!r}")
    assert summary.get("synthetic") == (basis == "synthetic_lognormal"), (
        f"synthetic 플래그({summary.get('synthetic')})가 실제 산출 근거({basis})와 불일치")
    assert expo.get("is_synthetic") == summary.get("synthetic"), (
        "assumptions.exposure.is_synthetic 과 최상위 synthetic 플래그가 어긋남")
    total_exp = float(summary["total_exposure_mkrw"])
    scen = summary["expected_loss_scenarios"]
    assert [float(s["lgd"]) for s in scen] == lgds, "요약 시나리오 LGD 순서/값 불일치"
    prev_el = -1.0
    for s in scen:
        el, sel = float(s["el_mkrw"]), float(s["stress_el_mkrw"])
        assert el <= total_exp + tol and sel <= total_exp + tol, \
            f"요약 EL이 총 exposure 초과 (lgd={s['lgd']})"
        assert el >= prev_el - tol, "요약 EL이 LGD 증가에 대해 비단조"
        assert sel >= el - tol, f"요약 스트레스 EL < 기본 EL (lgd={s['lgd']})"
        prev_el = el
    print(f"    portfolio: {len(port)} brands, total exposure={total_exp:,.0f} MKRW, "
          f"EL<=exposure & LGD-monotone & stress>=base OK")


# ---------------------------------------------------------------------------
# 러너
# ---------------------------------------------------------------------------

def test_correlation_math(ctx: dict) -> None:
    """브랜드 공통요인 상관 추정기의 **수학적 정확성**을 알려진 정답으로 검증한다.

    이 모듈은 "브랜드 상관이 실재한다"는 프로젝트의 핵심 주장을 떠받치므로,
    추정기가 맞다는 것을 실데이터가 아니라 **정답을 아는 합성 데이터**로 증명한다.
      ① 자산상관 역산 왕복: ρ를 정해 Φ₂로 p11을 만들고 되돌려 ρ를 복원
      ② 독립 데이터 → ρ ≈ 0 (거짓 양성이 나오지 않는가)
      ③ 상관 데이터 → 심어둔 ρ를 복원 (검정력이 있는가)
      ④ 층화가 공통충격을 실제로 제거하는가 (연도 효과만 있고 브랜드 효과 0인 데이터)
    """
    from scipy.stats import multivariate_normal, norm

    from src import correlation as corr

    # ① 역산 왕복
    for p_true, rho_true in ((0.10, 0.15), (0.40, 0.42), (0.25, 0.05)):
        thr = norm.ppf(p_true)
        p11 = float(multivariate_normal.cdf([thr, thr], mean=[0.0, 0.0],
                                            cov=[[1.0, rho_true], [rho_true, 1.0]]))
        back = corr.asset_correlation(p_true, p11)
        assert abs(back - rho_true) < 1e-3, \
            f"자산상관 역산 오차: 참값 {rho_true} → 복원 {back:.5f}"
    print("    (1) 자산상관 역산 왕복: 3개 조건에서 오차 <1e-3 확인")

    rng = np.random.default_rng(0)
    R, n_brands, years = 12, 900, [2020, 2021, 2022]

    def synth(rho: float, year_shift: dict | None = None) -> pd.DataFrame:
        """단일요인 모형으로 (브랜드×연도×지역) 감소 사건 생성."""
        rows = []
        for b in range(n_brands):
            for y in years:
                base = 0.40 + (year_shift or {}).get(y, 0.0)
                thr = norm.ppf(base)
                z = rng.standard_normal()
                e = rng.standard_normal(R)
                x = np.sqrt(rho) * z + np.sqrt(1.0 - rho) * e
                k = int((x < thr).sum())
                rows.append({"brand_mnno": f"B{b:04d}", "year": y, "R": R, "k": k,
                             "stores": R * 5.0, "industry": "외식"})
        return pd.DataFrame(rows).set_index(["brand_mnno", "year"])

    # ② 독립 → ρ ≈ 0 (거짓 양성 없음)
    r0 = corr.stratified_rho(synth(1e-9), ["year", "industry"])
    assert abs(r0["rho_asset"]) < 0.03, f"독립 데이터인데 상관이 검출됨: {r0['rho_asset']:.4f}"
    print(f"    (2) 독립 데이터: 자산상관 {r0['rho_asset']:.4f} ≈ 0 (거짓 양성 없음)")

    # ③ 상관 심기 → 복원
    rho_planted = 0.30
    r1 = corr.stratified_rho(synth(rho_planted), ["year", "industry"])
    assert abs(r1["rho_asset"] - rho_planted) < 0.05, \
        f"심어둔 상관 {rho_planted} 복원 실패: {r1['rho_asset']:.4f}"
    print(f"    (3) 상관 {rho_planted} 심음 → 복원 {r1['rho_asset']:.4f} (오차 <0.05)")

    # ④ 연도 공통충격만 있고 브랜드 효과 0 → 층화가 그것을 제거해야 한다
    shocked = synth(1e-9, year_shift={2020: -0.18, 2021: 0.0, 2022: +0.18})
    naive = corr.stratified_rho(shocked, None)["rho_asset"]
    controlled = corr.stratified_rho(shocked, ["year", "industry"])["rho_asset"]
    assert naive > controlled + 0.02, \
        f"연도 충격이 통제 전후로 차이를 만들지 않음 (naive {naive:.4f} / 통제 {controlled:.4f})"
    assert abs(controlled) < 0.03, \
        f"연도 통제 후에도 상관이 남음 — 층화가 공통충격을 못 걷어냄: {controlled:.4f}"
    print(f"    (4) 연도 공통충격만 존재: 통제전 {naive:.4f} → 통제후 {controlled:.4f} "
          f"(층화가 거시요인을 제거함)")

    # ⑤ 손실 시뮬레이션 재현성 + 단조성 — 같은 입력이면 같은 결과여야 하고,
    #    상관이 커지면 꼬리손실은 반드시 커져야 한다(모형이 옳다면 수학적으로 보장됨).
    thr = norm.ppf(np.array([0.05, 0.10, 0.02, 0.20, 0.08]))
    expo = np.array([100.0, 300.0, 50.0, 800.0, 200.0])
    a1 = corr._simulate_losses([0.0, 0.4], thr, expo, 0.45, 20_000, seed=7)
    a2 = corr._simulate_losses([0.0, 0.4], thr, expo, 0.45, 20_000, seed=7)
    assert np.array_equal(a1[0.0], a2[0.0]) and np.array_equal(a1[0.4], a2[0.4]), \
        "손실 시뮬레이션이 같은 seed에서 재현되지 않음 (재현성 원칙 위반)"
    assert abs(a1[0.0].mean() - a1[0.4].mean()) / a1[0.0].mean() < 0.05, \
        "상관은 기대손실을 바꾸면 안 된다 (꼬리만 두꺼워져야 함)"
    q99_indep = float(np.quantile(a1[0.0], 0.99))
    q99_corr = float(np.quantile(a1[0.4], 0.99))
    assert q99_corr > q99_indep, \
        f"상관을 넣었는데 꼬리손실이 커지지 않음 ({q99_corr:.1f} vs {q99_indep:.1f})"
    print(f"    (5) 손실 시뮬레이션: 동일 seed 재현 확인 · EL 불변 · "
          f"99% 손실 {q99_indep:.0f}→{q99_corr:.0f} (상관이 꼬리만 두껍게 함)")


TESTS = [
    ("label_rules", test_label_rules),
    ("time_leakage", test_time_leakage),
    ("time_split", test_time_split),
    ("predictions_schema", test_predictions_schema),
    ("portfolio_sanity", test_portfolio),
    ("correlation_math", test_correlation_math),
]


def main() -> int:
    cfg = load_config()
    set_seed(cfg["seed"])
    processed_dir = ROOT / "data" / "processed"
    fp_before = _fingerprint_dir(processed_dir)

    print("=== FranSCORE sanity suite (python -m tests.test_sanity) ===")
    print("    isolation: all runs write to a temp dir; real data/processed and "
          "outputs/ are never written")
    results: list[tuple[str, bool, str]] = []

    with tempfile.TemporaryDirectory(prefix="franscore_sanity_") as td:
        ctx = {"cfg": cfg, "tmp": Path(td)}
        for i, (name, fn) in enumerate(TESTS, 1):
            print(f"[{i}/{len(TESTS)}] {name} ...")
            try:
                fn(ctx)
                results.append((name, True, ""))
                print(f"[{i}/{len(TESTS)}] {name} ... PASS")
            except Exception as e:
                results.append((name, False, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
                print(f"[{i}/{len(TESTS)}] {name} ... FAIL")

    # 실데이터 불변 가드 (테스트 실패와 별개로 항상 확인)
    fp_after = _fingerprint_dir(processed_dir)
    guard_ok = fp_before == fp_after
    if not guard_ok:
        print("GUARD FAIL: data/processed changed during tests!")
        before, after = {f[0]: f for f in fp_before}, {f[0]: f for f in fp_after}
        changed = sorted(set(before) ^ set(after)
                         | {k for k in set(before) & set(after) if before[k] != after[k]})
        print(f"  changed entries: {changed}")
    else:
        print("guard: data/processed unchanged (files/sizes/mtimes identical)")

    n_pass = sum(ok for _, ok, _ in results)
    print(f"=== SUMMARY: {n_pass}/{len(TESTS)} passed ===")
    for name, ok, msg in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({msg})" if msg else ""))
    return 0 if (n_pass == len(TESTS) and guard_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
