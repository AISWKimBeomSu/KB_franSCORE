"""FranSCORE 공통 유틸: 설정 로드·로깅·시드·합성 스모크 패널.

모든 모듈이 이 파일만 통해 설정/경로에 접근한다 (단일 진실 원천).
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

_CFG_CACHE: dict | None = None
_SECRETS_LOADED = False

# 이 이름들만 시크릿으로 취급한다. 임의의 키를 환경변수로 밀어넣지 않기 위한 화이트리스트.
SECRET_KEYS = (
    "GEMINI_API_KEY", "DART_API_KEY", "DATA_GO_KR_KEY", "IFRMP_SERVICE_KEY",
    "NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET",   # 구 개발자센터 (검색만)
    "NCP_API_KEY_ID", "NCP_API_KEY",            # NAVER API HUB (검색 + 검색어트렌드)
    # 예비 Gemini 키. 주 키가 무료 한도(429)에 걸리면 src/llm.py 가 순서대로 갈아탄다.
    # 무료 한도는 **키 단위**로 걸리므로, 로고 판독처럼 호출이 수천 건인 작업은 키 수가
    # 곧 하루 처리량이다(실측: 키 2개로 13회 만에 소진). 없는 키는 건너뛴다.
    *[f"GEMINI_API_KEY_{i}" for i in range(2, 13)],
)

# ---------------------------------------------------------------------------
# 가동 능력 (capability) — 키가 없을 때 무엇이 어떻게 줄어드는가
#
# 왜 필요한가 (실제로 있었던 일)
#   일간 배치가 매일 GitHub Actions 에서 **success** 로 끝나고 있었다. 그런데 실행
#   로그를 열어 보니 `NCP_API_KEY_ID:` 가 빈 값이었고, 뉴스는 키가 필요 없는 Google
#   News RSS **제목만**으로 떨어졌으며, 사건 추출은 `LLM 미사용 — 규칙기반 폴백`
#   이었다. 폴백이 정상 동작한 덕분에 아무것도 실패하지 않았고, 그래서 **성능이
#   3분의 1로 줄어든 채 매일 초록불**이 켜졌다.
#
#   폴백 자체는 옳은 설계다. 잘못된 것은 그 사실이 어디에도 안 남는다는 점이다.
#   이 표는 축소 가동을 산출물(refresh_state.json)과 화면에 그대로 노출시킨다.
# ---------------------------------------------------------------------------

#  need: 이 기능이 살아나는 데 필요한 키 묶음 (묶음 중 **하나라도** 충족하면 가동)
CAPABILITIES: tuple[dict, ...] = (
    {"name": "뉴스 사건 구조화 추출",
     "need": [("GEMINI_API_KEY",),
              ("GEMINI_API_KEY_2",),
              ("GEMINI_API_KEY_3",),
              ("GEMINI_API_KEY_4",),
              ("GEMINI_API_KEY_5",),
              ("GEMINI_API_KEY_6",),
              ("GEMINI_API_KEY_7",),
              ("GEMINI_API_KEY_8",),
              ("GEMINI_API_KEY_9",),
              ("GEMINI_API_KEY_10",),
              ("GEMINI_API_KEY_11",),
              ("GEMINI_API_KEY_12",)],
     "fallback": "규칙기반 키워드 매칭",
     "impact": "신호에 llm_used=false 로 표기된다. 제목의 단어만 보므로 홍보 기사와 "
               "분쟁 기사를 가르는 정밀도가 떨어진다."},
    {"name": "뉴스 본문 요약 확보",
     "need": [("NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET"),
              ("NCP_API_KEY_ID", "NCP_API_KEY")],
     "fallback": "Google News RSS — 제목만",
     "impact": "판단 근거가 한 문단에서 한 줄로 줄어든다. 국내 프랜차이즈·지역 매체 "
               "색인도 얕아진다."},
    {"name": "검색수요 지표",
     "need": [("NCP_API_KEY_ID", "NCP_API_KEY")],
     "fallback": "없음 — 수집하지 않는다",
     "impact": "수요 관련 진단 소견이 아예 생성되지 않는다(없는 근거로 위험을 "
               "말하지 않기 위해). 다른 소견은 영향 없다."},
    {"name": "AI 상담·심사메모 생성",
     "need": [("GEMINI_API_KEY",),
              ("GEMINI_API_KEY_2",),
              ("GEMINI_API_KEY_3",),
              ("GEMINI_API_KEY_4",),
              ("GEMINI_API_KEY_5",),
              ("GEMINI_API_KEY_6",),
              ("GEMINI_API_KEY_7",),
              ("GEMINI_API_KEY_8",),
              ("GEMINI_API_KEY_9",),
              ("GEMINI_API_KEY_10",),
              ("GEMINI_API_KEY_11",),
              ("GEMINI_API_KEY_12",)],
     "fallback": "결정적 템플릿 (llm_used=false 표기)",
     "impact": "산출 수치는 같다. 문장이 템플릿으로 고정되고 자유질의 응답이 제한된다."},
    {"name": "가맹본부 재무 갱신",
     "need": [("DART_API_KEY",)],
     "fallback": "기존 수집분 사용",
     "impact": "일간 배치에는 영향 없다 — 재무는 연 1회 공시라 이미 받아 둔 값을 쓴다."},
)


def capability_report() -> dict:
    """지금 이 프로세스가 **실제로** 무엇을 할 수 있는지. 키 값은 담지 않는다.

    Returns:
        {"keys": {키이름: 있음/없음}, "features": [...], "degraded": [기능명, ...]}
    """
    load_secrets()
    present = {k: bool(os.environ.get(k, "").strip()) for k in SECRET_KEYS}
    feats, degraded = [], []
    for cap in CAPABILITIES:
        ok = any(all(present.get(k) for k in group) for group in cap["need"])
        if not ok:
            degraded.append(cap["name"])
        feats.append({
            "name": cap["name"], "full": ok,
            "needs": [" + ".join(g) for g in cap["need"]],
            "fallback": cap["fallback"], "impact": cap["impact"],
        })
    return {"keys": present, "features": feats, "degraded": degraded,
            "n_full": sum(1 for f in feats if f["full"]), "n_total": len(feats)}


def load_secrets() -> list[str]:
    """API 키를 환경변수로 올린다. 이미 설정된 값은 절대 덮어쓰지 않는다.

    왜 필요한가
      · 로컬: 키를 사용자 스코프에 setx 로 넣어도 **이미 떠 있는 프로세스와 그 자식**은
        그 값을 상속받지 못한다. 그래서 대시보드만 키를 못 보는 상황이 생긴다(실측).
      · 배포: 플랫폼마다 시크릿 전달 방식이 다르다. Streamlit Community Cloud 는
        st.secrets 로 주고, 컨테이너 배포는 환경변수로 준다. 두 경로를 모두 받는다.

    우선순위: 기존 환경변수 > .env 파일 > st.secrets.
    Returns: 이번 호출로 새로 채워진 키 이름 목록 (값은 반환하지 않는다).
    """
    global _SECRETS_LOADED
    filled: list[str] = []
    if _SECRETS_LOADED:
        return filled

    env_file = ROOT / ".env"
    if env_file.exists():
        try:
            # utf-8-sig: Windows PowerShell 의 Out-File 은 BOM 을 붙인다. 그냥 utf-8 로
            # 읽으면 첫 줄 키가 '﻿GEMINI_API_KEY' 가 되어 조용히 무시된다(실측).
            for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k in SECRET_KEYS and v and not os.environ.get(k):
                    os.environ[k] = v
                    filled.append(k)
        except OSError:
            pass

    # Streamlit 아래에서 돌 때만 st.secrets 를 본다 (파이프라인 실행에 streamlit 을
    # 끌어들이지 않기 위해 이미 임포트된 경우로 한정한다).
    if "streamlit" in sys.modules:
        try:
            import streamlit as st
            for k in SECRET_KEYS:
                if not os.environ.get(k) and k in st.secrets:
                    os.environ[k] = str(st.secrets[k])
                    filled.append(k)
        except Exception:
            pass

    _SECRETS_LOADED = True
    return filled


def load_config(path: str | Path | None = None) -> dict:
    """config.yaml 로드. cfg['_root']에 프로젝트 루트 Path를 담는다."""
    global _CFG_CACHE
    load_secrets()
    if path is None and _CFG_CACHE is not None:
        return _CFG_CACHE
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    for key, rel in cfg["paths"].items():
        p = ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = p
    if path is None:
        _CFG_CACHE = cfg
    return cfg


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"franscore.{name}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        try:
            fh = logging.FileHandler(ROOT / "outputs" / "pipeline.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError:
            pass
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 합성 스모크 패널 — ⚠️ 파이프라인 배선 점검(스모크테스트) 전용.
# 제출 지표·데모 수치에 절대 사용하지 않는다 (outputs/_smoke 격리).
# ---------------------------------------------------------------------------

_INDUSTRIES = {
    "외식": ["치킨", "커피", "한식", "분식", "피자", "주점"],
    "서비스": ["교육", "세탁", "이미용"],
    "도소매": ["편의점", "화장품"],
}


def make_synthetic_panel(cfg: dict, n_brands: int = 300, years: tuple[int, int] = (2019, 2024)) -> pd.DataFrame:
    """INTERFACES.md §1 스키마를 따르는 합성 brand×year 패널.

    일부 브랜드는 특정 연도에 '구조악화 국면'으로 전환(성장 둔화→점포 순감·매출 하락·
    계약종료 급증)하도록 설계해 라벨·모델 코드의 배선을 점검할 수 있게 한다.

    ⚠️ **실패널의 전 컬럼을 빠짐없이 생성해야 한다.** 컬럼이 하나 빠지면 build_features 의
       해당 분기(`if "biz_start_year" in df.columns` 등)가 통째로 건너뛰어지고,
       그 피처는 시점누출 테스트(tests/test_sanity.test_time_leakage)의 검사 범위에서
       조용히 빠진다 — "전 피처 누출 검증"이라는 주장이 거짓이 된다(적대적 리뷰 지적).
       panel.py 가 컬럼을 추가하면 여기도 반드시 함께 추가할 것.

    지역 분포는 임의값을 흩뿌리지 않고 **내부적으로 정합**하게 생성한다:
    상위지역 점유율 top_share 를 뽑고 나머지 지역에 균등 분배한 뒤 HHI 를 실제로 계산해
    (n_regions, region_hhi, top_region_share, region_max_stores) 가 서로 모순되지 않게 한다.
    """
    rng = np.random.default_rng(cfg["seed"])
    majors = list(_INDUSTRIES.keys())
    rows = []
    y0, y1 = years
    for i in range(n_brands):
        major = majors[rng.choice(len(majors), p=[0.7, 0.2, 0.1])]
        mid = _INDUSTRIES[major][rng.integers(len(_INDUSTRIES[major]))]
        brand_id = f"SYN{i:04d}"
        start = int(rng.integers(y0, y0 + 3))
        n_stores = float(rng.integers(15, 400))
        n_direct = max(0.0, round(n_stores * rng.uniform(0.0, 0.15)))
        avg_sales = float(rng.uniform(150_000, 700_000))  # 천원
        base_growth = rng.normal(0.06, 0.08)
        # 가맹사업 개시연도 (브랜드 고정) — 관측 시작보다 0~15년 앞선다
        biz_start_year = float(start - int(rng.integers(0, 16)))
        # 악화 전환 연도 (약 35% 브랜드, 관측 중반 이후)
        turn_year = int(rng.integers(start + 2, y1 + 1)) if rng.random() < 0.35 else None
        for year in range(start, y1 + 1):
            deteriorated = turn_year is not None and year >= turn_year
            g = rng.normal(-0.12, 0.06) if deteriorated else rng.normal(base_growth, 0.05)
            sg = rng.normal(-0.10, 0.05) if deteriorated else rng.normal(0.03, 0.05)
            end_rate = rng.uniform(0.12, 0.30) if deteriorated else rng.uniform(0.02, 0.10)
            prev_stores = n_stores
            n_stores = max(3.0, round(n_stores * (1 + g)))
            n_end = round(prev_stores * end_rate * rng.uniform(0.6, 1.0))
            n_cancel = round(prev_stores * end_rate * rng.uniform(0.0, 0.4))
            n_new = max(0.0, round(n_stores - prev_stores + n_end + n_cancel))
            avg_sales = max(30_000.0, avg_sales * (1 + sg))
            # 직영점수도 해마다 변한다. 브랜드 내 고정값으로 두면 f_struct_direct_growth 가
            # 전 행 0(상수)이 되어 **그 피처가 누출 테스트에서 사실상 검증되지 않는다**.
            n_direct = float(min(n_stores, max(0.0, round(n_direct * (1 + rng.normal(0.03, 0.15))))))
            # 본부 규모 (15109828 브랜드개요 대응) — 점포수에 느슨히 연동
            emp_cnt = float(max(3, round(n_stores * rng.uniform(0.05, 0.30))))
            exec_cnt = float(max(1, round(emp_cnt * rng.uniform(0.02, 0.15))))
            # 지역 분포 (15125490 지역별 대응) — 서로 모순 없도록 실제로 계산한다
            n_regions = int(min(17, max(1, round(n_stores / 12) + int(rng.integers(-1, 2)))))
            top_share = float(rng.uniform(1.0 / n_regions, min(1.0, 1.0 / n_regions + 0.5)))
            rest = (1.0 - top_share) / (n_regions - 1) if n_regions > 1 else 0.0
            region_hhi = top_share ** 2 + (n_regions - 1) * rest ** 2
            # 창업비용 (15110265 대응, 천원) — 브랜드 내에서는 완만히 변동
            startup_fee = float(round(rng.uniform(3_000, 12_000), -2))
            startup_edu = float(round(rng.uniform(1_000, 6_000), -2))
            startup_deposit = float(round(rng.uniform(0, 5_000), -2))
            startup_etc = float(round(rng.uniform(20_000, 150_000), -2))
            # 등록취소 (15125518 대응) — 악화 전환 브랜드의 마지막 연도에 드물게 발생
            cancelled = turn_year is not None and year == y1 and rng.random() < 0.15
            rows.append({
                "brand_id": brand_id,
                "brand_name": f"합성브랜드{i:04d}",
                "company_name": f"합성본부{i % 120:03d}",
                "industry_major": major,
                "industry_mid": mid,
                "year": year,
                "n_stores": n_stores,
                "n_direct": n_direct,
                "n_new": float(n_new),
                "n_contract_end": float(n_end),
                "n_contract_cancel": float(n_cancel),
                "n_name_change": float(rng.integers(0, max(2, int(prev_stores * 0.05)))),
                "avg_sales": avg_sales if rng.random() > 0.05 else np.nan,
                "avg_sales_per_area": avg_sales / rng.uniform(8, 25),
                "biz_start_year": biz_start_year,
                "emp_cnt": emp_cnt,
                "exec_cnt": exec_cnt,
                "n_regions": float(n_regions),
                "region_hhi": float(region_hhi),
                "top_region_share": float(top_share),
                "region_max_stores": float(round(n_stores * top_share)),
                # 창업비용 (천원) — 실패널은 15110265 공시값. 업종·브랜드별로 크게 다르다.
                "startup_fee": startup_fee,
                "startup_edu": startup_edu,
                "startup_deposit": startup_deposit,
                "startup_etc": startup_etc,
                "startup_total": startup_fee + startup_edu + startup_deposit + startup_etc,
                "cancel_flag": 1.0 if cancelled else 0.0,
                "cancel_type": ("자진취소" if rng.random() < 0.7 else "직권취소") if cancelled else None,
            })
    df = pd.DataFrame(rows).sort_values(["brand_id", "year"]).reset_index(drop=True)
    return df


def industry_group_col(panel: pd.DataFrame, min_group: int = 30) -> pd.Series:
    """업종그룹 결정: 해당 연도 industry_mid 그룹 크기 ≥ min_group이면 mid, 아니면 major.

    라벨·피처의 업종 내 분위수 계산에 공용으로 사용 (INTERFACES.md §1).
    """
    sizes = panel.groupby(["year", "industry_mid"])["brand_id"].transform("size")
    return panel["industry_mid"].where(sizes >= min_group, panel["industry_major"])
