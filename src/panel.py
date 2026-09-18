"""M1 패널: 스냅샷 → brand_id×year 롱포맷 패널 + 생존성 진단표.

핵심 규칙 (문서화):
- 패널 year = 정보공개서 기준연도(yr) − 1. 공시 수치(가맹점수·변동·평균매출)는 통상
  직전 회계연도 실적이므로 실적연도 기준으로 정렬한다. (모든 변수 동일 규칙 → 내부 정합)
- avg_sales(avrgSlsAmt)·avg_sales_per_area == 0 → 결측(NaN) 처리 (영세 브랜드 미기재 관행).
  행 자체는 유지한다 — 점포수 기반 신호는 살아있기 때문.
- n_direct(직영점수)는 실적 API(15110241)에는 없고, 별도 지역·직영 API(15125490)의
  areaNm=='전체' 행에서 병합한다(_merge_region_direct, 병합률 ~85%). brand_overview에서는
  업력(biz_start_year)·임직원수(emp_cnt, exec_cnt)를 병합한다.
- 표본 자격(eligible_t): industry_major ∈ config 범위에서, (brand,t) 행 단위로
  [t까지 누적 최대 점포수 ≥ min_stores] AND [t에서 끝나는 연속 관측 ≥ min_consecutive_years]
  — 과거 정보만 사용(룩어헤드 금지). 관측 단위 하한은 labels의 min_stores_at_t가 담당.
- 산출: data/processed/panel_full.parquet(전체), panel.parquet(업종 범위 + eligible_t),
  industry_baseline.parquet(업종 기준선), outputs/survival_report.csv(진단표)
"""
from __future__ import annotations

import pandas as pd

from src.collect import load_snapshots
from src.common import get_logger, load_config
from src.entity import build_master, normalize_name

log = get_logger("panel")

_NUM_COLS = ["n_stores", "n_new", "n_contract_end", "n_contract_cancel", "n_name_change",
             "avg_sales", "avg_sales_per_area"]


def build_panel(cfg: dict, master: pd.DataFrame | None = None) -> pd.DataFrame:
    if master is None:
        master = build_master(cfg)
    df = master.copy()

    df["year"] = df["_yr"].astype(int) - 1  # 실적연도 정렬 (상단 규칙 참조)
    df["n_stores"] = pd.to_numeric(df["frcsCnt"], errors="coerce")
    df["n_new"] = pd.to_numeric(df["newFrcsRgsCnt"], errors="coerce")
    df["n_contract_end"] = pd.to_numeric(df["ctrtEndCnt"], errors="coerce")
    df["n_contract_cancel"] = pd.to_numeric(df["ctrtCncltnCnt"], errors="coerce")
    df["n_name_change"] = pd.to_numeric(df["nmChgCnt"], errors="coerce")
    df["avg_sales"] = pd.to_numeric(df["avrgSlsAmt"], errors="coerce")
    df["avg_sales_per_area"] = pd.to_numeric(df["arUnitAvrgSlsAmt"], errors="coerce")
    df["n_direct"] = float("nan")  # 플레이스홀더 — _merge_region_direct(15125490)가 채움

    if cfg["sample"].get("drop_zero_sales", True):
        for c in ("avg_sales", "avg_sales_per_area"):
            df.loc[df[c] <= 0, c] = float("nan")

    # (brand_id, year) 중복 제거 — 점포수 최대 행 유지, 개수 로깅
    dups = df.duplicated(["brand_id", "year"], keep=False).sum()
    if dups:
        log.warning("(brand_id, year) 중복 %d행 → 점포수 최대 행만 유지", int(dups))
        df = (df.sort_values("n_stores", ascending=False)
                .drop_duplicates(["brand_id", "year"], keep="first"))

    # 유령 0점포 아티팩트 처리 (적대적 리뷰 확정 결함 수정):
    # 직전 관측연도 점포수 ≥10 브랜드가 당해 0으로 보고된 경우는 공시 미기재/재등록
    # 아티팩트로 간주(실측: 412→0→395 등 원복 사례 다수)하고 결측 처리한다.
    # 판정에 과거 정보만 사용하므로 실시간 안전(look-ahead 없음).
    df = df.sort_values(["brand_id", "year"])
    prev_stores = df.groupby("brand_id")["n_stores"].shift(1)
    zero_artifact = (df["n_stores"] == 0) & (prev_stores >= 10)
    if zero_artifact.any():
        log.warning("점포수 0 보고 아티팩트 %d행 → 결측 처리 (직전연도 10개+ 브랜드)", int(zero_artifact.sum()))
        df.loc[zero_artifact, "n_stores"] = float("nan")

    # 브랜드 개요 병합 (업력·임직원수 — 라벨 차원 밖 보조 피처)
    df = _merge_overview(df, cfg)
    # 직영점수 + 지역분산 병합 (명세 3.1 필수 피처 원천)
    df = _merge_region_direct(df, cfg)
    # 브랜드 등록취소(자진/직권) 병합 — 하드 실패 신호
    df = _merge_cancel(df, cfg)
    # 창업금액 병합 — 여신 익스포저 산출의 관측 기반 (합성 난수 대체)
    df = _merge_startup_cost(df, cfg)

    cols = ["brand_id", "id_source", "brand_mnno", "hq_mnno",
            "brand_name", "company_name", "industry_major", "industry_mid",
            "year", *PANEL_VALUE_COLS]
    cols = list(dict.fromkeys([c for c in cols if c in df.columns or c in
                               ("n_stores", "n_direct")]))
    panel_full = df[cols].sort_values(["brand_id", "year"]).reset_index(drop=True)

    # 업종 기준선 저장
    _save_industry_baseline(cfg)

    # 실시간 표본 자격 부여 (업종 범위 전체 행 유지 + eligible_t 플래그)
    panel = apply_sample_filter(panel_full, cfg)

    proc = cfg["paths"]["processed"]
    panel_full.to_parquet(proc / "panel_full.parquet", index=False)
    panel.to_parquet(proc / "panel.parquet", index=False)
    log.info("panel_full: %d행 %d브랜드 / panel(업종 범위): %d행 %d브랜드 / 자격(eligible_t) 행: %d",
             len(panel_full), panel_full["brand_id"].nunique(),
             len(panel), panel["brand_id"].nunique(), int(panel["eligible_t"].sum()))
    return panel


_REGION_COLS = ["n_regions", "region_hhi", "top_region_share", "region_max_stores"]

# 15110265 창업금액 (천원). 여신 익스포저를 **공시 실측값**에서 유도하기 위한 원천.
_COST_COLS = ["startup_fee", "startup_edu", "startup_deposit", "startup_etc", "startup_total"]


def _merge_startup_cost(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """15110265 브랜드별 창업 금액(가맹금·교육비·보증금·기타) 병합. 단위 천원.

    왜 필요한가: 포트폴리오 여신 익스포저를 난수로 만들면 집중도·손실 수치가 전부
    가정의 산물이 된다. 창업비용은 정보공개서 **필수 공시 항목**이라 브랜드별 실측값이
    존재하므로, 이것을 여신 규모의 관측 기반으로 삼는다(자세한 산식은 portfolio.py).
    """
    for c in _COST_COLS:
        df[c] = float("nan")
    try:
        rows = load_snapshots(cfg, "brand_startup_cost")
    except Exception:
        rows = []
    if not rows:
        log.warning("brand_startup_cost 스냅샷 없음 — 창업비용 병합 생략(여신 익스포저 근거 약화)")
        return df

    cs = pd.DataFrame(rows)
    cs["norm_brand"] = cs["brandNm"].map(normalize_name)
    cs["norm_corp"] = cs.get("corpNm").map(normalize_name)
    cs["industry_major"] = cs["indutyLclasNm"].astype(str).str.strip()
    # ⚠️ **`_yr` 은 공시연도 그대로 둔다.** 여기서 한 번 더 빼면 안 된다.
    #    패널 쪽 `_yr` 은 이미 **공시연도**이고(이 파일 상단 `df["year"] = df["_yr"] - 1`),
    #    실적연도 변환은 그쪽에서 끝난다. 그런데 이 병합만 `_yr` 을 실적연도로 만들어
    #    조인하고 있었다 — 같은 이름의 컬럼이 양쪽에서 다른 뜻이었던 것이다.
    #
    #    그 결과 **패널 연도 Y 행에 공시 Y+2(=실적 Y+1)의 창업비용**이 붙었다(실측):
    #      153셀라음악학원 공시2021=48,520 → 패널 2019 에 기록 (정상이라면 패널 2020)
    #      1943          공시2024=241,000 → 패널 2022 에 기록 (정상이라면 패널 2023)
    #    그리고 최신 연도(패널 2024)는 짝이 될 공시 2026 이 없어 **100% 결측**이었다.
    #
    #    모형은 영향받지 않는다 — features.py 는 startup_* 를 한 번도 쓰지 않는다(실측 0회).
    #    영향은 ① 진단 소견에 적히는 창업비용 **연도 라벨**, ② 최신 연도 결측이다.
    cs["_yr"] = pd.to_numeric(cs["yr"], errors="coerce")
    ren = {"jngBzmnJngAmt": "startup_fee", "jngBzmnEduAmt": "startup_edu",
           "jngBzmnAssrncAmt": "startup_deposit", "jngBzmnEtcAmt": "startup_etc",
           "smtnAmt": "startup_total"}
    for src, dst in ren.items():
        cs[dst] = pd.to_numeric(cs.get(src), errors="coerce")
    cs = cs.dropna(subset=["_yr"])

    base = df.drop(columns=_COST_COLS)
    # 1차: 법인명까지 포함한 정확 조인 (동명 브랜드의 금액 오귀속 방지)
    c1 = cs.groupby(["norm_brand", "norm_corp", "industry_major", "_yr"],
                    as_index=False)[_COST_COLS].first()
    merged = base.merge(c1, on=["norm_brand", "norm_corp", "industry_major", "_yr"], how="left")
    hit1 = merged["startup_total"].notna()

    # 2차 폴백: (브랜드명, 업종, 연도)가 법인 1개로 유일할 때만
    c2 = (cs.groupby(["norm_brand", "industry_major", "_yr"])
            .agg(**{c: (c, "first") for c in _COST_COLS}, _n=("norm_corp", "nunique"))
            .reset_index())
    c2 = c2[c2["_n"] == 1].drop(columns=["_n"])
    merged = merged.merge(c2, on=["norm_brand", "industry_major", "_yr"],
                          how="left", suffixes=("", "_fb"))
    for c in _COST_COLS:
        merged[c] = merged[c].fillna(merged[f"{c}_fb"])
    merged = merged.drop(columns=[f"{c}_fb" for c in _COST_COLS])

    rate = merged["startup_total"].notna().mean()
    log.info("창업비용 병합률 %.1f%% (1차 정확키 %.1f%%) — 중앙값 %s천원",
             100 * rate, 100 * hit1.mean(),
             f"{merged['startup_total'].median():,.0f}" if rate else "-")
    return merged

# 패널의 '값' 컬럼 (식별자·연도 제외) — 단일 정의.
# ⚠️ 이 목록은 세 곳이 공유한다: ① build_panel 의 출력 컬럼 순서,
#    ② common.make_synthetic_panel 이 생성해야 할 컬럼, ③ 시점누출 테스트의 커버리지 검사.
#    여기에 컬럼을 추가하면 ②도 함께 추가해야 한다 (테스트가 자동으로 잡아낸다).
PANEL_VALUE_COLS = ["n_stores", "n_direct", *_NUM_COLS[1:],
                    "biz_start_year", "emp_cnt", "exec_cnt",
                    *_REGION_COLS, "cancel_flag", "cancel_type", *_COST_COLS]


def _merge_region_direct(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """15125490 브랜드×연도×지역 → 직영점수(n_direct) + 지역분산 지표 병합.

    - areaNm == '전체' 행 = 브랜드-연도 합계 → frcsCnt/dmsCnt (직영점수 원천).
      ⚠️ '전체' 행과 17개 시도 행이 같은 응답에 함께 오므로 단순 합산하면 정확히 2배 중복.
    - 17개 시도 행 → 지역 분산: 진출 지역 수, 지역 HHI(집중도), 최다 지역 비중.
    - 조인 키: (brand_mnno, hq_mnno, 연도) 우선 → 실패 시 (brand_mnno, 연도) 폴백
      (본부관리번호가 연도 간 바뀌는 브랜드 대응).
    """
    for c in _REGION_COLS:
        df[c] = float("nan")
    if "brand_mnno" not in df.columns or df["brand_mnno"].isna().all():
        log.warning("brand_mnno 없음 — 직영점수·지역분산 병합 생략 (한계 기록)")
        return df
    try:
        rows = load_snapshots(cfg, "brand_region_direct")
    except Exception as e:
        log.warning("brand_region_direct 스냅샷 로드 실패 (%s) — 병합 생략", e)
        return df
    if not rows:
        log.warning("brand_region_direct 스냅샷 없음 — 직영점수·지역분산 병합 생략 (한계 기록)")
        return df

    rd = pd.DataFrame(rows)
    rd["brand_mnno"] = rd["brandMnno"].astype(str)
    rd["hq_mnno"] = rd["jnghdqrtrsMnno"].astype(str)
    rd["frcsCnt"] = pd.to_numeric(rd["frcsCnt"], errors="coerce")
    rd["dmsCnt"] = pd.to_numeric(rd["dmsCnt"], errors="coerce")
    rd["areaNm"] = rd["areaNm"].astype(str).str.strip()

    # (1) 전체 행 → 직영점수
    tot = rd[rd["areaNm"] == "전체"].copy()
    tot_agg = (tot.groupby(["brand_mnno", "hq_mnno", "_yr"], as_index=False)
                  .agg(n_direct=("dmsCnt", "max"), region_total_frcs=("frcsCnt", "max")))

    # (2) 시도 행 → 지역 분산 (HHI·진출지역수·최다지역비중)
    reg = rd[(rd["areaNm"] != "전체") & (rd["frcsCnt"].fillna(0) > 0)].copy()

    def _region_stats(g: pd.DataFrame) -> pd.Series:
        s = g["frcsCnt"].to_numpy(dtype=float)
        tot_s = s.sum()
        if tot_s <= 0:
            return pd.Series({"n_regions": float("nan"), "region_hhi": float("nan"),
                              "top_region_share": float("nan"),
                              "region_max_stores": float("nan")})
        share = s / tot_s
        return pd.Series({
            "n_regions": float(len(s)),
            "region_hhi": float((share ** 2).sum()),
            "top_region_share": float(share.max()),
            "region_max_stores": float(s.max()),
        })

    reg_agg = (reg.groupby(["brand_mnno", "hq_mnno", "_yr"])
                  .apply(_region_stats, include_groups=False).reset_index())

    side = tot_agg.merge(reg_agg, on=["brand_mnno", "hq_mnno", "_yr"], how="outer")
    val_cols = ["n_direct", "region_total_frcs", *_REGION_COLS]

    # 1차: (brand_mnno, hq_mnno, 연도)
    # ⚠️ df에 이미 있는 n_direct(전부 NaN 플레이스홀더)를 함께 제거해야 suffix 충돌이 없다
    drop_first = [c for c in ([*_REGION_COLS, "n_direct"]) if c in df.columns]
    merged = df.drop(columns=drop_first).merge(
        side, on=["brand_mnno", "hq_mnno", "_yr"], how="left")
    hit1 = merged["n_direct"].notna()

    # 2차 폴백: (brand_mnno, 연도) — **양쪽 모두** 유일할 때만 (리뷰 확정 결함 수정).
    # 지역데이터 쪽 유일성(_n_hq==1)만 보면, R3 충돌 분리로 서로 다른 엔티티가 된 패널 행들
    # (같은 brandMnno·같은 연도·다른 본부)에 한 엔티티의 지역·직영 데이터가 복제 귀속된다.
    # → 패널 쪽도 해당 (brand_mnno, 연도)에 brand_id가 정확히 1개일 때만 폴백을 적용한다.
    side_nb = side.groupby(["brand_mnno", "_yr"], as_index=False).agg(
        **{c: (c, "max") for c in val_cols},
        _n_hq=("hq_mnno", "nunique"))
    side_nb = side_nb[side_nb["_n_hq"] == 1].drop(columns=["_n_hq"])
    panel_uniq = (merged.groupby(["brand_mnno", "_yr"])["brand_id"].nunique()
                  .rename("_n_panel_ids").reset_index())
    side_nb = side_nb.merge(panel_uniq, on=["brand_mnno", "_yr"], how="left")
    n_amb = int((side_nb["_n_panel_ids"].fillna(0) > 1).sum())
    if n_amb:
        log.info("지역/직영 2차 폴백: 패널 쪽 모호 (brand_mnno,연도) %d그룹 제외 (오귀속 방지)", n_amb)
    side_nb = side_nb[side_nb["_n_panel_ids"].fillna(0) == 1].drop(columns=["_n_panel_ids"])
    merged = merged.merge(side_nb, on=["brand_mnno", "_yr"], how="left", suffixes=("", "_nb"))
    for c in val_cols:
        merged[c] = merged[c].where(merged[c].notna(), merged[f"{c}_nb"])
    merged = merged.drop(columns=[f"{c}_nb" for c in val_cols])

    # 정합 검증: 두 API의 가맹점수가 일치하는지 (조인 신뢰도 근거)
    both = merged["region_total_frcs"].notna() & merged["n_stores"].notna()
    if both.any():
        agree = (merged.loc[both, "region_total_frcs"] == merged.loc[both, "n_stores"]).mean()
        log.info("직영/지역 병합 검증: 가맹점수 일치율 %.1f%% (독립 API 간 교차검증)", 100 * agree)
    log.info("직영점수 병합률 %.1f%% (1차 키 %.1f%%) · 지역분산 병합률 %.1f%%",
             100 * merged["n_direct"].notna().mean(), 100 * hit1.mean(),
             100 * merged["region_hhi"].notna().mean())
    return merged.drop(columns=["region_total_frcs"])


def _merge_cancel(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """15125518 브랜드 등록취소 → 취소 연도 플래그(자진/직권). 명칭 기반 조인(관리번호 없음)."""
    df["cancel_flag"] = 0.0
    df["cancel_type"] = pd.Series(pd.NA, index=df.index, dtype=object)
    try:
        rows = load_snapshots(cfg, "brand_cancel")
    except Exception:
        rows = []
    if not rows:
        log.warning("brand_cancel 스냅샷 없음 — 등록취소 신호 생략")
        return df
    cx = pd.DataFrame(rows)
    cx["norm_brand"] = cx["brandNm"].map(normalize_name)
    cx["norm_corp"] = cx.get("jnghdqrtrsConmNm").map(normalize_name)
    cx = cx[cx["norm_brand"] != ""].copy()
    # 취소 '사건 연도' = 접수일(rcptDate)의 달력 연도 (리뷰 지적 반영: 공시연도(_yr)로 붙이면
    # 접수가 공시연도에 있는 ~8% 기록이 실제 발생 연도보다 한 해 앞 패널 행에 붙는다).
    # 패널 year는 달력(회계) 연도이므로 rcpt 연도로 조인하는 것이 사건 시점과 정합.
    rcpt_year = pd.to_numeric(cx.get("rcptDate", pd.Series(dtype=str)).astype(str).str[:4],
                              errors="coerce")
    cx["year"] = rcpt_year.fillna(pd.to_numeric(cx["_yr"], errors="coerce") - 1).astype("Int64")
    key = (cx.groupby(["norm_brand", "norm_corp", "year"], as_index=False)
             .agg(cancel_type=("rgsRtrcnTyNm", "first")))
    key["cancel_flag"] = 1.0
    key["year"] = key["year"].astype(int)
    out = df.drop(columns=["cancel_flag", "cancel_type"]).merge(
        key, on=["norm_brand", "norm_corp", "year"], how="left")
    out["cancel_flag"] = out["cancel_flag"].fillna(0.0)
    log.info("등록취소 병합(접수연도 기준): %d행 매칭 (자진/직권 %s)", int(out["cancel_flag"].sum()),
             out["cancel_type"].dropna().value_counts().to_dict())
    return out


def _merge_overview(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    try:
        rows = load_snapshots(cfg, "brand_overview")
    except Exception:
        rows = []
    if not rows:
        log.warning("brand_overview 스냅샷 없음 — 업력·임직원수 피처 생략")
        df["biz_start_year"] = float("nan")
        df["emp_cnt"] = float("nan")
        df["exec_cnt"] = float("nan")
        return df
    ov = pd.DataFrame(rows)
    ov["norm_brand"] = ov["brandNm"].map(normalize_name)
    ov["norm_corp"] = ov.get("corpNm").map(normalize_name)
    ov["industry_major"] = ov["indutyLclasNm"].astype(str).str.strip()
    ov["biz_start_year"] = pd.to_numeric(
        ov.get("jngBizStrtDate", pd.Series(dtype=str)).astype(str).str[:4], errors="coerce")
    ov["emp_cnt"] = pd.to_numeric(ov.get("empCnt"), errors="coerce")
    ov["exec_cnt"] = pd.to_numeric(ov.get("allExctvCnt"), errors="coerce")
    cols = ["biz_start_year", "emp_cnt", "exec_cnt"]

    # 1차: 법인명까지 포함해 정확 조인 (리뷰 지적 반영: 동명 브랜드가 서로 다른 본부일 때
    # 임의 법인의 임직원수·업력이 붙는 것을 방지 — 15109828은 corpNm을 제공한다)
    ov1 = ov.groupby(["norm_brand", "norm_corp", "industry_major", "_yr"], as_index=False)[cols].first()
    merged = df.merge(ov1, on=["norm_brand", "norm_corp", "industry_major", "_yr"], how="left")
    hit1 = merged["biz_start_year"].notna() | merged["emp_cnt"].notna()

    # 2차 폴백: (브랜드명, 업종, 연도)가 **법인 1개로 유일**할 때만
    ov2 = (ov.groupby(["norm_brand", "industry_major", "_yr"])
             .agg(**{c: (c, "first") for c in cols}, _n_corp=("norm_corp", "nunique"))
             .reset_index())
    ov2 = ov2[ov2["_n_corp"] == 1].drop(columns=["_n_corp"])
    merged = merged.merge(ov2, on=["norm_brand", "industry_major", "_yr"],
                          how="left", suffixes=("", "_nb"))
    for c in cols:
        merged[c] = merged[c].where(merged[c].notna(), merged[f"{c}_nb"])
    merged = merged.drop(columns=[f"{c}_nb" for c in cols])
    rate = merged["biz_start_year"].notna().mean()
    log.info("brand_overview 병합률: %.1f%% (법인명 정확 조인 %.1f%%)",
             100 * rate, 100 * hit1.mean())
    return merged


def _save_industry_baseline(cfg: dict) -> None:
    try:
        rows = load_snapshots(cfg, "industry_openclose")
    except Exception:
        return
    if not rows:
        return
    ib = pd.DataFrame(rows)
    ib = ib.rename(columns={
        "tpbizLclsfNm": "industry_major", "tpbizMclsfNm": "industry_mid",
        "newFrcsRt": "ind_new_rate", "endCncltnRt": "ind_end_rate",
        "frcsCnt": "ind_n_stores",
    })
    ib["year"] = pd.to_numeric(ib["jngBizCrtrYr"], errors="coerce").astype("Int64") - 1
    keep = ["year", "industry_major", "industry_mid", "ind_new_rate", "ind_end_rate", "ind_n_stores"]
    ib = ib[[c for c in keep if c in ib.columns]]
    ib.to_parquet(cfg["paths"]["processed"] / "industry_baseline.parquet", index=False)
    log.info("industry_baseline 저장: %d행", len(ib))


def apply_sample_filter(panel_full: pd.DataFrame, cfg: dict, relaxed: bool = False) -> pd.DataFrame:
    """실시간(과거 정보만) 표본 자격 부여 — 적대적 리뷰 확정 결함 수정.

    (구버전은 브랜드 전체 이력의 max(점포수)·최장 연속구간으로 편입을 결정해
    라벨 윈도우 안의 미래 성장이 표본 선별에 새어 들어갔다.)

    새 규칙 — (brand, t) 행 단위, t 이하 정보만 사용:
      eligible_t = [t까지의 누적 최대 점포수 ≥ min_stores]
                   AND [t에서 끝나는 연속 관측 구간 길이 ≥ min_consecutive_years]
    반환: 업종 범위 전체 행 + ``eligible_t`` 컬럼.
      - 라벨 분위수 풀·피처는 업종 범위 전체로 계산 (생존 편향 없는 모집단)
      - 학습/평가 표본은 eligible_t=True 행만 사용 (run_pipeline에서 필터)
    """
    s = cfg["sample"]["relaxed"] if relaxed else cfg["sample"]
    majors = s.get("industry_major", cfg["sample"]["industry_major"])
    min_stores = s.get("min_stores", cfg["sample"]["min_stores"])
    min_years = int(cfg["sample"]["min_consecutive_years"])

    df = (panel_full[panel_full["industry_major"].isin(majors)]
          .sort_values(["brand_id", "year"]).reset_index(drop=True).copy())

    g = df.groupby("brand_id")
    cummax_stores = g["n_stores"].cummax()          # t까지의 최대 점포수 (과거만)
    year_gap = g["year"].diff()
    new_run = year_gap.ne(1) | year_gap.isna()      # 연속 구간 시작점
    run_id = new_run.cumsum()                       # brand 정렬 상태라 브랜드 경계에서 항상 새 구간
    run_len = df.groupby(run_id).cumcount() + 1     # t에서 끝나는 연속 관측 길이

    df["eligible_t"] = (cummax_stores >= min_stores) & (run_len >= min_years)
    if relaxed:
        (cfg["paths"]["outputs"] / "sample_relaxed.flag").write_text("relaxed", encoding="utf-8")
    return df


def survival_report(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """생존성 진단표 (M1 DoD): 필터 단계별 표본 수 + (가능 시) 라벨 양성률."""
    proc = cfg["paths"]["processed"]
    full_path = proc / "panel_full.parquet"
    full = pd.read_parquet(full_path) if full_path.exists() else panel

    steps: list[dict] = []

    def _add(step: str, df: pd.DataFrame) -> None:
        steps.append({
            "단계": step,
            "행수": len(df),
            "브랜드수": df["brand_id"].nunique() if len(df) else 0,
            "연도범위": f"{df['year'].min()}~{df['year'].max()}" if len(df) else "-",
            "매출보유율%": round(100 * df["avg_sales"].notna().mean(), 1) if len(df) else 0,
        })

    _add("① 전체(정합 후)", full)
    scoped = apply_sample_filter(full, cfg)
    _add(f"② 업종 필터({','.join(cfg['sample']['industry_major'])})", scoped)
    eligible = scoped[scoped["eligible_t"]]
    _add(f"③ +실시간 자격(점포{cfg['sample']['min_stores']}+ 누적 & "
         f"{cfg['sample']['min_consecutive_years']}년연속 종료)", eligible)

    report = pd.DataFrame(steps)

    # 라벨 양성률 (labels 모듈이 준비된 경우) — 분위수 풀은 업종 전체, 표본은 자격 행만
    label_info = ""
    try:
        from src import labels as labels_mod
        lab = labels_mod.build_labels(scoped, cfg)
        lab = lab.merge(scoped[["brand_id", "year", "eligible_t"]], on=["brand_id", "year"], how="left")
        lab = lab[lab["eligible_t"].fillna(False)].drop(columns=["eligible_t"])
        overall = 100 * lab["label"].mean()
        by_year = (lab.groupby("year")["label"].agg(["count", "mean"])
                   .assign(양성률=lambda d: (100 * d["mean"]).round(1)))
        label_info = f"라벨 표본 {len(lab)}행, 양성률 {overall:.1f}%"
        by_year.to_csv(cfg["paths"]["outputs"] / "label_rate_by_year.csv", encoding="utf-8-sig")
        log.info("연도별 라벨: \n%s", by_year[["count", "양성률"]].to_string())
    except Exception as e:
        label_info = f"라벨 미산출 (labels 모듈 대기): {e}"
    log.info("생존성 진단: %s", label_info)

    out_path = cfg["paths"]["outputs"] / "survival_report.csv"
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n===== 생존성 진단표 (M1 DoD) =====")
    print(report.to_string(index=False))
    print(label_info)
    return report


if __name__ == "__main__":
    cfg = load_config()
    p = build_panel(cfg)
    survival_report(p, cfg)
