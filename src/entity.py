"""M1 엔티티 정합 — 브랜드 관리번호(brandMnno) 우선, 명칭은 보조.

명세 §2.2 준수:
    "연결 키 = 브랜드 관리번호(우선). 명칭은 보조(공백·특수문자·영문/한글 정규화).
     리브랜딩·본부변경·관리번호 변경·동명이업종 → 정합 규칙 문서화(못 잇는 건 버림, 버린 수 로깅)."

문제: 실적 API(15110241 getBrandFrcsStats)에는 관리번호가 없다(전 필드 실측 확인).
      관리번호는 별도 브랜드 등록 마스터(15125467 getBrandinfo)에만 존재한다.
해결: 2홉 정합.
    1홉 (명칭 크로스워크): 실적 행 (정규화 브랜드명, 정규화 법인명) → 마스터의 (brandMnno, jnghdqrtrsMnno).
          마스터는 **전 연도 합집합**을 쓴다 — 실적 연도에 등록이 없는(전/후년도 등록) 브랜드까지
          잇기 위함. 실측 매칭률: 연도별 마스터만 88.7% → 전연도 합집합 97.8%.
    2홉 (관리번호 정합): 확보한 관리번호로 연도 간 브랜드를 연결한다. 관리번호는 76.8%가
          복수 연도에 걸쳐 안정적이며, 리브랜딩(명칭 변경)에도 동일 ID를 유지한다.

정합 규칙 (실측 근거와 함께 문서화):
  R1. 정규화: NFKC → 법인 접두어(주식회사/(주)/㈜/(유)/유한회사/농업회사법인/재단·사단법인) 제거
      → 영숫자·한글 외 문자 제거 → 소문자.
  R2. 1홉 우선순위: ① (브랜드명, 법인명) 완전일치 → ② 브랜드명 단독일치(법인명 불일치 허용)
      → ③ 미매칭. 법인명을 함께 쓰면 동명 브랜드의 모호성이 줄어든다(실측: 고유해석 84.7%→88.6%).
  R3. 관리번호 단위 키 = (brandMnno, jnghdqrtrsMnno). brandMnno 단독은 연도 내에서도
      유일하지 않다(2023년 620개 ID가 1,256행에 중복 — 동일 브랜드명을 서로 다른 본부가 보유).
  R4. 1홉에서 후보 관리번호가 2개 이상이면 '모호'로 분류하고 관리번호를 부여하지 않는다(버림 아님).
      → 명칭 기반 대체 ID(NAME:정규화명@업종)로 패널에 남기고, 그 수를 로깅한다.
      (버리면 표본이 줄고 생존 편향이 생기므로, 남기고 출처를 표시하는 편이 정직하다.)
  R5. 정규화 후 브랜드명이 빈 행은 연결 불가 → 제거하고 수를 로깅.

산출: brand_id 및 id_source(MNNO/NAME) 부여된 실적 행 단위 DataFrame + 정합 리포트 CSV.
"""
from __future__ import annotations

import contextlib
import re
import unicodedata

import pandas as pd

from src.collect import load_snapshots
from src.common import get_logger, load_config

log = get_logger("entity")

_CORP_PREFIX = re.compile(r"(주식회사|\(주\)|㈜|\(유\)|유한회사|농업회사법인|\(재\)|재단법인|\(사\)|사단법인)")
_NON_ALNUM = re.compile(r"[^0-9a-z가-힣]")


def normalize_name(s: str | None) -> str:
    """R1 정규화: NFKC → 법인접두 제거 → 특수문자·공백 제거 → 소문자."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    s = _CORP_PREFIX.sub("", s)
    s = _NON_ALNUM.sub("", s)
    return s


def build_crosswalk(cfg: dict) -> pd.DataFrame:
    """브랜드 등록 마스터(전 연도 합집합) → 명칭 → 관리번호 크로스워크.

    반환: [norm_brand, norm_corp, brand_mnno, hq_mnno, brno, n_candidates]
          (norm_brand, norm_corp) 단위 1행. n_candidates>1 이면 모호(R4).
    """
    rows = load_snapshots(cfg, "brand_master")
    if not rows:
        log.warning("brand_master 스냅샷 없음 — 관리번호 정합 불가, 명칭 기반으로 폴백")
        return pd.DataFrame(columns=["norm_brand", "norm_corp", "brand_mnno",
                                     "hq_mnno", "brno", "n_candidates"])
    m = pd.DataFrame(rows)
    m["norm_brand"] = m["brandNm"].map(normalize_name)
    m["norm_corp"] = m.get("corpNm").map(normalize_name)
    m = m[m["norm_brand"] != ""].copy()
    m["brand_mnno"] = m["brandMnno"].astype(str)
    m["hq_mnno"] = m["jnghdqrtrsMnno"].astype(str)
    m["brno"] = m.get("brno").astype(str)

    # (브랜드명, 법인명) 단위로 관리번호 후보 집계 — 전 연도 합집합
    grp = (m.groupby(["norm_brand", "norm_corp"])
             .agg(brand_mnno=("brand_mnno", lambda s: sorted(set(s))[0]),
                  hq_mnno=("hq_mnno", lambda s: sorted(set(s))[0]),
                  brno=("brno", lambda s: sorted(set(s))[0]),
                  n_candidates=("brand_mnno", lambda s: s.nunique()))
             .reset_index())
    log.info("크로스워크(마스터 전연도 합집합): 마스터 %d행 → (브랜드명,법인명) %d조합, 모호 %d조합",
             len(m), len(grp), int((grp["n_candidates"] > 1).sum()))
    return grp


def build_crosswalk_name_only(crosswalk_src: pd.DataFrame) -> pd.DataFrame:
    """R2-② 폴백: 브랜드명 단독 크로스워크 (법인명 불일치 허용)."""
    if crosswalk_src.empty:
        return crosswalk_src
    g = (crosswalk_src.groupby("norm_brand")
         .agg(brand_mnno_nb=("brand_mnno", lambda s: sorted(set(s))[0]),
              hq_mnno_nb=("hq_mnno", lambda s: sorted(set(s))[0]),
              brno_nb=("brno", lambda s: sorted(set(s))[0]),
              n_candidates_nb=("brand_mnno", lambda s: s.nunique()))
         .reset_index())
    return g


def build_master(cfg: dict) -> pd.DataFrame:
    """실적 스냅샷 + 관리번호 크로스워크 → brand_id 부여된 행 단위 DataFrame.

    반환 컬럼: [_yr, brand_name, company_name, industry_major, industry_mid,
               norm_brand, norm_corp, brand_mnno, hq_mnno, brno, id_source, brand_id, ...원천]
    """
    rows = load_snapshots(cfg, "brand_frcs_stats")
    if not rows:
        raise FileNotFoundError("brand_frcs_stats 스냅샷 없음 — 먼저 `python run_pipeline.py --step collect`")
    df = pd.DataFrame(rows)
    n0 = len(df)

    df["brand_name"] = df["brandNm"].astype(str).str.strip()
    df["company_name"] = df["corpNm"].astype(str).str.strip()
    df["industry_major"] = df["indutyLclasNm"].astype(str).str.strip()
    df["industry_mid"] = df["indutyMlsfcNm"].astype(str).str.strip()
    df["norm_brand"] = df["brand_name"].map(normalize_name)
    df["norm_corp"] = df["company_name"].map(normalize_name)

    # R5. 정규화 불가 행 제거
    bad = df["norm_brand"] == ""
    if bad.any():
        log.warning("R5 브랜드명 정규화 불가 %d행 제거", int(bad.sum()))
    df = df[~bad].copy()

    # --- 1홉: 명칭 → 관리번호 -------------------------------------------------
    cw = build_crosswalk(cfg)
    if not cw.empty:
        df = df.merge(cw, on=["norm_brand", "norm_corp"], how="left")
        exact_hit = df["brand_mnno"].notna()
        # R2-② 브랜드명 단독 폴백
        cw_nb = build_crosswalk_name_only(cw)
        df = df.merge(cw_nb, on="norm_brand", how="left")
        need = df["brand_mnno"].isna() & df["brand_mnno_nb"].notna()
        for tgt, src in (("brand_mnno", "brand_mnno_nb"), ("hq_mnno", "hq_mnno_nb"),
                         ("brno", "brno_nb"), ("n_candidates", "n_candidates_nb")):
            df.loc[need, tgt] = df.loc[need, src]
        df = df.drop(columns=["brand_mnno_nb", "hq_mnno_nb", "brno_nb", "n_candidates_nb"])
        name_only_hit = need
        # R4. 모호(후보 2개+)는 관리번호 미부여
        ambiguous = df["brand_mnno"].notna() & (df["n_candidates"].fillna(1) > 1)
        df.loc[ambiguous, ["brand_mnno", "hq_mnno", "brno"]] = pd.NA
        log.info("1홉 정합: 완전일치 %.1f%% / 명칭단독 폴백 %.1f%% / 모호 무효화 %.1f%% / 최종 관리번호 확보 %.1f%%",
                 100 * exact_hit.mean(), 100 * name_only_hit.mean(),
                 100 * ambiguous.mean(), 100 * df["brand_mnno"].notna().mean())
    else:
        for c in ("brand_mnno", "hq_mnno", "brno", "n_candidates"):
            df[c] = pd.NA

    # --- 2홉: brand_id 확정 (관리번호 우선, 명칭 보조) ------------------------
    # R3 보완: brandMnno 단독은 연도 내 유일하지 않다(동일 ID를 서로 다른 본부가 보유).
    # 그러나 본부관리번호를 무조건 키에 붙이면, 본부관리번호가 연도 간 바뀌는 브랜드가
    # 패널에서 둘로 쪼개져 시계열 연속성이 끊긴다(실측: 4,136개 브랜드에서 발생).
    # → **실제 충돌(같은 brandMnno + 같은 연도 + 다른 법인)만** 본부번호로 분리한다.
    has_id = df["brand_mnno"].notna()
    df["id_source"] = pd.Series("NAME", index=df.index, dtype=object)
    df.loc[has_id, "id_source"] = "MNNO"

    idrows = df[has_id]
    if len(idrows):
        # 충돌 판정 키 = 본부관리번호(hq_mnno) (리뷰 지적 반영: norm_corp 기준은 같은 본부의
        # 법인명 표기 변형(예: 농업회사법인 표기 차이·한/영 표기)까지 별도 엔티티로 오분리한다.
        # hq_mnno는 공정위가 부여한 본부 식별자라 표기 변형에 불변)
        per_year_hqs = idrows.groupby(["brand_mnno", "_yr"])["hq_mnno"].nunique()
        collide_ids = set(per_year_hqs[per_year_hqs > 1]
                          .reset_index()["brand_mnno"].unique())
    else:
        collide_ids = set()
    log.info("R3: 연도 내 본부(hq_mnno) 충돌 brandMnno %d개 → 본부관리번호로 분리 "
             "(그 외는 brandMnno 단독 키로 연도 연속성 유지)", len(collide_ids))

    mnno_plain = df["brand_mnno"].astype(str)
    mnno_split = mnno_plain + "|" + df["hq_mnno"].astype(str)
    mnno_key = mnno_split.where(df["brand_mnno"].isin(collide_ids), mnno_plain)
    name_key = "NAME:" + df["norm_brand"] + "@" + df["industry_major"]
    df["brand_id"] = mnno_key.where(has_id, name_key)

    # 동명이브랜드 분리 진단 (명칭 기반 ID에만 해당 — 관리번호는 이미 본부까지 구분)
    name_rows = df[~has_id]
    if len(name_rows):
        collide = (name_rows.groupby(["brand_id", "_yr"])["norm_corp"].nunique() > 1).sum()
        log.info("명칭 기반 ID(%d행): 같은 연도 복수 법인 충돌 %d건 — 본부 구분 불가 한계로 기록",
                 len(name_rows), int(collide))

    # 본부(관리번호) 변경 이력 로깅
    hq_changes = df[has_id].groupby("brand_mnno")["hq_mnno"].nunique()
    log.info("정합 요약: 원천 %d행 → 유효 %d행 / 고유 브랜드 %d개 "
             "(관리번호 기반 %d, 명칭 기반 %d) / 본부관리번호 변경 브랜드 %d개",
             n0, len(df), df["brand_id"].nunique(),
             df.loc[has_id, "brand_id"].nunique(), df.loc[~has_id, "brand_id"].nunique(),
             int((hq_changes > 1).sum()))

    # 정합 리포트 저장 (제출 근거)
    rep = pd.DataFrame([{
        "원천_행수": n0,
        "유효_행수": len(df),
        "정규화불가_제거": int(bad.sum()),
        "관리번호_확보율%": round(100 * has_id.mean(), 2),
        "고유브랜드_전체": df["brand_id"].nunique(),
        "고유브랜드_관리번호기반": int(df.loc[has_id, "brand_id"].nunique()),
        "고유브랜드_명칭기반": int(df.loc[~has_id, "brand_id"].nunique()),
        "모호_관리번호_무효화": int((df["n_candidates"].fillna(1) > 1).sum()),
    }])
    with contextlib.suppress(OSError):
        rep.to_csv(cfg["paths"]["outputs"] / "entity_resolution_report.csv",
                   index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    cfg = load_config()
    m = build_master(cfg)
    print(m[["brand_id", "id_source", "brand_name", "company_name",
             "industry_major", "_yr"]].head(20).to_string())
