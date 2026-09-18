"""f_* 피처명 → 사람이 읽는 한국어 표시명.

화면·CSV·심사메모가 모두 같은 이름을 쓰도록 **한 곳에만** 둔다.
(원시 피처명 `f_ind_contract_end_pct` 가 그대로 화면에 나오면 아무도 못 읽는다.)
Streamlit 에 의존하지 않는 순수 모듈이라 파이프라인(score.py)에서도 쓸 수 있다.
"""
from __future__ import annotations

_FEATURE_KR_EXACT = {
    "f_lvl_n_stores": "[수준] 가맹점 수",
    "f_lvl_avg_sales": "[수준] 평균매출(log)",
    "f_lvl_avg_sales_log1p": "[수준] 평균매출(log)",
    "f_lvl_direct_ratio": "[수준] 직영점 비중",
    "f_lvl_brand_age": "[수준] 브랜드 연차",
    "f_chg_store_growth_rate": "[변화] 점포 증가율",
    "f_chg_sales_growth": "[변화] 매출 증가율",
    "f_chg_contract_end_rate": "[변화] 계약종료율",
    "f_chg_new_open_rate": "[변화] 신규개점률",
    "f_chg_name_change_rate": "[변화] 명의변경률",
}

_PREFIX_KR = [
    ("f_lvl_", "[수준] "),
    ("f_chg_", "[변화] "),
    ("f_trd_", "[추세] "),
    ("f_ind_", "[업종상대] "),
    ("f_struct_", "[구조] "),
]

# 긴 토큰 우선 (부분문자열 우선순위)
_TOKEN_KR = [
    # 긴 토큰 먼저 (부분문자열 우선순위) — 지역분산·면적당매출 피처 추가
    ("sales_per_area_growth", "면적당매출 증가율"),
    ("stores_per_region", "지역당 평균 점포수"),
    ("top_region_share", "최다 지역 비중"),
    ("real_sales_growth", "실질 매출 증가율"),
    ("store_growth_rate", "점포 증가율"),
    ("contract_end_rate", "계약종료율"),
    ("sales_per_area", "면적당매출"),
    ("direct_growth", "직영점 증감율"),
    ("region_hhi", "지역 집중도(HHI)"),
    ("n_regions", "진출 지역 수"),
    ("store_growth", "점포 증가율"),
    ("sales_growth", "매출 증가율"),
    ("contract_end", "계약종료"),
    ("direct_ratio", "직영점 비중"),
    ("emp_cnt", "본부 직원수"),
    ("biz_age", "가맹사업 업력"),
    ("brand_age", "브랜드 연차"),
    ("name_change", "명의변경"),
    ("new_open", "신규개점"),
    ("open_close_gap", "개점-종료 격차"),
    ("n_stores", "가맹점 수"),
    ("n_direct", "직영점 수"),
    ("avg_sales", "평균매출"),
    ("n_new", "신규개점"),
    ("major", "업종 대분류"),
    ("gap", "격차"),
]

_QUAL_KR = [
    ("mean", "평균"),
    ("slope", "기울기"),
    ("std", "변동성"),
    ("rank_pct", "업종 내 백분위"),
    ("rank", "업종 내 백분위"),
    ("pct", "백분위"),
    ("diff", "변화"),
    ("chg", "전년비 변화"),
    ("2y", "최근 2년"),
    ("3y", "최근 3년"),
    ("log1p", ""),
    ("log", ""),
]


def feature_korean_name(feat: str) -> str:
    """f_* 피처명을 한국어 표시명으로 변환. 미지의 이름도 안전하게 처리."""
    if feat in _FEATURE_KR_EXACT:
        return _FEATURE_KR_EXACT[feat]
    name = str(feat)
    prefix_kr = ""
    for pre, kr in _PREFIX_KR:
        if name.startswith(pre):
            prefix_kr, name = kr, name[len(pre):]
            break
    base_kr, rest = None, name
    for token, tkr in _TOKEN_KR:
        if token in name:
            base_kr = tkr
            rest = name.replace(token, "", 1)
            break
    if base_kr is None:
        return prefix_kr + name
    quals = []
    for token, tkr in _QUAL_KR:
        if token and token in rest and tkr:
            quals.append(tkr)
            rest = rest.replace(token, "", 1)
    label = base_kr + ((" " + "·".join(quals)) if quals else "")
    return prefix_kr + label
