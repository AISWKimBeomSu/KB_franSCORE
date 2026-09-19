"""평가 범위 보정 — 공정위 통계의 한 해 공백 때문에 대형 브랜드가 빠지는 문제.

실측 (2024년 실적 외식 9,333개 브랜드)
    가맹점 수 상위 60개 중 9개가 평가 대상에서 빠져 있었다 — 파리바게뜨(3,327)·BBQ(2,316)·
    투썸플레이스(1,510)·롯데리아·멕시카나·땅스부대찌개·피나치공·파스쿠찌·고봉민김밥인.
    실무자가 가장 먼저 검색할 브랜드들이다. 원인은 범위 설계가 아니라 자료 결함이었다.
      · 7곳: 가맹점 통계 API(15110241)에 한 해 행이 통째로 없다 → '3년 연속 관측' 조건 탈락.
        같은 해 지역·직영 API(15125490)에는 가맹점 수가 있다(BBQ 2022년: 2,041개).
      · 2곳: 한 해 이름 표기가 달라 브랜드 ID 가 둘로 갈렸다(피나치공, 고봉민김밥인).
    이런 한 해 공백은 해마다 반복된다(2020~2023년 외식 30개 이상 브랜드 26·45·49·44곳).

무엇을 보정하나 — 재학습 없이, 점수 산출 단계에서만
    '3년 연속 관측' 조건의 빈 해가 **다른 공식 기록으로 확인되면** 관측된 해로 본다.
      ① ID 분리    같은 관리번호, 또는 같은 가맹본부 + 이름 별칭("A(B)")이 겹치는 다른 ID 에
                   그 해 행이 있다
      ② 지역 통계  지역·직영 API(15125490) '전체' 행에 그 해 가맹점 수가 0보다 크다
      ③ 등록 이력  그 해 정보공개서 등록 목록(15125467)에 같은 관리번호가 있다
    단, **직전 연도 행이 자기 ID 에 있어야** 점수를 낸다. 모형의 핵심 입력(점포 증감·계약종료율·
    업종 내 위치)은 직전 연도와의 차이로 계산된다. 직전 연도가 비면 그 값들이 결측이 되는데,
    학습 표본에서는 한 번도 결측인 적이 없어(3,635행 중 0행) 모형이 결측을 '변화 없음·최저
    위험'으로 읽는다 — 점수가 낙관적으로 나온다(실측: 평균 0.107 → 직전 연도 보충 시 0.157).
    그런 브랜드는 점수를 내지 않고, 화면이 **정확한 사유**를 보여 준다.

한계 — 화면과 산출물에 표시한다
    보정 브랜드는 과거 백테스트에 없던 경로로 들어왔다(검증된 모집단 밖). 3년 추세 피처는
    결측이다(학습 행의 7.2%가 같은 상태였다).

산출: outputs/coverage_report.csv — 최신 연도 업종 범위 모든 브랜드의 평가 여부와 사유
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.collect import load_snapshots
from src.common import get_logger
from src.entity import normalize_name

log = get_logger("coverage")

BASIS_REGULAR = "정규"
STATUS_SCORED = "평가"
STATUS_BRIDGED = "평가 (공시 공백 보정)"
STATUS_UNSCORED = "평가 대상 아님"

_PAREN = re.compile(r"[\(\[（](.*?)[\)\]）]")


def _aliases(raw) -> set[str]:
    """'피자나라 치킨공주(피나치공)' → {전체, 괄호 밖, 괄호 안} 정규화 이름."""
    raw = str(raw or "")
    out = {normalize_name(raw), normalize_name(_PAREN.sub(" ", raw))}
    out |= {normalize_name(x) for x in _PAREN.findall(raw)}
    return {a for a in out if a and len(a) >= 2}


def _mnno_years(cfg: dict, service: str, keep=None) -> dict[str, set[int]]:
    """스냅샷의 관리번호 → 패널 연도 집합 (패널 연도 = 정보공개서 기준연도 − 1, panel.py 규칙)."""
    out: dict[str, set[int]] = {}
    try:
        rows = load_snapshots(cfg, service, keep=keep)
    except Exception as exc:                     # 스냅샷이 없으면 그 근거만 쓰지 않는다
        log.warning("%s 스냅샷을 읽지 못해 이 근거는 생략: %s", service, exc)
        return out
    for it in rows:
        m = it.get("brandMnno")
        if m in (None, ""):
            continue
        out.setdefault(str(m), set()).add(int(it["_yr"]) - 1)
    return out


_EV_CACHE: dict[tuple, dict] = {}


def evidence(cfg: dict) -> dict[str, dict[str, set[int]]]:
    """빈 해를 확인하는 외부 기록 — {'region': 관리번호→연도, 'registry': 관리번호→연도}.

    지역 스냅샷은 144만 행이라 읽는 데 몇 초 걸린다. 등급 이력은 연도마다 부르므로
    원본 파일이 그대로면 한 번만 읽는다.
    """
    raw = Path(cfg["paths"]["raw"])
    files = sorted(raw.glob("brand_region_direct_*.json*")) + sorted(raw.glob("brand_master_*.json*"))
    key = (str(raw), tuple((f.name, f.stat().st_mtime) for f in files))
    if key in _EV_CACHE:
        return _EV_CACHE[key]
    def _region_keep(it: dict) -> bool:
        try:
            return it.get("areaNm") == "전체" and float(it.get("frcsCnt") or 0) > 0
        except (TypeError, ValueError):
            return False

    out = {"region": _mnno_years(cfg, "brand_region_direct", _region_keep),
           "registry": _mnno_years(cfg, "brand_master")}
    _EV_CACHE[key] = out
    return out


def assess(cfg: dict, panel_full: pd.DataFrame, panel: pd.DataFrame, year: int,
           ev: dict[str, dict[str, set[int]]] | None = None) -> pd.DataFrame:
    """해당 연도 업종 범위 브랜드마다 평가 여부·보정 근거·미평가 사유.

    반환 열: brand_id, brand_name, n_stores, status, basis, gap_years, reason
      status  정규 평가 / 보정 평가 / 평가 대상 아님
      basis   보정 근거(예: '2022년: 지역·직영 통계 2,041개')
      reason  평가 대상이 아닌 이유(화면 문장)
    """
    s = cfg["sample"]
    min_stores, min_years = int(s["min_stores"]), int(s["min_consecutive_years"])
    ev = ev if ev is not None else evidence(cfg)
    region, registry = ev.get("region", {}), ev.get("registry", {})

    rng = panel.sort_values(["brand_id", "year"])
    cmax = rng.groupby("brand_id")["n_stores"].cummax()
    rng = rng.assign(cmax_stores=cmax)          # itertuples 는 밑줄로 시작하는 열 이름을 바꾼다
    cur = rng[rng["year"] == year]
    own_years = rng.groupby("brand_id")["year"].agg(lambda v: {int(x) for x in v})

    pf = panel_full[panel_full["year"] < year][["brand_id", "brand_mnno", "brand_name",
                                                "company_name", "year", "n_stores"]].copy()
    pf["_nc"] = pf["company_name"].map(normalize_name)
    pf["_alias"] = pf["brand_name"].map(_aliases)
    by_mnno = pf.dropna(subset=["brand_mnno"]).groupby("brand_mnno")
    mnno_ids = {str(k): set(g["brand_id"]) for k, g in by_mnno}
    years_of = pf.groupby("brand_id")["year"].agg(lambda v: {int(x) for x in v})
    stores_of = {(b, int(y)): n for b, y, n in pf[["brand_id", "year", "n_stores"]].itertuples(index=False)}

    rows = []
    for r in cur.itertuples(index=False):
        bid, name = str(r.brand_id), str(r.brand_name)
        n = r.n_stores
        base = {"brand_id": bid, "brand_name": name, "n_stores": n, "basis": "", "gap_years": "",
                "reason": ""}
        if bool(r.eligible_t):
            rows.append({**base, "status": STATUS_SCORED, "basis": BASIS_REGULAR})
            continue
        if pd.isna(n) or float(n) <= 0:
            rows.append({**base, "status": STATUS_UNSCORED,
                         "reason": "이번 공시의 가맹점 수가 0이거나 비어 있어 평가하지 않았습니다."})
            continue
        if not (r.cmax_stores >= min_stores):
            rows.append({**base, "status": STATUS_UNSCORED,
                         "reason": f"가맹점이 {min_stores}개에 이른 적이 없어(최대 {int(r.cmax_stores):,}개) "
                                   "성장률·종료율이 잡음에 좌우되므로 평가하지 않습니다."})
            continue
        own = own_years.get(bid, set())
        mnno = str(r.brand_mnno) if pd.notna(r.brand_mnno) else None
        # 연결 ID — 같은 관리번호, 또는 같은 본부 + 별칭 겹침
        links = set(mnno_ids.get(mnno, set())) if mnno else set()
        my_alias = _aliases(name)
        nc = normalize_name(r.company_name) if pd.notna(r.company_name) else ""
        if nc:
            same_corp = pf[(pf["_nc"] == nc) & pf["_alias"].map(lambda a, m=my_alias: bool(a & m))]
            links |= set(same_corp["brand_id"])
        links.discard(bid)
        link_years = set().union(*(years_of.get(b, set()) for b in links)) if links else set()
        link_mnnos = {str(m) for m in pf.loc[pf["brand_id"].isin(links), "brand_mnno"].dropna()}

        need = list(range(year - min_years + 1, year + 1))
        basis, gaps, broken = [], [], None
        for y in need:
            if y in own:
                continue
            if y in link_years:
                other = next(b for b in links if y in years_of.get(b, set()))
                cnt = stores_of.get((other, y))
                basis.append(f"{y}년: 다른 ID로 분리 등록된 같은 브랜드"
                             + (f" {int(cnt):,}개" if pd.notna(cnt) and cnt else ""))
            elif any(y in region.get(m, set()) for m in ({mnno} | link_mnnos) - {None}):
                basis.append(f"{y}년: 지역·직영 통계(15125490)에 가맹점 기록")
            elif any(y in registry.get(m, set()) for m in ({mnno} | link_mnnos) - {None}):
                basis.append(f"{y}년: 정보공개서 등록 기록")
            else:
                broken = y
                break
            gaps.append(str(y))
        prev_missing = (year - 1) not in own
        if broken is None and not prev_missing:
            rows.append({**base, "status": STATUS_BRIDGED, "basis": " · ".join(basis),
                         "gap_years": ",".join(gaps)})
            continue
        if prev_missing and broken is None:
            reason = (f"{year - 1}년 가맹점 통계가 이 브랜드 이름으로 비어 있어, 점포 증감·계약종료율 같은 "
                      "최근 변화 지표를 계산할 수 없습니다. 이 상태로 점수를 내면 위험이 낮게 나오므로 "
                      "평가하지 않습니다.")
            if basis:
                reason += f" (같은 해 다른 기록: {' · '.join(basis)})"
        elif any(y < broken for y in own | link_years):
            reason = (f"{broken}년 가맹점 통계가 비어 있고 다른 공식 기록으로도 확인되지 않아, "
                      f"'{min_years}년 연속 공시' 조건을 채우지 못했습니다.")
        else:
            first = min(own) if own else year
            reason = (f"공시 이력이 {year - first + 1}년으로 짧아 추세를 계산할 수 없습니다 "
                      f"(평가에는 {min_years}년 연속 공시가 필요합니다).")
        rows.append({**base, "status": STATUS_UNSCORED, "reason": reason})
    out = pd.DataFrame(rows, columns=["brand_id", "brand_name", "n_stores", "status", "basis",
                                      "gap_years", "reason"])
    return out


def write_report(cfg: dict, table: pd.DataFrame, year: int) -> Path:
    dest = Path(cfg["paths"]["outputs"]) / "coverage_report.csv"
    t = table.assign(year=int(year)).sort_values(["status", "n_stores"], ascending=[True, False])
    t.to_csv(dest, index=False, encoding="utf-8-sig")
    w = pd.to_numeric(t["n_stores"], errors="coerce").fillna(0)
    for st_ in (STATUS_SCORED, STATUS_BRIDGED, STATUS_UNSCORED):
        m = t["status"] == st_
        log.info("평가 범위 %s: %d개 브랜드 · 가맹점 %s개 (%.1f%%)", st_, int(m.sum()),
                 f"{int(w[m].sum()):,}", 100 * w[m].sum() / max(w.sum(), 1))
    return dest
