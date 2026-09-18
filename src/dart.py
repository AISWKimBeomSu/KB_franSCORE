"""M1.5 가맹본부 재무 (DART 전자공시) — 브랜드가 아니라 **본부**를 본다.

왜 이 층이 필요한가:
    지금까지 FranSCORE 가 보는 것은 전부 '브랜드의 겉모습'이다 — 가맹점 수, 평균매출,
    계약종료율. 그런데 브랜드가 무너지는 실제 경로는 대개 **본부가 먼저 무너지는 것**이다.
    본부가 자본잠식에 빠지거나 영업적자가 누적되면 물류·마케팅·신규출점 지원이 끊기고,
    그 결과가 가맹점 지표에 나타나기까지는 1~2년이 걸린다. 즉 본부 재무는 브랜드 지표보다
    **앞서는 신호**다. 공정위 공시에는 이 정보가 없다(정보공개서 원문에는 있으나 원문 API는
    별도 인증키가 필요 — src/ifrmp.py 참조).

무엇을 쓰는가:
    금융감독원 전자공시 OPEN DART (https://opendart.fss.or.kr). 인증키는 무료.
    ⚠️ 흔한 오해: "DART 는 상장사만"이 아니다. **외부감사 대상 법인**은 비상장이어도
    감사보고서를 DART 에 제출한다. 실측 결과 메가엠지씨커피(앤하우스)·컴포즈커피·이디야·
    BHC·배스킨라빈스·더벤티·굽네치킨·본죽 등 주요 비상장 본부가 모두 4년치 감사보고서를
    내고 있었다. 상장사만 세면 커버리지가 가맹점 가중 4.7% 지만, 감사보고서까지 포함하면
    **39.7%** 로 올라간다 (자격 브랜드 기준 점포 가중; 브랜드 수 기준은 12.2%).
    이 값은 주장이 아니라 매 실행마다 재측정돼 outputs/dart_match_report.json 에 남는다.

⛔ 엔티티 정합의 함정 (이 모듈의 핵심 설계):
    법인명 문자열 매칭만 쓰면 **동명이인 법인에 남의 재무제표가 붙는다.** 실제로 상위 18개
    브랜드를 검증했을 때 (주)한국일오삼(처갓집양념치킨)이 같은 이름의 다른 법인에 매칭됐다
    (FTC 사업자번호 2148720528 vs DART 5348703724). 재무제표가 통째로 틀리는 사고다.
    따라서 이 모듈은 이름 매칭을 **후보 생성에만** 쓰고, 공정위 원천이 가진
    brno(사업자등록번호)·crno(법인등록번호)를 DART company.json 의 bizr_no·jurir_no 와
    **대조해 일치한 것만 '확정'으로 채택**한다. 불일치는 조용히 버리지 않고 기록한다.
    (상위 18개 실측: 확정 17 / 불일치 1 — 검증이 없었으면 1건이 그대로 들어갔다.)

⛔ 시점 안전:
    회계연도 t 의 감사보고서는 t+1 년 3~4월에 공시된다. 따라서 패널 연도 t 의 피처에는
    **회계연도 ≤ t-1** 만 쓴다(features.py 에서 강제, 누출 테스트가 검사한다).
    보수적으로 한 해 더 물린 것이라 누출은 구조적으로 불가능하다.

산출:
    data/raw/dart/                  API 원본 스냅샷 (재현성 — 재실행 시 캐시로 재사용)
    data/processed/hq_financials.parquet   corp_code × fiscal_year 재무
    outputs/dart_match_report.json  매칭·확정·커버리지 진단 (허수아비 방지용 근거)

실행: `python -m src.dart`  또는  `python run_pipeline.py --step dart`
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.common import get_logger, load_config

log = get_logger("dart")

BASE = "https://opendart.fss.or.kr/api"
KEY_ENV = "DART_API_KEY"        # config.yaml dart.api_key_env 와 같은 값

# 정기보고서 코드 — 사업보고서만 재무제표 API(fnlttSinglAcntAll)로 조회된다.
REPRT_ANNUAL = "11011"

# 외감법인 감사보고서를 우선 사용한다. '연결'은 종속회사를 합친 수치라 가맹본부 단독의
# 재무건전성과 다를 수 있어, 별도(감사보고서)를 먼저 쓰고 없을 때만 연결을 쓴다.
_AUDIT_PREF = ("감사보고서", "연결감사보고서")
# 보고서명 앞에 붙는 정정 태그 — '[기재정정]감사보고서(2023.12)' 처럼 쓰인다
_TAG_PREFIX = re.compile(r"^\s*(\[[^\]]{1,20}\]\s*)+")

_PACE_SEC = 0.25          # DART 일일 20,000건 한도 — 예의상 간격
_TIMEOUT = 60


# ---------------------------------------------------------------------------
# 법인명 정규화
# ---------------------------------------------------------------------------
# ⚠️ 순서가 중요하다. 괄호를 먼저 지우면 "(주)앤하우스" → "주앤하우스" 가 되어
#    법인격 토큰이 이름의 일부로 남는다(초기 구현의 실제 버그 — 매칭률 8.6%p 손실).
#    반드시 법인격 토큰을 **먼저** 제거하고 그다음 구분기호를 지운다.
_LEGAL = re.compile(r"주식회사|㈜|\(주\)|유한회사|\(유\)|유한책임회사|합자회사|합명회사|"
                    r"농업회사법인|재단법인|사단법인|의료법인|학교법인")
_STRIP = re.compile(r"[\s()（）\[\]{}·・,.\-_/&'\"]+")


def norm_corp(s: str) -> str:
    """법인명 비교용 정규화: NFKC → 소문자 → 법인격 제거 → 구분기호 제거."""
    t = unicodedata.normalize("NFKC", str(s or "")).lower()
    return _STRIP.sub("", _LEGAL.sub("", t)).strip()


def _digits(s) -> str:
    return re.sub(r"\D", "", str(s or ""))


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def get_key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{KEY_ENV} 미설정 — https://opendart.fss.or.kr 에서 무료 인증키를 발급받아 "
            f"환경변수 {KEY_ENV} 에 설정하세요. (키를 코드·설정파일에 적지 말 것)")
    return key


def _mask(msg: str, key: str) -> str:
    return msg.replace(key, "***KEY***") if key else msg


# DART 응답 status 코드. HTTP 200 이어도 본문 status 가 오류일 수 있다 —
# 이를 검사하지 않으면 오류 응답이 '성공'으로 캐시에 굳어 영구적으로 잘못된 결과가 된다.
_DART_OK = "000"
_DART_EMPTY = "013"                    # 조회된 데이터 없음 — 정상적인 '없음'
_DART_RETRY = {"020", "021", "101", "800", "900", "901"}   # 한도초과·일시오류·점검
_HTTP_NO_RETRY = {400, 401, 403, 404, 405}                 # 재시도해도 같은 결과


class DartError(RuntimeError):
    """DART API 레벨 오류 — 캐시에 성공으로 굳히면 안 되는 응답."""


def _get(path: str, params: dict, *, key: str, binary: bool = False,
         retries: int = 4) -> requests.Response:
    """DART API 호출 (재시도·키 마스킹·본문 status 검사)."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{BASE}/{path}", params={"crtfc_key": key, **params},
                             timeout=_TIMEOUT)
            if r.status_code in _HTTP_NO_RETRY:
                # 키 오류·잘못된 경로에 20초를 낭비하지 않는다
                raise DartError(f"HTTP {r.status_code} (재시도 무의미)")
            r.raise_for_status()
            if not binary and r.headers.get("content-type", "").startswith("application/json"):
                st = str((r.json() or {}).get("status", _DART_OK))
                if st in _DART_RETRY:
                    raise RuntimeError(f"DART status={st} (일시 오류)")
                if st not in (_DART_OK, _DART_EMPTY):
                    raise DartError(f"DART status={st}")
            return r
        except DartError:
            raise
        except Exception as e:
            last = e
            wait = 2.0 * attempt
            log.warning("DART %s 실패 (%d/%d): %s → %.1fs 후 재시도",
                        path, attempt, retries, _mask(str(e), key)[:160], wait)
            time.sleep(wait)
    raise RuntimeError(f"DART {path} 호출 실패: {_mask(str(last), key)}")


# ---------------------------------------------------------------------------
# 1) 기업 인덱스
# ---------------------------------------------------------------------------
def corp_index(cfg: dict, *, refresh: bool = False) -> pd.DataFrame:
    """DART 전체 기업 코드표(corp_code·corp_name·stock_code)를 받아 스냅샷 보관."""
    snap = Path(cfg["paths"]["raw"]) / "dart" / "corp_index.parquet"
    snap.parent.mkdir(parents=True, exist_ok=True)
    if snap.exists() and not refresh:
        return pd.read_parquet(snap)

    key = get_key()
    r = _get("corpCode.xml", {}, key=key, binary=True)
    if r.content[:2] != b"PK":
        raise RuntimeError(f"corpCode.xml 이 ZIP 이 아님: {r.text[:200]}")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode("utf-8")

    rows = []
    for m in re.finditer(r"<list>(.*?)</list>", raw, re.S):
        b = m.group(1)

        def fld(tag: str, blk: str = b) -> str:
            mm = re.search(rf"<{tag}>(.*?)</{tag}>", blk, re.S)
            return (mm.group(1) or "").strip() if mm else ""

        rows.append({"corp_code": fld("corp_code"), "corp_name": fld("corp_name"),
                     "stock_code": fld("stock_code"), "modify_date": fld("modify_date")})
    df = pd.DataFrame(rows)
    df["key"] = df["corp_name"].map(norm_corp)
    df.to_parquet(snap, index=False)
    log.info("DART 기업 인덱스 %s개 (상장 %s) → %s",
             f"{len(df):,}", f"{(df['stock_code'] != '').sum():,}", snap.name)
    return df


# ---------------------------------------------------------------------------
# 2) 후보 매칭 → 3) 사업자·법인등록번호 대조로 '확정'
# ---------------------------------------------------------------------------
def _ftc_registry(cfg: dict) -> pd.DataFrame:
    """공정위 브랜드 마스터에서 법인별 brno·crno 를 모은다 (정합 검증의 기준값)."""
    rows = []
    for p in sorted(Path(cfg["paths"]["raw"]).glob("brand_master_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("%s 읽기 실패: %s", p.name, e)
            continue
        items = d if isinstance(d, list) else (d.get("items") or d.get("data") or [])
        rows += [x for x in items if isinstance(x, dict) and x.get("brno")]
    if not rows:
        raise FileNotFoundError(
            "brand_master_*.json 에 brno 가 없습니다 — 먼저 `--step collect` 를 실행하세요. "
            "사업자등록번호 없이는 동명 법인 오탐을 막을 수 없어 이 단계를 진행하지 않습니다.")
    df = pd.DataFrame(rows)
    df["key"] = df["corpNm"].map(norm_corp)
    df["brno_d"] = df["brno"].map(_digits)
    df["crno_d"] = df.get("crno", pd.Series(dtype=str)).map(_digits)
    # 같은 법인이 여러 해·여러 브랜드로 반복 등장 — 법인 단위로 축약한다.
    # ⛔ 그런데 **정규화 법인명이 같은 서로 다른 실제 법인**이 있다("우리개발" 등).
    #    거기서 임의로 하나를 골라 대조 기준으로 쓰면, DART 쪽 검증을 통과하더라도
    #    그 브랜드가 실제로 속한 법인이 아닌 재무가 붙는다 — 이 모듈이 막겠다고 한
    #    바로 그 사고가 다른 경로로 되살아난다(적대적 리뷰 지적).
    #    사업자번호가 하나로 모이지 않는 key 는 **판정 불가로 통째 제외**한다.
    df = df.sort_values("jngBizCrtraYr" if "jngBizCrtraYr" in df.columns else "key")
    n_brno = df.groupby("key")["brno_d"].nunique()
    ambiguous = set(n_brno[n_brno > 1].index)
    if ambiguous:
        log.warning("정규화 법인명이 같은데 사업자번호가 다른 key %d개 제외 — "
                    "어느 법인의 재무인지 확정할 수 없다 (예: %s)",
                    len(ambiguous), list(ambiguous)[:3])
    out = df[~df["key"].isin(ambiguous)].drop_duplicates("key", keep="last")
    return out[["key", "corpNm", "brno_d", "crno_d"]]


def confirm_matches(cfg: dict, *, limit: int | None = None) -> pd.DataFrame:
    """법인명으로 후보를 만들고, 사업자/법인등록번호 대조로 **확정된 것만** 남긴다.

    반환 컬럼: key·corp_name·corp_code·stock_code·brno_d·crno_d·bizr_no·jurir_no·verdict
    verdict ∈ {확정, 불일치, 조회불가}. 확정만 재무 수집 대상이 된다.
    """
    key = get_key()
    idx = corp_index(cfg)
    ftc = _ftc_registry(cfg)

    # 동명 법인이 DART 안에 여러 개면 어느 쪽인지 이름만으로는 결정할 수 없다.
    # 후보를 모두 남기고 사업자번호 대조로 가른다 (조용히 하나를 고르지 않는다).
    cand = ftc.merge(idx[["key", "corp_code", "corp_name", "stock_code"]], on="key", how="inner")
    n_amb = int(cand["key"].duplicated(keep=False).sum())
    log.info("법인명 후보 매칭: FTC 법인 %s개 중 %s건 후보 (동명 다중후보 %s건 포함)",
             f"{len(ftc):,}", f"{len(cand):,}", f"{n_amb:,}")
    if limit:
        cand = cand.head(limit)

    cache_dir = Path(cfg["paths"]["raw"]) / "dart" / "company"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, r in enumerate(cand.itertuples(index=False), 1):
        cp = cache_dir / f"{r.corp_code}.json"
        if cp.exists():
            info = json.loads(cp.read_text(encoding="utf-8"))
        else:
            try:
                info = _get("company.json", {"corp_code": r.corp_code}, key=key).json()
                # 성공 응답만 캐시한다 — 일시적 실패를 굳히면 그 법인이 영구히 '조회불가'가
                # 되어 재무가 붙지 않는다(캐시가 오류를 데이터로 승격시키는 사고).
                cp.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                log.warning("company.json %s 실패(캐시하지 않음): %s", r.corp_code, str(e)[:120])
                info = {"status": "ERR"}
            time.sleep(_PACE_SEC)
        bz, jr = _digits(info.get("bizr_no")), _digits(info.get("jurir_no"))
        if info.get("status") != "000":
            verdict = "조회불가"
        elif (bz and bz == r.brno_d) or (jr and jr == r.crno_d):
            verdict = "확정"
        else:
            verdict = "불일치"
        out.append({"key": r.key, "corp_name_ftc": r.corpNm, "corp_name_dart": r.corp_name,
                    "corp_code": r.corp_code, "stock_code": r.stock_code,
                    "brno_d": r.brno_d, "crno_d": r.crno_d, "bizr_no": bz, "jurir_no": jr,
                    "verdict": verdict})
        if i % 200 == 0:
            log.info("  대조 진행 %s/%s", f"{i:,}", f"{len(cand):,}")

    res = pd.DataFrame(out)
    vc = res["verdict"].value_counts().to_dict()
    log.info("사업자번호 대조 결과: %s", vc)
    log.info("⚠️ '불일치' %s건은 이름만 같은 **다른 법인**이다 — 채택하지 않는다 "
             "(채택했다면 그 브랜드에 남의 재무제표가 붙는다).", vc.get("불일치", 0))
    # 확정이 한 key 에 둘 이상이면 정합 실패로 간주하고 모두 버린다(조용한 오답 방지)
    ok = res[res["verdict"] == "확정"]
    dup = ok["key"].duplicated(keep=False)
    if dup.any():
        log.warning("한 법인에 확정 후보가 복수인 %s건은 판정 불가로 제외", int(dup.sum()))
        ok = ok[~dup]
    return res.assign(adopted=res.index.isin(ok.index))


# ---------------------------------------------------------------------------
# 4) 재무 수치 추출
# ---------------------------------------------------------------------------
# 재무제표 원문은 라벨에 공백이 섞여 있다 (실측: 이디야 감사보고서는 "자 산 총 계").
# 글자 사이 공백을 허용해야 하며, 그러지 않으면 부채비율 같은 핵심 피처가 조용히 결측된다.
_FIN_FIELDS: dict[str, tuple[tuple[str, int], ...]] = {
    # 필드: ((라벨, 부호), ...)  — 부호 -1 은 '손실' 표기(양수로 적히므로 뒤집는다)
    "assets": (("자산총계", 1),),
    "liabilities": (("부채총계", 1),),
    "equity": (("자본총계", 1),),
    # 자본잠식 판정에 쓴다 (자본총계 < 자본금). 텍스트에서 '자본잠식'을 찾는 방식은
    # 주석의 **피투자회사** 자본잠식까지 잡아 오탐이 난다(앤하우스 실측) — 숫자로 판정한다.
    "capital_stock": (("자본금", 1),),
    "revenue": (("매출액", 1), ("영업수익", 1)),
    "operating_income": (("영업이익", 1), ("영업손실", -1)),
    "net_income": (("당기순이익", 1), ("당기순손실", -1)),
}
# (주석12)·(주 14,17 ) 같은 참조는 숫자를 포함해 금액 추출을 방해한다 — 먼저 제거.
_NOTE_REF = re.compile(r"\(\s*주\s*석?\s*[\d,\s]*\)")
# 금액 토큰. 세 가지를 모두 받아야 한다:
#   ① 천단위 쉼표 표기 "139,112,054,292"  ② nil 표기 "-"  ③ "0"
# ⚠️ 쉼표 없는 4자리 토큰(\d{4,})은 **받지 않는다.** 연도·기간 표기(2018, 2024)가 그대로
#    금액으로 채택돼 revenue=2017.0 같은 행이 실제로 만들어졌다(적대적 리뷰 실측).
#    DART 감사보고서 금액은 예외 없이 쉼표를 쓰므로 잃는 것이 없다.
# ⚠️ nil('-')과 0 을 매칭해야 하는 이유: 당기 칸이 '-' 인데 이를 건너뛰면 **전기 열 금액**을
#    당기 값으로 잘못 채택한다(한 해 밀린 재무가 조용히 들어간다).
_AMOUNT = re.compile(r"\(\s*[\d,]*\d\s*\)|-?\d{1,3}(?:,\d{3})+|(?<![\d,.])[-0](?![\d,.\d])")
# 손익 계정을 가둘 구간의 시작점 (제목은 '손 익 계 산 서' 처럼 자간이 벌어져 있다)
_PL_HEAD = re.compile(r"(포\s*괄)?\s*손\s*익\s*계\s*산\s*서")
_PL_FIELDS = frozenset({"revenue", "operating_income", "net_income"})
# 제목만으로는 부족하다 — '손익계산서'는 **목차**("손익계산서----- 10")와 감사의견 문장에도
# 등장하고, 거기서 시작하면 주석의 서술문("매출액 23백만원, 매입액 638백만원과 기타비용
# 1,419백만원…")에서 숫자를 주워 매출액이 1,419원이 된다(실측 오염 사례의 실제 원인).
# 실제 재무제표 표는 제목 직후에 반드시 단위 표기가 따라온다 → 그것으로 진짜 표를 가린다.
_UNIT_HINT = re.compile(r"\(\s*단\s*위")
_UNIT_LOOKAHEAD = 400


def _pl_scope(flat: str) -> str:
    """손익계산서 **표**가 시작되는 위치부터의 구간을 돌려준다(못 찾으면 전체)."""
    for m in _PL_HEAD.finditer(flat):
        if _UNIT_HINT.search(flat[m.end():m.end() + _UNIT_LOOKAHEAD]):
            return flat[m.start():]
    return flat


def _to_num(tok: str) -> tuple[float, bool] | None:
    """금액 토큰 → (값, 이미_음수표기됨).

    회계 표기의 괄호와 마이너스는 모두 음수다. 단독 '-' 는 nil(해당 없음)이라 0 으로 읽는다.
    두 번째 원소를 함께 돌려주는 이유: 라벨이 '영업손실' 이면 부호를 뒤집어야 하는데,
    금액이 이미 괄호/마이너스로 음수라면 **두 번 뒤집혀 적자가 흑자가 된다**(적대적 리뷰 지적).
    호출부가 그 판단을 할 수 있도록 표기 여부를 넘긴다.
    """
    t = tok.strip()
    if t == "-":
        return 0.0, False
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").strip().replace(",", "")
    if t.startswith("-"):
        neg, t = True, t[1:]
    if not t.isdigit():
        return None
    v = float(t)
    return (-v if neg else v), neg


def _flatten(doc_xml: str) -> str:
    t = re.sub(r"<[^>]+>", " ", doc_xml)
    t = _NOTE_REF.sub(" ", t)
    return re.sub(r"\s+", " ", t)


def parse_financials(flat: str) -> dict[str, float | None]:
    """평문화된 재무제표에서 6개 계정을 뽑는다 (당기 = 첫 번째 금액).

    라벨 글자 사이 공백을 허용하고, 라벨 뒤 40자 이내의 **첫 금액**을 당기 값으로 본다.
    (재무제표는 '계정명 당기 전기' 순으로 배열된다.)
    """
    # ⚠️ 손익 계정은 **손익계산서 구간 안에서만** 찾는다.
    # 문서 전체를 훑으면 같은 라벨이 요약표·주석·목차에도 있어 **단위가 다른 표**의 값이
    # 잡힌다. 실측 피해: revenue 에 회계연도 문자열 "2017" 이 그대로 들어간 행, 자산
    # 3,670억인데 매출 221만원으로 기록된 행 등 — 1,988건 중 64건이 |영업이익률|>100%.
    # 재무상태표 계정(자산·부채·자본·자본금)은 라벨이 충분히 특이해 전체 스캔이 안전하다.
    pl_scope = _pl_scope(flat)
    res: dict[str, float | None] = {}
    for field, variants in _FIN_FIELDS.items():
        scope = pl_scope if field in _PL_FIELDS else flat
        val = None
        for label, sign in variants:
            pat = re.compile(r"\s*".join(re.escape(c) for c in label) + r"[^\d(\-]{0,40}")
            for m in pat.finditer(scope):
                am = _AMOUNT.search(scope[m.end():m.end() + 60])
                if am and (got := _to_num(am.group(0))) is not None:
                    v, already_negative = got
                    # '영업손실'·'당기순손실' 라벨은 금액을 양수로 적는 관행을 전제로 부호를
                    # 뒤집는다. 그런데 괄호·마이너스로 **이미 음수로 적힌** 경우 두 번
                    # 뒤집혀 적자가 흑자가 된다 — 이미 음수면 뒤집지 않는다.
                    val = v if (sign < 0 and already_negative) else v * sign
                    break
            if val is not None:
                break
        res[field] = val
    return res


# 감사의견 판정 — **제목**으로 가른다.
# 본문 문자열로 찾으면 오탐이 난다: 이디야 보고서의 "적정의견이 표명되었습니다"는
# '기타사항' 단락에서 **전기 타감사인의** 의견을 언급한 문장이지 당기 의견이 아니다.
# 변형의견은 전용 제목(한정의견근거·부적정의견·의견거절)을 반드시 달게 되어 있으므로
# 변형을 먼저 확인하고, 없을 때만 표준 적정 결어를 확인한다.
_OPINION_RULES = (
    (re.compile(r"의견\s*거절|의견을\s*표명하지\s*않습니다"), "의견거절"),
    (re.compile(r"부적정의견"), "부적정"),
    (re.compile(r"한정의견"), "한정"),
    (re.compile(r"공정하게\s*표시하고\s*있습니다"), "적정"),
)
# '계속기업'은 **모든** 감사보고서의 경영진·감사인 책임 문단에 상투적으로 등장한다
# (실측: 정상 기업 3곳 모두 등장). 회계감사기준 570 이 요구하는 별도 제목인
# "계속기업 관련 중요한 불확실성" 이 있을 때만 실제 신호로 본다 — 정상 3곳 모두 0건.
_GC_HEADING = re.compile(r"계속기업\s*관련\s*중요한\s*불확실성")


def parse_qualitative(flat: str) -> dict:
    op = None
    for pat, label in _OPINION_RULES:
        if pat.search(flat):
            op = label
            break
    return {"audit_opinion": op,
            "going_concern_flag": int(bool(_GC_HEADING.search(flat)))}


def _corp_filings(corp_code: str, years: tuple[int, ...], *, key: str,
                  cache: Path) -> list[dict]:
    """법인의 공시 목록을 **한 번만** 받아 캐시한다.

    연도마다 list.json 을 부르면 법인당 5회가 되어 DART 일일 한도(2만건)를 빠르게 먹는다.
    회계연도 t 의 감사보고서는 t+1 년에 접수되므로 전 구간을 한 번에 조회한다.
    """
    # ⚠️ 캐시 키에 조회 구간을 넣는다. 넣지 않으면 나중에 회계연도 범위를 넓혔을 때
    #    옛 구간으로 받은 목록이 그대로 재사용돼 새 연도가 **조용히 비어** 버린다.
    bgn, end = f"{min(years) + 1}0101", f"{max(years) + 1}1231"
    cp = cache / f"{corp_code}_filings_{min(years)}_{max(years)}.json"
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))
    try:
        js = _get("list.json", {"corp_code": corp_code, "bgn_de": bgn, "end_de": end,
                                "page_count": 100}, key=key).json()
    except Exception as e:
        # ⛔ 실패를 빈 목록으로 **캐시하지 않는다.** 캐시하면 그 법인은 재실행해도 영구히
        #    '재무 없음'이 되어, 일시적 네트워크 오류가 데이터셋에 영구 반영된다.
        log.warning("list.json %s 실패 — 캐시하지 않고 이번 실행에서만 건너뜀: %s",
                    corp_code, str(e)[:120])
        return []
    items = js.get("list") or []
    keep = [{"report_nm": x.get("report_nm", ""), "rcept_no": x.get("rcept_no"),
             "rcept_dt": x.get("rcept_dt")} for x in items
            if "감사보고서" in x.get("report_nm", "")]
    cp.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    time.sleep(_PACE_SEC)
    return keep


def _fetch_audit(corp_code: str, year: int, filings: list[dict], *,
                 key: str, cache: Path) -> dict | None:
    """해당 회계연도의 감사보고서 원문을 받아 재무·감사의견을 추출."""
    cp = cache / f"{corp_code}_{year}.json"
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8")) or None

    # 보고서명에 '(2024.12)' 처럼 결산기가 붙는다 — 연도가 맞는 것만 후보로 둔다.
    # ⚠️ '[기재정정]감사보고서' 를 startswith 로 걸러내면 **정정으로 폐기된 원본 수치**를
    #    항상 쓰게 된다(적대적 리뷰 지적). 정정본이 있으면 그쪽이 유효한 공시이므로,
    #    접두 태그를 떼고 종류를 판정한 뒤 **접수일이 가장 늦은 것**을 채택한다.
    pick = None
    for want in _AUDIT_PREF:
        cands = [it for it in filings
                 if _TAG_PREFIX.sub("", it["report_nm"]).lstrip().startswith(want)
                 and str(year) in it["report_nm"]]
        if cands:
            it = max(cands, key=lambda x: str(x.get("rcept_dt") or ""))
            kind = want if len(cands) == 1 else f"{want}(정정 반영)"
            pick = (it, kind)
            break
    if not pick:
        cp.write_text("null", encoding="utf-8")
        return None
    it, kind = pick

    try:
        d = _get("document.xml", {"rcept_no": it["rcept_no"]}, key=key, binary=True)
    except Exception as e:
        log.warning("document.xml %s 실패: %s", it["rcept_no"], str(e)[:100])
        return None
    if d.content[:2] != b"PK":
        return None
    z = zipfile.ZipFile(io.BytesIO(d.content))
    body = z.read(z.namelist()[0])
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            xml = body.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        xml = body.decode("utf-8", "replace")

    flat = _flatten(xml)
    rec = {"corp_code": corp_code, "fiscal_year": year, "source": kind,
           "rcept_no": it["rcept_no"], "rcept_dt": it["rcept_dt"],
           **parse_financials(flat), **parse_qualitative(flat)}
    cp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    time.sleep(_PACE_SEC)
    return rec


def _fetch_annual_report(corp_code: str, year: int, *, key: str, cache: Path) -> dict | None:
    """사업보고서 제출 법인은 구조화 재무제표 API 를 쓴다 (파싱 오차 없음)."""
    cp = cache / f"{corp_code}_{year}_afs.json"
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8")) or None
    try:
        js = _get("fnlttSinglAcntAll.json",
                  {"corp_code": corp_code, "bsns_year": str(year),
                   "reprt_code": REPRT_ANNUAL, "fs_div": "OFS"}, key=key).json()
    except Exception as e:
        log.warning("fnlttSinglAcntAll %s/%s 실패: %s", corp_code, year, str(e)[:100])
        return None
    time.sleep(_PACE_SEC)
    if js.get("status") != "000":
        cp.write_text("null", encoding="utf-8")
        return None
    acc = {}
    for x in js.get("list", []):
        nm = re.sub(r"\s+", "", str(x.get("account_nm", "")))
        got = _to_num(str(x.get("thstrm_amount", "")))
        if nm and got is not None and nm not in acc:
            acc[nm] = got[0]

    # 계정명은 제출사마다 다르다 — '영업이익' / '영업이익(손실)' / '영업손실' 이 모두 쓰인다.
    # 하나만 보면 조용히 결측이 되므로(실측: 팜스토리 4개 연도 손익 전부 NaN) 변형을 모두 본다.
    def pick(*names: str) -> float | None:
        for n in names:
            if (v := acc.get(n)) is not None:
                return v
        return None

    rec = {"corp_code": corp_code, "fiscal_year": year, "source": "사업보고서",
           "rcept_no": None, "rcept_dt": None,
           "assets": pick("자산총계"), "liabilities": pick("부채총계"),
           "equity": pick("자본총계", "자본총계(지배기업소유주지분)"),
           "capital_stock": pick("자본금"),
           "revenue": pick("매출액", "영업수익", "수익(매출액)", "매출"),
           "operating_income": pick("영업이익", "영업이익(손실)", "영업손실"),
           "net_income": pick("당기순이익", "당기순이익(손실)", "당기순손실",
                              "연결당기순이익"),
           # 사업보고서 경로는 구조화 API 라 감사의견·계속기업 문단 **텍스트 자체가 없다**.
           # 없는 것을 '적정'·'해당없음(0)'으로 채우면 거짓 안심이 되므로 둘 다 결측으로 둔다.
           # (going_concern_flag 를 0 으로 채웠던 초기 구현은 "확인하지 못했다"를
           #  "확인했고 문제없다"로 바꿔 말하는 셈이었다 — 적대적 리뷰 지적.)
           "audit_opinion": None, "going_concern_flag": None}
    if rec["assets"] is None and rec["revenue"] is None:
        # 별도(OFS)가 비어 있는 제출사가 있다 — 이 경우는 감사보고서 경로로 넘긴다.
        cp.write_text("null", encoding="utf-8")
        return None
    cp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


_CORE_FIELDS = ("assets", "liabilities", "equity", "revenue",
                "operating_income", "net_income")


def _completeness(rec: dict | None) -> int:
    """핵심 6개 계정 중 몇 개가 채워졌는지 (원천 선택 기준)."""
    return 0 if not rec else sum(rec.get(f) is not None for f in _CORE_FIELDS)


def collect_financials(cfg: dict, years: tuple[int, ...] | None = None,
                       *, limit: int | None = None) -> pd.DataFrame:
    """확정된 본부에 대해 회계연도별 재무를 수집한다."""
    if years is None:
        years = tuple(int(y) for y in
                      (cfg.get("dart") or {}).get("fiscal_years",
                                                  (2020, 2021, 2022, 2023, 2024)))
    key = get_key()
    matches = confirm_matches(cfg, limit=limit)
    adopted = matches[matches["adopted"]]
    log.info("재무 수집 대상: 확정 법인 %s개 × %d개 회계연도", f"{len(adopted):,}", len(years))

    cache = Path(cfg["paths"]["raw"]) / "dart" / "fin"
    cache.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, r in enumerate(adopted.itertuples(index=False), 1):
        listed = bool(str(r.stock_code or "").strip())
        filings = _corp_filings(r.corp_code, years, key=key, cache=cache)
        for y in years:
            # 같은 회계연도를 두 원천(사업보고서·감사보고서)에서 받을 수 있다.
            # 필드별로 섞으면 한 행의 출처가 불분명해지므로 **더 완전한 쪽을 통째로** 쓴다.
            cands = []
            if listed:
                cands.append(_fetch_annual_report(r.corp_code, y, key=key, cache=cache))
            if not cands or _completeness(cands[0]) < len(_CORE_FIELDS):
                cands.append(_fetch_audit(r.corp_code, y, filings, key=key, cache=cache))
            best = max((c for c in cands if c), key=_completeness, default=None)
            if best:
                recs.append({**best, "key": r.key})
        if i % 50 == 0:
            log.info("  재무 수집 %s/%s 법인 (누적 %s건)",
                     f"{i:,}", f"{len(adopted):,}", f"{len(recs):,}")

    fin = pd.DataFrame(recs)
    proc = Path(cfg["paths"]["processed"])
    out_dir = Path(cfg["paths"]["outputs"])
    quality: dict = {}
    if fin.empty:
        log.error("수집된 재무가 0건입니다 — 키·네트워크를 확인하세요.")
    else:
        fin = fin.drop_duplicates(["corp_code", "fiscal_year"], keep="first")
        fin, quality = _quality_gate(fin)
        fin.to_parquet(proc / "hq_financials.parquet", index=False)
        cov = {c: float(fin[c].notna().mean()) for c in
               ("assets", "liabilities", "equity", "revenue",
                "operating_income", "net_income")}
        log.info("본부 재무 %s건 (법인 %s개) → hq_financials.parquet",
                 f"{len(fin):,}", f"{fin['corp_code'].nunique():,}")
        log.info("계정별 추출률: %s", {k: f"{v:.1%}" for k, v in cov.items()})

    report = _match_report(cfg, matches, fin)
    report["quality"] = quality
    (out_dir / "dart_match_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("커버리지(자격 브랜드, 가맹점 가중): 확정 %.1f%% / 재무확보 %.1f%%",
             100 * report["eligible"]["confirmed_store_weighted"],
             100 * report["eligible"]["with_financials_store_weighted"])
    return fin


_BS_TOL = 0.01      # 자산 = 부채 + 자본 허용 오차 1%


def _quality_gate(fin: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """정규식 파싱은 **조용히 틀릴 수 있다** — 회계 항등식으로 스스로를 검산한다.

    자산총계 = 부채총계 + 자본총계 는 어떤 회계기준에서도 성립한다. 세 값이 모두 있는데
    이 항등식이 깨졌다면 라벨을 잘못 잡은 것이므로, 그 행의 재무상태표 3개 값을 버린다
    (손익 항목은 별개 표라 함께 버리지 않는다). 버린 건수를 반드시 보고한다.
    """
    d = fin.copy()
    have = d[["assets", "liabilities", "equity"]].notna().all(axis=1)
    lhs = d["assets"].where(have)
    rhs = (d["liabilities"] + d["equity"]).where(have)
    bad = have & ((lhs - rhs).abs() > _BS_TOL * lhs.abs().clip(lower=1.0))
    if bad.any():
        log.warning("회계 항등식(자산=부채+자본) 위배 %d건 — 해당 행의 재무상태표 값을 폐기",
                    int(bad.sum()))
        d.loc[bad, ["assets", "liabilities", "equity", "capital_stock"]] = pd.NA

    # 손익도 검산한다 — 재무상태표 항등식만 보면 손익 오염이 그대로 통과한다(실제로 통과했다).
    # 매출원가·판관비는 음수가 될 수 없으므로 **영업이익 ≤ 매출액** 은 어떤 손익계산서에서도
    # 성립하는 하드 제약이다. 깨졌다면 두 값이 서로 다른 표(다른 단위)에서 온 것이므로 버린다.
    pl = d[["revenue", "operating_income"]].notna().all(axis=1)
    pl_bad = (pl & (d["operating_income"] > d["revenue"] * (1 + _BS_TOL))) | d["revenue"].lt(0)
    if pl_bad.any():
        log.warning("손익 정합(영업이익 ≤ 매출액, 매출액 ≥ 0) 위배 %d건 — 해당 행의 손익 폐기",
                    int(pl_bad.sum()))
        d.loc[pl_bad, ["revenue", "operating_income", "net_income"]] = pd.NA

    q = {"rows": len(d),
         "balance_checked": int(have.sum()),
         "balance_failed": int(bad.sum()),
         "balance_pass_rate": float(1 - bad.sum() / have.sum()) if have.any() else None,
         "pl_checked": int(pl.sum()),
         "pl_failed": int(pl_bad.sum()),
         "pl_pass_rate": float(1 - pl_bad.sum() / pl.sum()) if pl.any() else None,
         "audit_opinion_counts": d["audit_opinion"].value_counts(dropna=False)
                                  .rename(index=str).to_dict(),
         "going_concern_flagged": int(d["going_concern_flag"].fillna(0).sum())}
    log.info("품질 검산: 재무상태표 항등식 %s건 중 통과율 %s | 손익 정합 %s건 중 통과율 %s",
             f"{q['balance_checked']:,}",
             "n/a" if q["balance_pass_rate"] is None else f"{q['balance_pass_rate']:.1%}",
             f"{q['pl_checked']:,}",
             "n/a" if q["pl_pass_rate"] is None else f"{q['pl_pass_rate']:.1%}")
    return d, q


def _match_report(cfg: dict, matches: pd.DataFrame, fin: pd.DataFrame) -> dict:
    """커버리지를 브랜드 수·가맹점 가중 두 기준으로 재측정한다 (주장 대신 숫자)."""
    panel = pd.read_parquet(Path(cfg["paths"]["processed"]) / "panel.parquet")
    latest = panel.sort_values("year").drop_duplicates("brand_id", keep="last").copy()
    latest["key"] = latest["company_name"].map(norm_corp)
    ok_keys = set(matches.loc[matches["adopted"], "key"])
    fin_keys = set(fin["key"]) if not fin.empty else set()

    def block(d: pd.DataFrame) -> dict:
        st = d["n_stores"].fillna(0)
        tot = float(st.sum()) or 1.0
        c = d["key"].isin(ok_keys)
        f = d["key"].isin(fin_keys)
        return {"n_brands": len(d),
                "confirmed_brands": int(c.sum()),
                "confirmed_pct": float(c.mean()),
                "confirmed_store_weighted": float(st[c].sum() / tot),
                "with_financials_brands": int(f.sum()),
                "with_financials_pct": float(f.mean()),
                "with_financials_store_weighted": float(st[f].sum() / tot)}

    elig = latest[latest["eligible_t"].fillna(False).astype(bool)]
    return {
        "verdicts": matches["verdict"].value_counts().to_dict(),
        "adopted_corps": int(matches["adopted"].sum()),
        "all_brands": block(latest),
        "eligible": block(elig),
        "note": ("법인명 매칭은 후보 생성에만 쓰고 사업자등록번호·법인등록번호 대조로 확정한 "
                 "것만 채택했다. '불일치'는 이름만 같은 다른 법인이므로 버린다."),
    }


# ---------------------------------------------------------------------------
# 합성 본부재무 — ⚠️ 누출 테스트 전용 (제출 지표에 절대 사용하지 않는다)
# ---------------------------------------------------------------------------
# 왜 여기에 두는가: 이 모듈이 hq_financials 스키마의 소유자다. 스키마와 합성 픽스처를
# 한 파일에 두어야 컬럼을 추가할 때 픽스처를 같이 고치게 되고, 그래야 신규 피처가
# 시점누출 검증에서 조용히 빠지는 사고(과거에 실제로 발생)를 막을 수 있다.
def make_synthetic_hq_financials(cfg: dict, panel: pd.DataFrame) -> pd.DataFrame:
    """합성 패널의 본부(company_name)마다 회계연도별 재무를 생성한다."""
    import numpy as np

    rng = np.random.default_rng(cfg["seed"] + 7)
    names = sorted(panel["company_name"].dropna().unique())
    y0, y1 = int(panel["year"].min()) - 2, int(panel["year"].max())
    rows = []
    for i, nm in enumerate(names):
        # ⚠️ 실데이터 커버리지를 **반드시 모사해야 한다.** 전 본부 × 전 회계연도를 빠짐없이
        #    채우면 f_hq_has_financials ≡ 1, f_hq_data_age ≡ 0 이 되어 상수 피처가 되고,
        #    tests/test_sanity.py 의 커버리지 가드 ③이 실패하면서 그 뒤의 교란-불변 비교가
        #    아예 실행되지 못한다(전 피처가 무검증 상태가 된다 — 적대적 리뷰가 실제로 검출).
        #    실데이터에서도 외감 대상 본부만 재무가 있고 최근 결산 시점도 제각각이다.
        if i % 7 == 0:
            continue                       # 외감 대상 아님 → 미매칭 (has_financials=0)
        last_fy = y1 - (i % 5)             # 최근 결산 노후도 0~4년 (3 초과는 stale 처리)
        assets = float(rng.uniform(5e9, 3e11))
        equity_ratio = float(rng.uniform(-0.1, 0.7))
        margin = float(rng.normal(0.05, 0.10))
        for fy in range(y0, last_fy + 1):
            assets *= float(1 + rng.normal(0.05, 0.12))
            equity_ratio = float(np.clip(equity_ratio + rng.normal(-0.01, 0.06), -0.5, 0.9))
            equity = assets * equity_ratio
            revenue = assets * float(rng.uniform(0.4, 2.5))
            op = revenue * float(margin + rng.normal(0, 0.05))
            rows.append({
                "key": norm_corp(nm), "corp_code": f"SYN{i:05d}", "fiscal_year": fy,
                "source": "합성", "rcept_no": None, "rcept_dt": None,
                "assets": assets, "liabilities": assets - equity, "equity": equity,
                "capital_stock": assets * 0.01,
                "revenue": revenue, "operating_income": op,
                "net_income": op * float(rng.uniform(0.4, 1.1)),
                "audit_opinion": "적정",
                "going_concern_flag": float(equity < 0),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    collect_financials(load_config())
