"""실제 여신 장부 — 은행이 가진 브랜드별 여신 잔액으로 포트폴리오 화면 전체를 다시 계산한다.

왜 필요한가
    여신 포트폴리오 화면의 기본 장부는 공정위 공시 창업비용에 대출조달비율·은행점유율
    가정을 곱해 **추정**한 60개 브랜드다(src/portfolio.py). 공개 데이터로 만들 수 있는
    최선이지만, 은행 안에서 이 도구를 쓰는 사람은 실제 잔액을 이미 갖고 있다. 그 장부를
    올리면 같은 화면(집중도·한도 점검·대리 예상손실·꼬리손실)이 실측 기준으로 바뀌어야
    한다 — 추정 장부로만 도는 도구는 시연용에 머문다.

이 모듈이 하는 일 (화면과 무관한 계산만 — streamlit 을 import 하지 않는다)
    read_book(name, data)          업로드 파일(CSV·엑셀) → 표준 열 이름의 행 단위 장부
    match_book(book, scores)       장부의 브랜드를 평가 대상 브랜드에 연결 (화면 검색과 같은 규칙)
    apply_fixes(matched, ...)      '확인 필요' 행에서 사용자가 고른 브랜드를 반영
    brand_book(matched, ...)       담보유형별 LGD 를 붙여 브랜드 단위로 합친다
    tail_risk(book, scores, ρ_W, ρ_B)  꼬리손실 분위수·UL 배수·브랜드별 성분 ES
    template_bytes()               업로드 양식 (실제 브랜드명 · 예시 금액)

단위
    내부 금액은 전부 백만원(MKRW) — 파이프라인 portfolio.csv 와 같다. 화면은 억원으로 보여 준다.

개인정보
    장부에는 차주 정보가 섞여 올 수 있다. 이 모듈은 파일을 디스크에 쓰지 않고, 브랜드·금액·
    담보유형·LGD·차주 수 열만 해석한다. 나머지 열은 결과에 싣지 않는다.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.brand_search import ALIASES, normalize, search
from src.correlation import _brand_tail_stats, _quantile_se, _simulate_nested

# ---------------------------------------------------------------------------
# 가정 — 값의 근거를 여기 한 곳에 적는다 (config.yaml 의 loanbook 절로 덮어쓸 수 있다)
# ---------------------------------------------------------------------------

# 담보유형별 손실률(LGD) 기본 가정. **은행 내부 LGD 모형이 없을 때의 출발값**이다.
#   신용 45%   바젤 II 기초 IRB(F-IRB)의 무담보 선순위 감독 LGD 45%와 같은 값이다(바젤 III
#              최종안은 일반 기업 40% — 소상공인 여신에 더 낮출 근거는 없어 45%를 둔다). 추정 장부의
#              대리 예상손실·꼬리손실이 쓰는 중간 시나리오(LGD 45%)와도 같아, 담보 정보가 없는
#              장부는 추정 장부와 **같은 가정**으로 계산된다 — 두 장부의 차이가 가정이 아니라
#              금액 구성에서만 나오게 하려는 것이다.
#   담보 25%   부동산 근저당·임차보증금 질권. 바젤 III F-IRB 는 부동산 담보분의 감독 LGD 를
#              20%로 둔다. 가맹점 여신의 담보는 상당 부분이 **점포 임차보증금**인데, 회수할 때
#              미납 차임·원상복구비가 먼저 공제되고 경매 부동산보다 회수가 불확실하다. 그래서
#              부동산 기준보다 5%p 높여 한 값으로 묶었다(유형을 더 나누면 행마다 판단이 필요해진다).
#   보증서 10%  신용보증기금·지역신용보증재단 보증서. 보증비율이 보통 85% 이상이고 보증기관이
#              정부 출연 기관이라 보증분 손실은 사실상 없다. 비보증분 15% × 신용 LGD 45% ≈ 7%에
#              보증 면책(약정 위반 시 보증 이행 거절)·대위변제 지연 위험을 더해 10%로 둔다.
#   모름 45%   담보유형이 비었거나 읽지 못하면 **없다고 본다**(신용과 같다). 모르는 담보를
#              있다고 치면 손실이 과소평가되고, 그 방향의 오류는 결재 문서에서 드러나지 않는다.
DEFAULT_LGD_BY_COLLATERAL: dict[str, float] = {"신용": 0.45, "담보": 0.25, "보증서": 0.10}
DEFAULT_LGD = 0.45
COLLATERAL_TYPES = ("보증서", "담보", "신용")    # 그 밖은 미기재·기타 → 기본값(모름)

# 금액 단위 → 백만원 환산 계수
UNIT_TO_MKRW: dict[str, float] = {"원": 1e-6, "천원": 1e-3, "만원": 1e-2, "백만원": 1.0, "억원": 100.0}
UNITS = tuple(UNIT_TO_MKRW)

MATCH_OK = "확정"
MATCH_CHECK = "확인 필요"
MATCH_NONE = "미매칭"
MATCH_DROP = "제외"
MATCH_ORDER = {MATCH_CHECK: 0, MATCH_NONE: 1, MATCH_OK: 2, MATCH_DROP: 3}

MAX_ROWS = 20_000          # 한 번에 읽는 장부 행 수 상한 (화면이 멈추지 않을 규모)
TEMPLATE_NAME = "FranSCORE_여신장부_양식.xlsx"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# 점수표에서 장부로 붙여 오는 열 — 화면(집중도·한도 점검)이 이 열들로 돈다
SCORE_COLS = ("brand_name", "industry_major", "industry_mid", "n_stores",
              "deterioration_1y", "risk_grade", "grade")

# ---------------------------------------------------------------------------
# 열 이름 해석
# ---------------------------------------------------------------------------

_HDR_STRIP = re.compile(r"[\s()（）\[\]{}·・,.\-_/:]+")


def _hnorm(h: object) -> str:
    """열 이름 비교용 정규화 — 괄호·공백·밑줄을 지운다('여신잔액(억원)' → '여신잔액억원')."""
    return _HDR_STRIP.sub("", unicodedata.normalize("NFKC", str(h if h is not None else "")).lower())


_ID_HEADERS = {"brandid", "브랜드id", "브랜드코드", "브랜드번호", "brandcode"}
_ID_VALUE = re.compile(r"^(BRD_\d+|NAME:.+)$")
_NAME_HEADERS = ("브랜드명", "브랜드", "브랜드이름", "가맹브랜드", "가맹브랜드명", "영업표지",
                 "영업표지명", "brandname", "brand", "프랜차이즈", "프랜차이즈명", "상호", "상호명")
_NAME_CONTAINS = ("브랜드", "영업표지", "brand")
_NOT_NAME = ("id", "코드", "번호", "code")

# 잔액 열은 등급을 두고 고른다 — '여신잔액'과 '신청금액'이 함께 있으면 잔액이 이긴다.
_EXPO_TIERS = (("잔액", "익스포저", "exposure", "balance", "ead"),
               ("여신", "대출", "loan"),
               ("금액", "amount", "amt"))
# ⚠️ '한도'는 잔액이 아니다. 미사용 한도까지 익스포저로 잡으면 쏠림이 부풀려진다.
_NOT_EXPO = ("건수", "차주", "lgd", "손실", "회수", "비중", "share", "비율", "율", "률", "%",
             "한도", "금리", "이자", "번호", "코드", "유형", "구분", "종류")

_LGD_HEADERS = ("lgd", "손실률", "손실율")
_RECOVERY_HEADERS = ("회수율", "회수률", "recovery")
_BORROWER_HEADERS = ("차주수", "차주건수", "borrower", "nborrowers", "차주", "건수", "계좌수", "좌수")
_COLLATERAL_HEADERS = ("담보유형", "담보구분", "담보종류", "담보", "보증구분", "보증종류", "보증",
                       "collateral", "security", "여신종류", "대출종류", "상품구분")

_TOTAL_WORDS = {"합계", "총계", "소계", "계", "합", "total", "subtotal", "sum", "grandtotal", "전체"}


def _is_name_header(h: str) -> bool:
    if h in _NAME_HEADERS:
        return True
    return any(k in h for k in _NAME_CONTAINS) and not any(k in h for k in _NOT_NAME)


def _expo_tier(h: str) -> int | None:
    if not h or any(k in h for k in _NOT_EXPO):
        return None
    for i, keys in enumerate(_EXPO_TIERS):
        if any(k in h for k in keys):
            return i
    return None


def _unit_from_header(header: str) -> str | None:
    """열 이름에 적힌 단위. 없으면 None — 그때만 값 크기로 추정한다."""
    t = _hnorm(header)
    if "억" in t or "ekw" in t:
        return "억원"
    if "백만" in t or "mkrw" in t or "million" in t:
        return "백만원"
    if "천원" in t:
        return "천원"
    if "만원" in t:
        return "만원"
    if t.endswith("원") or "krw" in t or t.endswith("won"):
        return "원"
    return None


# ---------------------------------------------------------------------------
# 값 해석
# ---------------------------------------------------------------------------

def _filled(x: object) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and np.isnan(x):
        return False
    return str(x).strip() not in ("", "nan", "None", "NaN")


def _num(s: pd.Series) -> pd.Series:
    """숫자 해석 — 천 단위 쉼표·통화기호·'원'·'%'·회계식 괄호 음수를 받아 준다."""
    t = s.map(lambda x: "" if not _filled(x) else str(x)).str.strip()
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    t = t.str.replace(r"[,\s₩$]", "", regex=True)
    t = t.str.replace(r"(원|krw|%)$", "", regex=True, case=False)
    return pd.to_numeric(t.replace({"": np.nan, "-": np.nan}), errors="coerce")


def _numeric_share(s: pd.Series) -> float:
    filled = s.map(_filled)
    if not filled.any():
        return 0.0
    return float(_num(s[filled]).notna().mean())


def classify_collateral(text: object) -> str:
    """담보유형 원문 → 보증서 / 담보 / 신용 / 미기재 / 기타.

    ⚠️ 검사 **순서가 규칙이다.** 은행 용어가 서로를 품고 있다.
       '보증서담보대출'은 담보가 아니라 보증서이고, '임차보증금'은 보증서가 아니라 담보다.
       '무담보'·'무보증'은 글자에 담보·보증이 있어도 신용이다.
    """
    t = _hnorm(text)
    if not t or t in ("nan", "none", "-"):
        return "미기재"
    if any(k in t for k in ("무담보", "무보증", "순수신용", "unsecured")):
        return "신용"
    if any(k in t for k in ("임차보증금", "보증금", "전세")):
        return "담보"
    if any(k in t for k in ("보증서", "신용보증", "신보", "기보", "보증재단", "보증기금", "보증",
                            "guarantee")):
        return "보증서"
    if any(k in t for k in ("부동산", "근저당", "저당", "질권", "예적금", "담보", "secured",
                            "mortgage", "collateral")):
        return "담보"
    if any(k in t for k in ("신용", "credit")):
        return "신용"
    return "기타"


def lgd_assumptions(cfg: dict | None = None) -> tuple[dict[str, float], float]:
    """(담보유형 → LGD, 미기재·기타 LGD). config.yaml 의 loanbook 절이 코드 기본값을 덮는다."""
    lb = (cfg or {}).get("loanbook") or {}
    m = dict(DEFAULT_LGD_BY_COLLATERAL)
    for k, v in (lb.get("lgd_by_collateral") or {}).items():
        m[str(k)] = float(v)
    return m, float(lb.get("lgd_default", DEFAULT_LGD))


# ---------------------------------------------------------------------------
# 1) 읽기
# ---------------------------------------------------------------------------

class BookReadError(ValueError):
    """장부를 해석하지 못함. 화면이 열을 직접 고르게 하도록 발견한 열 이름을 함께 싣는다."""

    def __init__(self, msg: str, columns: list[str] | None = None) -> None:
        super().__init__(msg)
        self.columns = list(columns or [])


def _decode(data: bytes) -> str:
    # 은행 내부 시스템의 CSV 내보내기는 아직 CP949 가 흔하다. UTF-8(BOM)을 먼저 본다.
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise BookReadError("CSV 인코딩을 읽지 못했습니다 — UTF-8 또는 CP949 로 저장해 주십시오.")


def _sniff(text: str) -> str:
    """구분자 추정 — 쉼표·탭·세미콜론·세로줄 중 줄마다 칸 수가 가장 많이 나오는 것."""
    sample = text.splitlines()[:40]
    best, best_score = ",", 1.0
    for d in (",", "\t", ";", "|"):
        counts = [len(r) for r in csv.reader(sample, delimiter=d) if r]
        score = float(np.median(counts)) if counts else 0.0
        if score > best_score:
            best, best_score = d, score
    return best


def _load_grids(name: str, data: bytes) -> list[tuple[str, list[list]]]:
    """파일 → [(시트 이름, 행 목록)]. 머리글 위치를 모르므로 머리글 없이 통째로 읽는다."""
    low = str(name or "").lower()
    if low.endswith((".xlsx", ".xlsm", ".xls")):
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=object)
        except ImportError as exc:                 # .xls 는 xlrd 가 있어야 읽힌다
            raise BookReadError("구형 엑셀(.xls)은 읽을 수 없습니다 — .xlsx 또는 CSV 로 저장해 "
                                "올려 주십시오.") from exc
        except Exception as exc:                   # 손상 파일·암호 걸린 파일
            raise BookReadError(f"엑셀 파일을 읽지 못했습니다 — {exc}") from exc
        return [(str(s), g.astype(object).where(g.notna(), None).values.tolist())
                for s, g in sheets.items()]
    text = _decode(bytes(data))
    return [("", list(csv.reader(io.StringIO(text), delimiter=_sniff(text))))]


def _find_header(grid: list[list]) -> int | None:
    """제목 줄('여신 잔액 현황 (8월말)')이 위에 붙은 내보내기가 흔하다 — 앞 15줄에서 머리글을 찾는다."""
    for r, row in enumerate(grid[:15]):
        hs = [_hnorm(x) for x in row if _filled(x)]
        brand = any(h in _ID_HEADERS or _is_name_header(h) for h in hs)
        expo = any(_expo_tier(h) is not None for h in hs)
        if brand and expo:
            return r
    return None


def _unique_headers(row: list) -> list[str]:
    out: list[str] = []
    for j, x in enumerate(row):
        h = str(x).strip() if _filled(x) else f"열{j + 1}"
        base, k = h, 2
        while h in out:
            h, k = f"{base} ({k})", k + 1
        out.append(h)
    return out


def _pick(cols: list[str], want, df: pd.DataFrame, numeric: bool | None) -> str | None:
    """열 이름 후보 목록(앞이 우선)에 맞는 첫 열. numeric 이면 숫자 열, False 면 글자 열만."""
    for key in want:
        for c in cols:
            h = _hnorm(c)
            if key not in h:
                continue
            share = _numeric_share(df[c])
            if numeric is True and share < 0.5:
                continue
            if numeric is False and share >= 0.5:
                continue
            return c
    return None


def _unit_from_values(v: np.ndarray, borrowers: np.ndarray | None,
                      loan_level: bool) -> tuple[str, str]:
    """열 이름에 단위가 없을 때 값 크기로 추정. (단위, 사람이 읽을 근거)

    원·백만원·억원만 가린다. 천원은 값 크기로 원과 구별되지 않아(10억 원 = 1,000,000천원)
    열 이름에 적혀 있을 때만 받는다. 추정은 추정일 뿐이라 화면이 근거를 밝히고 바꿀 수 있게 한다.
    """
    ok = np.isfinite(v) & (v > 0)
    if not ok.any():
        return "백만원", "금액이 없어 백만원으로 둠"
    med = float(np.median(v[ok]))
    if med >= 1e6:
        return "원", (f"금액 중앙값 {med:,.0f} — 백만원·억원이면 한 줄에 1조 원을 넘으므로 "
                     "원 단위로 봄")
    if borrowers is not None:
        # 차주 1명당 금액이 소상공인 대출로 말이 되는 쪽(3백만~30억 원)이 하나뿐이면 그것을 고른다
        both = ok & np.isfinite(borrowers) & (borrowers > 0)
        if both.any():
            per = float(np.median(v[both] / borrowers[both]))
            fits = [u for u in ("백만원", "억원") if 3.0 <= per * UNIT_TO_MKRW[u] <= 3000.0]
            if len(fits) == 1:
                u = fits[0]
                return u, (f"차주 1명당 {per * UNIT_TO_MKRW[u]:,.0f}백만원이 되는 단위 — "
                           f"소상공인 대출 규모(3백만–30억 원)에 맞아 {u}으로 봄")
    if med >= 1000:
        return "백만원", (f"금액 중앙값 {med:,.0f} — 억원이면 한 줄에 1,000억 원이 넘으므로 "
                         "백만원으로 봄")
    if loan_level:
        return "백만원", "같은 브랜드가 여러 줄에 반복돼 건별 대출로 보고 백만원으로 봄"
    v = v[ok]
    frac = float(np.mean(np.abs(v - np.round(v)) > 1e-9))
    if frac >= 0.2:
        return "억원", "소수점 금액이 많아(억 단위 표기에 흔함) 억원으로 봄"
    if med < 100:
        return "억원", (f"금액 중앙값 {med:,.1f} — 백만원이면 브랜드당 1억 원도 안 되므로 "
                       "억원으로 봄")
    return "백만원", f"금액 중앙값 {med:,.0f} — 억원이면 브랜드당 100억 원을 넘으므로 백만원으로 봄"


def read_book(name: str, data: bytes, *, brand_col: str | None = None,
              exposure_col: str | None = None, unit: str | None = None,
              max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """업로드한 여신 장부(CSV·엑셀) → 행 단위 표준 장부.

    반환 열
        src_row          파일에서의 행 번호(머리글 포함 1부터 — 엑셀에서 찾아가기용)
        brand_id_input   brand_id 열 값 (없으면 None)
        brand_input      브랜드명 열 값 (없으면 None)
        exposure_raw     파일에 적힌 금액 그대로
        exposure_mkrw    백만원 환산
        collateral_raw / collateral   담보유형 원문 / 분류(보증서·담보·신용·미기재·기타)
        lgd_input        행에 적힌 LGD (0–1, 없으면 NaN) — 담보유형 가정보다 우선한다
        n_borrowers      차주 수 (없으면 NaN)
    해석 과정(고른 열·단위와 그 근거·뺀 행)은 `df.attrs["loanbook"]` 에 담는다 — 화면이
    "이렇게 읽었습니다"를 밝히고, 틀렸으면 열·단위를 직접 고르게 하기 위해서다.
    """
    grids = _load_grids(name, data)
    chosen = next(((s, g, r) for s, g in grids if (r := _find_header(g)) is not None), None)
    if chosen is None:
        nonempty = [(s, g) for s, g in grids if any(any(_filled(x) for x in row) for row in g)]
        if not nonempty:
            raise BookReadError("파일이 비어 있습니다.")
        s, g = nonempty[0]
        first = next(i for i, row in enumerate(g) if any(_filled(x) for x in row))
        chosen = (s, g, first)
    sheet, grid, hr = chosen
    header = _unique_headers(grid[hr])
    width = len(header)
    body = [((list(r) + [None] * width)[:width], hr + 2 + i) for i, r in enumerate(grid[hr + 1:])]
    body = [(r, n) for r, n in body if any(_filled(x) for x in r)]
    truncated = len(body) > max_rows
    body = body[:max_rows]
    df = pd.DataFrame([r for r, _ in body], columns=header, dtype=object)
    src_rows = [n for _, n in body]
    cols = list(df.columns)

    # ── 브랜드 열 ────────────────────────────────────────────────────────────
    id_col = next((c for c in cols if _hnorm(c) in _ID_HEADERS), None)
    if id_col is None:
        for c in cols:
            vals = df[c][df[c].map(_filled)].astype(str).str.strip()
            if len(vals) and vals.str.match(_ID_VALUE).mean() >= 0.8:
                id_col = c
                break
    if brand_col and brand_col in cols:
        name_col = brand_col if brand_col != id_col else None
    else:
        # 머리글 목록의 **앞쪽이 우선**이다 — '상호'(가맹점 상호일 수 있다)와 '브랜드명'이 함께
        # 있으면 브랜드명을 쓴다. 이름이 목록에 없으면 '브랜드'를 품은 열을 찾는다.
        text_cols = [c for c in cols if c != id_col and _numeric_share(df[c]) < 0.5]
        name_col = next((c for key in _NAME_HEADERS for c in text_cols if _hnorm(c) == key), None)
        if name_col is None:
            name_col = next((c for c in text_cols if _is_name_header(_hnorm(c))), None)
    if id_col is None and name_col is None:
        raise BookReadError("브랜드 열을 찾지 못했습니다 — '브랜드명' 또는 'brand_id' 열이 필요합니다.",
                            cols)

    # ── 금액 열 ──────────────────────────────────────────────────────────────
    if exposure_col and exposure_col in cols:
        expo_col = exposure_col
    else:
        cand = [(t, i, c) for i, c in enumerate(cols)
                if c not in (id_col, name_col) and (t := _expo_tier(_hnorm(c))) is not None
                and _numeric_share(df[c]) >= 0.5]
        expo_col = min(cand)[2] if cand else None
    if expo_col is None:
        raise BookReadError("여신 잔액 열을 찾지 못했습니다 — '여신잔액(억원)'처럼 금액 열의 이름을 "
                            "적어 주십시오.", cols)

    taken = {id_col, name_col, expo_col}
    rest = [c for c in cols if c not in taken]
    coll_col = _pick(rest, _COLLATERAL_HEADERS, df, numeric=False)
    rest = [c for c in rest if c != coll_col]
    lgd_col = _pick(rest, _LGD_HEADERS, df, numeric=True)
    rec_col = None if lgd_col else _pick(rest, _RECOVERY_HEADERS, df, numeric=True)
    rest = [c for c in rest if c not in (lgd_col, rec_col)]
    bor_col = _pick(rest, _BORROWER_HEADERS, df, numeric=True)

    # ── 행 정리 ──────────────────────────────────────────────────────────────
    ids = (df[id_col].map(lambda x: str(x).strip() if _filled(x) else None)
           if id_col else pd.Series(None, index=df.index, dtype=object))
    names = (df[name_col].map(lambda x: str(x).strip() if _filled(x) else None)
             if name_col else pd.Series(None, index=df.index, dtype=object))
    amount = _num(df[expo_col])
    key = names.where(names.notna(), ids)
    # 은행 내보내기는 맨 아래에 합계 줄을 붙이는 일이 많다. 그대로 두면 총여신이 두 배가 된다.
    is_total = key.map(lambda k: k is not None and (
        _hnorm(k) in _TOTAL_WORDS or _hnorm(k).startswith(("합계", "총계", "소계"))))
    blank = key.isna()
    bad_amt = ~blank & ~is_total & ~(amount > 0)
    keep = ~blank & ~is_total & ~bad_amt

    borrowers = _num(df[bor_col]) if bor_col else None
    loan_level = bool(key[keep].duplicated(keep=False).mean() >= 0.3) if keep.any() else False
    header_unit = _unit_from_header(expo_col)
    if unit in UNIT_TO_MKRW:
        unit_used, unit_src, unit_why = unit, "사용자 지정", "화면에서 고른 단위"
    elif header_unit:
        unit_used, unit_src, unit_why = header_unit, "열 이름", f"'{expo_col}'에 적힌 단위"
    else:
        unit_used, unit_why = _unit_from_values(
            amount[keep].to_numpy(dtype=float),
            borrowers[keep].to_numpy(dtype=float) if borrowers is not None else None, loan_level)
        unit_src = "값 크기 추정"

    lgd_in = pd.Series(np.nan, index=df.index)
    n_lgd_bad = 0
    src = lgd_col or rec_col
    if src:
        v = _num(df[src])
        if (v[keep] > 1.0).any():                  # 45 → 45% 로 적은 장부
            v = v / 100.0
        if rec_col:
            v = 1.0 - v                            # 회수율 → 손실률
        bad = v.notna() & ((v < 0) | (v > 1))
        n_lgd_bad = int((bad & keep).sum())
        lgd_in = v.where(~bad)
    nb = pd.Series(np.nan, index=df.index)
    if borrowers is not None:
        nb = borrowers.where(borrowers > 0).round()

    coll_raw = (df[coll_col].map(lambda x: str(x).strip() if _filled(x) else "")
                if coll_col else pd.Series("", index=df.index))
    out = pd.DataFrame({
        "src_row": src_rows,
        "brand_id_input": ids,
        "brand_input": names,
        "exposure_raw": amount,
        "exposure_mkrw": amount * UNIT_TO_MKRW[unit_used],
        "collateral_raw": coll_raw,
        "collateral": coll_raw.map(classify_collateral),
        "lgd_input": lgd_in.astype(float),
        "n_borrowers": nb.astype(float),
    })[keep.to_numpy()].reset_index(drop=True)

    # 양식을 그대로 올리면 '예시' 표기가 남는다 — 가상 금액이 실제 장부로 읽히지 않게 표시한다
    example = any(
        df.loc[keep, c].map(lambda x: _filled(x) and "예시" in str(x)).mean() >= 0.5
        for c in rest if keep.any())
    notes = []
    if truncated:
        notes.append(f"행이 {max_rows:,}개를 넘어 앞 {max_rows:,}행만 읽었습니다.")
    if int(is_total.sum()):
        notes.append(f"합계·소계 줄 {int(is_total.sum())}개는 뺐습니다.")
    if int(bad_amt.sum()):
        notes.append(f"금액이 비었거나 0 이하인 {int(bad_amt.sum())}행은 뺐습니다.")
    if n_lgd_bad:
        notes.append(f"LGD 값이 0–100% 밖인 {n_lgd_bad}행은 담보유형 가정으로 계산합니다.")
    if loan_level and not bor_col:
        notes.append("같은 브랜드가 여러 줄에 나옵니다. 줄 하나가 차주 한 명이면 '차주 수' 열에 1을 "
                     "넣어 주십시오 — 없으면 가맹점 수만큼 차주가 있다고 가정합니다.")
    out.attrs["loanbook"] = {
        "file_name": str(name), "sheet": sheet, "header_row": hr + 1, "columns": cols,
        "brand_id_col": id_col, "brand_col": name_col, "exposure_col": expo_col,
        "collateral_col": coll_col, "lgd_col": lgd_col or rec_col,
        "lgd_is_recovery": bool(rec_col), "borrower_col": bor_col,
        "unit": unit_used, "unit_source": unit_src, "unit_reason": unit_why,
        "header_unit": header_unit, "loan_level": loan_level, "is_example": bool(example),
        "n_rows_read": len(df), "n_rows": len(out),
        "dropped": {"합계·소계 줄": int(is_total.sum()), "금액 없음·0 이하": int(bad_amt.sum()),
                    "브랜드 빈칸": int(blank.sum())},
        "notes": notes,
    }
    return out


# ---------------------------------------------------------------------------
# 2) 브랜드 매칭 — 화면 검색(src/brand_search.py)과 **같은 규칙**
# ---------------------------------------------------------------------------

def _match_key(bid: object, name: object) -> str:
    if _filled(name):
        return "nm:" + normalize(str(name))
    return "id:" + str(bid).strip() if _filled(bid) else ""


def _ids_by_name(scores: pd.DataFrame, names: list[str], limit: int = 5) -> list[str]:
    hit = scores[scores["brand_name"].astype(str).isin(names)]
    size = pd.to_numeric(hit.get("n_stores"), errors="coerce").fillna(0)
    return hit.assign(_n=size).sort_values("_n", ascending=False)["brand_id"].astype(str).head(limit).tolist()


def _match_one(bid: object, name: object, ids: set[str], scores: pd.DataFrame,
               all_brands: pd.DataFrame | None) -> dict:
    """행 하나의 매칭. 확정은 **정확 일치·통칭 일치·브랜드ID** 뿐이다.

    부분일치가 한 건뿐이어도 확정하지 않는다. '굽네'가 굽네치킨 한 곳에만 걸리는 것은
    오늘의 등록명 목록이 그렇다는 것일 뿐, 장부 작성자가 그 브랜드를 뜻했다는 증거가 아니다.
    잘못 붙은 여신은 조용히 다른 브랜드의 쏠림이 된다 — 그래서 사람이 한 번 본다.
    """
    none = {"match_status": MATCH_NONE, "match_rule": "공시에 없음", "brand_id": None,
            "candidates": []}
    if _filled(bid) and str(bid).strip() in ids:
        b = str(bid).strip()
        return {"match_status": MATCH_OK, "match_rule": "브랜드ID", "brand_id": b, "candidates": [b]}
    if not _filled(name):
        return {**none, "match_rule": "브랜드ID 없음"}
    q_raw = str(name)
    hit, near = search(scores, q_raw, limit=5)
    q = normalize(q_raw)
    if hit.empty:
        rule = "공시에 없음"
        if all_brands is not None and not all_brands.empty:
            h2, _ = search(all_brands, q_raw, limit=3)
            direct = {q} | ({normalize(ALIASES[q])} if q in ALIASES else set())
            if not h2.empty and h2["brand_name"].map(normalize).isin(direct).any():
                rule = "평가 대상 아님"       # 공시에는 있으나 외식업·가맹점 30개 이상이 아니다
        return {**none, "match_rule": rule, "candidates": _ids_by_name(scores, list(near))}
    norm = hit["brand_name"].astype(str).map(normalize)
    direct = {q} | ({normalize(ALIASES[q])} if q in ALIASES else set())
    exact = hit[norm.isin(direct)]
    token = normalize(ALIASES.get(q, ""))
    alias = hit[norm.str.contains(token, regex=False)] if token else hit.head(0)
    cands = hit["brand_id"].astype(str).tolist()
    if len(exact) == 1:
        return {"match_status": MATCH_OK, "match_rule": "정확 일치",
                "brand_id": str(exact.iloc[0]["brand_id"]), "candidates": cands}
    if len(exact) > 1:                     # 같은 이름으로 등록된 브랜드가 둘 이상(국수나무 등)
        ex = exact["brand_id"].astype(str).tolist()
        return {"match_status": MATCH_CHECK, "match_rule": "동명 브랜드",
                "brand_id": ex[0], "candidates": ex + [c for c in cands if c not in ex]}
    if len(alias) == 1:
        return {"match_status": MATCH_OK, "match_rule": "통칭 일치",
                "brand_id": str(alias.iloc[0]["brand_id"]), "candidates": cands}
    return {"match_status": MATCH_CHECK, "match_rule": "유사 일치",
            "brand_id": str(hit.iloc[0]["brand_id"]), "candidates": cands}


def _score_index(scores: pd.DataFrame) -> pd.DataFrame:
    s = scores.copy()
    s["brand_id"] = s["brand_id"].astype(str)
    return s.drop_duplicates("brand_id").set_index("brand_id")


def _attach_scores(df: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bid = out["brand_id"]
    for c in SCORE_COLS:
        out[c] = bid.map(idx[c]) if c in idx.columns else np.nan
    return out


def match_book(book: pd.DataFrame, scores: pd.DataFrame,
               all_brands: pd.DataFrame | None = None) -> pd.DataFrame:
    """행 단위 장부의 브랜드를 평가 대상 브랜드(scores)에 연결한다.

    상태
        확정       브랜드ID 일치, 등록명 정확 일치, 통칭 사전 일치(메가커피 → 메가엠지씨커피)
        확인 필요   같은 이름 브랜드가 둘 이상이거나 부분·유사 일치 — 가장 가까운 후보를 임시로
                   붙이고 `candidates` 에 후보를 남긴다(분석에는 포함, 화면이 강조한다)
        미매칭      평가 대상에서 찾지 못함 — 분석에서 빠지고 화면이 따로 모아 보여 준다.
                   `all_brands`(공시 전체 브랜드명)를 주면 '평가 대상 아님'과 '공시에 없음'을 가른다.
    같은 입력 이름은 한 번만 찾는다(수천 행 장부도 브랜드 수만큼만 검색한다).
    """
    idx = _score_index(scores)
    ids = set(idx.index)
    out = book.copy()
    bids = out["brand_id_input"] if "brand_id_input" in out else pd.Series(None, index=out.index)
    names = out["brand_input"] if "brand_input" in out else pd.Series(None, index=out.index)
    keys = [_match_key(b, n) for b, n in zip(bids, names, strict=True)]
    cache: dict[str, dict] = {}
    rows = []
    for k, b, n in zip(keys, bids, names, strict=True):
        if k not in cache:
            cache[k] = _match_one(b, n, ids, scores, all_brands)
        rows.append(cache[k])
    res = pd.DataFrame(rows, index=out.index,
                       columns=["match_status", "match_rule", "brand_id", "candidates"])
    out = pd.concat([out, res], axis=1)
    out["match_key"] = keys
    out["candidates"] = out["candidates"].map(lambda v: list(v) if isinstance(v, list) else [])
    return _attach_scores(out, idx)


def apply_fixes(matched: pd.DataFrame, scores: pd.DataFrame, fixes: dict | None) -> pd.DataFrame:
    """사용자가 고른 매칭을 반영한다. fixes = {match_key: {"brand_id": str, "reviewed": bool}}.

    brand_id 가 빈 문자열이면 그 이름의 행을 분석에서 뺀다(상태 '제외'). reviewed 가 참이면
    '확정'(근거 '사용자 확인')으로 올린다 — 사람이 본 것과 규칙이 정한 것을 구분해 남긴다.
    """
    if not fixes:
        return matched
    idx = _score_index(scores)
    out = matched.copy()
    for key, fx in fixes.items():
        m = out["match_key"] == key
        if not m.any() or not isinstance(fx, dict):
            continue
        bid = str(fx.get("brand_id") or "")
        if not bid:
            out.loc[m, "match_status"] = MATCH_DROP
            out.loc[m, "match_rule"] = "사용자 제외"
            out.loc[m, "brand_id"] = None
            continue
        if bid not in idx.index:
            continue
        out.loc[m, "brand_id"] = bid
        if fx.get("reviewed"):
            out.loc[m, "match_status"] = MATCH_OK
            out.loc[m, "match_rule"] = "사용자 확인"
        else:
            out.loc[m, "match_status"] = out.loc[m, "match_status"].replace(MATCH_NONE, MATCH_CHECK)
    return _attach_scores(out, idx)


def summarize(matched: pd.DataFrame) -> dict:
    """매칭 결과 요약 — 화면의 상태 칩·기준 장부 띠가 같은 숫자를 쓴다 (금액은 백만원)."""
    st_ = matched["match_status"]
    e = pd.to_numeric(matched["exposure_mkrw"], errors="coerce").fillna(0.0)
    used = st_.isin([MATCH_OK, MATCH_CHECK]) & matched["brand_id"].notna()
    return {
        "n_rows": len(matched),
        "n_ok": int((st_ == MATCH_OK).sum()), "n_check": int((st_ == MATCH_CHECK).sum()),
        "n_none": int((st_ == MATCH_NONE).sum()), "n_drop": int((st_ == MATCH_DROP).sum()),
        "expo_total_mkrw": float(e.sum()), "expo_used_mkrw": float(e[used].sum()),
        "expo_none_mkrw": float(e[st_ == MATCH_NONE].sum()),
        "expo_drop_mkrw": float(e[st_ == MATCH_DROP].sum()),
        "expo_check_mkrw": float(e[st_ == MATCH_CHECK].sum()),
        "n_brands": int(matched.loc[used, "brand_id"].nunique()),
        "n_lgd_override": int(matched["lgd_input"].notna().sum()) if "lgd_input" in matched else 0,
        "has_borrowers": bool(matched["n_borrowers"].notna().any()) if "n_borrowers" in matched else False,
    }


# ---------------------------------------------------------------------------
# 3) 브랜드 단위 장부
# ---------------------------------------------------------------------------

def row_lgd(matched: pd.DataFrame, lgd_by_collateral: dict | None = None,
            lgd_default: float | None = None) -> tuple[pd.Series, pd.Series]:
    """행별 LGD 와 그 출처. 행에 적힌 값 > 담보유형 가정 > 기본값 순서."""
    lmap = {**DEFAULT_LGD_BY_COLLATERAL, **(lgd_by_collateral or {})}
    default = DEFAULT_LGD if lgd_default is None else float(lgd_default)
    by_type = matched["collateral"].map(lmap)
    given = pd.to_numeric(matched.get("lgd_input"), errors="coerce") \
        if "lgd_input" in matched else pd.Series(np.nan, index=matched.index)
    lgd = given.where(given.notna(), by_type).fillna(default).astype(float)
    src = pd.Series(np.where(given.notna(), "행 입력",
                             np.where(by_type.notna(), "담보유형 가정", "기본값")), index=matched.index)
    return lgd, src


def brand_book(matched: pd.DataFrame, lgd_by_collateral: dict | None = None,
               lgd_default: float | None = None) -> pd.DataFrame:
    """행 단위(매칭 끝난) 장부 → 브랜드 단위. 포트폴리오 화면과 꼬리손실 계산의 입력이다.

    · 확정·확인 필요 행만 쓴다(미매칭·제외는 빠진다 — 빠진 금액은 summarize 가 밝힌다).
    · 한 브랜드가 여러 행(담보유형별·지점별)이면 금액은 더하고 LGD 는 **금액 가중 평균**을
      쓴다. 기대손실(Σ 금액×LGD)은 행 단위와 정확히 같고, 꼬리 계산에서는 그 브랜드 차주들이
      같은 담보 구성을 가졌다고 보는 근사다.
    · 차주 수가 없으면 그 브랜드 **가맹점 수**만큼 차주가 있다고 본다(파이프라인 추정 장부와
      같은 가정). 실제 차주가 더 적으면 여신이 덜 쪼개져 꼬리손실이 더 크다 — 화면이 밝힌다.
    """
    use = matched[matched["match_status"].isin([MATCH_OK, MATCH_CHECK])
                  & matched["brand_id"].notna()].copy()
    cols = ["brand_id", *SCORE_COLS, "exposure_mkrw", "lgd", "n_borrowers", "n_borrowers_basis",
            "collateral_mix", "match_status", "n_rows", "brand_input"]
    if use.empty:
        return pd.DataFrame(columns=cols)
    lgd, _ = row_lgd(use, lgd_by_collateral, lgd_default)
    use["_e"] = pd.to_numeric(use["exposure_mkrw"], errors="coerce").fillna(0.0)
    use["_el"] = use["_e"] * lgd
    g = use.groupby("brand_id", sort=False)
    first = g.first()
    out = pd.DataFrame({"brand_id": first.index.astype(str)})
    for c in SCORE_COLS:
        out[c] = first[c].to_numpy() if c in first.columns else np.nan
    out["exposure_mkrw"] = g["_e"].sum().to_numpy()
    out["lgd"] = (g["_el"].sum() / g["_e"].sum().where(lambda x: x > 0)).fillna(
        DEFAULT_LGD if lgd_default is None else float(lgd_default)).to_numpy()
    nb = g["n_borrowers"].sum(min_count=1).to_numpy(dtype=float)
    stores = pd.to_numeric(out["n_stores"], errors="coerce").to_numpy(dtype=float)
    out["n_borrowers_basis"] = np.where(np.isfinite(nb), "장부 차주 수", "가맹점 수 가정")
    out["n_borrowers"] = np.where(np.isfinite(nb), nb, np.where(np.isfinite(stores), stores, 1.0))
    out["n_borrowers"] = np.maximum(np.round(out["n_borrowers"]), 1.0)

    def _mix(sub: pd.DataFrame) -> str:
        tot = float(sub["_e"].sum())
        if tot <= 0:
            return ""
        s = sub.groupby("collateral")["_e"].sum().sort_values(ascending=False, kind="stable")
        return " · ".join(f"{k} {v / tot * 100:.0f}%" for k, v in s.items())

    out["collateral_mix"] = [_mix(sub) for _, sub in g]
    out["match_status"] = g["match_status"].agg(
        lambda s: MATCH_CHECK if (s == MATCH_CHECK).any() else MATCH_OK).to_numpy()
    out["n_rows"] = g.size().to_numpy()
    out["brand_input"] = g["brand_input"].agg(
        lambda s: " · ".join(dict.fromkeys(str(x) for x in s if _filled(x)))).to_numpy()
    out["brand_name"] = out["brand_name"].where(out["brand_name"].notna(), out["brand_id"])
    return out[cols]


# ---------------------------------------------------------------------------
# 4) 꼬리손실 — 파이프라인(src/correlation.py)과 같은 모형·같은 함수
# ---------------------------------------------------------------------------

def load_rho(out_dir: str | Path) -> dict | None:
    """측정된 브랜드 상관 (ρ_W 브랜드 내부 · ρ_B 브랜드 간). 파이프라인 산출물에서 읽는다.

    brand_correlation.json 이 원본이고, 없으면 correlation_impact.json 에 기록된 사용값을 쓴다.
    둘 다 없으면 None — 화면은 꼬리손실을 계산하지 않고 그렇게 말한다.
    """
    out = Path(out_dir)
    try:
        bc = json.loads((out / "brand_correlation.json").read_text(encoding="utf-8"))
        rw, btw = bc.get("rho_asset"), (bc.get("between_brand") or {})
        rb = btw.get("rho_between")
        if rw is not None and rb is not None and np.isfinite(rw) and np.isfinite(rb):
            return {"rho_w": float(rw), "rho_b": float(rb), "source": "brand_correlation.json",
                    "rho_w_ci": (bc.get("rho_asset_ci_lo"), bc.get("rho_asset_ci_hi")),
                    "rho_b_ci": (btw.get("rho_between_ci_lo"), btw.get("rho_between_ci_hi"))}
    except (OSError, ValueError, TypeError):
        pass
    try:
        ci = json.loads((out / "correlation_impact.json").read_text(encoding="utf-8"))
        return {"rho_w": float(ci["rho_within_brand"]), "rho_b": float(ci["rho_between_brand"]),
                "source": "correlation_impact.json", "rho_w_ci": (None, None),
                "rho_b_ci": (None, None)}
    except (OSError, ValueError, KeyError, TypeError):
        return None


TAIL_COLS = ("brand_id", "brand_name", "exposure_mkrw", "deterioration_1y", "n_borrowers",
             "n_stores", "lgd")


def book_hash(book: pd.DataFrame, *params) -> str:
    """꼬리손실 캐시 키 — 계산에 들어가는 열 + 매개변수의 지문. 같으면 결과도 같다."""
    cols = [c for c in TAIL_COLS if c in book.columns]
    h = hashlib.sha256()
    h.update(repr(cols).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(book[cols].reset_index(drop=True), index=False)
             .to_numpy().tobytes())
    h.update(repr(tuple(params)).encode("utf-8"))
    return h.hexdigest()[:32]


def _tail_inputs(book: pd.DataFrame, scores: pd.DataFrame | None,
                 lgd: float | None) -> tuple[pd.DataFrame, list[str]]:
    """브랜드 단위 입력(금액·브랜드 리스크·차주 수·LGD). 행 단위 장부가 와도 브랜드로 합친다."""
    df = book.copy()
    df["brand_id"] = df["brand_id"].astype(str)
    df["_e"] = pd.to_numeric(df["exposure_mkrw"], errors="coerce").fillna(0.0)
    if "lgd" in df.columns:
        lg = pd.to_numeric(df["lgd"], errors="coerce").fillna(DEFAULT_LGD if lgd is None else lgd)
    else:
        lg = pd.Series(DEFAULT_LGD if lgd is None else float(lgd), index=df.index)
    df["_el"] = df["_e"] * lg
    g = df.groupby("brand_id", sort=False)
    first = g.first()
    out = pd.DataFrame({"brand_id": first.index.astype(str)})
    out["exposure_mkrw"] = g["_e"].sum().to_numpy()
    tot = out["exposure_mkrw"].where(out["exposure_mkrw"] > 0)
    out["lgd"] = (g["_el"].sum().to_numpy() / tot).fillna(lg.iloc[0] if len(lg) else DEFAULT_LGD)
    idx = _score_index(scores) if scores is not None and not scores.empty else None

    def col(name: str) -> pd.Series:
        v = (pd.to_numeric(first[name], errors="coerce").to_numpy()
             if name in first.columns else np.full(len(out), np.nan))
        s = pd.Series(v, index=out.index, dtype=float)
        if idx is not None and name in idx.columns:
            s = s.fillna(pd.to_numeric(out["brand_id"].map(idx[name]), errors="coerce"))
        return s

    out["risk"] = col("deterioration_1y")
    nb = (pd.Series(g["n_borrowers"].sum(min_count=1).to_numpy(), index=out.index, dtype=float)
          if "n_borrowers" in df.columns else pd.Series(np.nan, index=out.index))
    # 차주 수는 정수여야 한다 — 이항분포는 정수 차주를 뽑는데 1인당 금액은 실수로 나누면
    # 브랜드 전액이 부실해도 손실이 금액에 못 미친다(12.5명 → 12명 × 금액/12.5).
    out["n_borrowers"] = np.round(nb.fillna(col("n_stores")).fillna(1.0)).clip(lower=1.0)
    names = (first["brand_name"].astype(str).to_numpy() if "brand_name" in first.columns
             else out["brand_id"].to_numpy())
    out["brand_name"] = names
    if idx is not None and "brand_name" not in first.columns:
        out["brand_name"] = out["brand_id"].map(idx["brand_name"]).fillna(out["brand_id"])
    bad = (out["exposure_mkrw"] <= 0) | out["risk"].isna()
    return out[~bad].reset_index(drop=True), out.loc[bad, "brand_id"].tolist()


def _chunk(n_brands: int) -> int:
    """몬테카를로 청크 크기 — 브랜드가 많아도 메모리가 폭주하지 않게 (청크당 약 240만 칸).

    60개 브랜드면 파이프라인 기본값 20,000 과 같아져, 같은 순서·같은 시드에서 파이프라인
    산출물을 그대로 재현한다(청크가 다르면 난수가 다른 시나리오에 배정된다).
    """
    return int(max(1_000, min(20_000, 2_400_000 // max(n_brands, 1))))


def tail_risk(book: pd.DataFrame, scores: pd.DataFrame | None, rho_w: float, rho_b: float,
              n_sims: int = 20_000, seed: int = 42, *, lgd: float | None = None,
              keep_order: bool = False, n_boot: int = 60,
              return_samples: bool = False) -> dict:
    """장부의 꼬리손실 — '차주가 서로 독립'이라는 통상 가정과 측정된 브랜드 상관을 비교한다.

    모형은 파이프라인 correlation_impact 와 **같은 함수**다(수식을 다시 쓰지 않는다).
        _simulate_nested   가맹점(차주) 단위 2단계 요인 모형 — 같은 브랜드 차주끼리 ρ_W,
                           다른 브랜드 차주끼리 ρ_B. 브랜드 리스크를 차주 악화 임계로 쓴다.
        _brand_tail_stats  같은 시드로 같은 시나리오를 다시 돌려 총손실이 99% 분위수를 넘는
                           시나리오의 브랜드별 손실을 모은다 → 성분 ES(Euler 배분)
        _quantile_se       분위수의 몬테카를로 표준오차(부트스트랩)

    book    brand_id·exposure_mkrw 필수. deterioration_1y·n_borrowers(없으면 n_stores)·lgd 가
            없으면 scores 에서 채우고, lgd 는 인자 → 0.45 순서로 채운다.
    LGD     브랜드마다 다르면 (금액×LGD)를 익스포저로 넘기고 LGD=1 로 계산한다 — 손실은
            차주 수 × 1인당 금액 × LGD 라 같은 값이다. 모두 같으면 파이프라인처럼 LGD 를 그대로 넘긴다.
    순서    기본은 brand_id 순으로 정렬해 난수를 배정한다 — 같은 장부를 행 순서만 바꿔 올려도
            같은 숫자가 나온다. keep_order=True 면 받은 순서 그대로(파이프라인 재현 감사용).

    반환 키는 correlation_impact.json 과 같은 이름(independent_p99_mkrw 등)에, 브랜드별 기여표
    `contrib`(brand_ul_contribution.csv 와 같은 정의)와 `euler`·`elapsed_sec` 를 더한다.
    """
    t0 = time.perf_counter()
    rho_w, rho_b = float(rho_w), float(rho_b)
    if not (0.0 <= rho_b <= rho_w < 1.0):
        raise ValueError(f"상관이 모형 조건(0 ≤ ρ_B ≤ ρ_W < 1)을 벗어났습니다: ρ_W={rho_w}, ρ_B={rho_b}")
    b, excluded = _tail_inputs(book, scores, lgd)
    if b.empty:
        raise ValueError("꼬리손실을 계산할 브랜드가 없습니다(금액·브랜드 리스크가 있는 브랜드 0개).")
    if not keep_order:
        b = b.sort_values("brand_id", kind="stable").reset_index(drop=True)

    thr = norm.ppf(np.clip(b["risk"].to_numpy(dtype=float), 1e-9, 1 - 1e-9))
    expo = b["exposure_mkrw"].to_numpy(dtype=float)
    counts = b["n_borrowers"].to_numpy(dtype=float)
    lgd_v = b["lgd"].to_numpy(dtype=float)
    uniform = bool(np.all(np.abs(lgd_v - lgd_v[0]) < 1e-12))
    e, lg = (expo, float(lgd_v[0])) if uniform else (expo * lgd_v, 1.0)
    chunk = _chunk(len(b))

    loss_i = _simulate_nested(thr, e, counts, lg, 0.0, 0.0, n_sims, seed, chunk=chunk)
    loss_c = _simulate_nested(thr, e, counts, lg, rho_w, rho_b, n_sims, seed, chunk=chunk)

    res: dict = {
        "model": "nested_two_factor_franchisee_level",
        "rho_within_brand": rho_w, "rho_between_brand": rho_b,
        "lgd": float(lgd_v[0]) if uniform else None, "lgd_uniform": uniform,
        "n_sims": int(n_sims), "seed": int(seed),
        "n_brands": len(b), "n_franchisees": int(np.maximum(counts, 0).astype(np.int64).sum()),
        "total_exposure_mkrw": float(expo.sum()),
        "expected_loss_mkrw": float(loss_i.mean()),
        "analytic_el_mkrw": float((expo * lgd_v * b["risk"].to_numpy(dtype=float)).sum()),
        "excluded_brands": excluded,
    }
    for lab, arr in (("independent", loss_i), ("brand_correlated", loss_c)):
        res[f"{lab}_mean_mkrw"] = float(arr.mean())
        for lvl, q in (("p95", 0.95), ("p99", 0.99), ("p999", 0.999)):
            res[f"{lab}_{lvl}_mkrw"] = float(np.quantile(arr, q))
    for lvl, q in (("p95", 0.95), ("p99", 0.99), ("p999", 0.999)):
        a, c = res[f"independent_{lvl}_mkrw"], res[f"brand_correlated_{lvl}_mkrw"]
        res[f"understatement_{lvl}_mkrw"] = c - a
        res[f"understatement_{lvl}_pct"] = (c / a - 1.0) if a > 0 else float("nan")
        # 파이프라인과 같은 시드 규칙(독립 seed, 상관 seed+1)으로 몬테카를로 오차를 함께 낸다
        res[f"mc_se_independent_{lvl}_mkrw"] = _quantile_se(loss_i, q, n_boot, seed)
        res[f"mc_se_correlated_{lvl}_mkrw"] = _quantile_se(loss_c, q, n_boot, seed + 1)
    for lab in ("independent", "brand_correlated"):
        res[f"{lab}_ul99_mkrw"] = res[f"{lab}_p99_mkrw"] - res[f"{lab}_mean_mkrw"]
    res["ul99_multiple"] = (res["brand_correlated_ul99_mkrw"] / res["independent_ul99_mkrw"]
                            if res["independent_ul99_mkrw"] > 0 else float("nan"))

    # ── 브랜드별 꼬리손실 기여 (Euler 성분 ES) — brand_ul_contribution.csv 와 같은 정의 ──
    var99 = res["brand_correlated_p99_mkrw"]
    tail_n, tail_sum, all_sum = _brand_tail_stats(thr, e, counts, lg, rho_w, rho_b, n_sims, seed,
                                                  var99, chunk=chunk)
    es_total = float(loss_c[loss_c >= var99].mean())
    es_i = tail_sum / max(tail_n, 1)
    el_i = all_sum / n_sims
    ul_i = es_i - el_i
    tot_e = max(float(expo.sum()), 1e-9)
    contrib = pd.DataFrame({
        "brand_id": b["brand_id"], "brand_name": b["brand_name"],
        "n_borrowers": counts.astype(int), "exposure_mkrw": expo,
        "exposure_share": expo / tot_e, "lgd": lgd_v,
        "el_mkrw": el_i, "es99_mkrw": es_i, "ul_contrib_mkrw": ul_i,
        "ul_share": ul_i / max(float(ul_i.sum()), 1e-9),
    })
    contrib["concentration_ratio"] = contrib["ul_share"] / contrib["exposure_share"].clip(lower=1e-9)
    res["contrib"] = contrib.sort_values("ul_contrib_mkrw", ascending=False).reset_index(drop=True)
    res["euler"] = {
        "n_tail_scenarios": int(tail_n), "es99_total_mkrw": es_total,
        # 성분의 합이 전체 ES 와 일치해야 배분이다 — 어긋나면 화면이 숫자를 믿지 않는다
        "component_addup_rel_err": abs(float(es_i.sum()) - es_total) / max(es_total, 1e-9),
        "top5_ul_share": float(res["contrib"]["ul_share"].head(5).sum()),
    }
    if return_samples:
        res["samples"] = {"independent": loss_i, "brand_correlated": loss_c}
    res["elapsed_sec"] = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------
# 5) 업로드 양식
# ---------------------------------------------------------------------------

# 양식의 줄마다 보여 주려는 규칙이 있다 — 이름은 전부 평가 대상 브랜드(scores_latest.csv)에서
# 확인된 것만 싣고, 금액·차주 수는 **가상**이다. 확인되지 않는 줄은 양식에서 뺀다
# (산출물이 바뀌어 브랜드가 빠져도 양식이 틀린 설명을 달고 나가지 않게).
#   (입력 이름, 억원, 담보유형, 차주 수, LGD%, 설명, 기대 매칭 근거)
_TEMPLATE_ROWS: tuple[tuple[str, float, str, int, float | None, str, str], ...] = (
    ("빽다방", 45.0, "보증서", 60, None, "등록명과 같으면 바로 확정", "정확 일치"),
    ("빽다방", 12.0, "신용", 15, None, "한 브랜드를 담보유형별로 나눠 적어도 브랜드로 합칩니다", "정확 일치"),
    ("메가커피", 38.0, "보증서담보", 52, None, "통칭으로 적어도 등록명을 찾습니다", "통칭 일치"),
    ("이디야커피", 30.0, "신용보증재단 보증서", 41, None, "", "정확 일치"),
    ("컴포즈커피", 25.0, "임차보증금", 33, None, "등록명과 글자가 달라 '확인 필요'로 표시됩니다",
     "유사 일치"),
    ("교촌치킨", 22.0, "부동산 담보", 20, None, "", "정확 일치"),
    ("굽네치킨", 18.0, "신용", 24, None, "", "정확 일치"),
    ("본죽&비빔밥", 15.0, "보증서", 21, None, "", "정확 일치"),
    ("투다리", 12.0, "담보", 16, 35.0, "LGD 를 적으면 담보유형 가정 대신 그 값을 씁니다", "정확 일치"),
    ("달리는커피", 9.0, "신용", 14, None, "", "정확 일치"),
    ("인생냉면", 6.0, "보증서", 9, None, "", "정확 일치"),
    ("국수나무", 8.0, "신용", 12, None, "같은 이름의 브랜드가 둘이라 '확인 필요'", "동명 브랜드"),
    ("빽다빵", 5.0, "신용", 6, None, "오타 — 찾지 못한 이름은 '미매칭'으로 따로 모으고 비슷한 "
     "이름을 제안합니다", "공시에 없음"),
)


def _default_scores() -> pd.DataFrame | None:
    from src.common import load_config
    p = Path(load_config()["paths"]["outputs"]) / "scores_latest.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


def template_bytes(scores: pd.DataFrame | None = None, cfg: dict | None = None) -> bytes:
    """업로드 양식(.xlsx) — '여신장부' 시트(예시 13줄)와 '작성 안내' 시트.

    브랜드명은 실제 평가 대상 브랜드이고 금액은 가상이다. 모든 줄의 비고에 '예시(가상 금액)'를
    적어 두어, 양식을 그대로 올리면 화면이 예시 장부라고 밝힌다.
    """
    if scores is None:
        scores = _default_scores()
    if cfg is None:
        from src.common import load_config
        cfg = load_config()
    rows = []
    from src import public
    if public.masked() and scores is not None:
        # 공개 배포는 실명 예시를 쓰지 않는다(모형 사용 명세 §3 · src/public.py) — 가명 브랜드에
        # 같은 금액·담보 구성을 입힌다. 통칭·동명·오타 예시는 실명에서만 성립해 뺀다.
        base = [r for r in _TEMPLATE_ROWS if r[6] == "정확 일치"]
        for name, (_, amt, coll, nb, lgd, why, _rule) in zip(public.example_names(scores, len(base)), base,
                                                              strict=False):
            rows.append({"브랜드명": name, "여신잔액(억원)": amt, "담보유형": coll, "차주 수": nb,
                         "LGD(%, 선택)": lgd,
                         "비고": "예시(가상 금액)" + (f" · {why}" if why and "LGD" in why else "")})
    for name, amt, coll, nb, lgd, why, rule in (() if rows else _TEMPLATE_ROWS):
        if scores is not None:
            got = _match_one(None, name, set(scores["brand_id"].astype(str)), scores, None)
            if got["match_rule"] != rule:
                continue
            if rule == "공시에 없음" and not got["candidates"]:
                continue                     # 제안할 이름이 없으면 오타 예시가 설명을 못 한다
        rows.append({"브랜드명": name, "여신잔액(억원)": amt, "담보유형": coll, "차주 수": nb,
                     "LGD(%, 선택)": lgd,
                     "비고": "예시(가상 금액)" + (f" · {why}" if why else "")})
    sample = pd.DataFrame(rows, columns=["브랜드명", "여신잔액(억원)", "담보유형", "차주 수",
                                         "LGD(%, 선택)", "비고"])
    lmap, ldef = lgd_assumptions(cfg)
    guide = [
        "이 양식의 금액·차주 수는 예시(가상)입니다. 브랜드명은 실제 평가 대상 브랜드입니다 — "
        "실제 잔액으로 바꿔 쓰십시오.",
        "필수 열: 브랜드명(또는 brand_id), 여신잔액. 금액 단위는 열 이름에 적어 주십시오 — "
        "(원)·(천원)·(백만원)·(억원). 단위가 없으면 값 크기로 추정하고 화면에 그 가정을 밝힙니다.",
        "선택 열: 담보유형(보증서·담보·신용), LGD(%), 차주 수.",
        "담보유형별 기본 손실률(LGD) 가정: "
        + " · ".join(f"{k} {lmap[k] * 100:.0f}%" for k in COLLATERAL_TYPES if k in lmap)
        + f" · 미기재 {ldef * 100:.0f}%. 화면에서 바꿀 수 있고, 행에 LGD 를 적으면 그 값이 우선합니다.",
        "차주 수가 없으면 여신이 그 브랜드 가맹점 전체에 고르게 나뉘어 있다고 가정합니다 — "
        "실제 차주가 더 적으면 꼬리손실이 과소평가됩니다.",
        "한 브랜드를 여러 줄(담보유형별·지점별)로 나눠 적어도 브랜드 단위로 합칩니다. "
        "합계·소계 줄은 자동으로 뺍니다.",
        "브랜드명이 등록명과 정확히 같지 않으면 '확인 필요'로 표시하고 가장 가까운 후보를 붙입니다. "
        "찾지 못한 이름은 '미매칭'으로 따로 모아 분석에서 뺍니다.",
        "차주 개인정보(성명·주민등록번호·계좌번호)는 넣지 마십시오 — 브랜드별 금액이면 충분합니다.",
        "올린 파일은 서버 디스크에 저장하지 않고 화면을 연 세션의 메모리에서만 처리합니다.",
    ]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        sample.to_excel(w, index=False, sheet_name="여신장부")
        ws = w.sheets["여신장부"]
        ws.freeze_panes = "B2"
        for letter, width in zip("ABCDEF", (18, 16, 20, 10, 14, 64), strict=True):
            ws.column_dimensions[letter].width = width
        pd.DataFrame({"작성 안내": guide}).to_excel(w, index=False, sheet_name="작성 안내")
        w.sheets["작성 안내"].column_dimensions["A"].width = 120
    return buf.getvalue()
