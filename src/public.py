"""공개 데모 가명 처리 — 공개 배포에서는 브랜드 실명 옆에 등급을 싣지 않는다.

왜 필요한가
    모형 사용 명세(docs/MODEL_USE_SPEC.md §3)는 "대외 공표·마케팅 자료에 브랜드 등급을 개별
    명시"하는 것을 금지한다 — 실명 브랜드에 대한 신용훼손 위험 때문이다. 누구나 접속하는 공개
    데모는 사실상 대외 공표다. 실명 소규모 브랜드 옆에 '주의 32.7%'를 띄우면 명세가 막으려던
    바로 그 위험을 만든다. 그래서 공개 배포에서는 브랜드의 정체를 가린다. 로컬·사내 실행은
    실명 그대로다(내부 참고용 — 명세가 허용하는 쓰임).

어떻게 — 화면이 읽는 산출물을 '가명 사본'으로 바꿔 끼운다
    화면·참고의견서·상담은 모두 설정의 outputs/processed 경로에서 산출물을 읽는다. 공개 모드에서는
    그 경로를 이 모듈이 만든 가명 사본(임시 폴더)으로 바꾼다(src/common.load_config). 한 곳에서
    바꾸므로 화면마다 가리는 코드를 흩뿌리지 않는다. 사본에는 앱이 읽는 파일만 허용 목록으로
    담는다 — 목록 밖 파일(뉴스 원문, 검색 색인 등)은 사본에 아예 없다.

무엇을 가리나
    브랜드명 → '{세부업종} {번호}'(예: '치킨 017') · 브랜드 ID → 'P' + 번호 · 가맹본부 법인명 →
    '가맹본부 {번호}' · 관리번호·DART 고유번호·접수번호 → 지움 · 소견 문장 속 브랜드명·법인명·검색어 →
    가명 · 뉴스 사건·검색 색인·로고 → 공개 모드에서 쓰지 않음.

한계 — 숨기지 않는다
    숫자(가맹점 수·매출·재무)는 그대로다. 공정위 자료와 대조하면 큰 브랜드는 다시 알아볼 수 있다.
    이 처리는 재식별이 불가능한 익명화가 아니라, 명세가 금지한 **실명 등급의 공표를 하지 않는 것**이
    목적이다. 누출 여부는 tests/test_public.py 가 사본 전체를 실명 목록으로 훑어 확인한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ENV = "FRANSCORE_PUBLIC_DEMO"
DIR_ENV = "FRANSCORE_PUBLIC_DIR"          # 사본 위치를 정할 때(테스트) — 기본은 임시 폴더
_SALT = "franscore-public-v1"
_LOCK = threading.Lock()

# 사본에 담는 파일 — 앱이 읽는 것만. (경로, 종류)
_OUT_FILES = (
    ("scores_latest.csv", "table"), ("grade_history.csv", "table"), ("coverage_report.csv", "table"),
    ("brand_diagnosis.parquet", "table"), ("brand_diagnosis_summary.csv", "table"),
    ("portfolio.csv", "table"), ("brand_ul_contribution.csv", "table"),
    ("localdata_signal.csv", "table"), ("localdata_flows.csv", "table"),
    ("scores_latest_meta.json", "json"), ("grade_bands.json", "json"), ("watch_base_rates.json", "json"),
    ("correlation_impact.json", "json"), ("brand_correlation.json", "json"),
    ("portfolio_summary.json", "json"), ("refresh_state.json", "json"), ("rag_stats.json", "json"),
    ("brand_diagnosis_meta.json", "json"), ("demand_trends.json", "json"),
    ("localdata_validation.json", "json"), ("validation/summary.json", "json"),
    ("split_years.json", "json"),
)
_PROC_FILES = (("panel.parquet", "table"), ("panel_full.parquet", "table"),
               ("hq_financials.parquet", "table"))
# 가명 사본에서 지우는 식별 열
_DROP_COLS = ("brand_mnno", "hq_mnno", "corp_code", "rcept_no", "reg_no", "brno", "crno")
_TEXT_COLS = ("title", "detail", "source", "headline", "headline_detail", "evidence")
_EVIDENCE_DROP = ("rcept_no", "corp_code", "reg_no", "brno", "crno", "url", "source_url", "keyword", "term")


def is_public() -> bool:
    """공개 배포인가 — FRANSCORE_PUBLIC_DEMO 가 우선, 없으면 Streamlit Community Cloud 경로로 판단."""
    flag = os.getenv(ENV, "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    return str(ROOT).startswith("/mount/src")


def _rank(key: str) -> str:
    return hashlib.sha1(f"{_SALT}:{key}".encode()).hexdigest()


_WORDY = re.compile(r"[가-힣A-Za-z]")


class _Scanner:
    """여러 이름을 한 번에 찾아 바꾼다 — 앞 두 글자 색인으로 훑고, 한 자리에서는 가장 긴 이름을 쓴다.

    법인명 1,400여 개를 문장마다 하나씩 `in` 으로 대조하면 사본 생성이 20초를 넘었다(실측).
    """

    def __init__(self, pairs):
        self.idx: dict[str, list[tuple[str, str]]] = {}
        for a, b in sorted(pairs, key=lambda t: -len(t[0])):
            if len(a) >= 2:
                self.idx.setdefault(a[:2], []).append((a, b))

    def sub(self, text: str) -> str:
        if not self.idx:
            return text
        out, i, n = [], 0, len(text)
        while i < n:
            for a, b in self.idx.get(text[i:i + 2], ()):
                if text.startswith(a, i):
                    out.append(b)
                    i += len(a)
                    break
            else:
                out.append(text[i])
                i += 1
        return "".join(out)


class Masker:
    """실명 → 가명 대응표와 치환 규칙. 같은 원본이면 늘 같은 가명이 나온다(링크가 안정적이다)."""

    def __init__(self, panel_full: pd.DataFrame, scores: pd.DataFrame | None = None,
                 terms: dict[str, str] | None = None):
        from src.dart import norm_corp
        pf = panel_full.copy()
        pf["brand_id"] = pf["brand_id"].astype(str)
        last = pf.sort_values("year").drop_duplicates("brand_id", keep="last")
        ids = list(last["brand_id"])
        if scores is not None:
            ids += [b for b in scores["brand_id"].astype(str) if b not in set(ids)]
        ind = dict(zip(last["brand_id"], last["industry_mid"].fillna(last["industry_major"])
                       .fillna("브랜드").astype(str), strict=True))
        order = sorted(ids, key=_rank)
        self.pid = {b: f"P{i + 1:05d}" for i, b in enumerate(order)}
        counter: dict[str, int] = {}
        self.pname: dict[str, str] = {}
        for b in order:
            g = ind.get(b, "브랜드")
            counter[g] = counter.get(g, 0) + 1
            self.pname[b] = f"{g} {counter[g]:03d}"
        # 법인명 → 가명, 정규화 키 → 가명의 정규화 키 (본부 재무 연결이 사본에서도 그대로 되게)
        comps = sorted({str(c).strip() for c in pf["company_name"].dropna() if str(c).strip()}, key=_rank)
        self.company = {c: f"가맹본부 {i + 1:04d}" for i, c in enumerate(comps)}
        self.key = {}
        for c, p in self.company.items():
            k = norm_corp(c)
            if k and k not in self.key:
                self.key[k] = norm_corp(p)
        # 문장 치환용 — 브랜드별 과거 이름·본부, 본부별 브랜드(같은 본부의 다른 브랜드 문장)
        self.names_of: dict[str, set[str]] = {}
        for b, n in zip(pf["brand_id"], pf["brand_name"].astype(str), strict=True):
            if n and n != "nan":
                self.names_of.setdefault(b, set()).add(n.strip())
        self.comp_of: dict[str, set[str]] = {}
        for b, c in zip(pf["brand_id"], pf["company_name"], strict=True):
            if pd.notna(c) and str(c).strip():
                self.comp_of.setdefault(b, set()).add(str(c).strip())
        self.brands_of_comp: dict[str, set[str]] = {}
        for b, cs in self.comp_of.items():
            for c in cs:
                self.brands_of_comp.setdefault(c, set()).add(b)
        # 검색수요 소견이 인용하는 검색어(브랜드명과 다를 수 있다) — demand_trends.json 의 term
        self.terms = {str(b): str(t) for b, t in (terms or {}).items() if t}
        # 전역 치환 사전 — 법인명(4자 이상)은 어느 문장에서든 가린다
        self._global = _Scanner((c, p) for c, p in self.company.items() if len(c.replace(" ", "")) >= 4)

    # ── 값 단위 ────────────────────────────────────────────────────────────
    def brand_id(self, v):
        s = str(v)
        return self.pid.get(s, v if pd.isna(v) else "P00000")

    def brand_name_for(self, bid, fallback=None):
        return self.pname.get(str(bid), fallback if fallback is not None else "브랜드")

    def text(self, s, bid: str | None = None):
        """소견·표제 문장 속 실명을 가린다 — 자기 이름·같은 본부 브랜드·법인명·검색어 인용."""
        if not isinstance(s, str) or not s or not _WORDY.search(s):
            return s                                   # 날짜·숫자뿐인 문자열은 볼 것이 없다
        out = s
        targets: list[tuple[str, str]] = []
        if bid is not None:
            b = str(bid)
            me = self.pname.get(b, "이 브랜드")
            targets += [(n, me) for n in self.names_of.get(b, ())]
            if b in self.terms:
                targets.append((self.terms[b], me))
            for c in self.comp_of.get(b, ()):
                targets += [(self._strip(n), self.pname.get(o, "같은 본부 브랜드"))
                            for o in self.brands_of_comp.get(c, ()) if o != b
                            for n in self.names_of.get(o, ())]
                targets.append((c, self.company.get(c, "가맹본부")))
        targets = [(a, p) for a, p in targets if a and len(a.replace(" ", "")) >= 2]
        for a, p in sorted(targets, key=lambda t: -len(t[0])):
            out = out.replace(a, p)
        return self._global.sub(out)

    def evidence(self, s, bid: str | None = None):
        """소견 근거(JSON 문자열) — 식별값(접수번호·법인번호·URL)은 지우고 문자열은 가린다."""
        if not isinstance(s, str) or not s.strip().startswith("{"):
            return self.text(s, bid)
        try:
            obj = json.loads(s)
        except ValueError:
            return self.text(s, bid)
        if isinstance(obj, dict):
            obj = {k: v for k, v in obj.items() if k not in _EVIDENCE_DROP}
        return json.dumps(self.json(obj, key=bid), ensure_ascii=False)

    @staticmethod
    def _strip(n: str) -> str:
        return str(n).strip()

    # ── 표 단위 ────────────────────────────────────────────────────────────
    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        bids = out["brand_id"].astype(str) if "brand_id" in out.columns else None
        if bids is not None:
            if "brand_name" in out.columns:
                out["brand_name"] = [self.pname.get(b, "브랜드") for b in bids]
            for col in _TEXT_COLS:
                if col in out.columns:
                    fn = self.evidence if col == "evidence" else self.text
                    out[col] = [fn(v, b) for v, b in zip(out[col], bids, strict=True)]
            out["brand_id"] = [self.pid.get(b, "P00000") for b in bids]
        elif "brand_name" in out.columns:
            out["brand_name"] = "브랜드"
        for col in ("company_name", "corp_name"):
            if col in out.columns:
                out[col] = [self.company.get(str(v).strip(), "가맹본부") if pd.notna(v) else v
                            for v in out[col]]
        if "key" in out.columns:                         # 본부 재무의 법인명 정규화 키
            out["key"] = [self.key.get(str(k), None) for k in out["key"]]
            out = out[out["key"].notna()]
        return out.drop(columns=[c for c in _DROP_COLS if c in out.columns])

    def json(self, obj, key: str | None = None):
        if isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                nk = self.pid.get(str(k), k)
                if k in ("term", "keyword", "query") and isinstance(v, str):
                    res[nk] = self.pname.get(str(key), "브랜드") if key else "브랜드"
                    continue
                res[nk] = self.json(v, key=str(k) if str(k) in self.pid else key)
            return res
        if isinstance(obj, list):
            return [self.json(v, key=key) for v in obj]
        if isinstance(obj, str):
            if obj in self.pid:
                return self.pid[obj]
            return self.text(obj, key) if key else self.text(obj)
        return obj


# ── 사본 만들기 ─────────────────────────────────────────────────────────────

def _signature(out: Path, proc: Path) -> str:
    h = hashlib.sha1(_SALT.encode())
    for base, files in ((out, _OUT_FILES), (proc, _PROC_FILES)):
        for name, _ in files:
            p = base / name
            if p.exists():
                st = p.stat()
                h.update(f"{name}:{st.st_size}:{st.st_mtime_ns}".encode())
    h.update(Path(__file__).read_bytes())               # 가리는 규칙이 바뀌어도 다시 만든다
    return h.hexdigest()[:16]


def _read(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, encoding="utf-8-sig")


def _write(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".parquet":
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False, encoding="utf-8-sig")


def build(dest: Path, out: Path, proc: Path) -> Path:
    """가명 사본을 dest/outputs, dest/processed 에 만든다. 원자적으로 바꿔 끼운다."""
    pf = pd.read_parquet(proc / "panel_full.parquet")
    sp = out / "scores_latest.csv"
    terms = {}
    dp = out / "demand_trends.json"
    if dp.exists():
        brands = (json.loads(dp.read_text(encoding="utf-8")) or {}).get("brands") or {}
        terms = {b: (v or {}).get("term") for b, v in brands.items() if isinstance(v, dict)}
    masker = Masker(pf, pd.read_csv(sp, encoding="utf-8-sig") if sp.exists() else None, terms)
    tmp = Path(tempfile.mkdtemp(prefix=".building-", dir=dest.parent))
    try:
        for base, files, sub in ((out, _OUT_FILES, "outputs"), (proc, _PROC_FILES, "processed")):
            for name, kind in files:
                src = base / name
                if not src.exists():
                    continue
                target = tmp / sub / name
                if kind == "table":
                    df = _read(src)
                    if name == "brand_diagnosis.parquet" and "code" in df.columns:
                        df = df[~df["code"].astype(str).str.startswith("NEWS")]   # 기사 제목·문장
                    _write(masker.frame(df), target)
                else:
                    obj = json.loads(src.read_text(encoding="utf-8-sig"))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(masker.json(obj), ensure_ascii=False, indent=1),
                                      encoding="utf-8")
        (tmp / "outputs").mkdir(parents=True, exist_ok=True)
        (tmp / "processed").mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return dest


def data_dirs(out: Path | None = None, proc: Path | None = None) -> tuple[Path, Path]:
    """공개 모드에서 화면이 읽을 (outputs, processed) — 원본이 바뀌었으면 사본을 다시 만든다."""
    out = Path(out or ROOT / "outputs")
    proc = Path(proc or ROOT / "data" / "processed")
    base = Path(os.getenv(DIR_ENV) or Path(tempfile.gettempdir()) / "franscore_public")
    dest = base / _signature(out, proc)
    if not (dest / "outputs" / "scores_latest.csv").exists():
        with _LOCK:
            if not (dest / "outputs" / "scores_latest.csv").exists():
                base.mkdir(parents=True, exist_ok=True)
                build(dest, out, proc)
    return dest / "outputs", dest / "processed"


# 공개 모드 예시는 중간 규모 브랜드에서 고른다. 가맹점 수가 업계 최상위인 브랜드는 숫자만 봐도
# 실명이 짐작돼 가명이 의미를 잃는다(커피 3천 곳대, 치킨 2천 곳대는 한두 곳뿐이다).
SHOWCASE_STORES = (100, 499)
_GRADE_CYCLE = ("FS1", "FS3", "FS2")        # 예시 표에 등급이 골고루 보이도록 번갈아 고른다


def _showcase(scores: pd.DataFrame) -> pd.DataFrame:
    s = scores.assign(n_sort=pd.to_numeric(scores.get("n_stores"), errors="coerce").fillna(0))
    band = s[s["n_sort"].between(*SHOWCASE_STORES)]
    return (band if len(band) >= 10 else s).sort_values(["n_sort", "brand_name"], ascending=[False, True])


def _industries(pool: pd.DataFrame) -> list[str]:
    """브랜드가 많은 업종부터 — 예시가 흔한 업종에서 나오도록."""
    vc = pool["industry_mid"].astype(str).value_counts()
    return sorted(vc.index, key=lambda i: (-vc[i], i))


def example_names(scores: pd.DataFrame, n: int = 6) -> list[str]:
    """공개 모드 예시에 쓸 가명 — 중간 규모에서, 업종이 겹치지 않게, 등급을 번갈아."""
    pool = _showcase(scores)
    grade = pool.get("grade", pd.Series("", index=pool.index)).astype(str)
    out = []
    for k, ind in enumerate(_industries(pool)[:n]):
        g = pool[pool["industry_mid"].astype(str) == ind]
        want = g[grade.loc[g.index] == _GRADE_CYCLE[k % len(_GRADE_CYCLE)]]
        out.append(str((want if len(want) else g)["brand_name"].iloc[0]))
    return out


def same_industry_pair_rows(scores: pd.DataFrame) -> pd.DataFrame:
    """비교 예시 두 브랜드 — 가장 흔한 업종의 중간 규모에서, 등급이 갈리면 안정 하나·나머지 하나."""
    pool = _showcase(scores)
    if pool.empty:
        return pool
    g = pool[pool["industry_mid"].astype(str) == _industries(pool)[0]]
    grade = g.get("grade", pd.Series("", index=g.index)).astype(str)
    safe, other = g[grade == "FS1"], g[grade != "FS1"]
    return pd.concat([safe.head(1), other.head(1)]) if len(safe) and len(other) else g.head(2)


def same_industry_pair(scores: pd.DataFrame) -> tuple[str, str]:
    """비교 질문 예시 — same_industry_pair_rows 의 두 이름."""
    two = same_industry_pair_rows(scores)["brand_name"].astype(str).tolist()
    return tuple((two + two)[:2]) if two else ("브랜드 A", "브랜드 B")


def unscored_example(panel_full: pd.DataFrame) -> str | None:
    """'평가 대상 아님' 예시 — 외식이 아닌 업종에서 가맹점이 가장 많은 브랜드(사본의 가명)."""
    p = panel_full[panel_full["industry_major"].astype(str) != "외식"]
    if p.empty:
        return None
    last = p[p["year"] == p["year"].max()]
    last = last.assign(n_sort=pd.to_numeric(last["n_stores"], errors="coerce").fillna(0))
    return str(last.sort_values("n_sort", ascending=False)["brand_name"].iloc[0])


def chat_examples(scores: pd.DataFrame) -> list[str]:
    """AI 상담 예시 질문 — 실명 대신 사본의 가명으로."""
    names = example_names(scores, 3) + ["브랜드"] * 3
    a, b = same_industry_pair(scores)
    return [f"{names[0]} 가맹점 대출 심사 전에 봐야 할 점을 정리해줘",
            f"{names[1]} 가맹점 수 추이를 가져와줘",
            f"{a}와 {b} 중 어디가 더 안정적이야?",
            "치킨 업종에서 지금 가장 위험한 브랜드는?"]
