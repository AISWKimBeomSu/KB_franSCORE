"""진짜 RAG — 검색 기반 근거 인용 (명세 COULD "진짜 RAG").

왜 필요한가:
    심사메모가 "근거만 인용"하려면 근거가 **검색 가능한 문서**여야 한다. 단순히
    프롬프트에 수치를 붙여 넣는 것은 RAG가 아니다. 본 모듈은 실제 문서 코퍼스를
    구축하고 색인·검색하여, 메모가 인용한 문장이 어느 문서(출처 URL·발행일)의
    몇 번째 청크에서 왔는지 추적 가능하게 만든다.

코퍼스 구성 (2종, 모두 실데이터):
    ① 뉴스 문서   — data/raw/news/*.json 스냅샷 (제목·출처·발행일·링크)
    ② 공시 문서   — 브랜드×연도 패널을 자연어 문장으로 렌더링
                    ("2023년 A브랜드 공시, 가맹점 57개, 직영점 1개, 가맹점 평균매출 1.16억원,
                      계약종료·해지 7건, 신규개점 3개, 직영비중 1.7%, 진출 8개 시도,
                      지역집중도(HHI) 0.210")
                    → 모형이 본 수치와 메모가 인용하는 근거가 **동일 원천**임을 보장

검색 방식 (재현성 우선):
    TF-IDF + 코사인 유사도. 한국어 형태소 분석기 없이도 동작하도록 **문자 n-gram
    (char_wb 2~4)** 과 단어 토큰을 결합한다. 외부 임베딩 API에 의존하지 않으므로
    키 없이 완전 재현 가능하고, seed 고정과 무관하게 결정적이다.
    (임베딩 기반 확장은 로드맵 — 본 프로토타입 범위에서는 재현성을 우선한다.)

산출물:
    data/processed/rag_corpus.parquet   문서 코퍼스 (doc_id·brand_id·text·출처)
    outputs/rag_index.joblib            벡터라이저 + 문서 행렬
    outputs/rag_stats.json              코퍼스 통계 (문서 수·유형별 분포)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common import get_logger, load_config

# hq_financials 에 섞여 들어온 정보공개서 열람분을 식별하는 표식(src/ifrmp_web.py 소유).
# 문자열을 여기서 다시 쓰지 않고 그쪽을 단일 출처로 둔다 — 한쪽만 바뀌면 출처가 뒤바뀐다.
from src.ifrmp_web import HQ_SOURCE_TAG as _IFRMP_TAG

try:  # 뉴스 개체 매칭 규칙 재사용 (단일 정의 원칙)
    from src.news_llm import _entity_match
except Exception:
    _entity_match = None  # type: ignore[assignment]

log = get_logger("rag")

CORPUS_FILE = "rag_corpus.parquet"
INDEX_FILE = "rag_index.joblib"
STATS_FILE = "rag_stats.json"

CHAR_MAX_FEATURES = 40_000   # 문자 n-gram 피처 상한 (메모리 — RagIndex 주석 참조)

_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS.sub(" ", str(s or "")).strip()


# ---------------------------------------------------------------------------
# 코퍼스 구축
# ---------------------------------------------------------------------------

def _news_documents(cfg: dict) -> list[dict]:
    """뉴스 스냅샷 → 문서. 브랜드명으로 brand 태깅(파일명 = 브랜드명 기반)."""
    news_dir = Path(cfg["paths"]["raw"]) / "news"
    docs: list[dict] = []
    if not news_dir.exists():
        return docs
    for fp in sorted(news_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        brand = _clean(data.get("brand") or fp.stem)
        for i, art in enumerate(data.get("articles") or []):
            title = _clean(art.get("title"))
            if not title:
                continue
            # 개체 매칭 필터: 일반명사·부분문자 오염 제거 (news_llm과 동일 규칙 재사용).
            # 없으면 문자 n-gram 검색이 무관 기사를 근거로 끌어올 수 있다(실측 오탐 확인).
            if _entity_match is not None and not _entity_match(brand, title):
                continue
            docs.append({
                "doc_id": f"news:{fp.stem}:{i}",
                "brand_name": brand,
                "source_type": "news",
                "text": title,
                "url": _clean(art.get("link")),
                "published": _clean(art.get("published")),
                "source_name": _clean(art.get("source")),
                "year": np.nan,
            })
    return docs


def _disclosure_documents(cfg: dict, max_rows: int | None = None) -> list[dict]:
    """브랜드×연도 패널 → 자연어 공시 문서 (모형 입력과 동일 원천)."""
    panel_path = Path(cfg["paths"]["processed"]) / "panel.parquet"
    if not panel_path.exists():
        return []
    p = pd.read_parquet(panel_path)
    if "eligible_t" in p.columns:                      # 표본 자격 행 우선
        p = pd.concat([p[p["eligible_t"]], p[~p["eligible_t"]]])
    if max_rows:
        p = p.head(max_rows)
    docs: list[dict] = []
    for r in p.itertuples():
        parts = [f"{int(r.year)}년 {r.brand_name} ({r.industry_major}/{r.industry_mid}) 공시"]
        if pd.notna(r.n_stores):
            parts.append(f"가맹점 {int(r.n_stores)}개")
        if pd.notna(getattr(r, "n_direct", np.nan)):
            parts.append(f"직영점 {int(r.n_direct)}개")
        if pd.notna(r.avg_sales):
            # avg_sales 단위는 천원(INTERFACES §0) → 억원 = /1e5 (리뷰 확정 결함 수정: /1e4는 10배 과장)
            parts.append(f"가맹점 평균매출 {r.avg_sales / 1e5:.2f}억원")
        end = (0 if pd.isna(r.n_contract_end) else r.n_contract_end) + \
              (0 if pd.isna(r.n_contract_cancel) else r.n_contract_cancel)
        parts.append(f"계약종료·해지 {int(end)}건")
        if pd.notna(r.n_new):
            parts.append(f"신규개점 {int(r.n_new)}개")
        n_d = getattr(r, "n_direct", np.nan)
        if pd.notna(n_d) and pd.notna(r.n_stores) and (n_d + r.n_stores) > 0:
            parts.append(f"직영비중 {100 * n_d / (n_d + r.n_stores):.1f}%")
        nreg = getattr(r, "n_regions", np.nan)
        if pd.notna(nreg):
            parts.append(f"진출 {int(nreg)}개 시도")
        hhi = getattr(r, "region_hhi", np.nan)
        if pd.notna(hhi):
            parts.append(f"지역집중도(HHI) {hhi:.3f}")
        if getattr(r, "cancel_flag", 0) == 1:
            parts.append(f"브랜드 등록취소({getattr(r, 'cancel_type', '') or '유형미상'})")
        docs.append({
            "doc_id": f"disc:{r.brand_id}:{int(r.year)}",
            "brand_name": r.brand_name,
            "source_type": "disclosure",
            "text": ", ".join(parts),
            "url": "공정거래위원회 가맹사업 정보공개 오픈API (15110241/15125490)",
            "published": f"{int(r.year)}",
            "source_name": "공정거래위원회",
            "year": int(r.year),
            "brand_id": r.brand_id,
        })
    return docs


def _hq_financial_documents(cfg: dict) -> list[dict]:
    """가맹본부 재무 → 자연어 문서 (감사보고서 + 정보공개서 열람분).

    ⭐ 이 층이 코퍼스의 성격을 바꾼다. 기존 disclosure 문서는 우리가 이미 가진 패널 수치를
    문장으로 다시 쓴 것이라, 심사메모가 그것을 '인용'해도 사실상 자기 인용이었다
    (자체 감사 지적). 반면 이 문서의 근거는 **외부 1차 문서**다 — 금감원에 접수된
    감사보고서(접수번호로 원문 URL 이 특정된다), 또는 공정위 가맹사업정보제공시스템의
    정보공개서. 어느 쪽이든 심사역이 클릭해 직접 확인할 수 있다.

    ⚠️ 두 원천을 **문서마다 구분해 적는다.** 정보공개서에서 온 수치에 '금융감독원
       전자공시'라는 이름표를 붙이면, 확인하러 간 심사역이 그 문서를 찾지 못한다.
       근거 문서에서 출처를 틀리는 것은 수치를 틀리는 것과 같은 급의 잘못이다.
    """
    fp = Path(cfg["paths"]["processed"]) / "hq_financials.parquet"
    pp = Path(cfg["paths"]["processed"]) / "panel.parquet"
    if not fp.exists() or not pp.exists():
        return []
    fin = pd.read_parquet(fp)
    panel = pd.read_parquet(pp)
    if "company_name" not in panel.columns or fin.empty:
        return []
    # 옛 스냅샷에 신규 컬럼이 없을 수 있다 — itertuples 가 AttributeError 로 죽지 않게 채운다
    for c in ("assets", "liabilities", "equity", "capital_stock", "revenue",
              "operating_income", "net_income", "audit_opinion", "going_concern_flag",
              "rcept_no", "rcept_dt", "source", "corp_code"):
        if c not in fin.columns:
            fin[c] = None

    from src.dart import norm_corp
    latest = panel.sort_values("year").drop_duplicates("brand_id", keep="last").copy()
    latest["key"] = latest["company_name"].map(norm_corp)
    # 한 본부가 여러 브랜드를 가질 수 있다 — 브랜드마다 문서를 만들어야 브랜드 검색에 걸린다
    link = latest[["key", "brand_id", "brand_name", "company_name"]].dropna(subset=["key"])

    def won(v, unit=1e8, suffix="억원"):
        return "미상" if pd.isna(v) else f"{v / unit:,.1f}{suffix}"

    docs: list[dict] = []
    for r in fin.merge(link, on="key", how="inner").itertuples():
        fy = int(r.fiscal_year)
        parts = [f"{fy}년 결산 기준 {r.company_name}(브랜드: {r.brand_name}) 가맹본부 재무"]
        parts.append(f"자산총계 {won(r.assets)}")
        parts.append(f"부채총계 {won(r.liabilities)}")
        parts.append(f"자본총계 {won(r.equity)}")
        parts.append(f"매출액 {won(r.revenue)}")
        parts.append(f"영업이익 {won(r.operating_income)}")
        parts.append(f"당기순이익 {won(r.net_income)}")
        if pd.notna(r.equity) and pd.notna(r.assets) and r.assets:
            parts.append(f"자기자본비율 {100 * r.equity / r.assets:.1f}%")
        if pd.notna(r.equity) and r.equity <= 0:
            parts.append("**자본총계가 음수로 완전자본잠식 상태**")
        elif pd.notna(r.equity) and pd.notna(r.capital_stock) and r.equity < r.capital_stock:
            parts.append("**자본총계가 자본금에 미달하여 부분 자본잠식**")
        if pd.notna(r.operating_income) and r.operating_income < 0:
            parts.append("영업손실 발생")
        if getattr(r, "audit_opinion", None):
            parts.append(f"감사의견 {r.audit_opinion}")
        # NaN(= 확인 못 함)은 truthy 라 notna 검사가 필요하다 — 없으면 결측이 경보로 뒤집힌다
        gc = getattr(r, "going_concern_flag", None)
        if pd.notna(gc) and gc:
            parts.append("**감사보고서에 계속기업 관련 중요한 불확실성 기재**")

        # ⚠️ 이 표에는 이제 두 원천이 섞여 있다 — DART 감사보고서와 **공정위 정보공개서
        #    열람분**(사람이 직접 받아 온 것). 출처를 구분하지 않으면 정보공개서에서 온
        #    수치가 '금융감독원 전자공시'라는 이름표를 달고 나가는데, 그것은 심사역이
        #    클릭해 확인할 수 없는 거짓 출처다. 근거 문서에서 출처를 틀리는 것은
        #    수치를 틀리는 것과 같은 종류의 잘못이다.
        src_txt = str(getattr(r, "source", "") or "")
        rc = getattr(r, "rcept_no", None)
        if src_txt == _IFRMP_TAG:
            url = "https://franchise.ftc.go.kr/mnu/00013/program/userRqst/list.do"
            src_name = "공정거래위원회 가맹사업정보제공시스템 정보공개서(열람)"
            doc_id = f"ifrmp:{r.key}:{fy}:{r.brand_id}"
        else:
            url = (f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rc}" if rc
                   else "https://opendart.fss.or.kr (전자공시)")
            src_name = f"금융감독원 전자공시 {src_txt or '감사보고서'}"
            doc_id = f"dart:{r.corp_code}:{fy}:{r.brand_id}"
        docs.append({
            "doc_id": doc_id,
            "brand_name": r.brand_name,
            "source_type": "hq_financial",
            "text": ", ".join(parts),
            "url": url,
            "published": str(getattr(r, "rcept_dt", "") or fy),
            "source_name": src_name,
            "year": fy,
            "brand_id": r.brand_id,
        })
    return docs


def _ifrmp_documents(cfg: dict) -> list[dict]:
    """정보공개서 원문 섹션 → 문서 (공정위 정식 키가 있을 때만 채워진다).

    본문 API 는 데모키에서 고정 1건만 주므로(src/ifrmp.py 참조) 지금은 보통 빈 리스트다.
    파일이 없으면 조용히 건너뛰되, 있으면 **법 위반·소송·계약해지 사유 원문**이 들어온다.
    """
    p = Path(cfg["paths"]["processed"]) / "ifrmp_sections.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    docs = []
    for r in df.itertuples():
        txt = _clean(str(r.text))[:2000]
        if len(txt) < 40:
            continue
        docs.append({
            "doc_id": f"ifrmp:{r.jngIfrmpSn}:{r.section}",
            "brand_name": r.brandNm,
            "source_type": "ifrmp",
            "text": f"[{r.brandNm} 정보공개서 — {r.title}] {txt}",
            "url": (f"https://franchise.ftc.go.kr/api/viewer.do?jngIfrmpSn={r.jngIfrmpSn}"),
            "published": str(r.yr),
            "source_name": "공정거래위원회 정보공개서",
            "year": int(r.yr),
            "brand_id": pd.NA,
        })
    return docs


def build_corpus(cfg: dict, max_disclosure_rows: int | None = 20000) -> pd.DataFrame:
    """뉴스 + 공시 + 본부 감사보고서 + 정보공개서 원문 코퍼스 구축 및 저장."""
    docs = (_news_documents(cfg) + _disclosure_documents(cfg, max_disclosure_rows)
            + _hq_financial_documents(cfg) + _ifrmp_documents(cfg))
    if not docs:
        log.warning("RAG 코퍼스 문서 0건 — 뉴스/패널 산출물 확인 필요")
        return pd.DataFrame()
    corpus = pd.DataFrame(docs)
    if "brand_id" not in corpus.columns:
        corpus["brand_id"] = pd.NA
    corpus = corpus.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    out = Path(cfg["paths"]["processed"]) / CORPUS_FILE
    corpus.to_parquet(out, index=False)
    stats = {
        "n_documents": len(corpus),
        "by_source_type": corpus["source_type"].value_counts().to_dict(),
        "n_brands_covered": int(corpus["brand_name"].nunique()),
        "retrieval": "TF-IDF (word 1-2gram + char_wb 2-4gram) + cosine similarity",
        "note": "외부 임베딩 API 미사용 — 키 없이 완전 재현 가능",
    }
    (Path(cfg["paths"]["outputs"]) / STATS_FILE).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("RAG 코퍼스: %d문서 (%s), 브랜드 %d개 → %s",
             len(corpus), stats["by_source_type"], stats["n_brands_covered"], out.name)
    return corpus


# ---------------------------------------------------------------------------
# 색인 / 검색
# ---------------------------------------------------------------------------

INDEX_FORMAT = 1  # 저장 포맷 버전 (구조가 바뀌면 올린다)


class RagIndex:
    """TF-IDF 색인 + 브랜드 필터 검색.

    ⚠️ 저장은 **객체 피클이 아니라 상태(state) 직렬화**로 한다.
       이유(실측 결함): 인스턴스를 그대로 피클하면 클래스가 정의된 모듈 경로가 함께 박힌다.
       색인을 `python -m src.rag` 로 만들면 그 경로가 `__main__.RagIndex` 가 되어,
       다른 진입점(Streamlit 대시보드, `python -m src.memo_llm`)에서 로드할 때
       `Can't get attribute 'RagIndex'` 로 **조용히 실패**한다 — 즉 실제 화면에서는
       RAG 근거 인용이 죽어 있었다. 상태만 저장하면 진입점과 무관하게 로드된다.
    """

    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus.reset_index(drop=True)
        texts = self.corpus["text"].fillna("").tolist()
        # 한국어: 형태소 분석기 없이도 동작하도록 단어 + 문자 n-gram 결합
        #
        # 메모리 (배포 제약): 기본 설정으로 만들면 색인이 91MB 파일 / 263MB 상주였다.
        # 무료 호스팅의 메모리 한도에 걸리므로 두 가지로 줄인다.
        #   · dtype=float32  — TF-IDF 값에 float64 정밀도가 필요 없다. 정확히 절반.
        #   · char 피처 상한 — char n-gram 이 nnz 의 85%를 차지한다. 상위 빈도만 남긴다.
        # 두 조치 모두 검색 품질을 실측으로 확인한 뒤 적용했다 (tools/check_rag_quality.py).
        self.vec_word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                        dtype=np.float32)
        self.vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                        min_df=3, sublinear_tf=True,
                                        max_features=CHAR_MAX_FEATURES, dtype=np.float32)
        self.mat_word = self.vec_word.fit_transform(texts)
        self.mat_char = self.vec_char.fit_transform(texts)

    def state(self) -> dict:
        """직렬화 가능한 상태 (우리 클래스 참조를 포함하지 않는다)."""
        return {"format": INDEX_FORMAT, "corpus": self.corpus,
                "vec_word": self.vec_word, "vec_char": self.vec_char,
                "mat_word": self.mat_word, "mat_char": self.mat_char}

    @classmethod
    def from_state(cls, state: dict) -> RagIndex:
        """state() 결과로부터 재구성 (재적합 없음)."""
        fmt = int(state.get("format", -1))
        if fmt != INDEX_FORMAT:
            raise ValueError(f"RAG 색인 포맷 불일치 (파일 {fmt} != 코드 {INDEX_FORMAT})")
        obj = cls.__new__(cls)
        obj.corpus = state["corpus"]
        obj.vec_word, obj.vec_char = state["vec_word"], state["vec_char"]
        obj.mat_word, obj.mat_char = state["mat_word"], state["mat_char"]
        return obj

    def _score(self, query: str) -> np.ndarray:
        qw = self.vec_word.transform([query])
        qc = self.vec_char.transform([query])
        sw = cosine_similarity(qw, self.mat_word).ravel()
        sc = cosine_similarity(qc, self.mat_char).ravel()
        return 0.5 * sw + 0.5 * sc

    def retrieve(self, query: str, k: int = 5, brand_name: str | None = None,
                 source_type: str | None = None, min_score: float = 0.02) -> pd.DataFrame:
        """질의에 대한 상위 k 문서. brand_name/source_type으로 사전 필터 가능."""
        s = self._score(query)
        mask = np.ones(len(self.corpus), dtype=bool)
        if brand_name:
            mask &= (self.corpus["brand_name"].astype(str) == str(brand_name)).to_numpy()
        if source_type:
            mask &= (self.corpus["source_type"].astype(str) == str(source_type)).to_numpy()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return self.corpus.head(0).assign(score=[])
        sub = s[idx]
        order = np.argsort(-sub)[:k]
        pick = idx[order]
        out = self.corpus.iloc[pick].copy()
        out["score"] = s[pick]
        return out[out["score"] >= min_score].reset_index(drop=True)


def build_index(cfg: dict, corpus: pd.DataFrame | None = None) -> RagIndex:
    if corpus is None:
        cp = Path(cfg["paths"]["processed"]) / CORPUS_FILE
        corpus = pd.read_parquet(cp) if cp.exists() else build_corpus(cfg)
    if corpus is None or corpus.empty:
        raise ValueError("RAG 코퍼스가 비어 색인을 만들 수 없습니다.")
    index = RagIndex(corpus)
    # 객체가 아니라 상태를 저장한다 (RagIndex 도크스트링의 진입점 의존 결함 참조)
    joblib.dump(index.state(), Path(cfg["paths"]["outputs"]) / INDEX_FILE)
    log.info("RAG 색인 저장: %s (문서 %d, word차원 %d, char차원 %d)", INDEX_FILE,
             len(corpus), index.mat_word.shape[1], index.mat_char.shape[1])
    return index


def load_index(cfg: dict) -> RagIndex | None:
    fp = Path(cfg["paths"]["outputs"]) / INDEX_FILE
    if not fp.exists():
        return None
    try:
        obj = joblib.load(fp)
    except Exception as e:  # 손상 파일·버전 불일치 등 — 폴백이 안전하다
        log.warning("RAG 색인 로드 실패 (%s) — `python -m src.rag` 로 재구축하세요", e)
        return None
    if isinstance(obj, RagIndex):  # 구버전(객체 피클) 파일 — 그대로 쓰되 재구축을 권고
        log.warning("RAG 색인이 구버전(객체 피클) 형식입니다 — `python -m src.rag` 로 재구축 권장")
        return obj
    if not isinstance(obj, dict):
        log.warning("RAG 색인 형식을 알 수 없습니다 (%s) — 재구축 필요", type(obj).__name__)
        return None
    try:
        return RagIndex.from_state(obj)
    except (KeyError, ValueError) as e:
        log.warning("RAG 색인 복원 실패 (%s) — `python -m src.rag` 로 재구축하세요", e)
        return None


def retrieve_evidence(cfg: dict, brand_name: str, risk_factors: list[str],
                      k_per_query: int = 3, index: RagIndex | None = None) -> list[dict]:
    """브랜드 + SHAP 위험요인을 질의로 근거 문서 검색 → 인용 가능한 리스트.

    반환 각 항목: {doc_id, source_type, text, url, published, source_name, score, query, scope}
    scope: "brand"  = 해당 브랜드 문서에서 검색된 근거
           "global" = 브랜드 문서가 없어 전체 코퍼스로 완화된 검색 결과 —
                      **타 브랜드/업계 참고 자료**이므로 소비자(메모)는 반드시 구분 표기해야 한다
                      (리뷰 지적 반영: 무표기 완화는 타 브랜드 문서를 해당 브랜드 근거처럼 보이게 함)
    """
    idx = index or load_index(cfg)
    if idx is None:
        log.warning("RAG 색인 없음 — 근거 검색 생략 (build_index 먼저 실행)")
        return []
    seen: set[str] = set()
    out: list[dict] = []
    queries = [f"{brand_name} 가맹점 계약종료 폐점 위험"] + \
              [f"{brand_name} {rf}" for rf in (risk_factors or [])[:4]]
    for q in queries:
        # 브랜드 한정 검색 → 결과 없으면 전체 코퍼스로 완화 (scope로 출처 구분)
        hits = idx.retrieve(q, k=k_per_query, brand_name=brand_name)
        scope = "brand"
        if hits.empty:
            hits = idx.retrieve(q, k=k_per_query)
            scope = "global"
        for r in hits.itertuples():
            if r.doc_id in seen:
                continue
            seen.add(r.doc_id)
            out.append({"doc_id": r.doc_id, "source_type": r.source_type,
                        "text": r.text, "url": r.url, "published": r.published,
                        "source_name": r.source_name, "score": round(float(r.score), 4),
                        "query": q, "scope": scope})
    out.sort(key=lambda d: -d["score"])
    return out


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    corpus = build_corpus(cfg)
    if corpus.empty:
        return
    index = build_index(cfg, corpus)
    # 데모: 첫 뉴스 브랜드로 검색 예시 출력
    sample_brand = corpus.loc[corpus["source_type"] == "news", "brand_name"]
    if len(sample_brand):
        b = sample_brand.iloc[0]
        ev = retrieve_evidence(cfg, b, ["계약종료율 급증", "가맹점 순감"], index=index)
        print(f"\n=== RAG 검색 예시: {b} ===")
        for e in ev[:5]:
            print(f"  [{e['source_type']} {e['score']:.3f}] {e['text'][:80]}")
            print(f"     ↳ {e['source_name']} {e['published']}")


if __name__ == "__main__":
    main()
