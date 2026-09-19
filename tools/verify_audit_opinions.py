"""감사의견 재판독 — 비적정으로 기록된 최신 결산의 감사보고서를 원문으로 다시 읽는다.

왜 필요한가
    data/processed/hq_financials.parquet 의 audit_opinion 은 src/dart.py::parse_qualitative 가
    감사보고서 원문에서 자동 판독한 값이다. 초기 판독기는 문서 전체에서 '의견거절'·'한정의견'
    같은 단어를 어디서든 찾아, 기타사항의 전기 의견 언급·연결재무제표 교차 언급·내부회계관리제도
    검토 문구·주석의 피투자회사 언급까지 당기 의견으로 읽었다(최신연도 본부의 약 15%가 비적정).
    판독기는 고쳤지만(감사인의 보고서 구간만 본다) 이미 만들어진 parquet 에는 옛 판독값이 남아
    있다. 게다가 DART 호출 캐시(data/raw/dart/fin/*.json)는 원문이 아니라 **판독 결과**를 저장하므로,
    캐시가 남아 있는 한 파이프라인을 다시 돌려도 옛 값이 되살아난다. 그래서 원문을 다시 받아
    새 판독기로 읽는 별도 도구를 둔다.

대상 — 소견이 실제로 읽는 행
    ① 진단 행: src/diagnosis.py 는 `_hq_frame` 으로 회계연도 ≤ (평가연도 − 1) 만 남기고
       `_hq_last` 로 그 마지막 행을 읽는다. 평가연도는 진단과 똑같이 outputs/scores_latest.csv 의
       year 첫 값에서 읽는다(없으면 scores_latest_meta.json 의 scored_year, 또는 --scored-year).
    ② 최신 행: 본부별 가장 최근 회계연도 — 다음 평가연도에 ①이 되는 행이자 화면·상담이 보여주는 행.
    두 집합의 합집합 중 audit_opinion 이 한정·부적정·의견거절인 DART 감사보고서 행(rcept_no 보유).
    (--scope diagnosis|latest 로 한쪽만 고를 수 있다. 이미 확인된 행은 --recheck 없이는 건너뛴다.)

원천
    OpenDART document.xml — 파이프라인(src/dart.py::_fetch_audit)과 같은 API·같은 호출 함수·같은 키
    (환경변수 DART_API_KEY; src.common.load_secrets 가 .env 도 읽는다). 키는 출력하지 않는다.
    ⛔ dart.fss.or.kr 웹 뷰어(/dsaf001/main.do, /report/viewer.do)는 robots.txt 가 모든 자동
       수집기에 금지하므로 쓰지 않는다.
    예의: 순차 호출, 요청 사이 --pace 초(기본 2초), 연속 실패 --max-fail 회(기본 3)면 중단.
    받은 ZIP 은 data/raw/dart/fin/docs/ (깃 추적 제외 경로)에 보관해 재실행 때 다시 받지 않는다.

ZIP 안의 파일 고르기
    _fetch_audit 는 ZIP 의 **첫 파일**만 읽는다. 첨부서류가 먼저 오면 보고서 본문을 못 보고,
    재무 수치가 통째로 비면서 첨부서류 문구가 의견으로 읽힐 수 있다(실측: '의견거절' 판독 행의
    42%가 재무 6개 계정 전부 결측 — 적정·한정 행은 0%). 여기서는 수신인까지 확인된 감사인의
    보고서가 든 파일을 고르고, 없으면 모든 파일을 이어 붙여 읽는다. 어느 파일을 썼는지 CSV 에 남긴다.

기록 (data/processed/hq_financials.parquet — 아래 관리 열 밖의 열·dtype·행 순서는 그대로)
    audit_opinion_parsed_v1        최초 판독값 보존 (재실행해도 덮어쓰지 않는다)
    going_concern_flag_parsed_v1   〃
    audit_opinion                  원문 재판독에 성공한 행만 새 값으로 교체
    going_concern_flag             〃 (같은 원문에서 함께 읽는다)
    audit_opinion_verified         True = 이번 원문 재판독으로 위 두 값을 확인함
    audit_opinion_checked_at       확인한 날짜(ISO, YYYY-MM-DD). 미확인 행은 None
    받지 못했거나 새 판독기로도 의견을 못 읽은 행은 값을 그대로 두고 verified=False 로 남긴다
    (못 읽은 것을 '적정'으로 바꾸면 확인하지 않은 것을 확인했다고 말하는 셈이다).
    쓰기 전에 원본 사본을 data/raw/dart/fin/ 에 남기고, 관리 열 밖의 모든 열이 값·dtype·순서까지
    같은지 검사한 뒤 임시 파일 → 교체로 쓴다. 검사가 실패하면 원본은 건드리지 않는다.

산출
    outputs/audit_opinion_verification.csv — 행별 old/new·상태·사용 파일·근거 발췌(사람이 대조할 수 있게)

실행
    python tools/verify_audit_opinions.py --dry-run   # 대상만 나열 (키·네트워크 불필요, 쓰기 없음)
    python tools/verify_audit_opinions.py             # 재판독 후 parquet 갱신 (DART_API_KEY 필요)
종료 코드: 0 완료 · 1 연속 실패로 중단(그때까지 확인한 행은 기록) · 2 키 없음
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.common import get_logger, load_config, load_secrets  # noqa: E402
from src.dart import (  # noqa: E402
    KEY_ENV,
    _auditor_report_span,
    _flatten,
    _get,
    _mask,
    _read_opinion,
    parse_qualitative,
)

log = get_logger("verify_audit_opinions")

NONCLEAN = ("한정", "부적정", "의견거절")
V1_OPINION, V1_GC = "audit_opinion_parsed_v1", "going_concern_flag_parsed_v1"
VERIFIED, CHECKED_AT = "audit_opinion_verified", "audit_opinion_checked_at"
# 이 도구가 값을 바꿀 수 있는 열. 나머지 열은 한 글자도 바뀌면 안 된다(쓰기 전 검사).
MANAGED = ("audit_opinion", "going_concern_flag", V1_OPINION, V1_GC, VERIFIED, CHECKED_AT)
PACE_MIN = 1.0              # --pace 를 이보다 줄이지 않는다 (공용 API 에 대한 예의)


# ---------------------------------------------------------------------------
# 대상 선정
# ---------------------------------------------------------------------------
def scored_year(out_dir: Path) -> int | None:
    """진단이 쓰는 평가연도 — diagnose_cohort 와 같은 출처(scores_latest.csv 의 year 첫 값)."""
    csv = out_dir / "scores_latest.csv"
    if csv.exists():
        s = pd.read_csv(csv, encoding="utf-8-sig")
        if "year" in s.columns and len(s):
            return int(s["year"].iloc[0])
    meta = out_dir / "scores_latest_meta.json"
    if meta.exists():
        with contextlib.suppress(ValueError, KeyError, TypeError):
            return int(json.loads(meta.read_text(encoding="utf-8"))["scored_year"])
    return None


def select_targets(hq: pd.DataFrame, year: int | None, scope: str, *, recheck: bool) -> pd.DataFrame:
    """재확인할 행 — 원본 인덱스(= parquet 행 위치)를 그대로 유지해 돌려준다."""
    if hq.duplicated(["key", "fiscal_year"]).any():
        # 중복이 있으면 _hq_last 가 무엇을 읽을지 정해지지 않는다(정렬이 안정적이지 않다)
        raise SystemExit("hq_financials 에 (key, fiscal_year) 중복이 있어 진단 행을 특정할 수 없습니다.")
    flags = pd.DataFrame(index=hq.index)
    flags["in_overall_latest"] = hq.index.isin(hq.groupby("key")["fiscal_year"].idxmax())
    if year is None:
        flags["in_diagnosis_row"] = False
    else:
        sub = hq[hq["fiscal_year"] <= year - 1]
        flags["in_diagnosis_row"] = hq.index.isin(sub.groupby("key")["fiscal_year"].idxmax())
    pick = {"both": flags["in_overall_latest"] | flags["in_diagnosis_row"],
            "latest": flags["in_overall_latest"],
            "diagnosis": flags["in_diagnosis_row"]}[scope]
    mask = pick & hq["audit_opinion"].isin(NONCLEAN) & hq["rcept_no"].notna()
    if not recheck and VERIFIED in hq.columns:
        mask &= ~hq[VERIFIED].fillna(False).astype(bool)
    cols = ["key", "corp_code", "fiscal_year", "rcept_no", "rcept_dt", "source",
            "audit_opinion", "going_concern_flag"]
    return hq.loc[mask, cols].join(flags.loc[mask]).sort_values(["key", "fiscal_year"])


def opinion_counts(hq: pd.DataFrame, year: int | None) -> dict[str, dict]:
    """본부당 한 행(최신 행·진단 행) 기준 감사의견 분포 — 전후 비교용."""
    out = {"latest": hq.loc[hq.groupby("key")["fiscal_year"].idxmax(), "audit_opinion"]}
    if year is not None:
        sub = hq[hq["fiscal_year"] <= year - 1]
        out["diagnosis"] = sub.loc[sub.groupby("key")["fiscal_year"].idxmax(), "audit_opinion"]
    return {k: v.fillna("None").value_counts().to_dict() for k, v in out.items()}


# ---------------------------------------------------------------------------
# 원문 수집·판독
# ---------------------------------------------------------------------------
def fetch_zip(rcept_no: str, *, key: str, cache: Path) -> tuple[bytes | None, str]:
    """document.xml ZIP → (본문, 출처). 캐시에 있으면 네트워크를 쓰지 않는다.

    ZIP 이 아닌 응답(오류 XML 등)은 **캐시하지 않는다** — 일시 오류가 영구 결과로 굳지 않게.
    실패는 예외로 올린다(메시지는 src.dart._get 에서 이미 키가 가려진다).
    """
    cp = cache / f"{rcept_no}.zip"
    if cp.exists():
        return cp.read_bytes(), "cache"
    body = _get("document.xml", {"rcept_no": rcept_no}, key=key, binary=True).content
    if body[:2] != b"PK":
        return None, "not_zip: " + _mask(body[:160].decode("utf-8", "replace"), key)
    cp.write_bytes(body)
    return body, "fetched"


def _decode(raw: bytes) -> str:
    """_fetch_audit 와 같은 순서로 복호화한다 (DART 원문은 UTF-8 이 대부분, 구형은 CP949)."""
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def read_document(blob: bytes) -> tuple[str, str, int]:
    """ZIP → (평문, 사용한 파일, 파일 수). 감사인의 보고서가 든 파일을 우선 고른다."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    names = z.namelist()
    flats = [(n, _flatten(_decode(z.read(n)))) for n in names]
    for n, flat in flats:
        if _auditor_report_span(flat, strict=True):
            return flat, n, len(names)
    return " ".join(f for _, f in flats), "+".join(names) or "(빈 ZIP)", len(names)


def reparse(flat: str) -> tuple[dict, str]:
    """고친 판독기로 읽는다 → (parse_qualitative 결과, 근거 발췌 120자)."""
    res = parse_qualitative(flat)
    span = _auditor_report_span(flat)
    _, evidence = _read_opinion(flat[span[0]:span[1]] if span else flat)
    if not evidence:                                   # 구간 밖에서 읽힌 경우(전체 문서 재적용)
        _, evidence = _read_opinion(flat)
    return res, evidence


def verify(targets: pd.DataFrame, *, key: str, cache: Path, pace: float,
           max_fail: int) -> tuple[pd.DataFrame, bool]:
    """대상 행을 순서대로 확인한다 → (행별 결과, 연속 실패로 중단했는가)."""
    rows, fails, stopped, last_net = [], 0, False, None
    for n, (idx, t) in enumerate(targets.iterrows(), 1):
        rec = {"row": idx, "key": t["key"], "corp_code": t["corp_code"],
               "fiscal_year": int(t["fiscal_year"]), "rcept_no": t["rcept_no"],
               "in_diagnosis_row": bool(t["in_diagnosis_row"]),
               "in_overall_latest": bool(t["in_overall_latest"]),
               "old": t["audit_opinion"], "new": None,
               "old_gc": t["going_concern_flag"], "new_gc": None,
               "status": "", "doc_file": "", "n_files": 0, "doc_is_first_file": None, "evidence": ""}
        if stopped:
            rec["status"] = "skipped_after_errors"
            rows.append(rec)
            continue
        cached = (cache / f"{t['rcept_no']}.zip").exists()
        if not cached and last_net is not None:
            time.sleep(max(0.0, pace - (time.monotonic() - last_net)))
        try:
            blob, how = fetch_zip(str(t["rcept_no"]), key=key, cache=cache)
        except Exception as e:                          # _get 은 재시도 후에만 예외를 올린다
            blob, how = None, f"fetch_failed: {_mask(str(e), key)[:160]}"
        if not cached:
            last_net = time.monotonic()
        if blob is None:
            fails += 1
            rec["status"] = how
            log.warning("[%d/%d] %s %s — %s", n, len(targets), t["key"], t["rcept_no"], how)
            if fails >= max_fail:
                log.error("연속 %d회 실패 — 남은 대상은 확인하지 않고 중단한다(서버 부담·차단 방지)", fails)
                stopped = True
            rows.append(rec)
            continue
        fails = 0
        try:
            flat, used, n_files = read_document(blob)
        except zipfile.BadZipFile as e:
            rec["status"] = f"bad_zip: {e}"
            rows.append(rec)
            continue
        res, evidence = reparse(flat)
        first = zipfile.ZipFile(io.BytesIO(blob)).namelist()[:1]
        rec.update(new=res["audit_opinion"], new_gc=res["going_concern_flag"], doc_file=used,
                   n_files=n_files, doc_is_first_file=bool(first) and used == first[0],
                   evidence=evidence,
                   status="verified" if res["audit_opinion"] is not None else "unreadable")
        log.info("[%d/%d] %s FY%s %s: %s → %s (계속기업 %s → %s, %s)", n, len(targets), t["key"],
                 t["fiscal_year"], t["rcept_no"], rec["old"], rec["new"], rec["old_gc"], rec["new_gc"], how)
        rows.append(rec)
    return pd.DataFrame(rows), stopped


# ---------------------------------------------------------------------------
# 기록
# ---------------------------------------------------------------------------
def apply_results(hq: pd.DataFrame, results: pd.DataFrame, today: str) -> pd.DataFrame:
    """확인된 행만 새 값으로 바꾼 사본을 돌려준다. 관리 열은 없을 때만 만든다(재실행 안전)."""
    out = hq.copy()
    if V1_OPINION not in out.columns:
        out[V1_OPINION] = out["audit_opinion"].copy()
    if V1_GC not in out.columns:
        out[V1_GC] = out["going_concern_flag"].copy()
    # 재실행 대비: src.ifrmp_web.merge_into_hq 가 정보공개서 행을 붙이면 이 열이 NaN 섞인
    # object 가 된다. 그 행들은 원문 확인 대상이 아니므로 False 로 되돌려 bool 을 유지한다.
    out[VERIFIED] = out[VERIFIED].fillna(False).astype(bool) if VERIFIED in out.columns else False
    if CHECKED_AT not in out.columns:
        out[CHECKED_AT] = None
    ok = results[results["status"] == "verified"] if len(results) else results
    for r in ok.itertuples(index=False):
        out.at[r.row, "audit_opinion"] = r.new
        out.at[r.row, "going_concern_flag"] = float(r.new_gc)
        out.at[r.row, VERIFIED] = True
        out.at[r.row, CHECKED_AT] = today
    return out


def _require(ok: bool, msg: str) -> None:
    # assert 문은 `python -O` 에서 사라진다 — 데이터 보존 검사는 항상 돌아야 하므로 직접 올린다
    if not ok:
        raise AssertionError(msg)


def check_preserved(before: pd.DataFrame, after: pd.DataFrame, touched: list[int]) -> None:
    """관리 열 밖은 값·dtype·열 순서·행 순서가 모두 같아야 한다. 어기면 AssertionError."""
    _require(len(before) == len(after) and before.index.equals(after.index), "행 수·순서가 바뀜")
    _require(list(after.columns[:len(before.columns)]) == list(before.columns), "기존 열 순서가 바뀜")
    for c in before.columns:
        if c in MANAGED:
            continue
        _require(after[c].dtype == before[c].dtype, f"{c}: dtype {before[c].dtype} → {after[c].dtype}")
        _require(after[c].equals(before[c]), f"{c}: 값이 바뀜")
    rest = ~before.index.isin(touched)
    for c in ("audit_opinion", "going_concern_flag"):
        _require(after[c].dtype == before[c].dtype, f"{c}: dtype 이 바뀜")
        _require(after.loc[rest, c].equals(before.loc[rest, c]), f"{c}: 확인하지 않은 행의 값이 바뀜")
    for v1, src in ((V1_OPINION, "audit_opinion"), (V1_GC, "going_concern_flag")):
        # 최초 판독값은 첫 실행에서 원래 값 그대로, 재실행에서는 이전 v1 그대로여야 한다
        base = before[v1] if v1 in before.columns else before[src]
        _require(after[v1].equals(base), f"{v1}: 최초 판독값 보존 실패")
    _require(after[VERIFIED].dtype == bool, f"{VERIFIED}: bool 이 아님")


def write_parquet(path: Path, before: pd.DataFrame, after: pd.DataFrame, touched: list[int],
                  backup_dir: Path, stamp: str) -> Path:
    """검사 → 원본 백업 → 임시 파일 기록 → 다시 읽어 재검사 → 교체."""
    check_preserved(before, after, touched)
    # 사본은 덮어쓰지 않는다 — 같은 초에 재실행하면 첫 실행 전의 **진짜 원본** 사본이 사라진다
    backup = backup_dir / f"hq_financials.before_verify_{stamp}.parquet"
    n = 1
    while backup.exists():
        n += 1
        backup = backup_dir / f"hq_financials.before_verify_{stamp}_{n}.parquet"
    shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp")
    after.to_parquet(tmp, index=False)
    check_preserved(before, pd.read_parquet(tmp), touched)      # 디스크 왕복 후에도 같은가
    os.replace(tmp, path)
    return backup


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def _fmt_counts(before: dict, after: dict | None = None) -> str:
    lines = []
    for scope, cnt in before.items():
        new = (after or {}).get(scope, {})
        parts = [f"{op} {cnt.get(op, 0)}" + (f"→{new.get(op, 0)}" if after else "")
                 for op in ("적정", *NONCLEAN, "None")]
        lines.append(f"  {scope:9s}: " + " · ".join(parts))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="비적정 감사의견 행을 OpenDART 원문으로 재판독한다.")
    ap.add_argument("--dry-run", action="store_true", help="대상만 나열 (키·네트워크 불필요, 쓰기 없음)")
    ap.add_argument("--scope", default="both", choices=["both", "diagnosis", "latest"])
    ap.add_argument("--scored-year", type=int, default=None, help="진단 평가연도 (기본: scores_latest.csv)")
    ap.add_argument("--pace", type=float, default=2.0, help=f"요청 간격 초 (최소 {PACE_MIN})")
    ap.add_argument("--max-fail", type=int, default=3, help="연속 실패 몇 번에 중단할지")
    ap.add_argument("--recheck", action="store_true", help="이미 확인된 행도 다시 본다")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="결과 CSV (기본: outputs/audit_opinion_verification.csv)")
    args = ap.parse_args(argv)

    cfg = load_config()
    proc, out_dir = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["outputs"])
    path = proc / "hq_financials.parquet"
    before = pd.read_parquet(path)
    year = args.scored_year or scored_year(out_dir)
    if year is None and args.scope != "latest":
        log.error("평가연도를 알 수 없습니다 — --scored-year 를 주거나 --scope latest 로 실행하세요.")
        return 2
    targets = select_targets(before, year, args.scope, recheck=args.recheck)
    counts0 = opinion_counts(before, year)
    log.info("평가연도 %s → 진단 행 = 회계연도 ≤ %s 중 본부별 최신 행", year, None if year is None else year - 1)
    log.info("재확인 대상 %d행 (진단 행 %d · 최신 행 %d · 겹침 %d) — %s", len(targets),
             int(targets["in_diagnosis_row"].sum()), int(targets["in_overall_latest"].sum()),
             int((targets["in_diagnosis_row"] & targets["in_overall_latest"]).sum()),
             targets["audit_opinion"].value_counts().to_dict())
    log.info("현재 본부당 감사의견 분포:\n%s", _fmt_counts(counts0))

    if args.dry_run:
        show = targets.drop(columns=["corp_code", "source"])
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(show.to_string())
        return 0

    load_secrets()
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        print(f"{KEY_ENV} 가 설정되어 있지 않습니다. OpenDART(https://opendart.fss.or.kr) 인증키를 환경변수나 "
              f".env 에 넣고 다시 실행하세요. 키 없이 대상만 보려면 --dry-run 을 쓰세요.", file=sys.stderr)
        return 2
    if targets.empty:
        log.info("재확인할 행이 없습니다.")
        return 0

    cache_root = Path(cfg["paths"]["raw"]) / "dart" / "fin"
    cache = cache_root / "docs"
    cache.mkdir(parents=True, exist_ok=True)
    results, stopped = verify(targets, key=key, cache=cache, pace=max(args.pace, PACE_MIN),
                              max_fail=max(1, args.max_fail))

    now = dt.datetime.now()
    after = apply_results(before, results, now.date().isoformat())
    touched = results.loc[results["status"] == "verified", "row"].astype(int).tolist()
    out_csv = args.out_csv or out_dir / "audit_opinion_verification.csv"
    results.drop(columns=["row"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    log.info("상태별: %s", results["status"].str.split(":").str[0].value_counts().to_dict())
    if not touched:
        # 확인된 행이 없으면 parquet 은 건드리지 않는다 (빈 관리 열만 붙이는 변경은 정보가 없다)
        log.warning("확인된 행이 없어 parquet 을 쓰지 않았습니다 · 행별 결과 %s", out_csv)
        return 1 if stopped else 0
    backup = write_parquet(path, before, after, touched, cache_root, now.strftime("%Y%m%d_%H%M%S"))
    log.info("본부당 감사의견 분포 (전→후):\n%s", _fmt_counts(counts0, opinion_counts(after, year)))
    log.info("기록: %s (확인 %d행) · 원본 사본 %s · 행별 결과 %s", path, len(touched), backup, out_csv)
    return 1 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
