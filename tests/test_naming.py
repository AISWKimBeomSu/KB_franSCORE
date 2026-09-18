"""명칭 규약 회귀 방지 — 산출물을 PD(부도확률)라고 부르지 않는다.

왜 테스트로 두는가 (실제로 있던 모순)
    두 문서가 정반대를 말하고 있었다.

        docs/MODEL_USE_SPEC.md  "'부도확률'·'PD'로 표기하거나 그렇게 인용" → **금지 용도**
        docs/MODEL_USE_SPEC.md  "어떤 산출물도 PD 가 아니다."
        docs/OPERATIONS.md      "PD로 쓰되 신용 부도율과는 다르다."   ← 정면 충돌

    그리고 공개 산출물 `outputs/scores_latest.csv` 의 컬럼명이 `pd_1y`·`pd_raw`·
    `pd_calibrated_step`·`pd_rank_pct` 였다. 은행 분석가가 이 CSV 를 열면 명세가
    금지한 바로 그 오독을 한다 — 그것도 우리가 이름으로 유도한 오독이다.

    라벨은 부도가 아니라 **공시 기반 구조악화 전환**이다. 실제 여신 부도·연체
    라벨을 한 건도 보지 못했으므로 PD 라고 부를 근거가 없다. 이름은 주장이다.

무엇을 막는가
    A. 옛 `pd_*` 식별자의 부활 (컬럼명·변수명·설정키·문서 표기 전부)
    B. 산출물을 PD 라고 부르는 문장 (EL 공식 포함)

무엇을 막지 않는가
    바젤 IRB 의 PD 개념을 **참조**하는 것은 정당하다 — 확률 하한 관행을 준용한
    근거를 밝히거나, 규제자본 산출에 쓸 수 없는 이유를 설명할 때가 그렇다.
    금지 대상은 '우리 숫자가 PD 다'라는 주장뿐이다.

실행: python tests/test_naming.py
"""
from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _s.reconfigure(encoding="utf-8", errors="replace")

EXTS = {".py", ".md", ".yaml", ".yml"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "data", "outputs"}
SKIP_FILES = {"test_naming.py"}          # 이 파일은 금지 표현을 예시로 담고 있다

# A. 되살아나면 안 되는 옛 식별자
OLD_TOKENS = re.compile(
    r"\bpd_(1y|raw|calibrated_step|rank_pct|min|max|floor|cap|col|column|"
    r"component|cut|definition|multiplier|pctile|stressed|sum|within_calibrator_range)\b")

# B. 산출물을 PD 라고 부르는 문장. 개념 참조와 구별되도록 **주장 형태**만 잡는다.
CLAIM_PATTERNS = [
    (re.compile(r"PD\s*로\s*쓰"),               "'PD로 쓰' — 산출물을 PD로 사용하라는 지시"),
    (re.compile(r"EL\s*=\s*exposure\s*×\s*PD"), "EL 공식에서 우리 확률을 PD 라고 표기"),
    (re.compile(r"EL_i\s*=\s*exposure_i\s*×\s*PD_i"), "EL 공식에서 우리 확률을 PD 라고 표기"),
    (re.compile(r"스트레스\s*PD"),               "'스트레스 PD' — 우리 확률에 PD 명칭 사용"),
    (re.compile(r"PD\s*대용"),                   "'PD 대용' — 대용이라도 PD 명칭을 앞세운다"),
    (re.compile(r"PD\s*컬럼"),                   "'PD 컬럼' — 산출물 컬럼을 PD 로 지칭"),
    (re.compile(r"PD\s*상[·・]?\s*하한\s*적용"),  "'PD 상·하한 적용' — 우리 확률에 PD 명칭 사용"),
]

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    RESULTS.append((ok, name, detail))


def _files():
    for p in _ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.parts) or p.name in SKIP_FILES:
            continue
        yield p


# 옛 표기를 **기록으로 인용해야만 하는** 단 한 곳.
#
# 기술설명서 §41-1 은 "무엇을 왜 개명했는가"를 남기는 절이고, 이전 이름이 없는
# 개명 대조표는 기록으로서 쓸모가 없다. 그렇다고 검사를 느슨하게 하면 진짜 잔존을
# 놓친다(이 프로젝트는 앞서 폐기 모델명이 문서에 남았을 때도 검사를 풀지 않고
# 문장을 고쳤다). 그래서 **파일 하나 × 제목 하나**로만 범위를 연다 —
# 다른 파일, 같은 파일의 다른 절에서 되살아나면 그대로 실패한다.
_HISTORY_FILE = "TECHNICAL_REPORT.md"
_HISTORY_HEADING = "PD 명칭"


def _scan(pattern_fn):
    """(파일, 줄번호, 표시문자열) 을 모으되 기록 인용 구역은 건너뛴다."""
    hits, allowed = [], 0
    for p in _files():
        heading = ""
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.startswith("#"):
                heading = line
            found = pattern_fn(line)
            if not found:
                continue
            if p.name == _HISTORY_FILE and _HISTORY_HEADING in heading:
                allowed += 1
                continue
            hits.append(f"{p.relative_to(_ROOT)}:{i} {found}")
    return hits, allowed


def test_no_old_pd_tokens() -> None:
    def find(line):
        m = OLD_TOKENS.search(line)
        return m.group(0) if m else ""

    hits, allowed = _scan(find)
    check(not hits, "옛 pd_* 식별자 없음",
          f"{len(hits)}건" + (f" 예: {hits[0]}" if hits else f" (개명 기록 인용 {allowed}건 제외)"))
    for h in hits[:10]:
        print(f"        {h}")


def test_no_pd_claim() -> None:
    def find(line):
        for pat, why in CLAIM_PATTERNS:
            if pat.search(line):
                return f"— {why}"
        return ""

    hits, allowed = _scan(find)
    check(not hits, "산출물을 PD 라 부르는 문장 없음",
          f"{len(hits)}건" + (f" 예: {hits[0]}" if hits else f" (개명 기록 인용 {allowed}건 제외)"))
    for h in hits[:10]:
        print(f"        {h}")


def test_published_columns() -> None:
    """**모든** 점수표의 컬럼명 — 코드만 고치고 CSV 를 안 돌리면 무의미하다.

    ⚠️ 이 검사는 처음에 `outputs/scores_latest.csv` **하나만** 봤다. 그래서 개명 후
       공표 트랙만 재산출하고 확장 트랙(`outputs/extended/`)을 빼먹은 것을 놓쳤고,
       배포 환경이 아직 `pd_1y` 인 확장 트랙을 골라 화면 전체가 KeyError 로 죽었다.
       로컬은 공표 트랙이 더 최신이라 통과했다 — 검사 범위가 좁으면 통과는 증거가
       아니다. 이제 저장소의 모든 scores_latest.csv 를 훑는다.
    """
    paths = sorted((_ROOT / "outputs").rglob("scores_latest.csv"))
    if not paths:
        check(True, "산출물 컬럼명", "점수표 없음 — 건너뜀")
        return
    need = {"deterioration_1y", "deterioration_step", "deterioration_rank_pct", "score_raw"}
    problems = []
    for p in paths:
        cols = [c.strip() for c in
                p.read_text(encoding="utf-8-sig").splitlines()[0].split(",")]
        bad = [c for c in cols if c.startswith("pd_")]
        missing = sorted(need - set(cols))
        if bad or missing:
            problems.append(f"{p.relative_to(_ROOT)}: 잔존 {bad} · 누락 {missing}")
    for x in problems:
        print(f"        {x}")
    check(not problems, "모든 점수표 컬럼명이 새 규약",
          f"{len(problems)}/{len(paths)}개 파일 불합격" if problems
          else f"{len(paths)}개 점수표 확인")


def test_service_reads_valid_schema() -> None:
    """화면이 실제로 읽는 파일에 화면이 요구하는 컬럼이 있는가.

    개명 사고의 본질은 '어느 파일을 서비스에 쓸지를 파일 수정시각이 정하고 있었다'는
    것이다. 지금은 공표 트랙 고정 + 스키마 검사이지만, 그 계약이 깨지지 않는지 본다.
    """
    import logging

    import pandas as pd
    p = _ROOT / "outputs" / "scores_latest.csv"
    if not p.exists():
        check(True, "화면 스키마 계약", "점수표 없음 — 건너뜀")
        return
    # streamlit 을 런타임 밖에서 import 하면 캐시 경고가 결과를 덮는다
    logging.getLogger("streamlit").setLevel(logging.CRITICAL)
    from src.views.common import SCORE_REQUIRED
    cols = set(pd.read_csv(p, encoding="utf-8-sig", nrows=1).columns)
    missing = [c for c in SCORE_REQUIRED if c not in cols]
    check(not missing, "화면 필수 컬럼 존재", f"누락 {missing}" if missing
          else f"{len(SCORE_REQUIRED)}개 전부 존재")


def test_spec_consistency() -> None:
    """사용 명세와 운영 문서가 같은 말을 하는가 — 이 둘이 어긋난 것이 사건의 시작이었다."""
    spec = (_ROOT / "docs" / "MODEL_USE_SPEC.md")
    ops = (_ROOT / "docs" / "OPERATIONS.md")
    if not (spec.exists() and ops.exists()):
        check(True, "명세·운영 문서 정합", "문서 없음 — 건너뜀")
        return
    s, o = spec.read_text(encoding="utf-8"), ops.read_text(encoding="utf-8")
    spec_forbids = "PD" in s and "가 아니다" in s
    ops_allows = bool(re.search(r"PD\s*로\s*(쓰|사용)", o))
    check(spec_forbids and not ops_allows, "명세·운영 문서가 같은 기준",
          "운영 문서가 PD 사용을 허용" if ops_allows else "두 문서 모두 PD 표기 금지")


def test_published_artifact_prose() -> None:
    """**반출되는 산출물**의 설명 문장이 명세와 같은 말을 하는가.

    지금까지 이 파일의 검사는 소스(.py/.md/.yaml)만 훑었다. 그런데 은행 분석가가
    실제로 읽는 것은 CSV·JSON 에 실려 나가는 문장이다. 실제로 pd_* → deterioration_*
    일괄 개명(156건) 때 portfolio_summary.json 의 risk_definition 이

        "악화확률은 … 확률이다. 부도확률(PD)이 사용한 것으로, 실제 부도율과 다를 수 있습니다."

    로 깨진 채 나갔다. 뜻이 통하지 않을 뿐 아니라 우리 산출물을 PD 라고 주장하는
    것처럼 읽혀 MODEL_USE_SPEC 의 금지 용도와 충돌한다. 소스 검사는 이걸 못 잡는다.

    명세가 요구하는 것은 "PD 라는 말을 쓰지 마라"가 아니라 **"숫자와 부인 문구가
    함께 다녀야 한다"**이다. 그래서 부인 문구의 존재를 계약으로 검사한다.
    """
    import json

    checks = [
        ("outputs/portfolio_summary.json", ("assumptions", "risk_definition")),
    ]
    bad = []
    for rel, path in checks:
        p = _ROOT / rel
        if not p.exists():
            continue
        node = json.loads(p.read_text(encoding="utf-8"))
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        text = str(node)
        if not text:
            bad.append(f"{rel}: {'.'.join(path)} 없음")
        elif "PD" in text and "아닙니다" not in text:
            bad.append(f"{rel}: PD 를 언급하면서 부인 문구가 없다 — {text[:60]}")
        elif OLD_TOKENS.search(text):
            bad.append(f"{rel}: 옛 식별자 잔존")
    check(not bad, "반출 산출물 설명문이 명세와 정합",
          bad[0] if bad else "악화확률 ≠ PD 부인 문구 확인")


def test_no_old_tokens_in_published_json() -> None:
    """`outputs/*.json` **전체**에 폐기된 식별자가 남아 있지 않은가.

    위 검사는 지목한 파일의 지목한 키만 본다. 그런데 실제로 새어 나간 자리는
    **아무도 지목하지 않은 파일**이었다 — `refresh_state.json` 의 `annual_scope` 에
    `pd_1y` 가 남아 있었다. 코드는 이미 `deterioration_1y` 로 고쳐졌는데, 그 코드가
    돌기 **전에** 만들어진 산출물이 그대로 커밋돼 있었던 것이다.

    소스 검사(`test_no_old_pd_tokens`)는 `outputs/` 를 SKIP_DIRS 로 제외하므로
    구조적으로 못 잡는다. 산출물은 심사자가 직접 열어 보는 파일이므로 여기서 훑는다.
    (JSON 만 본다 — 문장이 실려 나가는 것은 JSON 이고, CSV 는 컬럼명 검사가 따로 있다.)
    """
    bad = []
    for p in sorted((_ROOT / "outputs").rglob("*.json")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        m = OLD_TOKENS.search(text)
        if m:
            bad.append(f"{p.relative_to(_ROOT)}: '{m.group(0)}'")
    check(not bad, "반출 JSON 에 옛 식별자 없음",
          f"{len(bad)}건 예: {bad[0]}" if bad else "outputs/*.json 전량 확인")
    for b in bad[:10]:
        print(f"        {b}")


def test_logo_index_matches_disk() -> None:
    """화면이 띄우는 로고 = 색인이 인정한 로고인가.

    화면은 색인을 거치지 않고 브랜드명 해시로 파일을 찾아 **있으면 띄운다**.
    그래서 색인에서 철회한 로고라도 PNG 가 남으면 계속 뜬다 — 실제로 3건이
    배포 화면에 그렇게 나오고 있었다. 철회 사유는 "이 그림은 이 브랜드 것이
    아니다" 이므로 그건 틀린 로고를 띄우는 것이고, 로고가 없는 것보다 나쁘다.
    """
    import json

    from src.naver import LOGO_CACHE, LOGO_DIR, logo_file_name
    cache_p, img_dir = _ROOT / LOGO_CACHE, _ROOT / LOGO_DIR
    if not (cache_p.exists() and img_dir.exists()):
        check(True, "로고 색인·디스크 정합", "로고 자료 없음 — 건너뜀")
        return
    cache = json.loads(cache_p.read_text(encoding="utf-8"))
    disk = {p.name for p in img_dir.glob("*.png")}
    live = {logo_file_name(k) for k, v in cache.items()
            if isinstance(v, dict) and v.get("url")}
    ghost = disk - live          # 철회됐는데 화면에는 뜬다 (틀린 로고)
    phantom = live - disk        # 색인만 있고 실체가 없다 (커버리지 과대)
    check(not ghost and not phantom, "로고 색인·디스크 정합",
          f"철회 후 잔존 {len(ghost)}건 · 실체 없는 색인 {len(phantom)}건"
          if (ghost or phantom) else f"양쪽 모두 {len(disk)}건으로 일치")


def test_no_unreachable_code() -> None:
    """`return` 뒤에 남은 문장이 있는가 — 화면 하나가 통째로 죽었던 자리.

    실제로 있었던 일: 점검 큐의 우선순위 함수가 정렬 결과를 `return` 한 뒤에
    **카드 렌더링과 처리상태 저장 블록 47행이 그대로 남아 있었다.** 파이썬은
    경고하지 않고, 린트도 잡지 않았고, 화면은 예외 없이 잘 떴다 — 다만
    "상위 20건을 펼쳐 둡니다"라는 안내문 아래가 **비어 있었다.**
    사이드바에서 유일하게 '업무'라고 이름 붙은 화면이 그 상태였다.

    조용히 죽는 결함은 눈으로 못 잡는다. 구조로 잡는다.
    """
    import ast
    bad: list[str] = []
    for p in sorted((_ROOT / "src").rglob("*.py")) + sorted((_ROOT / "tools").rglob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as exc:                     # 문법 오류는 다른 테스트의 몫
            bad.append(f"{p.relative_to(_ROOT)}: 파싱 실패 {exc}")
            continue
        for node in ast.walk(tree):
            for field in ("body", "orelse", "finalbody"):
                blk = getattr(node, field, None)
                if not isinstance(blk, list):
                    continue
                for i, stmt in enumerate(blk):
                    exits = ast.Return | ast.Raise | ast.Continue | ast.Break
                    if isinstance(stmt, exits) and blk[i + 1:]:
                        bad.append(
                            f"{p.relative_to(_ROOT)}:{blk[i + 1].lineno} — "
                            f"{stmt.lineno}행 {type(stmt).__name__} 뒤 "
                            f"{len(blk) - i - 1}개 문장이 실행되지 않는다")
    check(not bad, "도달 불가 코드 없음",
          bad[0] if bad else "src·tools 전 함수에서 0건")
    for b in bad[:10]:
        print(f"        {b}")


def main() -> int:
    for fn in (test_no_old_pd_tokens, test_no_pd_claim, test_published_columns,
               test_service_reads_valid_schema, test_spec_consistency,
               test_published_artifact_prose, test_logo_index_matches_disk,
               test_no_unreachable_code, test_no_old_tokens_in_published_json):
        try:
            fn()
        except Exception as exc:
            check(False, fn.__name__, f"예외: {exc}")
    fails = [r for r in RESULTS if not r[0]]
    for ok, name, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}")
    print(f"=== {len(RESULTS) - len(fails)}/{len(RESULTS)} passed ===")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
