"""공개 데모 가명 처리(src/public.py) — 공개 배포에서 실명 브랜드 등급이 새지 않는가.

모형 사용 명세 §3 은 실명 브랜드 등급의 대외 공표를 금지한다. 공개 데모가 그 금지를 지키는지는
눈으로 확인할 수 없다(화면 수십 개 × 브랜드 1,500개). 그래서 사본 전체와 공개 모드 화면을 실명
목록으로 훑는다.
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src import public

ROOT = Path(__file__).resolve().parent.parent
OUT, PROC = ROOT / "outputs", ROOT / "data" / "processed"
# 누구나 아는 짧은 이름 — 3글자 이하라 전수 검사에서 빠지므로 따로 못 박는다
FAMOUS = ("빽다방", "메가엠지씨커피", "이디야", "컴포즈커피", "교촌치킨", "비비큐", "투다리", "굽네",
          "국수나무", "인생냉면", "달리는커피", "크린토피아", "파리바게뜨", "투썸플레이스", "롯데리아")


def _panel() -> pd.DataFrame:
    return pd.DataFrame({
        "brand_id": ["B1", "B1", "B2", "B3"], "year": [2023, 2024, 2024, 2024],
        "brand_name": ["알파치킨", "알파치킨(ALPHA)", "베타커피", "감마분식"],
        "company_name": ["(주)알파푸드", "(주)알파푸드", "(주)알파푸드", "감마에프앤비(주)"],
        "industry_mid": ["치킨", "치킨", "커피", "분식"], "industry_major": ["외식"] * 4})


def test_masker_is_deterministic_and_unique():
    a, b = public.Masker(_panel()), public.Masker(_panel())
    assert a.pid == b.pid and a.pname == b.pname
    assert len(set(a.pname.values())) == 3 and a.pname["B1"].startswith("치킨 ")
    assert all(v.startswith("P") for v in a.pid.values())


def test_text_scrubs_own_names_siblings_company_and_search_term():
    m = public.Masker(_panel(), terms={"B1": "알파치킨"})
    me, sib = m.pname["B1"], m.pname["B2"]
    s = m.text("네이버에서 '알파치킨'을 검색한 양이 줄었습니다. 같은 본부 '베타커피'도 · (주)알파푸드 부채비율", "B1")
    assert "알파" not in s and "베타" not in s
    assert me in s and sib in s and m.company["(주)알파푸드"] in s
    assert m.text("2024-07-01") == "2024-07-01"                  # 숫자·날짜는 그대로


def test_hq_join_key_survives_masking():
    from src.dart import norm_corp
    m = public.Masker(_panel())
    hq = pd.DataFrame({"key": [norm_corp("(주)알파푸드"), norm_corp("없는회사")], "fiscal_year": [2023, 2023],
                       "rcept_no": ["20240101000001", "x"], "corp_code": ["001", "002"], "equity": [1.0, 2.0]})
    out = m.frame(hq)
    panel = m.frame(_panel())
    company = panel.loc[panel["brand_id"] == m.pid["B1"], "company_name"].iloc[-1]
    assert list(out["key"]) == [norm_corp(company)]              # 매칭 안 되는 본부 행은 뺀다
    assert "rcept_no" not in out.columns and "corp_code" not in out.columns


def test_evidence_drops_identifiers():
    m = public.Masker(_panel())
    ev = m.evidence(json.dumps({"rcept_no": "20240101000001", "keyword": "알파치킨", "rate": 0.3}), "B1")
    assert "rcept_no" not in ev and "알파치킨" not in ev and "0.3" in ev


def test_public_examples_avoid_the_largest_brands():
    """가맹점 수 최상위 브랜드는 숫자만으로 실명이 짐작된다 — 공개 모드 예시는 중간 규모에서 고른다."""
    sizes = [3300, 2600, 2300, 1700, 1400, *range(480, 100, -20), 60, 40, 20]
    s = pd.DataFrame({"brand_id": [f"P{i:05d}" for i in range(len(sizes))],
                      "brand_name": [f"브랜드 {i:03d}" for i in range(len(sizes))],
                      "industry_mid": [("치킨", "커피", "한식")[i % 3] for i in range(len(sizes))],
                      "grade": [("FS1", "FS2", "FS3")[i % 3] for i in range(len(sizes))],
                      "n_stores": sizes})
    lo, hi = public.SHOWCASE_STORES
    n = s.set_index("brand_name")["n_stores"]
    names = public.example_names(s, 3)
    assert len(names) == 3 and all(lo <= n[x] <= hi for x in names)
    pair = public.same_industry_pair_rows(s)
    assert len(pair) == 2 and pair["industry_mid"].nunique() == 1
    assert pair["n_stores"].between(lo, hi).all()
    assert set(public.same_industry_pair(s)) == set(pair["brand_name"])


def _texts(path: Path) -> str:
    if path.suffix in (".parquet", ".csv"):
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype=str,
                                                                                 encoding="utf-8-sig")
        vals = set()
        for c in df.columns:
            if df[c].dtype == object:
                vals.update(str(v) for v in df[c].dropna().unique())
        return "\n".join(vals)
    return path.read_text(encoding="utf-8")


def _scan(text: str, names: set[str]) -> collections.Counter:
    idx: dict[str, list[str]] = collections.defaultdict(list)
    for n in names:
        idx[n[:2]].append(n)
    hits: collections.Counter = collections.Counter()
    for i in range(len(text) - 1):
        for n in idx.get(text[i:i + 2], ()):
            if text.startswith(n, i):
                hits[n] += 1
    return hits


@pytest.mark.skipif(not (PROC / "panel_full.parquet").exists(), reason="산출물 없음")
def test_mirror_contains_no_real_ids_names_or_companies(tmp_path):
    dest = public.build(tmp_path / "m", OUT, PROC)
    pf = pd.read_parquet(PROC / "panel_full.parquet",
                         columns=["brand_name", "company_name", "industry_mid", "industry_major"])
    labels = set(pf["industry_mid"].dropna().astype(str)) | set(pf["industry_major"].dropna().astype(str))
    long_names = {str(n).strip() for n in pf["brand_name"].dropna()
                  if len(str(n).replace(" ", "")) >= 4 and not str(n).replace(" ", "").isdigit()} - labels
    comps = {str(c).strip() for c in pf["company_name"].dropna()
             if len(str(c).replace(" ", "")) >= 4 and not str(c).replace(" ", "").isdigit()}   # '1250' 같은 이름
    targets = long_names | comps | set(FAMOUS)
    leaks: collections.Counter = collections.Counter()
    for f in dest.rglob("*"):
        if f.is_file():
            t = _texts(f)
            assert "BRD_" not in t and "JNG_" not in t, f"실제 브랜드 ID 가 남았다: {f.name}"
            leaks.update(_scan(t, targets))
    assert not leaks, f"실명이 사본에 남았다: {leaks.most_common(10)}"
    # 숫자는 그대로 — 등급·확률이 원본과 같아야 공개 데모가 같은 분석을 보여 준다
    a = pd.read_csv(OUT / "scores_latest.csv", encoding="utf-8-sig")
    b = pd.read_csv(dest / "outputs" / "scores_latest.csv", encoding="utf-8-sig")
    assert a["grade"].value_counts().to_dict() == b["grade"].value_counts().to_dict()
    assert sorted(a["deterioration_1y"].round(10)) == sorted(b["deterioration_1y"].round(10))


@pytest.mark.skipif(not (PROC / "panel_full.parquet").exists(), reason="산출물 없음")
def test_public_app_screens_show_no_real_brand_names(tmp_path):
    """공개 모드로 앱을 띄워 전 화면·예시 실행 결과를 훑는다 (별도 프로세스 — 설정 캐시가 섞이지 않게)."""
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        from streamlit.testing.v1 import AppTest
        APP = {str(ROOT / "src" / "app.py")!r}
        FAMOUS = {FAMOUS!r}
        MENUS = ["FRANSCORE", "브랜드 탐색·비교", "일괄 조회", "점검 큐", "여신 포트폴리오", "등급 검증",
                 "AI 상담", "서비스 소개"]
        CLICK = {{"일괄 조회": "예시로 바로 실행", "등급 검증": "가상 자료로 시연",
                  "여신 포트폴리오": "예시 장부로 해 보기"}}

        def texts(at):
            out = []
            for kind in ("markdown", "caption", "info", "warning", "error", "success", "title",
                         "header", "subheader"):
                out += [str(e.value) for e in getattr(at, kind)]
            out += [str(m.label) + str(m.value) for m in at.metric]
            out += [str(b.label) for b in at.button]
            for d in list(at.dataframe) + list(at.table):
                out.append(d.value.to_string())
            return "\\n".join(out)

        bad = []
        for menu in MENUS:
            at = AppTest.from_file(APP, default_timeout=180)
            at.session_state["nav_view"] = menu
            at.run()
            assert not at.exception, (menu, [e.message for e in at.exception])
            if menu in CLICK:
                b = next(x for x in at.button if x.label == CLICK[menu])
                b.click().run()
                assert not at.exception, (menu, [e.message for e in at.exception])
            t = texts(at)
            bad += [(menu, n) for n in FAMOUS if n in t]
            if menu == "FRANSCORE":
                assert "가명" in t, "공개 모드 고지가 없다"
        assert not bad, bad
        print("clean")
    """)
    env = {**os.environ, "FRANSCORE_PUBLIC_DEMO": "1", "FRANSCORE_PUBLIC_DIR": str(tmp_path),
           "FRANSCORE_QUEUE_STORE": "session"}
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=900,
                       cwd=str(ROOT))
    assert r.returncode == 0 and "clean" in r.stdout, (r.stdout[-1500:], r.stderr[-3000:])
