"""개인정보 최소수집 회귀 방지 — 개인 성명 필드가 원본 스냅샷에 다시 들어오지 않게 한다.

왜 테스트로 두는가
    공정위 15125467(brand_master) 응답에는 jnghdqrtrsRprsvNm(가맹본부 대표자 성명)이 섞여
    오고, 그것이 data/raw/brand_master_*.json 84,368행에 그대로 커밋돼 있었다. 코드 참조가
    0건이라 산출물 영향 없이 수집 단계에서 버리도록 바꿨고(src/collect.py PERSONAL_FIELDS),
    기존 스냅샷에서도 지웠다. 재수집·수작업 추가로 다시 들어오면 여기서 막는다.

무엇을 검사하는가
    A. data/raw 아래 어떤 파일에도 필드명이 없다 (.gz 는 메모리에서 풀어 바이트 스캔 — 1초 미만)
    B. drop_personal_fields 가 표본 레코드에서 필드만 지우고 나머지 키·값·순서를 보존한다
    C. 수집 경로(collect_all)가 쓴 스냅샷(.json / .json.gz)에 필드가 없다 — 네트워크 없이 모의 응답으로

실행: python -m pytest -q tests/test_privacy.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import collect
from src.collect import PERSONAL_FIELDS, drop_personal_fields

RAW = _ROOT / "data" / "raw"
# 공공데이터 표준 용어에서 'Rprsv…Nm' 은 대표자명이다 — 필드명이 달라도 같은 꼴이면 잡는다.
TOKENS = [*sorted(PERSONAL_FIELDS), "RprsvNm"]

# brand_master 응답 1행과 같은 모양의 표본 (값은 가공)
SAMPLE = {
    "jngBizCrtraYr": "2025", "brandMnno": "BRD_00000000000", "jnghdqrtrsMnno": "HQ_00000000000",
    "brno": "0000000000", "crno": "", "jnghdqrtrsRprsvNm": "홍길동", "brandNm": "표본브랜드",
    "indutyLclasNm": "외식", "indutyMlsfcNm": "치킨", "majrGdsNm": "치킨",
    "jngBizStrtDate": "20200101", "corpNm": "(주)표본",
}


def test_raw_snapshots_have_no_personal_fields() -> None:
    scanned: list[str] = []
    hits: list[str] = []
    for p in sorted(RAW.rglob("*")):
        if not p.is_file() or p.suffix == ".png":
            continue
        data = p.read_bytes()
        if p.suffix == ".gz":
            data = gzip.decompress(data)
        rel = p.relative_to(RAW).as_posix()
        scanned.append(rel)
        hits += [f"{rel}: {t}" for t in TOKENS if t.encode() in data]
    # 스냅샷이 없어서 '통과'하는 일이 없도록 — 원래 필드가 있던 파일을 실제로 훑었는지 확인
    assert any(s.startswith("brand_master_") for s in scanned), "brand_master 스냅샷을 찾지 못했다"
    assert not hits, f"data/raw 에 개인 성명 필드가 남아 있다: {hits[:10]}"


def test_drop_personal_fields_sample_record() -> None:
    out = drop_personal_fields(SAMPLE)
    assert not PERSONAL_FIELDS & out.keys()
    assert list(out) == [k for k in SAMPLE if k not in PERSONAL_FIELDS]   # 순서 보존
    assert all(out[k] == SAMPLE[k] for k in out)                           # 값 보존
    assert "jnghdqrtrsRprsvNm" in SAMPLE                                   # 입력은 변경하지 않는다


@pytest.mark.parametrize("service", ["brand_master", "brand_region_direct"])   # .json / .json.gz
def test_collect_all_writes_snapshot_without_personal_fields(
        service: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch_page(url: str, params: dict, cfg: dict) -> dict:
        return {"resultCode": "00", "totalCount": 1, "items": [dict(SAMPLE)]}

    monkeypatch.setattr(collect, "_fetch_page", fake_fetch_page)
    monkeypatch.delenv("FRANSCORE_TEST_NO_KEY", raising=False)
    cfg = {"paths": {"raw": tmp_path}, "collect": {"service_key_env": "FRANSCORE_TEST_NO_KEY"}}

    out = collect.collect_all(cfg, services=[service], years=[2025])
    (snap,) = out[service]
    assert snap.parent == tmp_path                                         # 실데이터 경로는 건드리지 않는다
    raw = gzip.decompress(snap.read_bytes()) if snap.suffix == ".gz" else snap.read_bytes()
    assert not [t for t in TOKENS if t.encode() in raw]
    assert json.loads(raw)["items"] == [drop_personal_fields(SAMPLE)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
