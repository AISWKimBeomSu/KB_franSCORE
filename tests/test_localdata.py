"""지방행정 인허가 → 브랜드 월간 신호(src/localdata.py) — 네트워크 없이 검증한다.

무엇을 지키는가
    A. 매칭 정밀도: 2026-09 실제 인허가 파일에서 관측한 사업장명(표기 변형 + 브랜드 토큰을 품은
       다른 가게)으로 만든 라벨 표본에서 오탐이 없고 재현율이 충분하다
    B. 지점 표기 떼기(strip_branch)
    C. 적재: 2026 파일 형식(39열 한글 헤더) — UTF-8·CP949(실제 파일 인코딩) 모두, API JSON 봉투
    D. 월간 집계: 개업·폐업·말 활성 점포, 같은 주소 재등록(명의 이전) 분리, 부분 월 제외
    E. 신호: 전년 동기 대비 악화/개선/유지/판단보류
    F. 파일 내려받기: WAF 헤더, 크기 한도(50MB) 초과 시 쓰지 않음

픽스처 tests/fixtures/localdata_sample.csv 는 실제 파일 헤더에 **실제 브랜드명 + 가상 지점명**을 넣은
합성 레코드다(개방자치단체코드 9990000·관리번호 TEST-… 는 존재하지 않는 값).

실행: python -m pytest -q tests/test_localdata.py
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import localdata as ld

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "localdata_sample.csv"

# scores_latest.csv(2024) 에서 옮긴 실제 브랜드 행 — 파일이 다시 생성돼도 테스트가 흔들리지 않게 고정한다.
BRANDS = pd.DataFrame([
    ("BRD_20151039", "60계", "치킨", 665),
    ("BRD_20220226", "PLUS 82(플러스 82)", "커피", 146),
    ("BRD_20120100533", "가장맛있는족발", "한식", 384),
    ("BRD_20180070", "감성커피", "커피", 283),
    ("BRD_20130100052", "공차", "음료 (커피 외)", 836),
    ("BRD_20140589", "교동짬뽕", "중식", 19),
    ("BRD_20080600002", "교촌치킨", "치킨", 1361),
    ("BRD_20090100195", "국수나무", "분식", 236),
    ("NAME:국수나무@외식", "국수나무", "분식", 173),
    ("BRD_20080200030", "굽네치킨", "치킨", 1154),
    ("BRD_20080100109", "김가네", "분식", 377),
    ("BRD_20080100085", "나뚜루", "아이스크림/빙수", 19),
    ("BRD_20160549", "노랑강정", "기타 외식", 43),
    ("BRD_20130176", "노랑통닭", "치킨", 751),
    ("BRD_20130100180", "뉴욕버거", "기타 외식", 161),
    ("BRD_20191099", "더맛있는족발보쌈", "한식", 54),
    ("BRD_20140722", "더벤티", "커피", 1230),
    ("BRD_20080100257", "더카페", "커피", 145),
    ("BRD_20201348", "떡볶이 참 잘하는 집, 떡참", "분식", 82),
    ("BRD_20080100039", "뚜레쥬르", "제과제빵", 1307),
    ("BRD_20080100157", "맘스터치", "패스트푸드", 1444),
    ("BRD_20220528", "맘스터치 피자앤치킨", "패스트푸드", 37),
    ("BRD_20230769", "맘스피자(MOM'S PIZZA)", "패스트푸드", 104),
    ("BRD_20161159", "매머드익스프레스(Mammoth Express)", "커피", 742),
    ("BRD_20140725", "매머드커피(Mammoth coffee)", "커피", 21),
    ("NAME:맥도날드@외식", "맥도날드", "패스트푸드", 55),
    ("BRD_20160628", "메가엠지씨커피(MEGA MGC COFFEE)", "커피", 3325),
    ("BRD_20231726", "메가혼밥", "한식", 58),
    ("BRD_20150930", "바른치킨", "치킨", 185),
    ("BRD_20080500015", "배스킨라빈스", "아이스크림/빙수", 1706),
    ("BRD_20080100243", "본죽", "한식", 575),
    ("BRD_20141221", "본죽&비빔밥", "한식", 1150),
    ("BRD_20190933", "봉구통닭", "치킨", 7),
    ("BRD_20080100324", "북창동순두부", "한식", 79),
    ("BRD_20130100283", "불닭발땡초동대문엽기떡볶이", "기타 외식", 650),
    ("BRD_20080100655", "비에이치씨(BHC)", "치킨", 2228),
    ("BRD_20090100502", "빽다방", "커피", 1712),
    ("BRD_20150828", "샐러디", "기타 외식", 333),
    ("BRD_20212607", "샐러디아", "한식", 70),
    ("BRD_20161219", "역전할머니맥주1982", "주점", 963),
    ("BRD_20130452", "유가네", "한식", 201),
    ("BRD_20150449", "유가네한우곰탕", "한식", 65),
    ("BRD_20080100014", "이디야커피", "커피", 2562),
    ("BRD_20080500031", "이삭토스트", "분식", 894),
    ("BRD_20080300013", "지코바양념치킨", "치킨", 743),
    ("BRD_20140942", "짝태&노가리", "주점", 69),
    ("BRD_20080200106", "처갓집양념치킨", "치킨", 1254),
    ("BRD_20141250", "컴포즈커피(COMPOSE COFFEE)", "커피", 2649),
    ("BRD_20130100320", "쿡1015", "기타 외식", 33),
    ("BRD_20180244", "텐퍼센트스페셜티커피", "커피", 814),
    ("BRD_20080100482", "토프레소", "커피", 98),
    ("BRD_20080200019", "투다리", "주점", 1292),
    ("BRD_20211363", "퍼스트커피", "커피", 27),
    ("BRD_20191192", "프랭크버거", "패스트푸드", 622),
    ("BRD_20230023", "프랭크커핀바", "커피", 35),
    ("BRD_20080100397", "피자헛", "피자", 260),
    ("BRD_20080100308", "한솥", "한식", 811),
    ("BRD_20120100672", "한스(HANS)", "제과제빵", 40),
    ("BRD_20120100121", "달.콤(dal.komm)", "커피", 76),
    ("BRD_20161241", "비비큐(BBQ)", "치킨", 2316),
    ("BRD_20080100199|JNG_000064", "할리스", "커피", 402),
    ("BRD_20151155", "쥬씨(JUICY)", "음료 (커피 외)", 175),
    ("BRD_20080100209", "본가", "한식", 16),
    ("BRD_20140974", "미술관", "주점", 9),
    ("BRD_20080100162", "기소야", "일식", 26),
    ("BRD_20160858", "두찜", "한식", 603),
], columns=["brand_id", "brand_name", "industry_mid", "n_stores"])

# (사업장명, 기대 브랜드 또는 None, 주소) — 전부 2026-09 실제 인허가 파일에서 관측한 표기다.
LABELED: list[tuple[str, str | None, str]] = [
    # 표기 변형 (정답이 있는 것)
    ("메가엠지씨커피 화서꽃뫼사거리점", "메가엠지씨커피(MEGA MGC COFFEE)", ""),
    ("메가MGC커피 수원지동시장점", "메가엠지씨커피(MEGA MGC COFFEE)", ""),
    ("메가커피 수원영통점", "메가엠지씨커피(MEGA MGC COFFEE)", ""),
    ("컴포즈 커피(COMPOSE COFFEE) 대치중앙점", "컴포즈커피(COMPOSE COFFEE)", ""),
    ("컴포즈 롯데캐슬스타점", "컴포즈커피(COMPOSE COFFEE)", ""),
    ("이디야아주대점", "이디야커피", ""),
    ("이디야커피 강남율현점", "이디야커피", ""),
    ("(주)이디야이비에스점", "이디야커피", ""),
    ("이디야 해운대우동점", "이디야커피", "부산광역시 해운대구 우동"),       # '우동'은 여기서 지명
    ("BHC(비에이치씨치킨) 송파구청점", "비에이치씨(BHC)", ""),
    ("bhc수원역점", "비에이치씨(BHC)", ""),
    ("비에이치씨(압구정로데오점)", "비에이치씨(BHC)", ""),
    ("B.H.C정자2지구점", "비에이치씨(BHC)", ""),
    ("가락시장역점 빽다방", "빽다방", ""),
    ("센텀SH밸리점빽다방", "빽다방", ""),
    ("빽다방(방이시장점)", "빽다방", ""),
    ("비알코리아(주)배스킨라빈스 수원역사3호점", "배스킨라빈스", ""),
    ("베스킨라빈스 센텀신세계2호점", "배스킨라빈스", ""),
    ("수원홈플러스베스킨라빈스", "배스킨라빈스", ""),
    ("맘스터치 수원북문점", "맘스터치", ""),
    ("맘스터치피자앤치킨 헬리오시티점", "맘스터치 피자앤치킨", ""),     # 긴 브랜드가 이긴다
    ("맘스피자 수원매교역점", "맘스피자(MOM'S PIZZA)", ""),
    ("교촌치킨매탄1호점", "교촌치킨", ""),
    ("교촌(정자2호점)", "교촌치킨", ""),
    ("까페뚜레쥬르 강남대로점", "뚜레쥬르", ""),
    ("뚜레주르 위윌락유 로열시어터", "뚜레쥬르", ""),
    ("투다리세류점", "투다리", ""),
    ("투다리(삼성점)", "투다리", ""),
    ("처갓집 양념치킨 수서점", "처갓집양념치킨", ""),
    ("더 벤티 영통중앙점", "더벤티", ""),
    ("호매실굽네치킨", "굽네치킨", "경기도 수원시 권선구 호매실동"),
    ("율전점 굽네치킨", "굽네치킨", ""),
    ("본죽&비빔밥 해운대백병원점", "본죽&비빔밥", ""),
    ("본죽 앤 비빔밥 방이시장점", "본죽&비빔밥", ""),
    ("본죽엔비빔밥 화서블루밍점", "본죽&비빔밥", ""),
    ("본죽&비빔밥카페 수원성빈센트병원점", "본죽&비빔밥", ""),
    ("본죽반송점", "본죽", "부산광역시 해운대구 반송동"),          # 두 글자 키 + 붙인 지점명 → 주소 지명으로 확인
    ("역전할머니맥주 신사가로수길점", "역전할머니맥주1982", ""),
    ("60계 치킨&맥주 서울잠실점", "60계", ""),
    ("(주)공차코리아(잠실롯데몰점)", "공차", ""),
    ("공차대치역점", "공차", ""),
    ("텐퍼센트커피 방이역점", "텐퍼센트스페셜티커피", ""),
    ("텐퍼센트 마천역점", "텐퍼센트스페셜티커피", ""),
    ("한솥도시락(강남대로 역삼점)", "한솥", ""),
    ("한솥 송파나루역점", "한솥", ""),
    ("지코바 숯불양념 치킨", "지코바양념치킨", ""),
    ("매머드 익스프레스 테헤란점(MAMMOTH EXPRESS)", "매머드익스프레스(Mammoth Express)", ""),
    ("매머드커피 역삼GFC점 (125호)", "매머드커피(Mammoth coffee)", ""),
    ("불닭발땡초동대문엽기떡볶이 수원터미널점", "불닭발땡초동대문엽기떡볶이", ""),
    ("동대문엽기떡볶이 언주역점", "불닭발땡초동대문엽기떡볶이", ""),
    ("한국맥도날드(유) 잠실역점", "맥도날드", ""),
    ("샐러디아 삼성점", "샐러디아", ""),
    ("샐러디 아주대점", "샐러디", ""),                    # '샐러디아…'로 읽으면 안 된다
    ("짝태&노가리연무점", "짝태&노가리", ""),
    ("떡볶이참잘하는집 떡참수원권선점", "떡볶이 참 잘하는 집, 떡참", ""),
    ("더(The) 맛있는 족발 보쌈 선릉본점", "더맛있는족발보쌈", ""),
    ("가장맛있는족발 수원율전점", "가장맛있는족발", ""),
    ("유가네닭갈비 반송점", "유가네", ""),
    ("유가네한우곰탕", "유가네한우곰탕", ""),
    ("김가네 영통클래시아점", "김가네", ""),
    ("진영푸드(주) 피자헛 해운대 신도시점", "피자헛", ""),
    ("쿡(cook)1015", "쿡1015", ""),
    ("국수나무 수원광교점", "국수나무", ""),
    ("노랑통닭 장산역점", "노랑통닭", ""),
    ("달콤커피 역삼에클라트점", "달.콤(dal.komm)", ""),
    ("비비큐치킨 논현점", "비비큐(BBQ)", ""),
    ("BBQ치킨 수원연무점", "비비큐(BBQ)", ""),
    ("오금점비비큐(BBQ)", "비비큐(BBQ)", ""),
    ("할리스커피 교통회관점", "할리스", ""),
    ("(주)공영식품 기소야삼성점", "기소야", ""),                    # 가맹점 운영법인 이름 뒤의 간판
    ("진영푸드(주)피자헛센텀점", "피자헛", ""),
    ("두마리찜닭 두찜수원고등점", "두찜", ""),
    ("비알코리아(주)훼이버릿디바이배스킨라빈스강남점", "배스킨라빈스", ""),   # 법인 표기에 맞닿은 토막 안쪽
    # 알려진 누락 — 재현율이 1이 아님을 그대로 둔다
    ("써티원베스킨라빈스", "배스킨라빈스", ""),
    ("롯데 나뚜루 아이스크림", "나뚜루", ""),
    # 어려운 음성: 브랜드 토큰을 품었지만 그 브랜드 가게가 아니다
    ("메가홀덤", None, ""),
    ("주식회사 메가젠임플란트 메가젠타워지점", None, ""),
    ("스무디킹영통메가박스점", None, ""),
    ("한솥집", None, ""),
    ("한솥밥", None, ""),
    ("처갓집야식", None, ""),
    ("처갓집한식", None, ""),
    ("장충동처갓집족발", None, ""),
    ("김가네 황태덕장", None, ""),
    ("김가네탕수육전문점", None, ""),
    ("빽다방빵연구소 수원영통점", None, ""),
    ("빽보이피자 수원권선점", None, ""),
    ("프랭크서울 경찰병원역점", None, ""),
    ("메종 프랭크(Maison Frank)", None, ""),
    ("라미나 더카페", None, ""),
    ("더카페라운지 앤 페르케노 수원성대역점", None, ""),
    ("더 베라즈 수원영통구청점", None, ""),
    ("(주)와인앤파트너스배라짜노", None, ""),
    ("교촌필방(한시적)", None, ""),
    ("올바른치킨", None, ""),
    ("신교동짬뽕", None, ""),
    ("가토프레소(Gattopresso)", None, ""),
    ("퍼스트 커피랩 문정점", None, ""),
    ("감성디저트", None, ""),
    ("엘에이 북창동 순두부타워펠리스점", None, ""),
    ("10뉴욕버거(Ten New York Burgger)", None, ""),
    ("봉구비어 신천역점", None, ""),
    ("이삭분식", None, ""),
    ("이삭튀김", None, ""),
    ("미니스톱매탄이삭점", None, ""),
    ("플러스(PLUSYAMI)", None, ""),
    ("Han's Coffee(한스커피)", None, ""),
    ("맘스푼일식", None, ""),
    ("벤티버거 거여송파파크센트럴점", None, ""),
    ("달콤상점", None, "서울특별시 송파구 석촌동"),                 # 두 글자 일반어 브랜드('달.콤')
    ("달콤바삭 마트본점", None, "서울특별시 송파구 석촌동"),
    ("달콤카페", None, "경기도 수원시 장안구 율전동"),
    ("고릴라 BBQ스테이크", None, ""),
    ("맥반BBQ", None, ""),
    ("그린나래비비큐펍(BBQ Pub)", None, "부산광역시 해운대구 우동"),
    ("주시브로스(JuicyBros) 압구정점", None, ""),
    ("백순대 본가새맛 인계점", None, ""),
    ("빈스빈스커피 경기도 미술관점(한시적)", None, ""),
]


@pytest.fixture(autouse=True)
def _quiet_log(monkeypatch):
    """모듈 로거는 outputs/pipeline.log 에 쓴다 — 테스트 중에는 파일을 건드리지 않게 바꿔 둔다."""
    monkeypatch.setattr(ld, "log", logging.getLogger("tests.localdata"))


def _labeled_frame() -> pd.DataFrame:
    return pd.DataFrame({"name": [n for n, _, _ in LABELED], "road_address": [a for _, _, a in LABELED],
                         "jibun_address": ""})


# ---------------------------------------------------------------------------
# A. 매칭 정밀도
# ---------------------------------------------------------------------------
def test_labeled_sample_is_large_and_has_hard_negatives():
    assert len(LABELED) >= 40
    assert sum(exp is None for _, exp, _ in LABELED) >= 20


def test_matching_precision_and_recall_on_labeled_sample():
    out = ld.match_brands(_labeled_frame(), BRANDS, keep_unmatched=True)
    got = [g if isinstance(g, str) else None for g in out["brand_name"]]
    tp = fp = fn = 0
    errors = []
    for (name, exp, _), g in zip(LABELED, got, strict=True):
        if g is not None and g == exp:
            tp += 1
        if g is not None and g != exp:
            fp += 1
            errors.append(f"오탐 {name!r} → {g!r} (정답 {exp!r})")
        if exp is not None and g != exp:
            fn += 1
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    assert precision == 1.0, "\n".join(errors)
    assert recall >= 0.9, f"재현율 {recall:.3f} (TP {tp}, FN {fn})"


def test_every_positive_except_known_misses_is_matched():
    out = ld.match_brands(_labeled_frame(), BRANDS, keep_unmatched=True)
    known_misses = {"써티원베스킨라빈스", "롯데 나뚜루 아이스크림"}
    wrong = [(n, exp, g) for (n, exp, _), g in zip(LABELED, out["brand_name"], strict=True)
             if exp is not None and n not in known_misses and g != exp]
    assert not wrong


def test_duplicate_brand_names_go_to_the_larger_brand():
    out = ld.match_brands(pd.DataFrame({"name": ["국수나무 수원광교점"]}), BRANDS)
    assert out.loc[0, "brand_id"] == "BRD_20090100195"          # 236곳 > 173곳


def test_match_rules_are_reported():
    names = ["투다리(삼성점)", "투다리세류점", "호매실굽네치킨", "교촌치킨매탄1호점"]
    addrs = ["", "", "경기도 수원시 권선구 호매실동", ""]
    out = ld.match_brands(pd.DataFrame({"name": names, "road_address": addrs}), BRANDS)
    assert out["match_rule"].tolist() == ["exact", "prefix_weak", "glued", "prefix"]


# ---------------------------------------------------------------------------
# B. 지점 표기 떼기
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("raw", "expected"), [
    ("메가엠지씨커피 강남역점", "메가엠지씨커피"),
    ("교촌치킨 매탄1호점", "교촌치킨"),
    ("투다리(삼성점)", "투다리"),
    ("가락시장역점 빽다방", "빽다방"),
    ("(주)교촌 역삼점", "교촌"),
    ("BHC 잠실 야구장 2호점(1루)", "BHC 잠실 야구장"),
    ("홍콩반점0410 수원역점", "홍콩반점0410"),
    ("한솥밥 강남점", "한솥밥"),
    ("교동반점", "교동반점"),                      # '반점'은 지점이 아니라 업종
    ("교촌치킨매탄1호점", "교촌치킨매탄1호점"),        # 붙여 쓴 것은 목록 없이 끊지 않는다
])
def test_strip_branch(raw, expected):
    assert ld.strip_branch(raw) == expected


# ---------------------------------------------------------------------------
# C. 적재
# ---------------------------------------------------------------------------
def test_load_fixture_utf8():
    rec = ld.load_records(FIXTURE, service="일반음식점")
    assert list(rec.columns) == ld.CANONICAL
    assert len(rec) == 30
    row = rec.set_index("mng_no").loc["TEST-0003"]
    assert row["name"] == "메가커피 해오름점" and row["status_code"] == "03"
    assert row["opened_on"] == pd.Timestamp("2022-01-05") and row["closed_on"] == pd.Timestamp("2026-07-03")
    assert pd.isna(rec.set_index("mng_no").loc["TEST-0001", "closed_on"])
    assert rec["data_updated_at"].max() == pd.Timestamp("2026-09-18 22:18:00")
    assert (rec["service"] == "일반음식점").all()


def test_load_cp949_file_with_service_inferred_from_name(tmp_path):
    """실제 파일은 헤더가 UTF-8 이라 말해도 바이트는 CP949 다 — 같은 내용이 같게 읽혀야 한다."""
    cp949 = tmp_path / "식품_휴게음식점_가상구.csv"
    cp949.write_bytes(FIXTURE.read_text(encoding="utf-8").encode("cp949"))
    a = ld.load_records(FIXTURE)
    b = ld.load_records(cp949)
    assert (b["service"] == "휴게음식점").all()
    cols = ["mng_no", "name", "opened_on", "closed_on", "status_code"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_overlapping_files_are_deduplicated(tmp_path):
    region = tmp_path / "general_restaurants_9990000.csv"
    whole = tmp_path / "general_restaurants_all.csv"
    shutil.copy(FIXTURE, region)
    shutil.copy(FIXTURE, whole)
    assert len(ld.load_records([region, whole])) == 30


def test_load_api_json_envelope(tmp_path):
    body = {"response": {"header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE"},
                         "body": {"items": {"item": [
                             {"OPN_ATMY_GRP_CD": "9990000", "MNG_NO": "API-1", "BPLC_NM": "투다리(가상점)",
                              "LCPMT_YMD": "20190101", "CLSBIZ_YMD": "20260702", "SALS_STTS_CD": "03",
                              "SALS_STTS_NM": "폐업", "ROAD_NM_ADDR": "서울특별시 가상구 가상로 1",
                              "DAT_UPDT_PNT": "20260918221800"}]},
                             "numOfRows": 100, "pageNo": 1, "totalCount": 1}}}
    p = tmp_path / "rest_cafes_api.json"
    p.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    rec = ld.load_records(p)
    assert rec.loc[0, "name"] == "투다리(가상점)" and rec.loc[0, "service"] == "휴게음식점"
    assert rec.loc[0, "closed_on"] == pd.Timestamp("2026-07-02")
    assert rec.loc[0, "data_updated_at"] == pd.Timestamp("2026-09-18 22:18:00")


def test_fixture_end_to_end_matching():
    matched = ld.match_brands(ld.load_records(FIXTURE, service="일반음식점"), BRANDS)
    by_no = matched.set_index("mng_no")["brand_name"].to_dict()
    assert len(matched) == 20
    assert by_no["TEST-0002"] == by_no["TEST-0003"] == "메가엠지씨커피(MEGA MGC COFFEE)"
    assert by_no["TEST-0012"] == "빽다방" and by_no["TEST-0014"] == "본죽&비빔밥" and by_no["TEST-0015"] == "본죽"
    assert by_no["TEST-0020"] == "배스킨라빈스"
    for neg in ("TEST-0021", "TEST-0022", "TEST-0023", "TEST-0024", "TEST-0026", "TEST-0027", "TEST-0028"):
        assert neg not in by_no


# ---------------------------------------------------------------------------
# D. 월간 집계
# ---------------------------------------------------------------------------
def _store(bid, opened, closed=None, addr="", no=None, updated=None):
    return {"brand_id": bid, "brand_name": bid, "mng_no": no or f"{bid}-{opened}-{closed}",
            "opened_on": pd.Timestamp(opened), "closed_on": pd.Timestamp(closed) if closed else pd.NaT,
            "status_code": "03" if closed else "01", "road_address": addr, "jibun_address": "",
            "data_updated_at": pd.Timestamp(updated) if updated else pd.NaT}


def test_monthly_flows_counts_and_active_stores():
    rows = [
        _store("A", "2026-01-15"),                                    # 1월 개업
        _store("A", "2025-12-01", "2026-02-10"),                      # 2월 폐업
        _store("A", "2010-05-05", "2026-03-31"),                      # 3월 말일 폐업
        _store("A", "2020-01-01", "2026-04-05", addr="가상로 1"),       # 4월: 같은 주소 같은 날 재등록
        _store("A", "2026-04-05", addr="가상로 1", updated="2026-05-02"),
    ]
    flows = ld.monthly_brand_flows(pd.DataFrame(rows), months=4)
    f = flows.set_index("month")
    assert [str(m) for m in f.index] == ["2026-01", "2026-02", "2026-03", "2026-04"]   # 5월은 부분 월 → 제외
    assert f["n_open"].tolist() == [1, 0, 0, 0]
    assert f["n_close"].tolist() == [0, 1, 1, 0]
    assert f["n_transfer"].tolist() == [0, 0, 0, 1]
    assert f["n_active_end"].tolist() == [4, 3, 2, 2]


def test_transfer_detection_can_be_disabled():
    rows = [_store("A", "2020-01-01", "2026-04-05", addr="가상로 1"), _store("A", "2026-04-05", addr="가상로 1")]
    f = ld.monthly_brand_flows(pd.DataFrame(rows), months=1, end_month="2026-04", transfer_days=0)
    assert f[["n_open", "n_close", "n_transfer", "n_active_end"]].iloc[0].tolist() == [1, 1, 0, 1]


def test_fixture_flows():
    matched = ld.match_brands(ld.load_records(FIXTURE, service="일반음식점"), BRANDS)
    flows = ld.monthly_brand_flows(matched, months=24)
    assert str(flows["month"].max()) == "2026-08" and str(flows["month"].min()) == "2024-09"
    mega = flows[flows["brand_id"] == "BRD_20160628"].set_index("month")
    assert mega.loc[pd.Period("2026-06", "M"), "n_open"] == 1
    assert mega.loc[pd.Period("2026-07", "M"), "n_close"] == 1
    assert mega.loc[pd.Period("2026-08", "M"), "n_active_end"] == 3
    compose = flows[flows["brand_id"] == "BRD_20141250"].set_index("month").loc[pd.Period("2026-05", "M")]
    assert (compose["n_close"], compose["n_open"], compose["n_transfer"]) == (0, 0, 1)


# ---------------------------------------------------------------------------
# E. 신호
# ---------------------------------------------------------------------------
def _brand_with_closures(bid, n_stores, prev_closures, cur_closures):
    """2015년에 n_stores 곳 개업, 전년 6~8월에 prev 곳·올해 6~8월에 cur 곳 폐업."""
    rows = []
    for i in range(n_stores):
        closed = None
        if i < prev_closures:
            closed = f"2025-0{6 + i % 3}-15"
        elif i < prev_closures + cur_closures:
            closed = f"2026-0{6 + i % 3}-15"
        rows.append(_store(bid, "2015-01-01", closed, no=f"{bid}-{i}"))
    return rows


def test_closure_signal_trends():
    rows = (_brand_with_closures("WORSE", 40, 2, 12) + _brand_with_closures("STABLE", 40, 2, 2)
            + _brand_with_closures("BETTER", 60, 12, 1) + _brand_with_closures("TINY", 5, 0, 3))
    flows = ld.monthly_brand_flows(pd.DataFrame(rows), months=24, end_month="2026-08")
    sig = ld.closure_signal(flows).set_index("brand_id")
    assert list(sig.columns[:5]) == ["close_3m", "open_3m", "net_3m", "close_rate_3m", "trend"]
    assert sig.loc["WORSE", "trend"] == "악화"
    assert sig.loc["WORSE", "close_3m"] == 12 and sig.loc["WORSE", "close_3m_prev_year"] == 2
    assert sig.loc["WORSE", "net_3m"] == -12
    assert sig.loc["WORSE", "close_rate_3m"] == pytest.approx(12 / 38)
    assert sig.loc["STABLE", "trend"] == "유지"
    assert sig.loc["BETTER", "trend"] == "개선"
    assert sig.loc["TINY", "trend"] == "판단보류"
    assert str(sig.loc["WORSE", "as_of_month"]) == "2026-08"


def test_zero_last_year_is_not_an_alarm_by_itself():
    """전년 0건 → 올해 3건(점포 18곳)은 경보감이 아니다 — 전년 건수의 표본오차를 반영해야 한다."""
    flows = ld.monthly_brand_flows(pd.DataFrame(_brand_with_closures("SMALL", 18, 0, 3)), months=24,
                                   end_month="2026-08")
    assert ld.closure_signal(flows).loc[0, "trend"] == "유지"


def test_short_history_is_withheld():
    flows = ld.monthly_brand_flows(pd.DataFrame(_brand_with_closures("WORSE", 40, 2, 12)), months=6,
                                   end_month="2026-08")
    sig = ld.closure_signal(flows)
    assert sig.loc[0, "trend"] == "판단보류" and "개월" in sig.loc[0, "reason"]


# ---------------------------------------------------------------------------
# F. 파일 내려받기 (모의)
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, status=200, headers=None, chunks=()):
        self.status_code = status
        self.headers = headers or {}
        self._chunks = list(chunks)

    def iter_content(self, size):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Get:
    def __init__(self, *responses):
        self.calls = []
        self._responses = list(responses)

    def __call__(self, url, params=None, headers=None, **kw):
        self.calls.append({"url": url, "params": params, "headers": dict(headers or {})})
        return self._responses.pop(0)


def test_download_writes_file_with_waf_headers(monkeypatch, tmp_path):
    data = FIXTURE.read_bytes()
    get = _Get(_Resp(200), _Resp(200, {"Content-Type": "text/csv;charset=UTF-8",
                                       "Content-Length": str(len(data))}, [data]))
    monkeypatch.setattr(ld.requests, "get", get)
    out = ld.download_csv("rest_cafes", tmp_path, org_code="3220000")
    assert out.name == "rest_cafes_3220000.csv" and out.read_bytes() == data
    dl = get.calls[1]
    assert dl["url"].endswith("/file/download/rest_cafes/info") and dl["params"] == {"orgCode": "3220000"}
    assert dl["headers"]["Referer"] == f"{ld.FILE_BASE}/rest_cafes/info"
    assert "Mozilla" in dl["headers"]["User-Agent"]


def test_download_refuses_files_over_the_limit(monkeypatch, tmp_path):
    big = _Resp(200, {"Content-Type": "text/csv", "Content-Length": str(697_606_263)}, [b"x"])
    monkeypatch.setattr(ld.requests, "get", _Get(_Resp(200), big))
    with pytest.raises(RuntimeError, match="한도"):
        ld.download_csv("general_restaurants", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_stops_streaming_past_the_limit(monkeypatch, tmp_path):
    stream = _Resp(200, {"Content-Type": "text/csv"}, [b"a" * 600, b"b" * 600])
    monkeypatch.setattr(ld.requests, "get", _Get(_Resp(200), stream))
    with pytest.raises(RuntimeError, match="한도"):
        ld.download_csv("bakeries", tmp_path, max_bytes=1000)
    assert list(tmp_path.iterdir()) == []


def test_download_reports_waf_block_and_rate_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(ld.requests, "get", _Get(_Resp(200), _Resp(403, {"Content-Type": "text/html"})))
    with pytest.raises(RuntimeError, match="WAF"):
        ld.download_csv("bakeries", tmp_path)
    monkeypatch.setattr(ld.requests, "get", _Get(_Resp(429)))
    with pytest.raises(RuntimeError, match="429"):
        ld.download_csv("bakeries", tmp_path)
    with pytest.raises(ValueError):
        ld.download_csv("hospitals", tmp_path)


def test_bulk_one_month_closures_are_flagged_for_review_not_called_deterioration():
    """한 달에 영업 점포의 30%·20곳 이상이 한꺼번에 폐업 처리되면 '악화'가 아니라 '확인 필요'.

    실측: 직전 달 영업 184곳 브랜드가 한 달에 132곳 폐업 — 3개월 추세 검정이 다룰 사건이 아니다(원인은 원자료로).
    """
    import pandas as pd

    from src import localdata as L
    months = pd.period_range("2024-09", "2026-08", freq="M")
    rows = []
    active = 150
    for m in months:
        close = 120 if str(m) == "2026-06" else 1
        active = active - close + 1
        rows.append({"brand_id": "B", "brand_name": "가상", "month": m, "n_open": 1, "n_close": close,
                     "n_transfer": 0, "n_active_end": active})
    sig = L.closure_signal(pd.DataFrame(rows)).set_index("brand_id")
    assert sig.loc["B", "trend"] == L.CHECK
    assert "한꺼번에" in sig.loc["B", "reason"] and "2026-06" in sig.loc["B", "reason"]
