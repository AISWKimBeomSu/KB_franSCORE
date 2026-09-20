<div align="center">

<img src="assets/screenshots/franscore-main.png" alt="FranSCORE — 프랜차이즈 브랜드 리스크 현황 화면" width="860"/>

# 🏦 FranSCORE — 가맹점 여신에 '브랜드' 리스크 축을 더하다

**KB국민은행 제8회 AI Challenge 출품작 · 개인 프로젝트**

은행은 가맹점주를 **차주 단위**로 심사하지만, 부실은 **브랜드 단위로 함께** 옵니다.<br/>
공정위 가맹사업 공시와 가맹본부 감사보고서로 **외식 프랜차이즈 1,521개 브랜드의 1년 내 구조악화 위험**을 산출하고,<br/>
심사 · 협약 · 사후관리 · 편중 관리 · 모형 검증 업무 화면으로 연결한 **2선 리스크 관리 서비스**입니다.

<br/>

[![Live Demo](https://img.shields.io/badge/▶_Live_Demo-FFCC00?style=for-the-badge&logoColor=black)](https://kb-franscore.streamlit.app)
[![Results](https://img.shields.io/badge/검증_결과_상세-26221E?style=for-the-badge)](docs/RESULTS.md)
[![Technical Report](https://img.shields.io/badge/기술설명서-26221E?style=for-the-badge)](docs/TECHNICAL_REPORT.md)

[![CI](https://github.com/AISWKimBeomSu/KB_franSCORE/actions/workflows/ci.yml/badge.svg)](https://github.com/AISWKimBeomSu/KB_franSCORE/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python_3.13-0f172a?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-0f172a?style=flat&logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-0f172a?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0f172a?style=flat&logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0f172a?style=flat)
![pandas](https://img.shields.io/badge/pandas-0f172a?style=flat&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-0f172a?style=flat&logo=plotly&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-0f172a?style=flat&logo=googlegemini&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0f172a?style=flat&logo=pytest&logoColor=white)

<sub>공시로 공개된 자료로 만든 <b>참고 지표</b>입니다 — 신용평가가 아니며 등급을 인용·재배포하지 마십시오. 실명을 쓸 수 없는 자리에서는 사이드바의 <b>가명 모드</b>를 켜면 브랜드 이름만 가명으로 바뀝니다.</sub>

</div>

---

## 📌 Overview

> **🔍 문제** — 가맹점주 대출은 개인사업자 CB·대안정보(매출·상권·업력·결제정보)로 심사하는데, 이 정보는 모두 **점포 단위**입니다. 같은 브랜드 가맹점들이 함께 무너지는 **브랜드 공통 위험**은 어느 심사 자료에도 없습니다.
>
> **🛠️ 해결** — 공정위 가맹사업 공시(오픈API **7종**)·DART 감사보고서·네이버 검색수요로 브랜드×연도 패널을 만들고, **1년 내 구조악화 확률 → 고정 등급(FS1~FS3) → 근거 소견**을 산출해 업무 화면과 문서로 제공합니다.
>
> **📈 결과** — 2022년 자료로 맞춘 등급을 2023년에 그대로 적용했을 때 실제 악화율이 **안정 2.3% → 관찰 8.8% → 주의 19.5%** 로 갈립니다(시점 밖 검증). 워크포워드 표본 밖 AUC **0.707**.
>
> **📅 공시 사이는 월 단위로** — 공시는 연 1회·1~2년 늦습니다. 그 사이는 전국 지자체 인허가 **301만여 건**을 브랜드에 이어 만든 **월별 폐점 신호**로 메웁니다. 공시와의 일치도는 폐점률 순위상관 **0.734**, 점포 순증감 **0.843**(2024년 실적)입니다.
>
> **🏦 은행 자료로 이어지게** — 행내 여신 장부를 올리면 편중·꼬리손실을 그 장부로, 연체 자료를 올리면 등급 검증을 그 자리에서 다시 계산합니다. 차주 사업자번호로 국세청 휴·폐업도 함께 확인합니다.

### 누가, 어느 업무에서 쓰는가

| 여신 업무 | 실무자의 질문 | FranSCORE 가 주는 것 |
|---|---|---|
| **신규 취급** (심사) | "이 가맹점주가 속한 브랜드, 괜찮은가?" | 브랜드 상세 · 품의서용 **참고의견서** · 신청 목록 **일괄 조회**(차주 사업자 휴·폐업 동시 확인) |
| **협약대출** (브랜드 관리) | "어느 본부와 협약하고, 언제 재심사하나?" | **브랜드 탐색·비교** — 업종·등급·규모·중대 신호로 거르고 후보 2~4개를 나란히 비교 |
| **사후관리** (조기경보) | "올해 무엇부터 점검하나?" | **점검 큐** — 중대 신호 → 1년 내 악화 위험 × 가맹점 수 순서, 담당·메모·**변경 이력**·엑셀 반출. 공시 사이에 폐점이 몰린 브랜드(**월별 폐점 신호**)는 등급과 무관하게 올라옵니다 |
| **포트폴리오** (편중 관리) | "어느 브랜드에 쏠렸고, 꼬리위험은 어디서 오나?" | **여신 포트폴리오** — 행내 장부 업로드, 담보유형별 LGD, 브랜드 상관을 반영한 꼬리손실(Euler ES) 실시간 계산 |
| **모형 검증** (심사 반영 전) | "이 등급이 우리 은행 연체를 설명하나?" | **등급 검증** — 연체 자료를 올리면 취급 당시 등급 기준으로 서열성·내부등급 대비 추가 정보를 판정 |

> 차주 심사를 **대체하지 않습니다.** 차주 심사가 구조적으로 볼 수 없는 **브랜드 축 하나를 더합니다.**
> 등급은 부도확률(PD)이 아니라 브랜드 공시 지표의 구조악화 확률이며, 여신 승인·거절, 한도·금리 결정에 쓰지 않습니다.

---

## 💡 Why brand? — 문제 정의

- **시장이 크고 약하다** — 자영업자 대출 **1,095.5조원**, 숙박·음식업 사업자대출 연체율 **3.02%** 로 부동산업(1.10%)의 약 3배(한국은행 『금융안정보고서』 2025.12). 가맹점 379,739개 · 브랜드 13,725개(공정위, 2025).
- **차주 부도 = 담보 가치 동시 소멸** — 가맹사업법상 가맹점사업자의 파산·부도는 가맹계약 **즉시해지** 사유입니다. 영업권이 함께 사라져 **PD와 LGD가 같이 나빠집니다.**
- **위험이 브랜드에 모인다** — 같은 브랜드 가맹점끼리의 자산상관 **0.416**, 서로 다른 브랜드 사이 **0.005**. 한 브랜드의 모든 지역이 동시에 줄어든 빈도는 지역 독립 가정의 **21.6배**입니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/brand_correlation-dark.png">
  <img src="assets/charts/brand_correlation-light.png" alt="같은 브랜드 가맹점끼리의 자산상관 0.416, 서로 다른 브랜드 사이 0.005, 바젤 기업여신 가정 0.12~0.24" width="100%">
</picture>

- **그래서 차주를 서로 독립으로 보면 꼬리를 과소평가한다** — 60개 브랜드 · 차주 9,180명 몬테카를로에서 비예상손실(99%)이 **1.7억 → 12.7억, 7.69배** 로 커집니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/tail_loss-dark.png">
  <img src="assets/charts/tail_loss-light.png" alt="손실 분위수 95·99·99.9%에서 차주 독립 가정 대비 브랜드 상관 반영 손실이 더 크다" width="100%">
</picture>

<sub>상관은 지역별 점포 감소의 동시성으로 잰 대리지표이고, 익스포저는 공시 창업비용으로 만든 추정치입니다(실제 여신 잔액은 은행 내부 자료). 절대 금액보다 **독립 가정 대비 배수**가 메시지입니다. → [전체 표와 한계](docs/RESULTS.md)</sub>

---

## 🖥️ Service — 화면

<sub>모든 그림은 공개 데모 화면입니다 — 브랜드 실명과 실제 공시 수치 그대로입니다.</sub>

<table>
<tr>
<td width="50%"><img src="assets/screenshots/brand-detail.png" alt="브랜드 상세 화면"/><br/><b>브랜드 상세</b> — 등급·근거 소견·모형 요인(SHAP)·월별 폐점 신호·공시 추이·본부 재무. 공시 공백 보정 브랜드는 그 근거를 함께 표시</td>
<td width="50%"><img src="assets/screenshots/batch-screening.png" alt="일괄 조회 화면"/><br/><b>일괄 조회</b> — 신청 목록 업로드 → 브랜드 자동 매칭(통칭·동명 구분) → 등급·중대 신호·확인 서류, 사업자번호가 있으면 국세청 휴·폐업까지 붙여 엑셀 반출</td>
</tr>
<tr>
<td><img src="assets/screenshots/explore-compare.png" alt="브랜드 탐색·비교 화면"/><br/><b>브랜드 탐색·비교</b> — 협약 후보를 조건으로 거르고 2~4개를 나란히 비교(등급·상태·가맹점 흐름·본부 재무)</td>
<td><img src="assets/screenshots/review-queue.png" alt="점검 큐 화면"/><br/><b>점검 큐</b> — 중대 신호 우선, 1년 내 악화 위험 × 가맹점 수 순서. 월별 폐점 신호 브랜드는 등급 무관 편입. 담당 배정·처리상태·메모, 판단 당시 근거를 남기는 변경 이력</td>
</tr>
<tr>
<td><img src="assets/screenshots/portfolio.png" alt="여신 포트폴리오 꼬리손실 화면"/><br/><b>여신 포트폴리오</b> — 행내 장부를 올리면 화면 전체가 그 장부로 바뀌고, 담보유형별 LGD로 꼬리손실·브랜드별 기여를 즉시 재계산 (그림은 양식 예시 장부)</td>
<td><img src="assets/screenshots/grade-validation.png" alt="등급 검증 화면"/><br/><b>등급 검증</b> — 은행 연체 자료로 서열성·내부등급 대비 추가 정보를 판정. 시점 정합·생존 편향 방지·브랜드 군집 강건 검정 (그림은 <b>가상 시연 자료</b>)</td>
</tr>
<tr>
<td><img src="assets/screenshots/credit-memo.png" alt="참고의견서"/><br/><b>참고의견서</b> — 여신 품의서에 첨부하는 인쇄·PDF용 문서. 확인·징구 서류 체크리스트 포함</td>
<td><img src="assets/screenshots/assistant.png" alt="AI 상담 화면"/><br/><b>AI 상담</b> — 자연어 질의에 공시·감사보고서·뉴스 근거로 답변(공개 데모는 기사 제외). API 키가 없어도 표 형태로 답합니다</td>
</tr>
</table>

### 📅 공시 사이를 메우는 월별 폐점 신호

<img src="assets/screenshots/monthly-signal.png" alt="공시 추이 탭 — 연간 가맹점 수·매출·개점/종료와 그 아래 최근 24개월 월별 개점·폐점 막대" width="100%"/>

공정위 공시는 연 1회이고 1~2년 늦습니다. 그 사이 점포가 빠지는 일은 지자체 인허가에 먼저 찍힙니다. 전국 일반·휴게음식점·제과점 인허가 **301만여 건**을 사업장명으로 브랜드에 잇고, 최근 3개월 폐업이 전년 같은 때보다 **유의하게 많은지**(조건부 이항검정) 매월 판정합니다.

| 공시와 같은 것을 재는가 — 2024년 실적, 725개 브랜드 | 결과 |
|---|---|
| 폐점률 순위상관 — 인허가 폐업 vs 공시 계약종료·해지 | **0.734** |
| 순증감 순위상관 — 인허가 영업 점포 vs 공시 가맹점 수 | **0.843** |
| 매칭 완전성 — 인허가에서 찾은 영업 점포 ÷ 공시 가맹점 수 (중앙값) | **98.3%** |

- **점수에는 넣지 않습니다.** 브랜드 상세·참고의견서·탐색 필터·일괄 조회 열에 보이고, '악화'·'확인 필요' 브랜드는 **등급과 무관하게 점검 큐에 오릅니다.**
- 한 달에 영업 점포의 30% 이상이 한꺼번에 폐업 처리되면 추세 판정 대신 **'확인 필요'** 로 두고 점검 큐에 올립니다. 실제로 한 브랜드에서 184곳 중 132곳이 한 달에 폐업 처리됐는데, 여러 지자체·여러 날짜에 흩어져 있고 같은 주소 재인허가도 없어 행정 정리보다 실제 일괄 이탈에 가까웠습니다 — 원자료와 본부 사정을 먼저 확인할 대상입니다.

### 실무자가 쓰기 편하도록

- **중대 신호는 등급과 따로 본다** — 본부 계속기업 불확실성·자본잠식·정보공개서 등록취소는 사건 그 자체입니다. 모형 등급이 '안정'이어도 점검 큐 맨 앞에 올립니다.
- **악화가 이미 난 브랜드는 모형 대신 실적으로** — 올해 공시에 악화가 나타난 브랜드는 학습 표본 밖입니다. 확률 대신 **같은 사건 수 브랜드의 다음 해 재발동률**(건전 9.4% · 1건 24.1% · 2건 45.9% · 3건 64.8%)로 판단합니다.
- **모든 숫자에 문장과 출처** — 규칙 34종이 "2024년에 계약이 끝난 가맹점이 82개로, 연초 점포의 58.6%입니다. 한식 업종 상위 8%" 처럼 그 브랜드의 실제 수치로 소견을 씁니다.
- **화면끼리 같은 숫자** — 등급 경계·표시 반올림·점검 순서를 한 모듈(`src/grading.py`)에서 정하고, 화면과 AI 상담이 같은 규칙을 씁니다.
- **공유와 반출** — 브랜드 링크(`?brand=`), 한글 머리글 엑셀, 품의서용 참고의견서.
- **평가 범위의 빈틈을 메운다** — 공정위 통계의 한 해 공백 때문에 가맹점이 천 곳 넘는 대형 브랜드까지 평가에서 빠지고 있었습니다. 빈 해를 다른 공식 기록(지역·직영 통계, 등록 이력)으로 확인해 79개 브랜드를 더 평가합니다(가맹점 기준 73.6% → 80.1%, 재학습 없음). 평가하지 않은 브랜드는 이유를 브랜드별로 알려 줍니다.
- **연도를 헷갈리지 않게** — "2024년 실적 · 2025년 정보공개서"로 적어 공정위 자료와 바로 대조됩니다.

---

## 📊 Validation — 검증

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/grade_validation-dark.png">
  <img src="assets/charts/grade_validation-light.png" alt="2022년 자료로 맞춘 등급을 2023년에 적용했을 때 등급별 실제 악화율: 안정 2.3%, 관찰 8.8%, 주의 19.5%" width="100%">
</picture>

| 항목 | 결과 |
|---|---|
| 판별력 — 워크포워드 표본 밖 (2021~2023, n=2,063) | AUC **0.707** [0.671, 0.742] · KS **0.316** |
| 순위 성능 — fold 평균 Lift@10% | LightGBM **2.745** · 로지스틱 2.526 · 계약종료율 단일변수 2.403 · 전년 상태 유지 1.688 |
| 보정 — 2023 시점 밖 | Hosmer-Lemeshow p=0.158 · ECE 0.035 |
| 등급 분리 — 2022 → 2023 시점 밖 | 안정 2.3% · 관찰 8.8% · 주의 19.5% |
| 뉴스 사건 추출 — 정답셋 59건 | Gemini 정확도 **0.966** vs 규칙기반 0.475 |

> **정직한 주장 범위** — LightGBM 이 통계적으로 유의하게 넘어서는 것은 '전년 상태 유지' 기준모형뿐이고, 로지스틱·단일변수와는 **통계적으로 동등**합니다. LightGBM 을 쓰는 이유는 성능 우위가 아니라 **SHAP 요인 분해로 브랜드별 근거를 설명**할 수 있어서입니다. 신뢰구간·불리한 결과까지 전부 → [docs/RESULTS.md](docs/RESULTS.md)

**은행 자료가 들어오면** — 라벨이 연체가 아니므로, 은행 연체 자료로 곧바로 돌릴 검증 절차를 코드로 고정해 두었습니다(**등급 검증** 화면). 대출을 **취급 당시 볼 수 있던 등급**에 맞추고(실적은 다음 해에 공개되므로 취급연도 − 2년), 그사이 사라진 브랜드까지 찾아 **생존 편향**을 막고, 같은 브랜드 연체의 동조를 반영한 **브랜드 군집 강건 검정**으로 판정합니다. 판정 기준은 귀무 모의실험으로 오탐률을 확인했습니다.

**품질 장치** — 시점 누출 자동 검사 · 라벨 규칙 학습 전 동결 · 두 표본 트랙 사전 선언 · 문서에 적힌 수치 **210건**을 산출물과 자동 대조(`tools/check_doc_numbers.py`) · pytest 전 화면 스모크 테스트 · 가명 모드 누출 검사(사본 전체 + 전 화면)와 실명↔가명 토글 왕복 검사 · 푸시마다 GitHub Actions CI

---

## 🏗️ Architecture & Pipeline

```mermaid
flowchart LR
    subgraph SRC["공개 데이터"]
        direction TB
        A1["공정위 가맹사업 공시<br/>오픈API 7종"]
        A2["DART 감사보고서<br/>가맹본부 재무"]
        A3["네이버 검색수요·뉴스"]
        A4["국세청 휴·폐업 · 지자체 인허가<br/>(월·일 단위)"]
    end
    subgraph PIPE["배치 파이프라인 · run_pipeline.py"]
        direction TB
        B1["엔티티 정합<br/>브랜드 관리번호"] --> B2["브랜드×연도 패널"]
        B2 --> B8["평가 범위 보정<br/>공시 공백 확인"]
        B8 --> B3["피처 49개<br/>시점 누출 검사"]
        B3 --> B4["LightGBM + 보정<br/>워크포워드 검증"]
        B4 --> B5["고정 등급 FS1~FS3<br/>+ 실적연도별 등급 이력"]
        B2 --> B6["진단 규칙 34종<br/>한국어 소견"]
        B2 --> B7["브랜드 상관·<br/>꼬리손실(Euler ES)"]
        B9["월별 폐점 신호<br/>인허가 매칭·이항검정"]
    end
    subgraph APP["서비스 · Streamlit"]
        direction TB
        C1["FRANSCORE<br/>현황·상세·참고의견서"]
        C6["브랜드 탐색·비교"]
        C2["일괄 조회<br/>+ 사업자 휴·폐업"]
        C3["점검 큐<br/>+ 변경 이력·월별 경보"]
        C4["여신 포트폴리오<br/>행내 장부·꼬리손실"]
        C7["등급 검증<br/>은행 연체 자료"]
        C5["AI 상담<br/>RAG + Gemini"]
    end
    SRC --> PIPE --> APP
```

| 단계 | 핵심 설계 |
|---|---|
| **정합** | 이름이 아니라 공정위 **브랜드·본부 관리번호**로 연도를 잇습니다(이름 매칭은 오매칭이 절반 수준). |
| **라벨** | '악화' = 다음 해 공시 지표(가맹점 순감소·실질매출 감소·계약종료율 급등)가 **업종×연도 하위 구간**에 진입. 규칙은 학습 전에 고정합니다. |
| **검증** | 완전 시간분할 + 워크포워드 3-fold, 기준모형 3종(전년 상태 유지·단일변수·로지스틱)과 같은 지표로 비교합니다. |
| **등급** | 순위가 아닌 **고정 확률 구간**(4.5% · 16.0%) — 업계 전체가 나빠지면 하위 등급이 늘어납니다. |
| **월별 신호** | 인허가 사업장명 → 브랜드 매칭, 같은 주소 90일 내 재개업은 명의 이전. 최근 3개월 폐업을 전년 동기와 **조건부 이항검정**으로 비교합니다. 점수에는 넣지 않습니다. |
| **AI** | LLM 은 뉴스 사건 구조화와 상담에만 씁니다. 점수에는 넣지 않고, 정답셋으로 규칙기반과 비교 평가했습니다. |
| **가명 모드** | 실명을 쓸 수 없는 자리(제3자 배포 자료·기관 내부 배포·브랜드 측 요청)에서는 사이드바 토글 한 번으로 브랜드·본부 이름과 식별번호가 가명으로 바뀝니다(`src/public.py`). 읽는 산출물 경로만 바꾸므로 화면 코드는 그대로입니다. |

---

## 🛠️ Tech Stack

| 구분 | 기술 |
|---|---|
| **Data** | ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![PyArrow](https://img.shields.io/badge/Parquet_·_PyArrow-555555?style=flat-square) |
| **ML·통계** | ![LightGBM](https://img.shields.io/badge/LightGBM-3E7E3E?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![SHAP](https://img.shields.io/badge/SHAP-FF0D57?style=flat-square) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white) |
| **LLM·검색** | ![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-8E75B2?style=flat-square&logo=googlegemini&logoColor=white) ![RAG](https://img.shields.io/badge/TF--IDF_RAG-555555?style=flat-square) |
| **App** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) |
| **Quality** | ![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white) ![Ruff](https://img.shields.io/badge/Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=black) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white) |
| **Data sources** | 공정거래위원회 가맹사업정보 오픈API · 금융감독원 OPEN DART · 네이버 데이터랩·검색 API · 국세청 사업자등록 상태조회 · 지방행정 인허가 |

---

## 🚀 Run Locally

```bash
git clone https://github.com/AISWKimBeomSu/KB_franSCORE.git && cd KB_franSCORE
pip install -r requirements.txt           # Python 3.13 (버전 고정)
streamlit run src/app.py                  # 산출물이 포함돼 있어 API 키 없이 전 화면 동작 (실명 — 내부 참고용)
FRANSCORE_PUBLIC_DEMO=1 streamlit run src/app.py   # 가명 모드로 시작 (사이드바 토글로도 켜고 끕니다)
```

```bash
python run_pipeline.py --step all         # data/raw 스냅샷으로 전 파이프라인 재실행 (API 키 불필요)
pytest -q                                 # 단위 · 정합성 · 전 화면 스모크 · 공개 모드 누출 테스트
python tools/check_doc_numbers.py         # 문서에 적힌 수치 ↔ 산출물 대조
python tools/readme_charts.py             # 이 README 의 차트를 산출물에서 다시 그리기
```

<details>
<summary>월별 폐점 신호 갱신 (월 1회)</summary>

1. [지방행정 인허가 데이터](https://file.localdata.go.kr)에서 전국 **일반음식점 · 휴게음식점 · 제과점영업** 파일(CSV)을 내려받아 `data/raw/localdata/` 에 둡니다(합계 약 1GB, 저장소에는 올리지 않음).
2. 신호표와 월별 흐름을 만듭니다.

```bash
python -m src.localdata --csv data/raw/localdata/*.csv --scope 전국 --months 24 \
    --out outputs/localdata_signal.csv --flows-out outputs/localdata_flows.csv
```

3. 새 공시가 들어온 해에는 공시와의 일치도를 다시 잽니다 — `python tools/validate_localdata.py --csv data/raw/localdata/*.csv`
</details>

<details>
<summary>API 키 (전부 선택 — 없으면 해당 기능만 대체 경로로 동작)</summary>

| 환경변수 | 발급처 | 없을 때 |
|---|---|---|
| `GEMINI_API_KEY` | aistudio.google.com | AI 상담은 수집된 사실을 표로 정리해 답변, 뉴스 추출은 규칙기반 |
| `NAVER_CLIENT_ID` / `_SECRET` | developers.naver.com | 뉴스는 Google News RSS(제목만) |
| `DART_API_KEY` | opendart.fss.or.kr | 본부 재무 재수집·감사의견 원문 대조(`tools/verify_audit_opinions.py`) 불가 — 포함된 산출물은 그대로 사용 |
| `DATA_GO_KR_KEY` | data.go.kr | 새 공시 수집, 국세청 휴·폐업 조회(데이터셋 15081808 활용신청 필요) 불가 — 저장된 스냅샷으로 전 화면 동작 |

**키 넣는 곳** — 로컬은 저장소 루트의 `.env` 파일(`DART_API_KEY=...` 한 줄씩, `.gitignore` 에 들어 있어 커밋되지 않음), Streamlit Cloud 는 앱 **Settings → Secrets** 에 `DART_API_KEY = "..."` 형식으로 넣습니다. 우선순위는 환경변수 > `.env` > Secrets 입니다.

배포는 [DEPLOY.md](DEPLOY.md) — Streamlit Community Cloud(Python 3.13 지정)·사내망 이식 절차.
</details>

---

## 🗂️ Project Structure

```
KB_franSCORE/
├─ run_pipeline.py          # 파이프라인 러너 (--step collect … diagnose)
├─ config.yaml              # 라벨 임계값·분할·모형·포트폴리오 가정 — 학습 전 고정
├─ src/
│  ├─ collect.py entity.py panel.py         # 공정위 수집 · 관리번호 정합 · 패널
│  ├─ dart.py ifrmp.py naver.py             # 본부 재무(DART) · 정보공개서 · 검색수요/뉴스
│  ├─ features.py labels.py                 # 피처 49개 · 라벨 (시점 누출 검사)
│  ├─ model.py evaluate.py backtest.py      # 학습 · 기준모형 · 보정 · 워크포워드
│  ├─ score.py coverage.py grading.py       # 최신 점수·실적연도별 등급 이력 · 평가 범위 보정 · 등급/순서 규칙
│  ├─ diagnosis.py guidance.py report.py    # 진단 규칙 34종 · 확인 서류 가이드 · 참고의견서
│  ├─ portfolio.py correlation.py loanbook.py   # 집중도 · 브랜드 상관 · 꼬리손실 · 행내 장부 업로드
│  ├─ batch.py delinquency.py               # 일괄 조회 · 은행 연체 자료 등급 검증
│  ├─ nts.py localdata.py                   # 국세청 휴·폐업 · 지자체 인허가 월별 폐점 신호
│  ├─ public.py                             # 가명 모드 — 실명을 못 쓰는 자리용 사본 (명세 §3)
│  ├─ llm.py news_llm.py rag.py chat.py     # Gemini · 뉴스 사건 추출 · RAG · 상담
│  ├─ app.py theme.py                       # Streamlit 셸 · 디자인 시스템
│  └─ views/                                # 화면 8종
├─ tests/                   # 단위 · 정합성 · 개인정보 · 전 화면 스모크 · 공개 모드 누출
├─ tools/                   # 문서 수치 대조 · README 차트 · 감사의견 원문 대조 · 월별 신호 타당도 · 품질 점검
├─ .github/workflows/       # CI(린트·테스트·문서 대조) · 수동 갱신 배치
├─ data/raw/ data/processed/ outputs/       # 원본 스냅샷 · 가공 데이터 · 산출물
└─ docs/                    # 검증 결과 · 기술설명서 · 방법론 · 사용 명세 · 운영 설계
```

## 📚 Docs

| 문서 | 내용 |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | 검증 결과 상세 — 전체 표·신뢰구간·불리한 결과·데이터 출처 |
| [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | 전 과정 기술설명서 — 문제·데이터·모형·운영, 스스로 잡은 결함 전수 기록 |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | 방법론 공개서 |
| [docs/MODEL_USE_SPEC.md](docs/MODEL_USE_SPEC.md) | 모형 사용 범위·금지 용도 |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | 운영·거버넌스 설계 (은행 실여신 연결 인터페이스 포함) |
| [docs/LEAKAGE_CHECKLIST.md](docs/LEAKAGE_CHECKLIST.md) | 시점 누출 점검표 |

---

## ⚠️ Limitations

- 라벨은 부도가 아니라 **공시 지표 기반 구조악화 사건**입니다. 은행 연체와의 관계는 **등급 검증** 화면으로 바로 검증하도록 절차를 만들어 두었지만, 실제 은행 연체 자료로 돌린 결과는 아직 없습니다(화면 시연은 가상 자료).
- 여신 익스포저 기본값은 **공시 창업비용 기반 추정치**입니다. 행내 장부를 올리면 그 장부로 계산합니다(세션 메모리에서만 처리).
- 본부 재무는 자격 브랜드 기준 커버리지 13.8%(가맹점 가중 48.1%)입니다. 감사의견 판독은 감사인의 보고서 구간만 읽도록 고쳤지만, DART 원문 대조(키 필요)를 돌리기 전까지는 **'원문 확인 필요'** 로만 표시합니다.
- 공시 공백 보정으로 평가한 79개 브랜드는 과거 백테스트 표본 밖입니다 — 화면에 그 사실과 근거를 함께 적습니다.
- 월별 폐점 신호는 사업장명으로 브랜드를 이은 값입니다 — 상호에 브랜드명이 없는 가맹점은 빠지고(매칭 완전성 중앙값 98.3%, 브랜드별 편차 있음), 인허가 폐업은 가맹계약 종료와 같은 사건이 아닙니다. 그래서 점수에 넣지 않고 조기경보로만 씁니다. 전국 파일은 월 1회 손으로 내려받습니다(자동화는 공공데이터 API 키로 가능).
- 국세청 휴·폐업 조회는 `DATA_GO_KR_KEY` 와 데이터셋 15081808 활용신청 뒤에 동작합니다.
- 공개 데모는 **실명**으로 동작합니다. 공시로 이미 공개된 자료를 다시 계산한 참고 지표이지 신용평가가 아니고, 특정 브랜드에 대한 평가·권유도 아닙니다 — 등급을 인용·재배포하지 마십시오. 실명을 쓸 수 없는 자리에서는 **가명 모드**를 켜고, 브랜드 측 요청이 오면 해당 브랜드를 내리거나 가명 모드로 전환합니다(창구: 저장소 이슈). 저장소의 `outputs/` 는 재현용 연구 자료입니다.

---

## 👤 Author

**김범수 (Beomsu Kim)** · [@AISWKimBeomSu](https://github.com/AISWKimBeomSu)

문제 정의 · 데이터 수집/정합 파이프라인 · 모형·검증 설계 · 서비스 설계와 구현 — KB국민은행 제8회 AI Challenge 출품
