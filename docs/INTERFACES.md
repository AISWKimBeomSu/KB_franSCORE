# FranSCORE 모듈 인터페이스 계약 (구현 규약)

> 모든 모듈은 이 계약을 따른다. 계약 변경 시 이 문서를 먼저 수정한다.
> 원칙: 단일책임 · 시점누출 금지 · 모든 파라미터는 config.yaml · seed 고정.
>
> **개정 이력** (계약은 살아있는 문서다 — 변경은 숨기지 않고 기록한다):
> - v1 (개발 착수 시): 초기 계약 동결. 병렬 모듈 구현의 기준.
> - v2 (적대적 리뷰 반영): 표본 자격 실시간화(`eligible_t`), 라벨 관측가능성 요건
>   (`n_obs_metrics`), 분위수 풀 게이트, 보정 tie-break — 리뷰 확정 결함 수정.
> - v3 (명세 빈틈 해소): 브랜드 관리번호 정합(`brand_mnno`), 직영·지역·취소
>   병합 컬럼, valid 기반 복잡도 선택, 워크포워드(backtest)·RAG 모듈 추가.
> - v4 (무결점 재검토 반영, 현행):
>   ① 런타임 LLM 공급자를 Google Gemini로 전환하고 호출을 `src/llm.py` 한 곳으로 통합.
>   ② 워크포워드 집계를 fold 평균(`macro_avg_folds`) 기준으로 수정 — 원점수 전역 풀링은
>      fold별 확률 스케일 차이로 재학습 모형에 불리한 편향이 있음을 실측 확인(§5.2).
>   ③ 업종 더미는 범주 2개 이상일 때만 생성 — 데이터 의존 상수컬럼 제거가 피처 **스키마**를
>      미래 값에 의존하게 만드는 문제 제거(시점누출 테스트로 검출).
>   ④ 패널 값 컬럼 단일 정의 `panel.PANEL_VALUE_COLS` — 합성 패널·누출 테스트가 이를 공유.

## 0. 공통 (src/common.py)

```python
from src.common import load_config, get_logger, PATHS, set_seed, make_synthetic_panel
cfg = load_config()          # config.yaml → dict. cfg["_root"]에 프로젝트 루트 Path
log = get_logger("module")   # 표준 로거 (stdout + outputs/pipeline.log)
set_seed(cfg["seed"])        # random/numpy 시드 고정
panel = make_synthetic_panel(cfg)  # ⚠️ 스모크테스트 전용 합성 패널
```

- 실행은 항상 **프로젝트 루트에서** `python -m src.모듈` 또는 `python run_pipeline.py`.
- 금액 단위: `avg_sales`·`avg_sales_per_area`는 **천원**(원천 그대로), 포트폴리오 exposure는
  **백만원(MKRW)**, 화면 표시는 **억원**(=백만원/100).
- 산출 경로는 항상 `cfg["paths"]`를 **호출 시점에** 참조한다
  (`--demo`는 `outputs/_smoke/`, `--scope extended`는 `outputs/extended/`로 격리 스왑).

## 1. 패널 스키마 (data/processed/panel.parquet) — 모든 모듈의 공용 입력

brand_id × year 롱포맷. 1행 = 한 브랜드의 한 해(실적연도).
**panel.parquet = 업종 범위(기본: 외식) 전체 행** + 자격 플래그. 필터된 표본이 아니다 —
라벨 분위수 풀·피처 횡단면은 전체 모집단으로 계산하고, 학습 표본만 `eligible_t`로 거른다.
`panel_full.parquet`은 업종 필터 전의 전체 패널이다.

| 컬럼 | 타입 | 의미 |
|---|---|---|
| brand_id | str | **브랜드 관리번호(brandMnno) 우선** — 연도 내 법인 충돌 시 `brandMnno\|hq_mnno`로 분리, 관리번호 미확보분(2.2%)은 `NAME:정규화명@업종` |
| id_source | str | `MNNO`(관리번호 기반) / `NAME`(명칭 기반 폴백) — 정합 출처 투명화 |
| brand_mnno, hq_mnno | str? | 브랜드/가맹본부 관리번호 (15125467 크로스워크, 97.8% 확보) |
| brand_name, company_name | str | 브랜드명·가맹본부(법인)명 |
| industry_major, industry_mid | str | 업종 대분류(외식/서비스/도소매)·중분류 |
| year | int | **실적 기준 연도 = 공시 기준연도(yr) − 1** (공시 수치가 직전 회계연도 실적; `acntgYr`로 교차 확인) |
| n_stores | float | 가맹점 수. ⚠️ 직전연도 10개+ 브랜드의 0 보고는 공시 아티팩트로 NaN 처리(과거 정보만 사용) |
| n_direct | float | 직영점 수 (15125490 `areaNm=='전체'` 행, 병합률 84.6%) |
| n_new / n_contract_end / n_contract_cancel / n_name_change | float | 신규개점·계약종료·계약해지·명의변경 수 |
| avg_sales / avg_sales_per_area | float | 가맹점 평균매출액·3.3㎡당 매출액 (천원, 0→NaN) |
| biz_start_year, emp_cnt, exec_cnt | float | 가맹사업개시연도·본부 직원/임원수 (15109828) |
| n_regions, region_hhi, top_region_share, region_max_stores | float | 지역 분산: 진출 시도 수·지역 HHI·최다지역 비중·최다지역 점포수 (15125490 시도 행. ⚠️ `전체`행과 시도행 합산 금지 — 정확히 2배 중복) |
| cancel_flag, cancel_type | float/str | 브랜드 등록취소 여부·유형(자진/직권) (15125518, 명칭 조인) |
| eligible_t | bool | **실시간 표본 자격**: [t까지 누적 최대 점포수 ≥ min_stores] AND [t에서 끝나는 연속 관측 ≥ min_consecutive_years]. 과거 정보만 사용 (룩어헤드 금지) |

정렬: (brand_id, year). 중복 (brand_id, year) 없음 보장.

### 파생 지표 정의 (labels.compute_derived_metrics — 단일 정의 원칙)
- `store_growth_rate(t)` = (n_stores_t − n_stores_{t−1}) / n_stores_{t−1}
- `contract_end_rate(t)` = (n_contract_end_t + n_contract_cancel_t) / n_stores_{t−1}
- `sales_growth(t)` = avg_sales_t / avg_sales_{t−1} − 1
- `sales_per_area_growth(t)` = avg_sales_per_area_t / avg_sales_per_area_{t−1} − 1
- `real_sales_growth(t)` = sales_growth(t) − (업종그룹 내 sales_growth(t) 중앙값)
- `direct_ratio(t)` = n_direct_t / (n_direct_t + n_stores_t)
- 전년 지표는 **연속 관측(consec)** 일 때만 사용 (연도 갭이면 NaN). 전년 점포/매출 ≤ 0이면 NaN.
- 업종그룹 = industry_mid (해당 연도 그룹 크기 ≥ 30) else industry_major

## 2. M2 — labels.py / features.py

### labels.build_labels(panel, cfg) -> DataFrame
반환: `[brand_id, year, label, healthy_at_t, ev_store, ev_sales, ev_contract, n_events, extreme_fired]`
- **`label=1` ⇔ t년에 '검증 가능하게 양호'했던 브랜드가 t+1년에 악화 전환.** 행의 year는 t.
- 악화 사건 3종(t+1 지표, 업종그룹×연도 분위수): ① store_growth_rate 하위 20% AND |Δ점포|≥min_abs_store_change ② real_sales_growth 하위 20% ③ contract_end_rate 상위 20%.
  발동: **2개 이상 or 1개 극단**(extreme_quantile=5%).
- **분위수 임계값 풀 = 셀 내 비교 가능 규모 동료군**(전년 점포수 ≥ min_stores_at_t — 지표
  시점에 이미 알 수 있는 정보라 실시간 안전). 임계값은 풀에서 계산하되 전 행에 적용.
- **healthy_at_t = (t년 사건 0건) AND (t년 지표 3종 중 ≥1개 관측 가능)** — 전부 결측이면
  "양호"가 아니라 "판정 불가"로 표본 제외 (v2 수정).
- 게이트: n_stores_t ≥ min_stores_at_t. t+1 없는 행 제외. 학습 표본은 이후
  run_pipeline에서 `eligible_t`로 추가 필터.
- 로깅: 사건별 비율·중복 분포 → `outputs/label_composition.csv`.
- 해석 규칙(문서화): 엄격 부등호(동률 미발동) · 극단은 사건의 부분집합(점포 게이트 동일 적용)
  · NaN 지표는 절대 미발동 · |Δ점포|는 절대값.

### features.build_features(panel, cfg) -> DataFrame
반환: `[brand_id, year] + f_*` **35개** (기본 트랙 실측; 확장 트랙은 업종 더미 3종이 붙어 38개).
업종 더미는 **범주가 2개 이상일 때만** 생성한다 — 단일 업종 트랙에서는 전 행 동일값이라
정보가 없고, 값을 보고 사후에 버리면 피처 스키마가 미래 값에 의존하게 된다(§5.2 참조).
행 (brand_id, t)는 **t 이하 연도 값만** 사용(⛔ t+1 금지 — bit-exact 자동검증 대상).

| 그룹 | 피처 |
|---|---|
| f_lvl_ (7) | n_stores, avg_sales_log, sales_per_area_log, direct_ratio, brand_age(관측연차), biz_age(가맹사업 업력), emp_cnt_log |
| f_chg_ (6) | store_growth, sales_growth, sales_per_area_growth, contract_end_rate, new_open_rate, name_change_rate |
| f_trd_ (9) | store_growth·sales_growth·sales_per_area 각각의 rolling(3, min 2) mean/std/slope |
| f_ind_ (3+더미) | store_growth·real_sales_growth·contract_end의 업종그룹×연도 rank pct + 업종 대분류 더미 |
| f_struct_ (10) | direct_ratio_chg, direct_growth, open_close_gap, n_regions(+chg), region_hhi(+chg), top_region_share(+chg), stores_per_region |

- 윈저라이즈: **연도별 횡단면** 상하위 1% 클립 (더미 제외 전 연속 피처).
- 요약 → `outputs/feature_summary.csv`.

## 3. M3 — model.py / evaluate.py / backtest.py

### model.train_all(features, labels, cfg)
- inner join(brand_id, year) → 완전 시간분할 auto_last_two (test=최대 라벨연도, valid=그 전).
- 기준모형: `p_persistence`(t년 사건 발동수+심도 점수), `p_single`(계약종료율 rank pct),
  `p_logistic`(train-fit median impute + 표준화 + 로지스틱).
- 주모형: **`fit_lgbm_valid_selected`** — config `model.complexity_candidates`(사전 선언 3개)
  중 **valid AUC로만** 선택(test 미사용, 그리드서치 아님), valid AUC 조기종료
  (`first_metric_only=True`).
- 저장: predictions.parquet `[brand_id, year, split, y_true, p_lgbm, p_persistence, p_single,
  p_logistic, is_new_brand]`, features_matrix.parquet, labels_joined.parquet,
  model_lgbm.txt(부스터 텍스트 — 비ASCII 경로 대응 문자열 I/O), model_logistic.joblib,
  split_years.json(분할·피처명 매핑·복잡도 선택 로그).

### evaluate.evaluate_all(cfg)
`outputs/`: metrics.csv(**5모형**×3분할 — lgbm_calibrated 포함; Lift@10=Prec@10/base,
k=max(1,floor(n·10%))), calibration.png/csv + calibrator.joblib(valid에서만 적합; 보정확률에
원점수 순위 tie-break ≤1e-5 부여 — 동률 붕괴 방지), shap_summary.png + shap_values.parquet
(test 행별 상위 10, 원본 피처명 복원), label_decomposition.csv, newbrand_split.csv,
ablation.csv, metrics_bootstrap_ci.csv(재표집 200회 95% CI), lift_delta_bootstrap.csv
(페어드 부트스트랩 — LightGBM−기준 차이 CI·승률).

### backtest.run_walkforward(cfg)  (검정력 보강 — 보조 근거)
확장창 워크포워드: fold별 train(≤T−2)/valid(T−1)/test(T), 매 fold 재학습(복잡도 선택 포함).
풀링 OOS + **연도블록 부트스트랩** delta CI → walkforward_metrics.csv /
walkforward_predictions.parquet / walkforward_delta_ci.csv.
`run_both_tracks.py`: 기본·확장 두 트랙을 항상 함께 실행·공개 → track_comparison.csv.

## 4. M4 — portfolio.py / news_llm.py / memo_llm.py / rag.py

### portfolio.build_portfolio(cfg)
- 입력: predictions.parquet(test, p_calibrated 우선) + panel(점포수; (brand,year) 정확 매칭
  실패 시 브랜드 최신 관측 폴백·로깅).
- 익스포저 3단 우선순위: ① 실여신 CSV → ② **공시 창업비용(15110265) 실측**(기본) →
  ③ 합성 LogNormal 폴백. 어느 경로를 탔는지 `assumptions.exposure.basis` 와 최상위
  `synthetic` 플래그가 **실제 경로를 따라** 기록된다.
- 등급 = test 전체 분포 내 **순위(pct rank, method='first')** 기준 High(상위 10%)/Medium(30%)/
  Low — 값 임계 컷은 동률 왜곡 때문에 사용하지 않는다(참고 컷값만 기록).
- EL = exposure × 악화확률 × LGD{0.3,0.45,0.6}; 스트레스 = 악화확률 상위 10% × 1.5 (cap 1.0).
  악화확률은 PD 가 아니다(구조악화 전환 확률) — EL 은 대리 손실 지표다.
- 산출: portfolio.csv, portfolio_summary.json(가정 명세 필수 포함).

### news_llm.py
- `fetch_news`: Google News RSS(feedparser), 브랜드+접미사 질의, 링크 dedupe, 브랜드당 상한,
  원본 스냅샷 → data/raw/news/. 실패 무시(빈 리스트).
- `extract_signals`: 개체 매칭 필터 → (키 있으면) Gemini 구조화 추출(JSON Schema 강제,
  refusal/예외/비JSON 시 규칙 폴백) → outputs/news_signals.json.
  신호: `{brand, event_type(본부분쟁|재무이슈|집단폐점|기타|무관), date, evidence_sentence,
  confidence(상|중|하), source_url, published, llm_used}`.
  로그는 **실제 LLM 추출/폴백 건수**를 분리 보고. ⚠️ 점수 미투입(화면 전용).

### memo_llm.generate_memo(context, cfg) -> str (markdown)
- context에 `rag_evidence`(retrieve_evidence 결과) 자동 주입.
- LLM 경로: 입력 근거만 인용 제약 + **프롬프트 인젝션 방어 지시**(뉴스 텍스트 내 지시문 무시)
  + 필수 고지("2선 리스크 참고용·자동 여신결정 아님") 누락 시 강제 부착.
- 키 없음/실패 시 결정적 템플릿 폴백(동일 입력→동일 출력, `LLM 미사용 폴백` 각주).

### rag.py (COULD "진짜 RAG")
- `build_corpus`: 뉴스 스냅샷(개체 매칭 필터 적용) + 패널 공시 문서(자연어 렌더링) →
  data/processed/rag_corpus.parquet.
- `build_index`/`load_index`: TF-IDF(word 1-2gram + char_wb 2-4gram) → outputs/rag_index.joblib.
  외부 임베딩 API 미사용(키 없이 완전 재현).
- `retrieve_evidence(cfg, brand, risk_factors)`: 브랜드 한정 검색→전체 완화, dedupe,
  `{doc_id, source_type, text, url, published, source_name, score, query}` 리스트.

## 5. M5 — app.py (Streamlit)

- `streamlit run src/app.py`. 산출물 파일만 읽음(재계산 없음), 부재 시 실행 안내(크래시 금지).
- 화면① 브랜드 상세: 위험등급 배지·보정확률·SHAP 상위 4(한글 요인명 매핑)·공시 지표 추이·
  뉴스 경보("점수 미반영" 고지)·집중 여신/상관손실 카드·등급별 권고("자동 결정 아님")·
  심사메모 생성(RAG 근거 포함).
- 화면② 포트폴리오 뷰: KPI·위험×익스포저 산점·상위 20 랭킹·LGD별 EL·**동일 LGD 짝** 스트레스
  비교·백테스트 요약. 합성 경고 배너 상시.

## 6. 러너

- `run_pipeline.py --step collect|panel|features|labels|model|evaluate|portfolio|news|all`
  `[--scope primary|extended] [--demo]`
- `run_both_tracks.py`: 두 트랙 전체 + 워크포워드 + 비교표.
- `check_keys.py [--data|--llm]`: 키 발급 직후 진단(데이터셋별 승인·키 형식·동기화·크레딧).

## 7. tests/

- `python tests/test_sanity.py` (6종): 라벨 규칙(수제 패널 기대값)·시점 누출(bit-exact)·
  시간분할·predictions 스키마·포트폴리오 수리·상관 수학. 임시 디렉토리 격리 +
  data/processed 불변 가드.
- `python -m tests.test_llm_paths` (4종): `src.llm._http_post`(HTTP 경계)만 모의로 치환해
  Gemini 클라이언트 전 경로 검증 — 스키마 정제(미지원 키워드 제거)·사고(thought) 파트 제거·
  안전차단·토큰 절단·재시도·키 미설정, 그리고 뉴스 구조화 추출(모델 출력 재검증 포함)·
  메모 고지 강제·폴백 결정성·RAG 검색/개체필터. **실키 불필요.**
- `python -m ruff check .` : 린트 (규칙 `ruff.toml`, 버전 `requirements-dev.txt` 고정).
- `python tools/check_doc_numbers.py` : 문서에 적힌 수치가 실제 산출물과 일치하는지 자동 대조.
