# 시점 누출 점검 체크리스트 (명세 §4.2)

> 각 항목은 **자동 테스트 또는 코드 위치**로 증빙된다. 수동 주장만으로 통과 처리하지 않는다.
> 실행: `python tests/test_sanity.py` (6/6 통과 필요), `python tests/test_llm_paths.py` (4/4)

## 1. 라벨 시점 > 피처 시점

| 점검 | 방법 | 증빙 |
|---|---|---|
| 라벨은 t+1 사건, 피처는 t 이하 | `labels.build_labels`가 t+1 사건 플래그를 t 행에 붙임 (`nxt["year"] -= 1`) | [labels.py](../src/labels.py) `build_labels` |
| 피처가 t+1 값에 불변 | t+1 이후 **모든 수치 컬럼을 교란**한 뒤 (brand, t≤2022) 피처 행이 bit-exact 동일한지 assert | `tests/test_sanity.py` [2/6] time_leakage |
| 라벨 게이트도 t 시점 정보만 | healthy_at_t·min_stores_at_t 모두 t년 지표로만 판정 | [labels.py](../src/labels.py) `build_labels` |

## 2. 미래 파생 없음 (횡단면 통계)

| 위험 지점 | 처리 | 증빙 |
|---|---|---|
| 윈저라이즈 경계 | **연도별 횡단면** 분위수로만 계산 (전 기간 풀 사용 금지) | [features.py](../src/features.py) `build_features` 윈저라이즈 블록 |
| 업종 내 rank pct | 업종그룹 × **해당 연도** 내에서만 순위 | [features.py](../src/features.py) `f_ind_*` |
| 실질매출 업종 중앙값 | 업종그룹 × **해당 연도** 중앙값 차감 | [labels.py](../src/labels.py) `compute_derived_metrics` |
| 라벨 분위수 임계값 | 업종그룹 × **t+1 연도** 실측 분포 (라벨 정의이므로 피처 아님) | [labels.py](../src/labels.py) `_event_flags` |
| 추세(rolling) | `rolling(window=3)`은 과거 방향만 (pandas 기본) | [features.py](../src/features.py) `f_trd_*` |
| 결측 대치 | 로지스틱의 median impute는 **train 분할에서 fit** 후 transform | [model.py](../src/model.py) `Pipeline` |
| 확률 보정 | isotonic 보정기는 **valid 분할에서만 fit**, test에는 적용만 | [evaluate.py](../src/evaluate.py) `_fit_calibrator` |
| 복잡도 선택 | 3개 사전 후보 중 **valid AUC로만** 선택, test 미사용 | [model.py](../src/model.py) `fit_lgbm_valid_selected` |

## 3. 표본 선별에 미래 정보 없음 (적대적 리뷰로 발견·수정)

| 점검 | 처리 | 증빙 |
|---|---|---|
| 점포수 하한 | t까지의 **누적 최대**(cummax)로 판정 — 전체 이력 max 금지 | [panel.py](../src/panel.py) `apply_sample_filter` |
| 연속 관측 요건 | **t에서 끝나는** 연속 구간 길이로 판정 — 최장 구간 금지 | 동일 |
| 분위수 풀 | 업종 범위 전체 패널로 계산 (선별된 표본으로 계산 금지) | [run_pipeline.py](../run_pipeline.py) labels 스텝 |
| 0점포 아티팩트 | 직전연도 정보만으로 판정 (t+1 원복 여부 미사용) | [panel.py](../src/panel.py) `build_panel` |

> ⚠️ **이력:** 초기 구현은 브랜드 전체 이력의 max(점포수)·최장 연속구간으로 표본을 선별해
> 라벨 윈도우 안의 미래 성장이 선별에 새어 들어갔다. 적대적 코드리뷰에서 확정되어 위와 같이
> 실시간(과거 정보만) 규칙으로 교체했다. 교체 전후 지표를 모두 보고한다.

## 4. 시간 분할 정합

| 점검 | 방법 | 증빙 |
|---|---|---|
| 연도 교집합 없음 | train/valid/test 연도 집합 교집합 공집합 assert | `tests/test_sanity.py` [3/5] time_split |
| test가 최신 연도 | test = max(라벨 연도) assert | 동일 |
| 랜덤 분할 없음 | 분할은 연도 기준 `np.select`만 사용 | [model.py](../src/model.py) `train_all` |
| 브랜드 단위 정합 | `is_new_brand` = train 연도에 없던 brand_id → 성능 분리 보고 | `outputs/newbrand_split.csv` |
| 워크포워드 각 fold | fold별로 train ≤ valid < test, 매 fold 재학습 | [backtest.py](../src/backtest.py) `run_walkforward` |

## 5. 뉴스·LLM 신호 누출 차단

| 점검 | 처리 | 증빙 |
|---|---|---|
| 뉴스 신호 점수 미투입 | `config.llm.score_injection: false`, 피처 행렬에 뉴스 컬럼 없음 | [config.yaml](../config.yaml), `outputs/feature_summary.csv` |
| 뉴스 발행일 | 수집 시 발행일 기록, 화면에 표기 | [news_llm.py](../src/news_llm.py) |
| RAG 근거 | 검색 결과는 메모 서술에만 사용, 모델 입력 아님 | [rag.py](../src/rag.py), [memo_llm.py](../src/memo_llm.py) |

## 6. 재현성

| 점검 | 처리 |
|---|---|
| seed 고정 | `config.seed: 42`, `set_seed()` 전 모듈 호출 |
| 버전 고정 | `requirements.txt` 정확한 버전 |
| 원본 스냅샷 | `data/raw/` 보존 (API 변경에도 재현) — 대용량은 gzip |
| 파라미터 단일 원천 | 전부 `config.yaml` |
| 라벨 규칙 동결 | 학습 전 git 커밋(`cfcf164`)으로 증빙 |
| 산출물 격리 | `--demo`(스모크)·`--scope extended`(확장 트랙) 별 디렉토리 분리 |
