# 배포 안내

KB 심사역이 브라우저로 접속해 바로 쓸 수 있도록 웹에 올리는 절차입니다.
저장소는 이미 배포 가능한 상태로 준비돼 있습니다 — **새로 clone 한 상태에서 API 키
없이도 5개 화면이 모두 정상 동작하는 것을 실측 확인**했습니다.

---

## 1. Streamlit Community Cloud (권장 · 무료)

가장 빠릅니다. GitHub 계정으로 로그인하면 3분 안에 URL이 나옵니다.

### 1-1. 배포

아래 주소를 열면 저장소·브랜치·진입점이 미리 채워진 배포 화면이 뜹니다.

```
https://share.streamlit.io/deploy?repository=AISWKimBeomSu%2FKB_franSCORE&branch=main&mainModule=src%2Fapp.py
```

화면에서 확인할 값:

| 항목 | 값 |
|---|---|
| Repository | `AISWKimBeomSu/KB_franSCORE` |
| Branch | `main` |
| Main file path | `src/app.py` |
| Python version | **`3.13`** — `Advanced settings` 에서 **반드시 직접 지정** (아래 1-1-c) |
| App URL | 원하는 주소 (예: `kb-franscore`) |

`Deploy!` 를 누르면 의존성 설치에 3~5분 걸립니다.
완료되면 `https://<정한이름>.streamlit.app` 으로 누구나 접속할 수 있습니다.

### 1-1-a. ⚠️ 저장소를 비공개로 바꿨다면 먼저 권한을 줘야 합니다 (실제로 여기서 막혔음)

이 저장소는 현재 **공개**입니다. 비공개로 운영한다면 — Streamlit Cloud 가 처음 GitHub 로그인 시
받는 권한은 **공개 저장소까지만** 이고, GitHub 은 권한 없는 리소스에 404 를
돌려주므로 배포 화면에 이렇게 뜹니다 — 저장소는 멀쩡한데도 그렇습니다.

```
This repository does not exist
This branch does not exist
This file does not exist
```

App URL 칸에 `Domain is available` 이 초록색으로 뜬다면 로그인은 정상이고
**권한만 없는 것**입니다. 아래로 해결합니다.

1. https://share.streamlit.io 접속
2. 좌측 상단 **GitHub 사용자명** 클릭 → **`Settings`**
3. 왼쪽 사이드바 **`Linked accounts`**
4. `Source control` 의 **`Connect here`**
5. **`Authorize streamlit`** — "access your private repositories" 가 있는 두 번째 승인

`Connect here` 가 안 보이거나 이미 연결됨으로 표시되면
https://github.com/settings/applications 에서 `Streamlit` 을 **Revoke** 한 뒤 다시 하십시오.
기존 승인이 남아 있으면 GitHub 이 재승인 화면을 띄우지 않습니다.

공개 전환으로 해결할 수도 있으나 코드와 수집 데이터가 전부 공개됩니다.

```bash
gh repo edit AISWKimBeomSu/KB_franSCORE --visibility public --accept-visibility-change-consequences
```

### 1-1-b. `Paste GitHub URL` 모드를 쓸 때

이 모드는 저장소 주소가 아니라 **`.py` 파일을 직접 가리키는 주소**를 요구합니다.

```
https://github.com/AISWKimBeomSu/KB_franSCORE/blob/main/src/app.py
```

저장소 루트 URL 을 넣으면
`The field needs to contain a Github URL pointing to a .py file` 로 거부됩니다.
이 모드 역시 비공개 저장소면 1-1-a 의 권한 승인이 선행돼야 합니다.

### 1-1-c. ⚠️ Python 은 3.13 이어야 합니다 (실제로 여기서 막혔음)

`Advanced settings` 에서 지정하지 않으면 Community Cloud 가 **최신 Python(3.14)** 을
잡습니다. 그러면 아래 네 패키지가 **3.14용 wheel 이 없어 소스 빌드로 넘어가고**,
컨테이너에 컴파일러·meson·BLAS 가 없으므로 끝내 완료되지 않습니다.

| 패키지 | Python 3.13 | Python 3.14 |
|---|---|---|
| `numpy==2.2.4` | wheel 있음 | **없음** |
| `pandas==2.2.3` | wheel 있음 | **없음** |
| `scikit-learn==1.6.1` | wheel 있음 | **없음** |
| `matplotlib==3.10.1` | wheel 있음 | **없음** |
| 나머지 8개 (lightgbm·shap·pyarrow·streamlit·plotly·joblib·openpyxl·PyYAML) | 있음 | 있음 |

`Resolved 67 packages in 655ms` 까지는 정상적으로 지나가므로 성공한 것처럼 보입니다.
**의존성 해석은 성공하고 그다음 빌드에서 막히는 것**이라, 로그가 조용한 채로
20 분 넘게 흐르다 머신이 재기동되면 이 증상입니다.

버전 핀을 푸는 방식은 권하지 않습니다. 학습된 모델이 `scikit-learn 1.6.1` ·
`lightgbm 4.7.0` 으로 직렬화돼 있어 상위 버전에서 언피클 경고·오류가 날 수 있고,
재현성 보증이 이 프로젝트의 주장 중 하나이기 때문입니다. Python 쪽을 맞춥니다.

**이미 3.14 로 배포해 버렸다면** — Community Cloud 는 배포 후 Python 버전 변경을
지원하지 않습니다. 앱을 지우고 다시 만들어야 합니다. 서브도메인은 삭제 즉시
재사용할 수 있으므로 주소는 그대로 유지됩니다.

1. 앱 목록에서 해당 앱 우측 **점 세 개** → **`Delete app`** (확인창에 앱 이름 입력)
2. **`Create app`** → 저장소·브랜치·`src/app.py` 지정
3. App URL 에 **같은 이름** 다시 입력
4. **`Advanced settings`** → `Python version` **`3.13`** → Secrets 입력 → `Save` → `Deploy!`

### 1-2. API 키 등록 (선택)

**키가 하나도 없어도 앱은 정상 동작합니다.** 새로 clone 한 상태에서 키 없이 5개
화면이 전부 뜨는 것을 실측 확인했습니다. 이미 수집해 둔 산출물(점수·진단 소견·
로고 611건·검색수요 292건)이 저장소에 들어 있기 때문입니다.

키는 **다시 수집하거나 AI 상담 답변을 생성할 때만** 필요합니다.

| 키 | 없을 때 화면 | 발급처 |
|---|---|---|
| `GEMINI_API_KEY`<br>`GEMINI_API_KEY_2` | AI 상담이 답변을 만들지 않고 **수집된 사실만 정리**해 보여줍니다 | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `NCP_API_KEY_ID`<br>`NCP_API_KEY` | 이미 수집된 292개 브랜드의 검색수요는 **그대로 보입니다**. 갱신만 안 됩니다 | NCP 콘솔 → NAVER API HUB → Application |
| `DART_API_KEY` | 이미 수집된 본부 재무는 그대로 보입니다. 갱신만 안 됩니다 | [opendart.fss.or.kr](https://opendart.fss.or.kr) |

등록: 배포된 앱 → 우측 하단 `Manage app` → `Settings` → `Secrets` 에 붙여넣습니다.

```toml
GEMINI_API_KEY = "여기에-키"
GEMINI_API_KEY_2 = "예비-키 (선택)"
NCP_API_KEY_ID = "여기에-아이디"
NCP_API_KEY = "여기에-시크릿"
DART_API_KEY = "여기에-키"
```

저장하면 앱이 자동 재시작합니다. `src/common.load_secrets()` 가 이 값을 환경변수로
올리므로 코드 수정은 필요 없습니다.

**예비 키를 `GEMINI_API_KEY_2` ~ `_5` 로 넣어 두면 자동으로 돌려씁니다.** 주 키가
무료 한도(429)에 걸리거나 인증에 실패하면 백오프 없이 즉시 다음 키로 넘어갑니다
(400·안전차단·토큰절단은 키를 바꿔도 같으므로 전환하지 않습니다).

> Gemini 키 형식은 `AIza…` 와 `AQ.…` 두 가지가 있습니다. AI Studio 가 현재 발급하는
> 것은 `AQ.` 이며 둘 다 정상입니다 — 형식으로 키를 판단하지 마십시오.

> ⚠️ **키를 저장소에 커밋하지 마십시오.** `.env` 는 `.gitignore` 에 있고, 배포는
> 플랫폼 Secrets 로만 전달합니다. 이 대화에 붙여넣은 키들은 제출 전 재발급을 권합니다.

### 1-3. 자원 한도

| 항목 | 실측값 | Streamlit Cloud 한도 |
|---|---|---|
| 저장소 크기 | 251MB (clone 기준) | 제한 없음 (파일당 100MB) |
| 최대 파일 | `rag_index.joblib` 64MB | 100MB |
| RAG 색인 상주 메모리 | 236MB | — |
| 앱 전체 상주 (추정) | 약 500MB | 약 2.7GB |

첫 배포는 저장소 clone + 의존성 설치로 5~8분 걸립니다. 이후 재시작은 빠릅니다.

RAG 색인은 배포를 감안해 `float32` + 문자 n-gram 상한 40,000 으로 줄였습니다
(91MB → 67MB, 상주 263MB → 236MB). 줄이기 전후 검색 결과를 실측 비교해
**상위 5건 겹침 95% · 1위 일치 11/12** 로 품질이 유지되는 것을 확인했습니다
(`python tools/check_rag_quality.py --compare`).

---

## 2. Hugging Face Spaces (대안)

메모리 여유가 더 필요할 때 씁니다. (비공개 저장소는 Streamlit Cloud 로도 배포
가능합니다 — 1-1-a 참고.)

1. https://huggingface.co/new-space → SDK `Streamlit`, Hardware `CPU basic (무료)`
2. 생성된 Space 저장소에 이 저장소 내용을 push
3. `app_file: src/app.py` 를 `README.md` 프런트매터에 지정
4. Settings → `Variables and secrets` 에 위 키들을 등록

67MB 색인 파일은 Git LFS 로 올리는 편이 안정적입니다.

```bash
git lfs install
git lfs track "outputs/rag_index.joblib"
```

---

## 3. 사내 서버 (KB 내부망 가정)

```bash
pip install -r requirements.txt
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
```

내부망 배포 시에는 `config.yaml` 의 `portfolio.exposure_source` 에 실제 여신 CSV
경로를 넣으십시오. 그러면 화면의 익스포저가 **추정값이 아니라 실제 여신 잔액**으로
바뀌고, 고지 문구도 자동으로 그에 맞게 바뀝니다 (필요 컬럼: `brand_id` 또는
`brand_name`, `exposure_mkrw`).

### 3-1. 폐쇄망 이식 — 무엇이 밖으로 나가고, 막히면 무엇이 남는가

은행 도입 심사에서 반드시 나오는 두 질문에 미리 답한다.
**"이 앱이 어디로 통신하는가"** 와 **"전부 막으면 무엇이 되는가"**.

**아웃바운드 목적지 전량** (이 표에 없는 호스트로는 나가는 요청이 없다)

| 호스트 | 용도 | 주기 | 막히면 |
|---|---|---|---|
| `apis.data.go.kr` | 공정위 가맹사업 공시 6종 수집 | **연 1회** | 기존 스냅샷(`data/raw/`)으로 전 지표 재현 — 점수·등급 무영향 |
| `opendart.fss.or.kr` | 가맹본부 감사보고서 재무 | 연 1회 | 기존 수집분 사용 — 무영향 |
| `generativelanguage.googleapis.com` | Gemini — 뉴스 사건 추출·AI 상담·로고 판독 | 일간 | 규칙기반 폴백 자동 전환, `llm_used=false` 표기 — 점수 무영향 |
| `openapi.naver.com` / `naverapihub.apigw.ntruss.com` | 뉴스 본문·검색수요 | 일간 | 뉴스는 제목만(RSS), 수요 소견은 생성 안 함 — 점수 무영향 |
| `news.google.com` | 뉴스 RSS 보완 | 일간 | 뉴스 층 축소 |
| 브랜드 공식 도메인 다수 | 로고 1회 수집 | 1회성 | 글자 마크로 대체 — 기능 무영향 |

핵심 설계: **점수·등급·포트폴리오 지표는 외부 연결이 하나도 없어도 산출된다.**
연 1회 공시 스냅샷이 `data/raw/` 에 커밋돼 있고, 새 클론에서 API 호출 없이 전 지표가
bit-identical 재현됨을 실측으로 확인했다(→ OPERATIONS §6). 일간 층(뉴스·수요·로고)은
전부 폴백이 있으며, 어느 것이 축소 가동인지 `python tools/check_credentials.py` 가
기동 전에 찍고 화면에도 표기된다 — `src/common.py` 의 `CAPABILITIES` 표가 단일 진실이다.

**오프라인 설치 (외부 PyPI 차단 시)**

```bash
# 인터넷 되는 곳에서: 바퀴 전부 내려받기
pip download -r requirements.txt -d wheelhouse/
# 폐쇄망에서: 인덱스 없이 설치
pip install --no-index --find-links wheelhouse/ -r requirements.txt
```

버전은 `requirements.txt` 에 고정돼 있어 두 환경의 패키지가 동일하다.
Python 3.13 고정 사유는 §5 참조.

**사내 LLM 교체 지점 — 파일 한 개**

모든 LLM 호출은 `src/llm.py` 의 `generate()` / `generate_json()` **두 함수만** 지난다
(화면·뉴스·메모 등 호출부 어디에도 공급자 코드가 없다). 사내 승인 LLM 으로 바꾸려면
이 두 함수의 HTTP 호출부만 교체하면 되고, 폴백·재시도·`llm_used` 표기는 그대로
동작한다. 키가 전혀 없는 상태의 동작은 지금도 매일 검증된다 — 규칙기반 폴백이
정상 경로이기 때문이다.

**데이터 반출입 경계**

- 들어오는 것: 실여신 CSV 1장 (`brand_id/brand_name`, `exposure_mkrw`) — §3 위 참조
- 나가는 것: **없음.** 여신 금액·차주 정보는 어떤 외부 API 로도 전송되지 않는다.
  AI 상담(`src/chat.py`)의 LLM 프롬프트에는 공시 기반 산출물만 들어가며, 포트폴리오
  익스포저 값은 포함되지 않는다(코드 검증: chat.py 에 exposure 참조 0건).

---

## 4. 데이터 갱신

공정거래위원회 공시는 매년 갱신됩니다. 새 연도 공시가 나오면:

```bash
python run_pipeline.py --step collect     # 공시 원본 수집 (DATA_GO_KR_KEY 필요)
python run_pipeline.py --step panel
python run_pipeline.py --step dart        # 가맹본부 재무 (DART_API_KEY 필요)
python run_pipeline.py --step features
python run_pipeline.py --step labels
python run_pipeline.py --step model
python run_pipeline.py --step score
python run_pipeline.py --step demand      # 검색수요 + 브랜드 로고 (NCP 키 필요, 선택)
python run_pipeline.py --step news        # 뉴스 (선택)
python run_pipeline.py --step diagnose    # 브랜드별 진단 소견
python -m src.rag                         # RAG 색인 재구축
```

또는 한 번에 `python run_pipeline.py --step all`.

갱신된 산출물을 커밋해 push 하면 배포된 앱이 자동으로 다시 뜹니다.

---

## 5. 배포 전 점검

```bash
python -m ruff check .
python tests/test_sanity.py
python tests/test_diagnosis.py
python tests/test_llm_paths.py
python tools/check_doc_numbers.py
```

전부 통과해야 합니다. 마지막 확인 시점 기준: ruff 통과 · sanity 6/6 ·
diagnosis 8/8 · LLM/RAG 4/4 · 문서 수치 88건 일치.

**새 clone 재현 확인** — 배포 환경과 같은 조건(키 없음)을 흉내 냅니다.

```bash
git clone https://github.com/AISWKimBeomSu/KB_franSCORE.git /tmp/franscore-check
cd /tmp/franscore-check && pip install -r requirements.txt
streamlit run src/app.py --server.port 8788
```

5개 화면이 모두 뜨고 오류가 없어야 합니다.

---

## 6. 네이버 API 자격증명 얻는 곳 (혼동하기 쉬움)

네이버가 개발자센터 오픈API를 **NAVER API HUB**(네이버 클라우드)로 이관했습니다.
둘은 자격증명도 엔드포인트도 다릅니다.

| | 구 개발자센터 | **NAVER API HUB (현재 사용)** |
|---|---|---|
| 콘솔 | developers.naver.com | ncloud.com → NAVER API HUB |
| 자격증명 | Client ID / Secret | Client ID / Secret (`X-NCP-APIGW-…`) |
| 검색 API | 가능 | 가능 |
| **검색어트렌드** | **불가** (Scope Status Invalid) | **가능** |

Client ID 는 **10자 안팎**, Client Secret 은 **40자**입니다. 둘의 길이가 크게
다르므로 헷갈리면 길이로 구분하십시오.

경로: NCP 콘솔 → `NAVER API HUB` → `Application` → 앱 선택 → `인증 정보`
