# 제출 안내 — KB 제8회 AI Challenge · FranSCORE

> 마감 **2026-08-03 16:00**. 이 문서는 "무엇을 어떻게 내는가"와 **"내고 나서 무엇을 하면 안 되는가"** 를 적는다.

---

## 0. 🛑 가장 먼저 — 제출 후 커밋 금지

공모 안내문:

> 구현코드의 경우 깃허브 링크 제출 가능하나, **참가접수기간 이후 수정·변경 이력 확인 시
> 심사 대상 제외**

즉 깃허브 링크로 내는 순간 **커밋 히스토리 자체가 심사 대상**이 된다. 접수 마감(16:00) 이후에
찍힌 커밋이 하나라도 있으면 실격 사유다.

**지켜야 할 것**

| 언제 | 무엇 |
|---|---|
| 제출 **전** | 커밋·푸시 자유. 마지막 푸시를 마치고 `git log -1 --date=iso` 로 시각을 확인한다 |
| 제출 **직전** | 아래 §1 체크리스트를 돌리고, 통과하면 태그를 찍는다 |
| 제출 **후** | **어떤 커밋도 하지 않는다.** 오타 하나도 고치지 않는다 |

✅ **자동 커밋은 이미 차단해 두었다.** 이 저장소에는 산출물을 실제로 커밋·푸시하는
GitHub Actions 워크플로(`.github/workflows/daily-refresh.yml`)가 있었고, 매일 06:00 KST 에
돌도록 되어 있었다(실제로 봇 커밋 1건이 히스토리에 남아 있다). 그대로 두면 **마감 다음 날
아침에 봇 커밋이 찍혀 그것만으로 실격**이므로, 제출 전에 `schedule:` 트리거를 제거하고
`workflow_dispatch`(수동 실행 버튼)만 남겼다. 배치 기능은 그대로 살아 있고 Actions 탭에서
눌러 확인할 수 있다. **심사가 끝난 뒤** 워크플로 파일의 주석을 풀면 일간 자동 갱신이 돌아온다.

---

## 1. 제출 직전 체크리스트

```bash
cd franscore

# ① 코드·문서·산출물이 서로 맞는가
python -m pytest tests -q                 # 기대: 31 passed
python -m ruff check .                    # 기대: All checks passed!
python tools/check_doc_numbers.py         # 기대: 문서 수치 186건 전부 일치

# ② 비밀값이 커밋에 섞이지 않았는가 (.env 는 gitignore 대상)
git grep -nE "AIza[0-9A-Za-z_-]{30,}|AQ\.[0-9A-Za-z_-]{20,}" -- . ; echo "(위가 비어야 정상)"

# ③ 남은 변경이 없는가
git status --short                        # 기대: 출력 없음
```

전부 통과하면 태그를 찍는다.

```bash
git tag -a submit-kb8 -m "KB 제8회 AI Challenge 제출본" && git push origin submit-kb8
git log -1 --date=iso --format="최종 커밋: %h %ad %s"
```

---

## 2. 자동 커밋 차단 — 완료됨 (확인만 하면 된다)

`.github/workflows/daily-refresh.yml` 의 `schedule:` 트리거를 제거했다. 남은 트리거는
`workflow_dispatch`(수동 버튼) 하나뿐이므로 **저절로 커밋되는 일이 없다.**

확인:

```bash
grep -n "cron\|schedule:" .github/workflows/daily-refresh.yml   # 주석 안에만 있어야 정상
```

한 겹 더 안전하게 하려면 GitHub 웹 → Actions 탭 → `daily-refresh` → `⋯` →
**Disable workflow** 를 눌러 둔다(저장소 파일을 건드리지 않아 커밋이 생기지 않는다).

**심사 종료 후** 워크플로 파일의 주석 두 줄을 풀면 일간 자동 갱신이 되살아난다.

---

## 3. 제출물 구성

| 제출 항목 | 무엇을 낼 것인가 |
|---|---|
| **구현 코드** | 깃허브 저장소 링크 — `https://github.com/AISWKimBeomSu/KB_franSCORE` |
| **가동 서비스** | <https://kb-franscore.streamlit.app> (심사위원이 바로 눌러 볼 수 있는 화면) |
| **기술설명서** | [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) — 전 과정 + 자기적발 결함 전수 |

### 저장소 공개 범위 — ⚠️ 반드시 확인

제출 당시에는 두 가지 이유로 저장소를 비공개로 두었다 — 공정위 공개 미리보기 키가 코드에
하드코딩돼 있었고, `data/raw/brand_master_*.json` 에 대표자 성명 84,368행이 있었다.
**심사가 끝난 뒤 공개로 전환하면서 둘 다 정리했다.**

1. 미리보기 키는 코드에서 뺐다. 새로 수집할 때만 본인 키(`DATA_GO_KR_KEY`)가 필요하다.
2. 대표자 성명은 수집 단계에서 버리고 기존 스냅샷에서도 삭제했다(산출물 영향 0 — `tests/test_privacy.py`).

### 키 재발급 권고

`.env` 는 커밋되지 않았고 스테이징 diff 를 매 커밋마다 검사해 노출 0건을 확인했다.
그럼에도 개발 중 여러 키를 다뤘으므로, **제출 후 사용하지 않는 키는 폐기·재발급**을 권한다.
(Gemini · data.go.kr · DART · 네이버)

---

## 4. 심사위원이 10분 안에 볼 동선

이 순서로 보도록 README 상단을 구성해 두었다.

1. **README 최상단** — 가동 URL → 문제 정의(한국은행·공정위 출처 명시) → 실측 표
   (ρ_W 0.416 vs ρ_B 0.005, 비예상손실 7.69배)
2. **URL 클릭 → FRANSCORE 화면** — 브랜드명 두 글자 입력 → 「진단 근거 보기」
   → 그 브랜드가 왜 그 등급인지 공시 원수치까지 내려간다
3. **점검 큐** — 담당자 지정·처리상태·메모까지 남는 업무 화면
4. **여신 포트폴리오** — 브랜드 집중도와 상관 손실을 원화로
5. **기술설명서 제7부** — 우리가 스스로 잡은 결함 전수와 **철회한 주장 7건**

---

## 5. 재현 방법 (심사위원이 직접 돌려 볼 때)

```bash
git clone <저장소> franscore && cd franscore
pip install -r requirements.txt
streamlit run src/app.py            # API 키 없이 전 화면 동작
```

`data/raw/` 스냅샷과 `outputs/` 산출물을 저장소에 포함했으므로 **API 키 없이** 화면과 지표가
그대로 재현된다. 수집을 다시 돌릴 때만 키가 필요하다.
