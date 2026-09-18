"""API 키 진단 도구 — 키 발급 직후 실행해 바로 쓸 수 있는지 확인한다.

사용법 (프로젝트 루트에서):
    python check_keys.py            # 둘 다 점검
    python check_keys.py --data     # data.go.kr 키만
    python check_keys.py --llm      # Gemini 키만

무엇을 알려주나:
  · 키가 설정돼 있는지, 어떤 형태인지(Encoding/Decoding 판별)
  · 데이터셋 각각에 활용신청이 승인됐는지 (하나씩 실제 호출)
  · 실패 시 원인을 구분해서 안내 (미등록 / 아직 미활성 / 이중 인코딩 / 트래픽 초과)
  · Gemini 키가 실제로 호출 가능한지 (모델 목록 조회 + 설정 모델로 1회 실호출)
"""
from __future__ import annotations

import argparse
import os
import sys

import requests

from src.collect import SERVICES
from src.common import load_config

DATASET_IDS = {
    "brand_frcs_stats": "15110241",
    "brand_overview": "15109828",
    "industry_openclose": "15157660",
    "brand_master": "15125467",
    "brand_region_direct": "15125490",
    # ⚠️ 창업비용(15110265)이 이 표에 없어서 화면에 데이터셋 번호가 '?' 로 찍히고
    #    신청 페이지 링크가 `/data/?/openapi.do` 로 깨져 나갔다. 원천을 하나 추가할
    #    때마다 여기도 함께 늘려야 하는데, 그 사실이 어디에도 적혀 있지 않았다
    #    — 같은 누락이 README 에서도 '6종 vs 7종'으로 한 번 났었다.
    "brand_startup_cost": "15110265",
    "brand_cancel": "15125518",
}

OK, NG, WARN = "  [OK]  ", "  [실패]", "  [주의]"


def _classify(body: dict | None, raw_text: str, status: int) -> tuple[bool, str]:
    """응답 → (성공여부, 사람이 읽을 원인 설명)."""
    txt = (raw_text or "")[:400]
    if status == 401 or "Unauthorized" in txt:
        # apis.data.go.kr 게이트웨이는 잘못되거나 아직 동기화 안 된 키에 XML이 아니라 401을 준다
        return False, ("인증 실패(401). 키 오타이거나 **활용신청 후 동기화 대기 중**입니다. "
                       "포털 게이트웨이는 보통 10~30분 뒤 반영됩니다. 재발급 금지.")
    if body is None:
        if "SERVICE_KEY_IS_NOT_REGISTERED" in txt:
            return False, ("키 미등록(코드 30). 활용신청 직후라면 **인증키 동기화 대기**일 수 "
                           "있습니다(포털 게이트웨이 10~30분). 키를 재발급하지 마세요 — "
                           "재발급하면 기존 키가 폐기되고 동기화가 처음부터 다시 시작됩니다.")
        if "SERVICE_ACCESS_DENIED" in txt:
            return False, ("코드 20 — 이 데이터셋에 **활용신청을 하지 않았습니다**. "
                           "키는 계정당 1개로 모든 API 공용이지만, 데이터셋별 활용신청은 따로 해야 합니다.")
        if "LIMITED_NUMBER_OF_SERVICE_REQUESTS" in txt:
            return False, "일일 트래픽 초과(개발계정 10,000건/일). 내일 다시 시도하세요."
        return False, f"JSON 파싱 실패(응답 앞부분: {txt[:120]!r})"
    # ⚠️ 게이트웨이 오류는 **JSON 파싱에 성공한다.** 다만 형태가 다르다:
    #      {"OpenAPI_ServiceResponse": {"cmmMsgHeader": {"errMsg": "...", "returnAuthMsg": "..."}}}
    #    이 경로를 안 보면 code·msg 가 빈 문자열이 되어 화면에 "resultCode= msg=" 라는
    #    아무 정보 없는 줄이 찍힌다. 실제로 그 상태였고, 원인(활용신청 미승인)을
    #    도구가 알고 있으면서도 말해 주지 못했다.
    gw = (body.get("OpenAPI_ServiceResponse") or {}).get("cmmMsgHeader") or {}
    if gw:
        err = str(gw.get("errMsg") or "")
        auth = str(gw.get("returnAuthMsg") or "")
        return _classify(None, f"{err} {auth}", status)
    code = str(body.get("resultCode", body.get("response", {}).get("header", {}).get("resultCode", "")))
    msg = str(body.get("resultMsg", ""))
    if code in ("00", "0"):
        return True, "정상"
    if code == "30" or "NOT_REGISTERED" in msg:
        return False, "활용신청 미승인 또는 키 미등록(resultCode=30)."
    if code == "22":
        return False, "일일 트래픽 초과(resultCode=22)."
    if code in ("10", "11"):
        return False, f"파라미터 오류(resultCode={code}) — 코드 문제일 수 있으니 알려주세요."
    return False, f"resultCode={code} msg={msg}"


def check_data_key(cfg: dict) -> bool:
    print("=" * 72)
    print("data.go.kr (DATA_GO_KR_KEY) 점검")
    print("=" * 72)
    key = os.environ.get("DATA_GO_KR_KEY", "").strip()
    if not key:
        print(f"{NG} 환경변수 DATA_GO_KR_KEY 가 설정돼 있지 않습니다.")
        print("       PowerShell 영구 설정:")
        print('         [Environment]::SetEnvironmentVariable("DATA_GO_KR_KEY","<키>","User")')
        print("       (설정 후 터미널을 새로 열어야 반영됩니다)")
        return False

    print(f"{OK} 키 감지 (길이 {len(key)})")
    # 키 형식 판별.
    #  · 2025-08-21 이후 발급분: 특수문자 없는 영문·숫자 → 인코딩/디코딩 구분이 폐지됨.
    #    그대로 쓰면 되고, requests의 자동 인코딩도 무해(변환할 문자가 없음).
    #  · 그 이전 발급분(레거시): 원본 base64라 + / = 포함 → 이 프로젝트는 requests가
    #    자동 인코딩하므로 **Decoding(원본) 키**를 써야 한다.
    if "%" in key and not any(c in key for c in "+/="):
        print(f"{WARN} **이미 URL 인코딩된 키(Encoding)로 보입니다** (% 포함).")
        print("       이 프로젝트는 requests가 자동 인코딩하므로 이중 인코딩이 되어 실패합니다.")
        print("       레거시 키라면 'Decoding(원본)' 값을 쓰세요.")
    elif any(c in key for c in "+/="):
        print(f"{OK} 레거시 Decoding(원본) 키 형태 (+ / = 포함) — 이 프로젝트에 맞습니다.")
    elif key.isalnum():
        print(f"{OK} 신형 키 형태 (영문·숫자만) — 2025-08-21 이후 발급분.")
        print("       인코딩/디코딩 구분이 폐지되어 그대로 사용하면 됩니다.")
    else:
        print(f"{WARN} 형식을 단정할 수 없습니다. 실패하면 원본(Decoding) 값으로 바꿔보세요.")

    print()
    print("데이터셋별 활용신청 승인 여부 (각 1건씩 실제 호출):")
    all_ok = True
    for name, spec in SERVICES.items():
        ds = DATASET_IDS.get(name, "?")
        params = {"serviceKey": key, "pageNo": 1, "numOfRows": 1,
                  "resultType": "json", spec["year_param"]: 2023}
        try:
            r = requests.get(spec["url"], params=params, timeout=20)
            try:
                body = r.json()
            except ValueError:
                body = None
            ok, why = _classify(body, r.text, r.status_code)
        except requests.RequestException as e:
            ok, why = False, f"네트워크 오류: {type(e).__name__}"
        total = ""
        if ok and isinstance(body, dict):
            total = f" (totalCount={body.get('totalCount', '?')})"
        print(f"{OK if ok else NG} {ds}  {name:22s}{total}")
        if not ok:
            print(f"           → {why}")
            print(f"           → 신청 페이지: https://www.data.go.kr/data/{ds}/openapi.do")
            all_ok = False

    print()
    if all_ok:
        print(f"  ✅ {len(SERVICES)}개 데이터셋 전부 사용 가능 — "
              "`python run_pipeline.py --step collect` 실행하면")
        print("     본인 키로 원본 스냅샷을 재수집합니다.")
    else:
        print("  ⚠️ 일부 데이터셋이 아직 사용 불가합니다. 위 신청 페이지에서 활용신청 후")
        print("     (자동승인이라도 반영에 시간이 걸릴 수 있음) 다시 실행하세요.")
    return all_ok


def check_llm_key(cfg: dict) -> bool:
    from src import llm

    env_name = llm.api_key_env(cfg)
    model = llm.model_name(cfg)
    print()
    print("=" * 72)
    print(f"Google Gemini ({env_name}) 점검")
    print("=" * 72)
    key = llm.api_key(cfg)
    if not key:
        print(f"{NG} 환경변수 {env_name} 가 설정돼 있지 않습니다.")
        print("       PowerShell 영구 설정:")
        print(f'         [Environment]::SetEnvironmentVariable("{env_name}","<키>","User")')
        print("       (설정 후 터미널을 새로 열어야 반영됩니다)")
        print("       미설정 상태에서도 파이프라인은 규칙 폴백으로 정상 동작합니다.")
        return False
    masked = key[:6] + "..." + key[-4:] if len(key) > 14 else "(짧음)"
    print(f"{OK} 키 감지: {masked} (길이 {len(key)})")
    if not (key.startswith("AIza") or key.startswith("AQ.")):
        print(f"{WARN} Gemini 키는 보통 'AIza'(레거시) 또는 'AQ.'(신형)로 시작합니다. 형식 확인 필요.")

    # 1) 모델 목록 조회 — 키 자체의 유효성과 설정 모델의 접근 가능 여부를 함께 본다.
    base = str(cfg["llm"].get("api_base", "https://generativelanguage.googleapis.com/v1beta"))
    try:
        r = requests.get(f"{base.rstrip('/')}/models",
                         headers={"X-goog-api-key": key}, timeout=30)
        if r.status_code != 200:
            print(f"{NG} 모델 목록 조회 실패 (HTTP {r.status_code})")
            body = r.text[:400].replace(key, "***KEY***")
            print(f"       → {body}")
            if r.status_code in (401, 403):
                print("       → 키가 유효하지 않거나 Generative Language API가 비활성입니다.")
                print("       → https://aistudio.google.com/apikey 에서 키·프로젝트를 확인하세요.")
            return False
        names = [m.get("name", "").removeprefix("models/") for m in r.json().get("models", [])
                 if "generateContent" in (m.get("supportedGenerationMethods") or [])]
        print(f"{OK} 모델 목록 조회 성공 (generateContent 지원 {len(names)}종)")
        if model in names:
            print(f"{OK} 설정 모델 사용 가능: {model}")
        else:
            print(f"{NG} 설정 모델 '{model}' 이(가) 목록에 없습니다. config.yaml 의 llm.model 을 바꾸세요.")
            print(f"       → 사용 가능한 예: {', '.join(names[:6])}")
            return False
    except requests.RequestException as e:
        print(f"{NG} 네트워크 오류: {type(e).__name__}")
        return False

    # 2) 실제 파이프라인이 쓰는 코드 경로(src.llm.generate)로 1회 호출 — 진짜 동작 확인.
    try:
        text, meta = llm.generate(
            cfg, system="You are a connectivity check.",
            user="Reply with exactly: OK", max_tokens=512,
        )
        u = meta.get("usage", {})
        print(f"{OK} 실호출 성공 (model={meta['model']}, 응답={text.strip()[:40]!r})")
        print(f"       토큰: 입력 {u.get('promptTokenCount', '?')} / "
              f"출력 {u.get('candidatesTokenCount', '?')} / "
              f"사고 {u.get('thoughtsTokenCount', 0)}")
        print()
        print("  ✅ LLM 층 활성화 — 다음 실행 시 실제 Gemini가 사용됩니다:")
        print("     python run_pipeline.py --step news     (뉴스 신호 구조화 추출)")
        print("     streamlit run src/app.py               (심사메모 실제 생성)")
        return True
    except llm.LLMError as e:
        print(f"{NG} 실호출 실패: {type(e).__name__}")
        print(f"       → {str(e)[:400]}")
        low = str(e).lower()
        if "quota" in low or "429" in low:
            print("       → 무료 등급 분당/일일 한도일 수 있습니다. 잠시 후 재시도하세요.")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="FranSCORE API 키 진단")
    ap.add_argument("--data", action="store_true", help="data.go.kr 키만 점검")
    ap.add_argument("--llm", action="store_true", help="Gemini 키만 점검")
    args = ap.parse_args()
    both = not (args.data or args.llm)
    cfg = load_config()
    ok = True
    if both or args.data:
        ok &= check_data_key(cfg)
    if both or args.llm:
        ok &= check_llm_key(cfg)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
