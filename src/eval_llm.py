"""M4 평가 — LLM 추출이 규칙기반보다 **실제로 나은지** 정량 측정한다.

왜 필요한가 (자체 감사 critical 지적):
    기존에는 LLM이 "동작한다"만 검증돼 있었고 "규칙기반보다 낫다"는 근거가 0건이었다.
    근거 없이 LLM을 붙이면 그 자체가 보여주기다. 그래서 정답셋을 만들고 두 방식을
    **같은 입력·같은 지표**로 채점한다.

정답셋: data/eval/news_gold.jsonl (생성기 tools/build_news_gold.py 에 라벨 기준 명시)
    · real      — 실제 수집된 기사 제목에 사람이 라벨을 붙인 것
    · synthetic — 실수집분에 사례가 0건인 위험 유형의 재현율을 재기 위한 프로브
    두 부분을 **분리 보고**한다. 합쳐서 하나의 점수로 뭉개면 해석이 불가능해진다.

지표: 클래스별 precision/recall/F1 + macro-F1, 그리고 여신 실무에서 실제로 중요한
    **위험 탐지 지표**(위험 4종 vs 비위험 2종의 이진 분류)를 따로 낸다.
    운영에서는 위험을 놓치는 비용이 오탐 비용보다 크므로 recall 을 함께 본다.

⚠️ 표본이 작다(수십 건). 소수점 이하를 유의미한 차이로 읽지 말고, 방향과 실패 유형을 보라.
   실패 사례는 전부 outputs/llm_eval_errors.csv 에 남겨 사람이 직접 확인할 수 있게 한다.

실행: python -m src.eval_llm            (키 없으면 규칙기반만 채점하고 그 사실을 밝힘)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src import llm
from src.common import get_logger, load_config, set_seed
from src.news_llm import (
    EVENT_TYPES,
    RISK_EVENT_TYPES,
    _extract_with_llm,
    _extract_with_rules,
)

log = get_logger("eval_llm")

GOLD = Path("data/eval/news_gold.jsonl")


def load_gold(root: Path) -> pd.DataFrame:
    p = root / GOLD
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — `python tools/build_news_gold.py` 먼저 실행")
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return pd.DataFrame(rows)


def _score(gold: list[str], pred: list[str]) -> dict:
    """클래스별 precision/recall/F1 + macro-F1 + 정확도."""
    labels = [c for c in EVENT_TYPES if c in set(gold) | set(pred)]
    per, f1s = {}, []
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"n_gold": tp + fn, "precision": round(prec, 3),
                  "recall": round(rec, 3), "f1": round(f1, 3)}
        if tp + fn:                      # 정답에 존재하는 클래스만 macro 에 포함
            f1s.append(f1)
    acc = sum(1 for g, p in zip(gold, pred, strict=True) if g == p) / max(len(gold), 1)

    # 여신 실무 관점: '위험이냐 아니냐'의 이진 판정
    gb = [g in RISK_EVENT_TYPES for g in gold]
    pb = [p in RISK_EVENT_TYPES for p in pred]
    tp = sum(1 for g, p in zip(gb, pb, strict=True) if g and p)
    fp = sum(1 for g, p in zip(gb, pb, strict=True) if not g and p)
    fn = sum(1 for g, p in zip(gb, pb, strict=True) if g and not p)
    rp = tp / (tp + fp) if tp + fp else 0.0
    rr = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(gold), "accuracy": round(acc, 3),
        "macro_f1": round(sum(f1s) / len(f1s), 3) if f1s else 0.0,
        "risk_precision": round(rp, 3), "risk_recall": round(rr, 3),
        "risk_f1": round(2 * rp * rr / (rp + rr), 3) if rp + rr else 0.0,
        "per_class": per,
    }


def _predict_rules(df: pd.DataFrame) -> list[str]:
    """규칙기반은 브랜드 개체 판정을 못 하므로 '무관'을 낼 수 없다 — 그 한계 그대로 채점."""
    out = []
    for _, r in df.iterrows():
        art = [{"title": r["title"], "link": "", "published": "", "source": ""}]
        out.append(_extract_with_rules(r["brand"], art)[0]["event_type"])
    return out


def _predict_llm(df: pd.DataFrame, cfg: dict) -> tuple[list[str], int]:
    """브랜드 단위로 묶어 실제 운영과 동일한 배치 호출 경로를 태운다.

    반환: (예측 목록, **폴백 발생 브랜드 수**).
    ⚠️ 폴백이 하나라도 있으면 그 결과는 'LLM 성능'이 아니다 — 호출자가 무효 처리한다.
       (쿼터 초과로 절반이 폴백된 점수를 LLM 점수로 보고하면 그 자체가 허위다.)
    """
    pace = float(cfg["llm"].get("eval_pace_sec", 0.0))
    preds: dict[int, str] = {}
    n_fallback = 0
    groups = list(df.groupby("brand", sort=False))
    for i, (brand, sub) in enumerate(groups):
        arts = [{"title": t, "link": "", "published": "", "source": ""} for t in sub["title"]]
        got = _extract_with_llm(str(brand), arts, cfg)
        if got is None:
            n_fallback += 1
            log.warning("LLM 추출 실패(brand=%s) — 이 실행의 LLM 지표는 무효 처리된다", brand)
            got = [{"event_type": "기타"} for _ in arts]
        for idx, sig in zip(sub.index, got, strict=False):
            preds[idx] = sig["event_type"]
        if pace > 0 and i < len(groups) - 1:
            time.sleep(pace)   # 무료 등급 분당 한도 회피 (429 폭주 방지)
    return [preds.get(i, "기타") for i in df.index], n_fallback


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    root = Path(cfg.get("_root", "."))
    df = load_gold(root).reset_index(drop=True)
    out_dir = Path(cfg["paths"]["outputs"])

    df["pred_rules"] = _predict_rules(df)
    use_llm = llm.is_enabled(cfg)
    n_fallback = 0
    if use_llm:
        preds, n_fallback = _predict_llm(df, cfg)
        df["pred_llm"] = preds
        if n_fallback:
            # 폴백이 섞인 점수는 LLM 성능이 아니다. 숫자를 내보내지 않는다.
            log.error("LLM 폴백 %d개 브랜드 발생(쿼터·차단 등) — **LLM 지표를 무효 처리**한다. "
                      "쿼터 회복 후 재실행하거나 config llm.eval_pace_sec 을 늘리세요.", n_fallback)
            use_llm = False
    else:
        log.warning("LLM 키 없음 — 규칙기반만 채점한다 (LLM 열은 비움)")
        df["pred_llm"] = None

    result: dict = {"model": llm.model_name(cfg) if use_llm else None,
                    "llm_evaluated": bool(use_llm),
                    "llm_fallback_brands": int(n_fallback),
                    "llm_invalidated_reason": (
                        f"{n_fallback}개 브랜드에서 폴백 발생 — 폴백 예측이 섞인 점수는 "
                        "LLM 성능이 아니므로 보고하지 않는다" if n_fallback else None),
                    "gold_file": str(GOLD)}
    for scope in ("all", "real", "synthetic"):
        sub = df if scope == "all" else df[df["source"] == scope]
        if sub.empty:
            continue
        block = {"rules": _score(sub["gold"].tolist(), sub["pred_rules"].tolist())}
        if use_llm:
            block["llm"] = _score(sub["gold"].tolist(), sub["pred_llm"].tolist())
        result[scope] = block

    (out_dir / "llm_eval.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 오분류 전수 기록 — 숫자만 보고 넘어가지 않도록 사람이 볼 수 있게 남긴다
    err = df[(df["gold"] != df["pred_rules"]) |
             (df["pred_llm"].notna() & (df["gold"] != df["pred_llm"]))]
    err[["source", "brand", "title", "gold", "pred_rules", "pred_llm"]].to_csv(
        out_dir / "llm_eval_errors.csv", index=False, encoding="utf-8-sig")

    for scope in ("real", "synthetic", "all"):
        if scope not in result:
            continue
        r = result[scope]["rules"]
        line = (f"[{scope:9s}] n={r['n']:3d} | 규칙: acc {r['accuracy']:.3f} "
                f"macroF1 {r['macro_f1']:.3f} 위험F1 {r['risk_f1']:.3f}")
        if use_llm:
            m = result[scope]["llm"]
            line += (f"  ||  LLM: acc {m['accuracy']:.3f} macroF1 {m['macro_f1']:.3f} "
                     f"위험F1 {m['risk_f1']:.3f} (재현율 {m['risk_recall']:.3f})")
        log.info(line)
    log.info("오분류 %d건 → llm_eval_errors.csv (사람이 직접 확인할 것)", len(err))
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LLM vs 규칙기반 정량 평가")
    ap.add_argument("--model", help="config 대신 사용할 모델 ID (예: gemini-2.5-flash). "
                                    "모델 간 비교나 쿼터 회피에 사용")
    ap.add_argument("--pace", type=float, help="브랜드 배치 간 대기(초) 재정의")
    _a = ap.parse_args()
    _cfg = load_config()
    if _a.model:
        _cfg["llm"]["model"] = _a.model
    if _a.pace is not None:
        _cfg["llm"]["eval_pace_sec"] = _a.pace
    run(_cfg)
