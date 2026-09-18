"""FranSCORE 파이프라인 러너.

사용법 (프로젝트 루트에서):
    python run_pipeline.py --step all            # M1→M2→M3→M4 전체
    python run_pipeline.py --step collect        # 개별 스텝
    python run_pipeline.py --step panel|features|labels|model|evaluate|portfolio|correlation|news
    python run_pipeline.py --step all --demo     # 합성 패널 스모크 (outputs/_smoke 격리)

각 스텝은 파일 기반 중간 산출물로 독립 실행 가능하다.
--demo 는 제출 지표를 오염시키지 않도록 outputs/_smoke 로 격리된다.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time

import pandas as pd

from src.common import get_logger, load_config, make_synthetic_panel, set_seed

log = get_logger("pipeline")

# dart·ifrmp 는 panel 뒤·features 앞에 온다: 본부 재무를 붙이려면 패널의 법인명이 필요하고,
# 그 결과(hq_financials.parquet)를 features 가 읽는다.
# demand(검색수요)·news 는 diagnose 앞에 온다: 진단 규칙이 그 산출물을 읽어
# 수요·평판 소견을 만든다. 둘 다 없어도 diagnose 는 공시·재무 소견만으로 동작한다.
# ⚠️ 순서가 의미를 갖는다. bands 는 evaluate 산출물(워크포워드 예측)을 읽어 등급 컷과
#    배포용 보정기를 만들고, score 는 그 둘을 읽어 등급을 매긴다. validate 는 마지막에
#    전부를 검증한다. 순서를 바꾸면 조용히 옛 컷으로 점수가 나간다.
STEPS = ["collect", "panel", "dart", "ifrmp", "features", "labels", "model", "evaluate",
         "bands", "portfolio", "score", "correlation", "demand", "news", "diagnose",
         "eval_llm", "validate"]


def _extended_cfg(cfg: dict) -> dict:
    """확장 표본 트랙(명세 §11 플랜B): 전 업종·점포20+. 산출물은 outputs/extended/ 로 격리.

    ⚠️ 체리피킹 방지 — 기본 트랙과 **둘 다** 공개하는 것을 전제로 한 사전 선언 트랙이다.
    """
    ext = copy.deepcopy(cfg)
    rel = cfg["sample"]["relaxed"]
    ext["sample"]["industry_major"] = list(rel["industry_major"])
    ext["sample"]["min_stores"] = int(rel["min_stores"])
    base_out = cfg["_root"] / "outputs" / "extended"
    ext["paths"]["outputs"] = base_out
    ext["paths"]["processed"] = base_out / "processed"
    for p in (ext["paths"]["outputs"], ext["paths"]["processed"]):
        p.mkdir(parents=True, exist_ok=True)
    return ext


def _demo_cfg(cfg: dict) -> dict:
    """--demo: 산출물 경로를 스모크 디렉토리로 격리한 cfg 사본."""
    demo = copy.deepcopy(cfg)
    smoke = cfg["_root"] / "outputs" / "_smoke"
    demo["paths"]["outputs"] = smoke
    demo["paths"]["processed"] = smoke / "processed"
    for p in (demo["paths"]["outputs"], demo["paths"]["processed"]):
        p.mkdir(parents=True, exist_ok=True)
    return demo


def _load_panel(cfg: dict, demo: bool) -> pd.DataFrame:
    if demo:
        return make_synthetic_panel(cfg)
    panel_path = cfg["paths"]["processed"] / "panel.parquet"
    if not panel_path.exists():
        log.error("panel.parquet 없음 — 먼저 `--step panel` 실행 필요: %s", panel_path)
        sys.exit(1)
    return pd.read_parquet(panel_path)


def run_step(step: str, cfg: dict, demo: bool) -> None:
    t0 = time.time()
    log.info("=== step: %s%s ===", step, " (DEMO/합성)" if demo else "")

    if step == "collect":
        if demo:
            log.info("demo 모드에서는 수집 생략 (합성 패널 사용)")
            return
        from src import collect
        collect.collect_all(cfg)

    elif step == "panel":
        if demo:
            panel = make_synthetic_panel(cfg)
            panel.to_parquet(cfg["paths"]["processed"] / "panel.parquet", index=False)
            from src import panel as panel_mod
            panel_mod.survival_report(panel, cfg)
        else:
            from src import entity
            from src import panel as panel_mod
            master = entity.build_master(cfg)
            panel = panel_mod.build_panel(cfg, master)
            panel_mod.survival_report(panel, cfg)

    elif step == "dart":
        # 키가 없거나 네트워크가 막혀도 파이프라인 전체를 세우지 않는다 —
        # 본부 재무는 '있으면 더 좋은' 층이고, 없으면 features 가 f_hq_* 를 만들지 않는다.
        if demo:
            log.info("demo 모드 — 외부 API 호출(DART) 생략")
        else:
            try:
                from src import dart
                dart.collect_financials(cfg)
            except Exception as exc:
                log.error("DART 수집 실패 — 본부 재무 피처 없이 계속 진행합니다: %s",
                          str(exc)[:200])

    elif step == "ifrmp":
        if demo:
            log.info("demo 모드 — 외부 API 호출(정보공개서) 생략")
        else:
            try:
                from src import ifrmp
                ifrmp.build(cfg)
            except Exception as exc:
                log.error("정보공개서 수집 실패 — 계속 진행합니다: %s", str(exc)[:200])
        # 웹 열람으로 사람이 받아 둔 본부 재무를 hq_financials 에 합친다.
        # ⚠️ 반드시 dart 다음·features 앞이어야 한다. dart 가 파일을 다시 쓰므로
        #    앞에 두면 합친 행이 그대로 날아가고, 뒤에 두면 모형이 못 본다.
        #    (실측: 이 배선이 없어서 손으로 받은 30개 브랜드를 모형이 한 번도 못 봤다.)
        try:
            from src import ifrmp_web
            log.info("본부 재무 통합: %s", ifrmp_web.merge_into_hq(cfg))
        except Exception as exc:
            log.error("본부 재무 통합 실패 — 계속 진행합니다: %s", str(exc)[:200])

    elif step == "features":
        from src import features
        panel = _load_panel(cfg, demo)
        feats = features.build_features(panel, cfg)
        feats.to_parquet(cfg["paths"]["processed"] / "features.parquet", index=False)
        log.info("features: %s rows × %s cols", *feats.shape)

    elif step == "labels":
        from src import labels
        panel = _load_panel(cfg, demo)
        # 분위수 풀은 업종 범위 전체 패널로 계산하고,
        # 학습/평가 표본은 실시간 자격(eligible_t) 행만 남긴다 (룩어헤드 선별 방지).
        lab = labels.build_labels(panel, cfg)
        if "eligible_t" in panel.columns:
            elig = panel[["brand_id", "year", "eligible_t"]]
            n0 = len(lab)
            lab = (lab.merge(elig, on=["brand_id", "year"], how="left"))
            lab = lab[lab["eligible_t"].fillna(False)].drop(columns=["eligible_t"])
            log.info("labels: 실시간 자격 필터 %d → %d행", n0, len(lab))
        # 학습·검증에 실제로 쓰이는 표본의 구성표. build_labels 안에서 쓰면 자격
        # 필터 이전 프레임이 기록돼 문서가 다른 숫자를 인용하게 된다(실제로 그랬다).
        labels.write_composition(lab, cfg, scope=labels.SAMPLE_SCOPE)
        lab.to_parquet(cfg["paths"]["processed"] / "labels.parquet", index=False)
        log.info("labels: %s rows, 양성률 %.1f%%", len(lab), 100 * lab["label"].mean())

    elif step == "model":
        from src import model
        feats = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
        lab = pd.read_parquet(cfg["paths"]["processed"] / "labels.parquet")
        model.train_all(feats, lab, cfg)

    elif step == "evaluate":
        from src import evaluate
        evaluate.evaluate_all(cfg)

    elif step == "portfolio":
        from src import portfolio
        portfolio.build_portfolio(cfg)

    elif step == "eval_llm":
        # LLM이 규칙기반보다 실제로 나은지 정량 측정 (정답셋 대조)
        from src import eval_llm
        eval_llm.run(cfg)

    elif step == "score":
        # 운영 점수 산출 — 라벨이 아직 없는 최신 코호트 (학습 없이 산출물만 사용)
        from src import score
        score.score_latest(cfg)

    elif step == "correlation":
        # 브랜드 공통요인 상관 실증 (portfolio 이후 — 손실 영향 계산에 exposure·PD가 필요)
        if demo:
            log.info("demo 모드에서는 상관 실증 생략 (지역 원본 스냅샷 필요)")
            return
        from src import correlation
        correlation.run(cfg)

    elif step == "news":
        from src import news_llm
        if demo:
            # --demo 는 합성 스모크다. 실제 RSS를 때리고 git 추적 대상 data/raw/news 를
            # 오염시키면 안 된다(감사 지적 — 격리 약속이 깨져 있었다).
            log.info("demo 모드에서는 뉴스 수집 생략 (실제 RSS·스냅샷 오염 방지)")
            return
        names = news_llm.select_brands(cfg)
        news_llm.run_news(names, cfg)

    elif step == "demand":
        # 네이버 데이터랩 검색어트렌드 — 브랜드 수요 신호 (키 없으면 스스로 건너뛴다)
        if demo:
            log.info("demo 모드에서는 검색수요 수집 생략")
            return
        from src import naver
        naver.collect_demand(cfg)
        naver.collect_logos(cfg)          # 브랜드 로고도 같은 자격증명으로 함께 채운다

    elif step == "diagnose":
        # 브랜드 개별 진단 — 공시·본부재무·수요·보도를 규칙으로 훑어 소견을 만든다
        from src import diagnosis
        diagnosis.run(cfg)

    elif step == "bands":
        # 등급 밴드 도출 — 관측 악화율로 앵커링된 절대 컷 + 배포용 보정기
        from tools.derive_grade_bands import main as derive_bands
        derive_bands()

    elif step == "validate":
        # 모형 검증 산출물 — 판별력·정확성·안정성 (PSI·KS·HL·이행행렬)
        from src import validation
        validation.run(cfg)

    else:
        raise ValueError(f"unknown step: {step}")

    log.info("=== step %s 완료 (%.1fs) ===", step, time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description="FranSCORE pipeline")
    ap.add_argument("--step", default="all", choices=[*STEPS, "all"])
    ap.add_argument("--demo", action="store_true", help="합성 패널 스모크 (outputs/_smoke 격리)")
    ap.add_argument("--scope", default="primary", choices=["primary", "extended"],
                    help="primary=명세 표본(외식·점포30+) / extended=플랜B 확장(전업종·점포20+)")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    if args.scope == "extended":
        cfg = _extended_cfg(cfg)
        log.info("확장 표본 트랙(플랜B): 업종=%s, 점포하한=%s → 산출물 %s",
                 cfg["sample"]["industry_major"], cfg["sample"]["min_stores"],
                 cfg["paths"]["outputs"])
    if args.demo:
        cfg = _demo_cfg(cfg)

    steps = STEPS if args.step == "all" else [args.step]
    if args.step == "all":
        # news·demand 는 외부 API 의존이라 실패를 허용한다. 다만 diagnose 는 그 뒤에
        # 와야 하므로 순서를 유지한 채 두 스텝만 예외로 감싼다 (실패해도 diagnose 는
        # 공시·재무 소견만으로 계속 진행 — 소견이 통째로 사라지지 않는다).
        soft = {"news", "demand"}
        for s in steps:
            if s in soft:
                try:
                    run_step(s, cfg, args.demo)
                except Exception as e:
                    log.warning("%s 스텝 실패 (파이프라인은 계속): %s", s, e)
            else:
                run_step(s, cfg, args.demo)
    else:
        run_step(args.step, cfg, args.demo)


if __name__ == "__main__":
    main()
