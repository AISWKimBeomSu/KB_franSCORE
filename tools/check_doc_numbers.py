"""문서에 적힌 수치가 실제 산출물과 일치하는지 자동 대조한다.

왜 필요한가 (정직성):
    파이프라인을 다시 돌리면 지표가 바뀐다. 문서의 숫자는 사람이 손으로 고치므로
    어딘가 하나는 반드시 낡은 채로 남는다. "문서 수치가 산출물과 일치한다"는 주장은
    주장으로 두지 말고 **기계가 매번 확인**해야 한다.

동작:
    ① 산출물(outputs/, data/processed/)에서 핵심 수치를 다시 계산한다.
       — 기대값을 이 파일에 적어두지 않는다. 적어두면 그 자체가 또 낡는다.
    ② 그 수치를 문서 표기 형태(예: `2.456`, `3,635`, `9.2%`)로 만들어,
       지정한 문서 파일에 그 문자열이 실제로 들어 있는지 확인한다.
    ③ 폐기된 표기(STALE)가 문서 어디에도 남아 있지 않은지 확인한다.
       (공급자 교체·지표 변경 후 잔존물을 잡는다.)

한계(명시): 문자열 포함 검사이므로 "숫자가 문서에 있다"는 것까지만 보증하고
    그 숫자가 올바른 문맥에 쓰였는지는 사람이 본다. 그래도 낡은 수치는 확실히 잡는다.

실행: python tools/check_doc_numbers.py      (종료코드 0 = 전부 일치)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
PROC = ROOT / "data" / "processed"

DOCS = {
    "README": ROOT / "README.md",
    # README 는 포트폴리오용 요약이 됐고, 전체 표·신뢰구간은 docs/RESULTS.md 로 옮겼다.
    # 상세 수치는 RESULTS 에서, README 가 인용하는 헤드라인 수치는 README 에서 따로 대조한다.
    "RESULTS": ROOT / "docs" / "RESULTS.md",
    "IMPL": ROOT / "docs" / "IMPLEMENTATION.md",
    "IFACE": ROOT / "docs" / "INTERFACES.md",
    "LEAK": ROOT / "docs" / "LEAKAGE_CHECKLIST.md",
    # ⚠️ 아래 4개는 오랫동안 이 목록에 없었다. 그 결과 대조는 88/88 통과인데
    #    이 문서들 안의 수치는 낡아 있었다 — FS3 등급이 404개로 적혀 있었지만
    #    실제는 367개였고, 학습 모집단 밖은 708개(49.1%)로 적혀 있었지만 550개(38.1%)
    #    였다. 검사 대상에 없으면 통과 숫자는 안전의 증거가 아니라 착시다.
    "SPEC": ROOT / "docs" / "MODEL_USE_SPEC.md",
    "CONCEPT": ROOT / "docs" / "RATING_CONCEPT.md",
    "METHOD": ROOT / "docs" / "METHODOLOGY.md",
    "OPS": ROOT / "docs" / "OPERATIONS.md",
    # DEPLOY.md 도 같은 이유로 뒤늦게 들어왔다 — "로고 535건" 이 실제 611건이
    # 되도록 수집이 진행되는 동안에도 검사에 걸리지 않았다.
    "DEPLOY": ROOT / "DEPLOY.md",
    # 기술설명서는 전 과정을 한 파일에 모으므로 낡을 여지가 가장 크다.
    # ⚠️ 등록만 하고 need() 를 0건 두면 OPERATIONS 때와 같은 착시가 된다 —
    #    아래 §TECH 블록에 실제 대조를 배선했다.
    "TECH": ROOT / "docs" / "TECHNICAL_REPORT.md",
}

# 폐기된 표기 — 문서 어디에도 남아 있으면 안 된다 (교체 누락 탐지).
STALE: list[tuple[str, str]] = [
    ("walkforward_pooled_oos", "워크포워드 주 지표는 macro_avg_folds"),
    # 실제로 호출하는 모델은 config.yaml 의 llm.model (gemini-2.5-flash) 이다.
    # 문서에만 다른 이름이 남으면 '쓰지 않는 모델을 썼다'고 적는 셈이라 잡아낸다.
    ("gemini-3.6", "실제 호출 모델은 config.yaml llm.model 값"),
    ("src/ui.py", "화면 코드는 src/views/ 로 분리됨"),
    ("모델 근거", "해당 화면은 AI 상담으로 대체됨"),
]

_fails: list[str] = []
_checked = 0


def _txt(key: str) -> str:
    return DOCS[key].read_text(encoding="utf-8")


EMIT = "--emit" in sys.argv     # label<TAB>value 만 찍는다 (버전 간 대조용)


def need(label: str, value: str, *doc_keys: str) -> None:
    """value 문자열이 지정한 문서들에 모두 있어야 한다."""
    global _checked
    _checked += 1
    if EMIT:
        # 표기가 바뀔 때 어느 값이 어느 값으로 갈아타야 하는지 기계로 짝지으려면
        # '문서를 검사한 결과'가 아니라 '산출물이 말하는 값' 자체가 필요하다.
        print(f"{label}\t{value}\t{','.join(doc_keys)}")
        return
    missing = [k for k in doc_keys if value not in _txt(k)]
    if missing:
        _fails.append(f"{label}: 실측 표기 '{value}' 가 {', '.join(missing)} 에 없음")
        print(f"  [MISS] {label:46s} '{value}'  → {', '.join(missing)}")
    else:
        print(f"  [OK  ] {label:46s} '{value}'  ({', '.join(doc_keys)})")


def _checked_inc() -> None:
    global _checked
    _checked += 1


def info(label: str, value: object) -> None:
    print(f"  [INFO] {label:46s} {value}")


def head(t: str) -> None:
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


def thou(n: float) -> str:
    return f"{round(n):,}"


def main() -> int:
    head("① 패널 · 엔티티 정합")
    pf = pd.read_parquet(PROC / "panel_full.parquet")
    need("panel_full 행수", thou(len(pf)), "IMPL")
    n_mnno = pf.loc[pf["id_source"] == "MNNO", "brand_id"].nunique()
    n_name = pf.loc[pf["id_source"] == "NAME", "brand_id"].nunique()
    need("고유 브랜드 수", thou(pf["brand_id"].nunique()), "IMPL")
    need("관리번호 기반 브랜드", thou(n_mnno), "IMPL")
    need("명칭 기반 브랜드", thou(n_name), "IMPL")

    head("② 라벨 · 피처")
    lab = pd.read_parquet(PROC / "labels.parquet")
    need("라벨 표본 행수", thou(len(lab)), "RESULTS", "IMPL")
    need("라벨 양성률", f"{100 * lab['label'].mean():.1f}%", "RESULTS", "IMPL")
    fs = pd.read_csv(OUT / "feature_summary.csv")
    fs_ext = pd.read_csv(OUT / "extended" / "feature_summary.csv")
    need("피처 수(기본 트랙)", str(len(fs)), "RESULTS", "IMPL", "IFACE", "README")
    info("피처 수(확장 트랙)", len(fs_ext))
    dead = int((fs["n"] == 0).sum()) if "n" in fs.columns else 0
    info("전량 결측(죽은) 피처 수", dead)
    if dead:
        _fails.append(f"죽은 피처 {dead}개 — 문서의 '죽은 피처 0개' 주장과 불일치")

    head("③ 단일 시간분할 test (기본 트랙)")
    te = pd.read_csv(OUT / "metrics.csv")
    te = te[te["split"] == "test"].set_index("model")
    need("test 표본수", thou(te["n"].max()), "RESULTS", "IMPL")
    need("test 양성률", f"{100 * te['base_rate'].max():.1f}%", "RESULTS")
    for m in ("persistence", "single", "logistic", "lgbm"):
        need(f"{m} Lift@10", f"{te.loc[m, 'lift_at_10']:.3f}", "RESULTS")
    need("보정 Brier", f"{te.loc['lgbm_calibrated', 'brier']:.3f}", "RESULTS")

    head("④ 워크포워드 (주 지표 = fold 평균)")
    wm = pd.read_csv(OUT / "walkforward_metrics.csv")
    macro = wm[wm["scope"] == "macro_avg_folds"].set_index("model")
    need("WF 표본수(풀링 OOS)", thou(macro["n"].max()), "RESULTS", "README")
    for m in ("persistence", "single", "logistic", "lgbm"):
        need(f"WF fold평균 {m} Lift@10", f"{macro.loc[m, 'lift_at_10']:.3f}", "RESULTS", "README")
    # ⚠️ 예전에는 walkforward_delta_ci.csv(bootstrap_unit=year_block, 블록 3개)를
    #    기준으로 삼았다. 블록이 3개뿐인 백분위 부트스트랩은 t-구간의 약 1/3.4 폭밖에
    #    나오지 않아 없는 유의성을 만들어낸다 — 같은 저장소의 lift_delta_bootstrap.csv
    #    는 같은 비교를 비유의로 낸다. 두 산출물이 모순되므로, fold 값에서 직접 계산한
    #    t-구간을 문서 기준으로 삼는다.
    folds = wm[wm["valid_year"] > 0].pivot_table(
        index="valid_year", columns="model", values="lift_at_10")
    for base in ("persistence", "single", "logistic"):
        d = (folds["lgbm"] - folds[base]).to_numpy()
        m, n = float(d.mean()), len(d)
        se = float(d.std(ddof=1) / np.sqrt(n))
        t = float(stats.t.ppf(0.975, n - 1))
        need(f"lgbm - {base} Δ(t)", f"{m:+.3f}", "RESULTS")
        need(f"lgbm - {base} CI(t)", f"[{m - t * se:+.3f}, {m + t * se:+.3f}]", "RESULTS")
    bias = pd.read_csv(OUT / "walkforward_pool_bias.csv")
    worst = bias.loc[bias["bias_ratio"].sub(1.0).abs().idxmax()]
    need("원점수 풀링 최대 편향배수", f"{worst['bias_ratio']:.2f}", "IMPL")

    head("⑤ 두 트랙 비교")
    tc = pd.read_csv(OUT / "track_comparison.csv")
    ext_ss = tc[(tc["track"] == "extended") & (tc["eval"] == "single_split_test")].set_index("model")
    ext_wf = tc[(tc["track"] == "extended")
                & (tc["eval"] == "walkforward_macro_avg_folds")].set_index("model")
    ext_lab = pd.read_parquet(OUT / "extended" / "processed" / "labels.parquet")
    need("확장 라벨 표본", thou(len(ext_lab)), "RESULTS")
    need("확장 단일분할 lgbm", f"{ext_ss.loc['lgbm', 'lift_at_10']:.3f}", "RESULTS")
    need("확장 단일분할 logistic", f"{ext_ss.loc['logistic', 'lift_at_10']:.3f}", "RESULTS")
    need("확장 WF lgbm", f"{ext_wf.loc['lgbm', 'lift_at_10']:.3f}", "RESULTS")
    need("확장 WF logistic", f"{ext_wf.loc['logistic', 'lift_at_10']:.3f}", "RESULTS")

    head("⑥ 포트폴리오 (합성 예시 여신)")
    port = pd.read_csv(OUT / "portfolio.csv")
    s = json.loads((OUT / "portfolio_summary.json").read_text(encoding="utf-8"))
    need("총 익스포저(억원)", thou(port["exposure_mkrw"].sum() / 100), "RESULTS", "IMPL")
    need("EL LGD45(억원)", f"{port['el_lgd45_mkrw'].sum() / 100:.1f}", "RESULTS", "IMPL")
    need("스트레스 EL LGD45(억원)", f"{port['stress_el_lgd45_mkrw'].sum() / 100:.1f}",
         "RESULTS", "IMPL")
    need("상위10 집중도", f"{100 * s['concentration']['top10_share']:.1f}%", "RESULTS", "IMPL")
    need("HHI", f"{s['concentration']['hhi']:.3f}", "RESULTS", "IMPL")
    need("High 등급 브랜드 수", f"{int(s['risk_grades']['counts']['High'])}/60", "IMPL")

    head("⑥-2 가맹본부 재무 (DART) · 정보공개서 원문")
    dp = OUT / "dart_match_report.json"
    if not dp.exists():
        print("  [SKIP] dart_match_report.json 없음 — `--step dart` 실행 필요")
    else:
        dm = json.loads(dp.read_text(encoding="utf-8"))
        v = dm["verdicts"]
        need("DART 확정 법인", thou(v.get("확정", 0)), "IMPL")
        need("DART 불일치 법인", thou(v.get("불일치", 0)), "IMPL")
        # 이름 매칭만 믿었을 때의 오매칭 비율 — 이 층의 존재 이유를 숫자로 고정한다
        tot = sum(v.values()) or 1
        need("이름 매칭 오매칭률", f"{100 * v.get('불일치', 0) / tot:.0f}%", "IMPL")
        el = dm["eligible"]
        info("DART 단독 커버리지(자격·가맹점가중)",
             f"{100 * el['with_financials_store_weighted']:.1f}%")
        info("자격 브랜드 확정 커버리지(가맹점가중)",
             f"{100 * el['confirmed_store_weighted']:.1f}%")
        q = dm.get("quality") or {}
        if q.get("balance_pass_rate") is not None:
            info("회계 항등식 통과율", f"{100 * q['balance_pass_rate']:.1f}% "
                                       f"({q['balance_checked']:,}건 검증)")
        info("계속기업 불확실성 기재", q.get("going_concern_flagged"))

    # 🛑 위 값은 **DART 매칭 리포트**의 것이라 정보공개서 열람분이 빠져 있다. 그리고 그
    #    리포트는 `--step dart` 를 돌려야만 갱신되므로, 열람분을 아무리 합쳐도 숫자가
    #    움직이지 않는다. 실제로 이 검사는 모형 입력이 39.7% → 48.1% 로 바뀐 뒤에도
    #    **통과했다.** 눈에 보이는 불일치보다 나쁘다 — 틀린 값을 옳다고 확인해 준다.
    #
    #    커버리지의 정의는 '모은 양'이 아니라 **'모형이 실제로 본 양'** 이다.
    #    그래서 features.parquet 의 f_hq_has_financials 에서 직접 잰다.
    head("⑥-3 본부재무 커버리지 — 모형이 실제로 본 양")
    feat = pd.read_parquet(PROC / "features.parquet", columns=["brand_id", "year",
                                                               "f_hq_has_financials"])
    pnl = pd.read_parquet(PROC / "panel.parquet", columns=["brand_id", "year",
                                                           "n_stores", "eligible_t"])
    yr = int(feat["year"].max())
    cov = feat[feat["year"] == yr].merge(pnl[pnl["year"] == yr], on=["brand_id", "year"],
                                         how="left")
    cov = cov[cov["eligible_t"].fillna(False).astype(bool)]
    has = cov["f_hq_has_financials"].fillna(0)
    w = cov["n_stores"].fillna(0)
    need("본부재무 커버리지(자격·브랜드)", f"{100 * has.mean():.1f}%", "RESULTS", "IMPL", "README")
    need("본부재무 커버리지(자격·가맹점가중)",
         f"{100 * (w * has).sum() / max(w.sum(), 1):.1f}%", "RESULTS", "IMPL")
    info("본부재무 보유 브랜드 수", f"{int(has.sum())} / {len(cov)}")
    # 원천 분해 — 어디서 온 값인지 문서가 말하는 대로인지 본다
    hq_full = pd.read_parquet(PROC / "hq_financials.parquet")
    hq = hq_full[["source", "key"]]
    if str(ROOT) not in sys.path:          # tools/ 를 직접 실행해도 src 가 보이게
        sys.path.insert(0, str(ROOT))
    from src.ifrmp_web import HQ_SOURCE_TAG
    n_web = int((hq["source"] == HQ_SOURCE_TAG).sum())
    info("본부재무 행 원천", f"DART {len(hq) - n_web:,}행 · 정보공개서 열람 {n_web:,}행")
    # ⚠️ need() 는 **부분문자열**로 찾는다. '28' 같은 두 자리 수는 문서 안 아무 숫자에나
    #    걸려 통과해 버린다 — 실제로 문서에 29 라고 적혀 있는데 통과했다. 작은 수를
    #    대조할 때는 값만 넘기지 말고 **문구째** 넘겨 우연한 일치를 막는다.
    need("정보공개서 열람 법인 수",
         f"열람 법인 **{hq.loc[hq['source'] == HQ_SOURCE_TAG, 'key'].nunique()}**개", "IMPL")

    # 🛑 여기 없던 동안 실제로 이런 일이 있었다: 문서 세 곳이 등록취소 병합을
    #    '317행(자진 195 / 직권 122)' 이라고 적고 있었는데 실제 패널은 **241행
    #    (자진 154 / 직권 87)** 이었다. 값이 산출물에서 오지 않고 로그를 손으로
    #    옮겨 적은 자리였고, 자동대조 목록에 없어 172건 통과 속에 그대로 숨어 있었다.
    #    "문서 수치 전부 일치" 가 안전의 증거가 되려면 **문서에 있는 수치가 전부
    #    목록에 있어야 한다** — 빠진 자리는 통과가 아니라 미검사다.
    cf = pd.read_parquet(PROC / "panel.parquet", columns=["cancel_flag", "cancel_type"])
    cf = cf[pd.to_numeric(cf["cancel_flag"], errors="coerce").fillna(0) > 0]
    vc = cf["cancel_type"].value_counts()
    need("등록취소 병합 행수", f"{len(cf):,}행", "IMPL", "TECH")
    need("등록취소 병합 유형", f"자진 {int(vc.get('자진', 0))} / 직권 {int(vc.get('직권', 0))}",
         "IMPL")

    # 🔬 파서를 **독립 원천으로 검증한다.** 두 원천이 같은 (법인, 회계연도)를 갖는 구간이
    #    있는데, 한쪽은 금감원에 접수된 감사보고서이고 다른 쪽은 사람이 화면에서 옮긴
    #    정보공개서다. 두 값이 맞으면 파서가 옳다는 것을 **자기 주장이 아닌 제3자 문서로**
    #    말할 수 있다. 겹치는 구간을 합병 단계에서 DART 로 덮으므로, 대조는 덮기 전
    #    원본(ifrmp_web_financials.parquet)과 한다.
    from src.dart import norm_corp
    webp = PROC / "ifrmp_web_financials.parquet"
    if webp.exists():
        w = pd.read_parquet(webp).copy()
        pfx = pd.read_parquet(PROC / "panel_full.parquet",
                              columns=["brand_id", "company_name", "year"])
        pfx = pfx.sort_values("year").drop_duplicates("brand_id", keep="last")
        kmap = {str(b)[4:]: norm_corp(str(c))
                for b, c in zip(pfx["brand_id"], pfx["company_name"], strict=False)
                if isinstance(c, str) and c.strip()}
        w["key"] = w["reg_no"].astype(str).map(kmap)
        both = w.dropna(subset=["key"]).merge(
            hq_full[hq_full["source"] != HQ_SOURCE_TAG], on=["key", "fiscal_year"],
            suffixes=("_w", "_d"))
        vals, agree = 0, 0
        for c in ("assets", "liabilities", "equity", "revenue",
                  "operating_income", "net_income"):
            a = pd.to_numeric(both.get(c + "_w"), errors="coerce")
            b = pd.to_numeric(both.get(c + "_d"), errors="coerce")
            ok = a.notna() & b.notna()
            if not ok.any():
                continue
            rel = ((a - b).abs() / b.abs().clip(lower=1))[ok]
            vals += int(ok.sum())
            agree += int((rel <= 0.001).sum())
        info("두 원천 겹침", f"(법인, 회계연도) {len(both)}쌍 · 대조값 {vals}개")
        _checked_inc()
        if vals and agree == vals:
            need("정보공개서·감사보고서 교차검증",
                 f"{vals}개 값 전부 0.1% 이내 일치", "IMPL")
        elif vals:
            _fails.append(f"교차검증 불일치 — {vals}개 중 {vals - agree}개가 0.1% 초과")
            print(f"  [FAIL] 교차검증 {vals}개 중 {vals - agree}개 불일치")

    ip = OUT / "ifrmp_status.json"
    if not ip.exists():
        print("  [SKIP] ifrmp_status.json 없음")
    else:
        st = json.loads(ip.read_text(encoding="utf-8"))
        pv = st.get("parser_validation") or {}
        need("정보공개서 표준 섹션 수", str(pv.get("sections_found", "?")), "IMPL")
        info("데모 고정응답 탐지", st.get("demo_check", {}).get("is_demo"))
        info("브랜드별 수집 건수", st.get("collected_brands"))
        # 데모 상태인데 '전량 수집했다'고 적혀 있으면 즉시 실패시킨다
        if st.get("demo_check", {}).get("is_demo"):
            for k in ("IMPL", "RESULTS"):
                if "정식 키" not in _txt(k):
                    _fails.append(f"정보공개서가 데모 상태인데 {k} 에 키 대기 사실이 없음")

    head("⑦ 브랜드 공통요인 상관 (전제 검증)")
    bc = json.loads((OUT / "brand_correlation.json").read_text(encoding="utf-8"))
    dec = bc["decomposition"]
    need("상관 통제없음", f"{dec['no_control']['rho_asset']:.3f}", "RESULTS")
    need("상관 연도통제", f"{dec['year_controlled']['rho_asset']:.3f}", "RESULTS")
    need("상관 연도+업종통제(headline)", f"{bc['rho_asset']:.3f}", "RESULTS", "IMPL", "README")
    need("상관 95% CI", f"[{bc['rho_asset_ci_lo']:.3f}, {bc['rho_asset_ci_hi']:.3f}]", "RESULTS")
    need("분석 브랜드-연도", thou(bc["n_brand_years"]), "RESULTS")
    need("분석 지역쌍", thou(bc["n_region_pairs"]), "RESULTS")
    need("전지역 동시감소 실측", f"{100 * bc['all_regions_decline_observed']:.2f}%", "RESULTS")
    need("전지역 동시감소 배수", f"{bc['all_regions_decline_ratio']:.1f}배", "RESULTS", "README")

    btw = bc["between_brand"]
    need("브랜드 간 상관 rho_B", f"{btw['rho_between']:.3f}", "RESULTS", "README")
    need("rho_B CI", f"[{btw['rho_between_ci_lo']:.3f}, {btw['rho_between_ci_hi']:.3f}]", "RESULTS")

    ci_ = json.loads((OUT / "correlation_impact.json").read_text(encoding="utf-8"))
    need("차주(가맹점) 수", thou(ci_["n_franchisees"]), "RESULTS", "README")
    for lvl, _lab in (("p95", "95%"), ("p99", "99%"), ("p999", "99.9%")):
        need(f"{lvl} 독립가정 손실(억)", f"{ci_[f'independent_{lvl}_mkrw'] / 100:.1f}", "RESULTS")
        need(f"{lvl} 상관반영 손실(억)", f"{ci_[f'brand_correlated_{lvl}_mkrw'] / 100:.1f}", "RESULTS")
        need(f"{lvl} 과소추정(억)", f"{ci_[f'understatement_{lvl}_mkrw'] / 100:.1f}", "RESULTS")
    need("UL99 배수", f"{ci_['ul99_multiple']:.2f}배", "RESULTS", "README")
    need("UL99 독립(억)", f"{ci_['independent_ul99_mkrw'] / 100:.1f}", "RESULTS", "README")
    need("UL99 상관(억)", f"{ci_['brand_correlated_ul99_mkrw'] / 100:.1f}", "RESULTS", "README")

    head("⑦-3 LLM 정량 평가 (규칙기반 대비)")
    ep = OUT / "llm_eval.json"
    if not ep.exists():
        print("  [SKIP] llm_eval.json 없음 — `--step eval_llm` 실행 필요")
    else:
        ev = json.loads(ep.read_text(encoding="utf-8"))
        if not ev.get("llm_evaluated"):
            _fails.append(f"LLM 평가가 무효 상태다: {ev.get('llm_invalidated_reason')}")
            print(f"  [FAIL] LLM 지표 무효 — {ev.get('llm_invalidated_reason')}")
        else:
            for scope in ("real", "synthetic", "all"):
                if scope not in ev:
                    continue
                for who in ("rules", "llm"):
                    m = ev[scope][who]
                    need(f"{scope}/{who} 정확도", f"{m['accuracy']:.3f}", "RESULTS")
                    need(f"{scope}/{who} macro-F1", f"{m['macro_f1']:.3f}", "RESULTS")
                    need(f"{scope}/{who} 위험F1", f"{m['risk_f1']:.3f}", "RESULTS")
            need("평가 대상 실수집 건수", f"{ev['real']['rules']['n']}건", "RESULTS")
            need("평가 프로브 건수", f"{ev['synthetic']['rules']['n']}건", "RESULTS")
            info("평가 모델", ev.get("model"))

    # 🛑 배포 산출물이 **실제로 LLM 으로 만들어진 것인가**를 여기서 확인한다.
    #    IMPLEMENTATION.md 에 "실제 Gemini 호출 63/63건(폴백 0)" 이라고 적혀 있었는데,
    #    같은 시점의 news_signals.json 은 45건 전량 `llm_used=False`(규칙 폴백)였다.
    #    AI 활용을 묻는 항목에서 문서와 산출물이 정반대를 말하고 있었던 것이다.
    #    수치(건수)는 수집 때마다 바뀌므로 **'폴백 0건'이라는 성질**을 검사한다.
    np_ = OUT / "news_signals.json"
    if not np_.exists():
        print("  [SKIP] news_signals.json 없음 — `--step news` 실행 필요")
    else:
        sig = json.loads(np_.read_text(encoding="utf-8"))
        n_all = len(sig)
        n_llm = sum(1 for s in sig if s.get("llm_used"))
        info("뉴스 신호 건수", f"{n_llm}/{n_all}건이 LLM 추출")
        _checked_inc()
        if n_all and n_llm == n_all:
            need("뉴스 LLM 폴백", "폴백 0건", "IMPL")
        else:
            _fails.append(
                f"news_signals.json {n_all}건 중 LLM 추출은 {n_llm}건 — "
                f"'폴백 0건' 주장을 문서에 적어 둘 수 없다 (규칙 폴백 {n_all - n_llm}건)")
            print(f"  [FAIL] 뉴스 LLM 폴백 {n_all - n_llm}건 — 문서 주장과 대조 필요")

    head("⑦-2 폐기값 잔존 검사 (같은 자리에 옛 수치가 남아 있는가)")
    # need() 는 '정답이 문서에 있는가'만 본다 — 옛 값이 함께 남아 있어도 통과한다.
    # 실제로 이 사각지대 때문에 헤드라인에 옛 수치가 남는 사고가 있었다(자체 감사 검출).
    # 그래서 '같은 패턴에 매칭되는 다른 값'이 남아 있으면 실패시킨다.
    import re as _re
    for label, pattern, truth in (
        ("과소추정 억원", r"과소추정\**\s*\+?([\d,]+\.\d)억",
         f"{ci_['understatement_p99_mkrw'] / 100:.1f}"),
        ("몬테카를로 횟수", r"몬테카를로\s*(\d+)만\s*회", str(ci_["n_sims"] // 10000)),
        ("UL 배수", r"([\d.]+)배\*{0,2}다", f"{ci_['ul99_multiple']:.2f}"),
    ):
        txt = _txt("README") + "\n" + _txt("RESULTS")
        found = set(_re.findall(pattern, txt))
        stale = {v for v in found if v.replace(",", "") != truth}
        _checked_inc()
        if stale:
            _fails.append(f"{label}: 폐기값 {sorted(stale)} 이 README·RESULTS 에 남아 있음 (정답 {truth})")
            print(f"  [STALE] {label:28s} 잔존 {sorted(stale)} (정답 {truth})")
        else:
            print(f"  [OK  ] {label:28s} 정답 '{truth}' 외 다른 값 없음")

    head("⑧ 폐기 표기 잔존 검사")
    for token, why in STALE:
        hits = [k for k, p in DOCS.items() if token in p.read_text(encoding="utf-8")]
        if hits:
            _fails.append(f"폐기 표기 '{token}' 잔존: {', '.join(hits)} ({why})")
            print(f"  [STALE] '{token}' → {', '.join(hits)}  ({why})")
        else:
            print(f"  [OK  ] '{token}' 잔존 없음")

    head("⑧-2 본부 축 실증 (측정했으나 모형에 넣지 않은 축)")
    hp = OUT / "hq_axis.json"
    if not hp.exists():
        print("  [SKIP] hq_axis.json 없음 — `python tools/hq_axis.py` 실행 필요")
    else:
        # 산출물은 있는데 어느 문서도 참조하지 않으면 **없는 것과 같다.** 실제로 이 파일은
        # 만들어진 뒤 한동안 생성기 자신 말고는 저장소 어디에서도 언급되지 않았다.
        ha = json.loads(hp.read_text(encoding="utf-8"))
        # 값만 넘기면 문서 안 다른 97.8%(엔티티 정합 확보율)에 걸려 통과한다 —
        # 실제로 README 에 98.8% 라 적혀 있는데 통과했다. 문구째 대조한다.
        need("본부 축 패널 관리번호 보유율",
             f"패널 전체의 **{100 * ha['hq_coverage_in_panel']:.1f}%**", "RESULTS")
        for r in ha["table"]:
            if r["state"] != "요주의" or r["n_events_at_t"] not in (1, 2):
                continue
            k = r["n_events_at_t"]
            need(f"형제 정상 재발동률(k={k})", f"{100 * r['rate_sib_ok']:.1f}%", "RESULTS")
            need(f"형제 악화 재발동률(k={k})", f"{100 * r['rate_sib_bad']:.1f}%", "RESULTS")
            need(f"형제 효과 Fisher p(k={k})", f"{r['fisher_p']:.4f}", "RESULTS")
        for e in ha.get("hq_size_effect", []):
            if e["n_events_at_t"] != 1:
                continue
            need("본부 규모 효과 소", f"{100 * e['rate_small']:.1f}%", "RESULTS")
            need("본부 규모 효과 대", f"{100 * e['rate_big']:.1f}%", "RESULTS")
            need("본부 규모 효과 p", f"{e['fisher_p']:.4f}", "RESULTS")
        need("본부 축 관측 Δ", f"{ha['observed_delta']:+.4f}", "RESULTS")
        need("본부 축 귀무 평균", f"{ha['null']['mean']:+.4f}", "RESULTS")
        need("본부 축 p값", f"p={ha['null']['p_value']:.3f}", "RESULTS")
        need("본부 축 교란 설명분", f"{100 * ha['confound_share']:.1f}%", "RESULTS")

    head("⑨ 화면 코드 대조 (src/views) — 문서만 고치면 잡히지 않는 자리")
    # 🛑 이 절이 없던 동안 실제로 이런 일이 있었다: README 는 '7.30배'인데 서비스 소개
    #    화면(src/views/about.py)은 '5.31배'를 그대로 띄우고 있었다. 검사 대상이 .md
    #    뿐이면 "문서 수치 142건 전부 일치"는 안전의 증거가 아니라 **착시**다.
    #    같은 이유로 DEPLOY·TECH 를 뒤늦게 등록했던 전례를 여기서 되풀이하지 않는다.
    VIEWS = sorted((ROOT / "src" / "views").glob("*.py"))
    for token, why in STALE:
        _checked_inc()
        hits = [p.name for p in VIEWS if token in p.read_text(encoding="utf-8")]
        if hits:
            _fails.append(f"폐기 표기 '{token}' 가 화면 코드에 잔존: {', '.join(hits)} ({why})")
            print(f"  [STALE] '{token}' → {', '.join(hits)}  ({why})")
        else:
            print(f"  [OK  ] '{token}' 화면 코드 잔존 없음")

    # 핵심 수치는 화면에 **문자열로 박지 않는다.** 박는 순간 산출물과 갈라질 수 있고,
    # 갈라진 것을 사람이 알아채는 유일한 방법이 '심사위원이 먼저 보는 것'이 된다.
    # 그래서 '값이 맞는가'가 아니라 **'박아 두었는가'** 를 검사한다 — 값 대조는 다음
    # 재산출에서 또 낡지만, 이 검사는 구조를 강제하므로 낡지 않는다.
    LITERAL_BANS = (
        (r'"\s*\d+\.\d+\s*배\s*"', "UL 배수", "outputs/correlation_impact.json ul99_multiple"),
        (r'"\s*0\.\d{3}\s*"', "상관계수", "outputs/correlation_impact.json rho_*"),
    )
    for pattern, what, whence in LITERAL_BANS:
        _checked_inc()
        hits = []
        for p in VIEWS:
            for m in _re.finditer(pattern, p.read_text(encoding="utf-8")):
                hits.append(f"{p.name}:{m.group(0).strip()}")
        if hits:
            _fails.append(f"{what} 를 화면에 문자열로 박았다: {', '.join(hits)} "
                          f"— {whence} 에서 읽어야 한다")
            print(f"  [HARD] {what:12s} 박힌 값 {hits}")
        else:
            print(f"  [OK  ] {what:12s} 화면에 박힌 값 없음 ({whence} 에서 읽는다)")

    head("⑪ 운영 코호트 구성 · 요주의 실현율")
    sc = pd.read_csv(OUT / "scores_latest.csv", encoding="utf-8-sig")
    n_co = len(sc)
    need("코호트 규모", thou(n_co), "SPEC", "CONCEPT")
    for g in ("FS1", "FS2", "FS3"):
        info(f"{g} 개수", int((sc["grade"] == g).sum()))
    need("FS3 등급 수", thou(int((sc["grade"] == "FS3").sum())), "SPEC")
    st = sc["brand_state"].value_counts()
    for s in ("건전", "요주의", "평가불가"):
        need(f"상태 {s}", thou(int(st.get(s, 0))), "SPEC", "METHOD")
        need(f"상태 {s} 비중", f"{100 * st.get(s, 0) / n_co:.1f}%", "SPEC", "METHOD")
    # 학습 모집단 밖 = 요주의 + 평가불가
    out_n = int(n_co - st.get("건전", 0))
    need("학습 모집단 밖", thou(out_n), "SPEC", "CONCEPT")
    need("학습 모집단 밖 비중", f"{100 * out_n / n_co:.1f}%", "SPEC", "CONCEPT")
    fs3 = sc[sc["grade"] == "FS3"]
    w3 = int((fs3["brand_state"] == "요주의").sum())
    need("FS3 중 요주의", thou(w3), "SPEC")
    need("FS3 중 요주의 비중", f"{100 * w3 / len(fs3):.1f}%", "SPEC")

    wb_p = OUT / "watch_base_rates.csv"
    if wb_p.exists():
        wb = pd.read_csv(wb_p, encoding="utf-8-sig")
        for r in wb.to_dict("records"):
            k = int(r["n_events_at_t"])
            lbl = "건전" if r["state"] == "건전" else f"요주의 {k}건"
            need(f"실현율 {lbl}", f"{100 * r['rate']:.1f}%", "SPEC", "METHOD")
            need(f"실현율 {lbl} 표본", thou(r["n"]), "SPEC", "METHOD")
    else:
        info("요주의 실현율표", "없음 — tools/watch_base_rates.py 미실행")

    head("⑫ 운영·거버넌스 문서 (OPERATIONS) — 게이트를 규정한 문서가 게이트를 받는다")
    # ⚠️ 이 절이 없던 동안 OPS 는 DOCS 레지스트리에 **등록만 되고 검사 0건**이었다.
    #    그 사이 §2 큐 규모(145 vs 실측 367), §3 게이트 근거(+0.775 vs +1.059),
    #    fold 변동폭(2.617/2.252/2.500 vs 2.818/2.752/2.667)이 전부 낡았다.
    #    등록돼 있다는 사실이 검사받는다는 뜻이 아니다 — 그래서 착시가 더 컸다.
    g = sc["grade"].value_counts()
    need("OPS 큐 규모(FS3)", thou(int(g.get("FS3", 0))), "OPS")
    need("OPS 큐 소요 주수", f"{round(int(g.get('FS3', 0)) / 10)}주", "OPS")
    dci_p = OUT / "walkforward_delta_ci.csv"
    if dci_p.exists():
        dci = pd.read_csv(dci_p, encoding="utf-8-sig").set_index("comparison")
        r = dci.loc["lgbm - persistence"]
        need("OPS 게이트 근거 격차", f"+{r['mean_delta_lift']:.3f}", "OPS")
        need("OPS 게이트 근거 CI", f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", "OPS")
    wfm = pd.read_csv(OUT / "walkforward_metrics.csv", encoding="utf-8-sig")
    lg = wfm[wfm["scope"].astype(str).str.startswith("fold_test_") & (wfm["model"] == "lgbm")]
    if len(lg):
        need("OPS fold별 Lift 나열",
             " / ".join(f"{v:.3f}" for v in lg["lift_at_10"]), "OPS")
    # ---- TECH: 기술설명서 전용 대조 ------------------------------------------
    # 이 문서는 전 과정을 한 파일에 모으므로 다른 문서가 갱신될 때마다 낡을 수 있다.
    # 각 부에서 결론에 해당하는 값만 골라 산출물에서 다시 계산해 대조한다.
    vsum = json.loads((OUT / "validation" / "summary.json").read_text(encoding="utf-8"))
    dp = vsum["discrimination_pooled"]
    need("TECH 판별력 AUC", f"{dp['auc']:.3f}", "TECH")
    need("TECH 판별력 AUC CI", f"[{dp['auc_lo']:.3f}, {dp['auc_hi']:.3f}]", "TECH")
    need("TECH 검증 OOS 표본", thou(dp["n"]), "TECH")
    cal = vsum["calibration"]
    need("TECH HL p", f"p={cal['hosmer_lemeshow']['p_value']:.3f}", "TECH")
    need("TECH Spiegelhalter Z", f"Z={cal['spiegelhalter']['z']:.3f}", "TECH")

    gb = json.loads((OUT / "grade_bands.json").read_text(encoding="utf-8"))
    need("TECH 등급 컷", f"[{gb['cuts'][0]}, {gb['cuts'][1]}]", "TECH")
    need("TECH 최소 CI 분리", f"{100 * gb['min_ci_separation']:.2f}%p", "TECH")
    need("TECH 절차 OOT 컷",
         f"[{gb['cut_procedure_oot']['cuts_from_anchor'][0]}, "
         f"{gb['cut_procedure_oot']['cuts_from_anchor'][1]}]", "TECH")
    need("TECH 귀무검정 반복", f"{gb['null_test']['n_iter']}회", "TECH")
    for row in gb["pooled"]:
        need(f"TECH 실현율 {row['grade']}", f"{100 * row['rate']:.2f}%", "TECH")

    need("TECH UL 배수", f"{ci_['ul99_multiple']:.2f}배", "TECH")
    need("TECH 상위5 UL 비중",
         f"{100 * ci_['euler_allocation']['top5_ul_share']:.1f}%", "TECH")
    need("TECH 브랜드 내부 상관", f"{bc['rho_asset']:.3f}", "TECH")
    need("TECH 브랜드 간 상관", f"{btw['rho_between']:.3f}", "TECH")

    wb = json.loads((OUT / "watch_base_rates.json").read_text(encoding="utf-8"))
    need("TECH 요주의 실현율 나열",
         " / ".join(f"{100 * r['rate']:.1f}%" for r in wb["table"]), "TECH")

    le = json.loads((OUT / "llm_eval.json").read_text(encoding="utf-8"))
    need("TECH LLM 정확도(전체)", f"{le['all']['llm']['accuracy']:.3f}", "TECH")
    need("TECH 규칙 정확도(전체)", f"{le['all']['rules']['accuracy']:.3f}", "TECH")

    dg = json.loads((OUT / "brand_diagnosis_meta.json").read_text(encoding="utf-8"))
    need("TECH 진단 소견 건수", thou(dg["n_findings"]), "TECH")
    need("TECH 소견 조합 종수", thou(dg["distinct_finding_sets"]), "TECH")

    rs = json.loads((OUT / "rag_stats.json").read_text(encoding="utf-8"))
    need("TECH RAG 문서수", thou(rs["n_documents"]), "TECH")
    need("TECH RAG 커버 브랜드", thou(rs["n_brands_covered"]), "TECH")

    # 결함 전수 목록은 **표를 세어서** 건수를 정한다. 사람이 헤아려 적으면 항목을
    # 추가할 때마다 어긋난다 — 그 어긋남이 바로 이 문서가 고발하는 종류의 결함이다.
    tech = _txt("TECH")
    sec = tech.split("## 43. 스스로 잡은 결함")[1].split("## 44.")[0]
    rows = [int(m.group(1)) for m in re.finditer(r"^\|\s*(\d+)\s*\|", sec, re.M)]
    if rows:
        if sorted(rows) != list(range(1, len(rows) + 1)):
            _fails.append(f"TECH 결함 목록 번호가 연속이 아니다 (n={len(rows)})")
        # ⚠️ 카테고리 문자를 A-H 로 고정해 두었더니, I 를 새로 만들었을 때 그 21건이
        #    합계에서 통째로 빠졌다. 검사 자신이 낡는 자리다 — 범위를 열어 둔다.
        cats = [int(m.group(3)) for m in re.finditer(r"### ([A-Z])\. (.+?) \((\d+)건\)", sec)]
        if cats and sum(cats) != len(rows):
            _fails.append(f"TECH 카테고리 합 {sum(cats)} ≠ 실제 행수 {len(rows)}")
        need("TECH 결함 전수 건수", f"결함 {len(rows)}건", "RESULTS", "TECH")

    # 수집 데이터셋 종수는 **원본 스냅샷 파일군**에서 센다. 창업비용(15110265)을
    # 뒤늦게 추가했을 때 README 가 6종에 멈춰 있었다 — 원천이 늘어나는 것은
    # 이 프로젝트에서 자주 일어나는 일이라 사람 기억에 맡기지 않는다.
    raw = ROOT / "data" / "raw"
    if raw.exists():
        fam = {re.sub(r"_\d{4}\.json.*$", "", p.name) for p in raw.glob("*.json*")}
        if fam:
            need("공정위 수집 데이터셋 종수", f"오픈API **{len(fam)}종**", "RESULTS", "README")

    # 라벨 구성표는 **자격 통과 표본**을 말한다. 예전에는 같은 파일이 자격 이전
    # 프레임(7,163행·11.4%)을 담아 OPERATIONS 와 IMPLEMENTATION 이 서로 다른 값을
    # 인용했다. 두 값 모두 산출물에서 다시 계산해 대조한다.
    lc = pd.read_csv(OUT / "label_composition.csv")
    yr = lc[lc["section"] == "year_positive_rate"]["value"]
    need("라벨 양성률(자격 표본)",
         f"{100 * float(lc.loc[lc['key'] == 'positive_rate', 'value'].iloc[0]):.1f}%", "OPS")
    need("라벨 연도별 양성률 범위",
         f"{100 * yr.min():.1f}~{100 * yr.max():.1f}%", "OPS")

    # 로고는 색인이 아니라 **디스크의 PNG** 가 화면에 뜨는 실체다 (→ tests/test_naming.py
    # 로고 색인·디스크 정합). 그래서 세는 대상도 파일이다.
    logo_dir = ROOT / "data" / "raw" / "naver" / "logo_img"
    if logo_dir.exists():
        need("DEPLOY 동봉 로고 건수", f"로고 {len(list(logo_dir.glob('*.png')))}건", "DEPLOY")

    head("⑩ README 헤드라인 — 포트폴리오 요약이 인용하는 수치")
    sc = pd.read_csv(OUT / "scores_latest.csv", encoding="utf-8-sig")
    need("평가 브랜드 수", thou(len(sc)), "README")
    gb = json.loads((OUT / "grade_bands.json").read_text(encoding="utf-8"))
    for r in gb["out_of_time"]["by_grade"]:
        need(f"시점 밖 등급 실현율 {r['grade']}", f"{100 * r['rate']:.1f}%", "README")
    vs = json.loads((OUT / "validation" / "summary.json").read_text(encoding="utf-8"))
    dp = vs["discrimination_pooled"]
    need("풀링 OOS AUC", f"{dp['auc']:.3f}", "README")
    need("풀링 OOS AUC CI", f"[{dp['auc_lo']:.3f}, {dp['auc_hi']:.3f}]", "README")
    need("풀링 OOS KS", f"{dp['ks']:.3f}", "README")
    cal = vs["calibration"]
    need("보정 HL p", f"p={cal['hosmer_lemeshow']['p_value']:.3f}", "README")
    need("보정 ECE", f"ECE {cal['ece']:.3f}", "README")
    wr = json.loads((OUT / "watch_base_rates.json").read_text(encoding="utf-8"))
    for r in wr["table"]:
        need(f"다음 해 재발동률 k={r['n_events_at_t']}", f"{100 * r['rate']:.1f}%", "README")
    le = json.loads((OUT / "llm_eval.json").read_text(encoding="utf-8"))
    need("뉴스 추출 정확도 LLM(전체)", f"{le['all']['llm']['accuracy']:.3f}", "README")
    need("뉴스 추출 정확도 규칙(전체)", f"{le['all']['rules']['accuracy']:.3f}", "README")
    need("뉴스 정답셋 건수", f"{le['all']['rules']['n']}건", "README")

    # 월별 신호의 타당도는 README 헤드라인에도 오른다. 신호표를 새로 만들면(월 1회) 판정 분포는
    # 바뀌지만 이 값들은 validate_localdata.py 를 다시 돌릴 때만 바뀐다 — 그래서 분포는 문서에
    # 적지 않고, 타당도만 대조한다.
    lv_p = OUT / "localdata_validation.json"
    if lv_p.exists():
        head("⑪ 월별 폐점 신호 — 인허가 ↔ 공시 동시 타당도 (tools/validate_localdata.py)")
        lv = json.loads(lv_p.read_text(encoding="utf-8"))
        years = sorted(lv["years"], key=int)
        for y in years:
            cv, gv = lv["years"][y]["close_vs_churn"], lv["years"][y]["growth_vs_growth"]
            docs = ("RESULTS", "TECH", "README") if y == years[-1] else ("RESULTS", "TECH")
            need(f"{y} 폐점률 순위상관", f"{cv['rho']:.3f}", *docs)
            need(f"{y} 폐점률 순위상관 CI", f"[{cv['ci_lo']:.3f}, {cv['ci_hi']:.3f}]", "RESULTS", "TECH")
            need(f"{y} 순증감 순위상관", f"{gv['rho']:.3f}", *docs)
            need(f"{y} 순증감 순위상관 CI", f"[{gv['ci_lo']:.3f}, {gv['ci_hi']:.3f}]", "RESULTS", "TECH")
            need(f"{y} 폐점률 비교 브랜드 수", f"{cv['n']:,}개", *docs)
            need(f"{y} 순증감 비교 브랜드 수", f"{gv['n']:,}개", "RESULTS", "TECH")
            need(f"{y} 매칭 완전성 중앙값", f"{100 * lv['years'][y]['coverage_median']:.1f}%", *docs)
        need("인허가 기록 수", thou(lv["n_records"]), "RESULTS", "TECH")
        need("인허가 기록 수(README 표기)", f"{lv['n_records'] // 10_000:,}만여", "README")
        need("인허가 매칭 브랜드 수", thou(lv["n_brands_matched"]), "RESULTS", "TECH")
        need("인허가 매칭 기록 수", thou(lv["n_matched"]), "TECH")

    need("OPS 자동대조 건수", f"{_checked + 1}건", "OPS", "README")   # 이 줄 자신을 포함한 수

    head("결과")
    if _fails:
        print(f"불일치 {len(_fails)}건 / 검사 {_checked}건")
        for f in _fails:
            print(f"   - {f}")
        return 1
    print(f"문서 수치 {_checked}건 전부 산출물 실측과 일치, 폐기 표기 잔존 없음")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
