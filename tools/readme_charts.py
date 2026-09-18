"""README 차트 — 산출물에서 직접 그린다 (손으로 옮긴 숫자 0개).

왜 스크립트로 두는가
    README 의 그림은 문서 수치와 같은 규칙을 따라야 한다. 숫자를 그림에 손으로 적으면
    산출물이 바뀔 때 그림만 낡는다. 그래서 그림도 outputs/ 의 JSON 을 읽어 다시 그린다.

무엇을 그리는가 (세 장 × 라이트/다크)
    brand_correlation   같은 브랜드 안 vs 브랜드 사이 자산상관 + 바젤 IRB 기업여신 가정 구간
    grade_validation    2022년에 맞춘 등급을 2023년에 적용했을 때 등급별 실제 악화율 (시점 밖)
    tail_loss           차주 독립 가정 vs 브랜드 상관 반영 — 신뢰수준별 손실 분위수

색
    등급은 순서가 있는 값이므로 한 색상(주황→적갈) 명도 단계로 칠한다(서열 램프).
    강조 그림은 강조 1색 + 회색. 모든 막대에 값 라벨이 붙어 색만으로 뜻을 전하지 않는다.
    GitHub 라이트/다크 배경(#ffffff / #0d1117) 각각에서 대비를 검사한 값이다.

실행: python tools/readme_charts.py      → assets/charts/*-light.png, *-dark.png
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DEST = ROOT / "assets" / "charts"

THEMES = {
    "light": {
        "ink": "#26221E", "sub": "#6B635A", "muted": "#8A8279", "grid": "#E8E4DE", "axis": "#D6D0C7",
        "accent": "#CC6536", "context": "#CFC8BE", "band": "#26221E",
        "ramp": ["#E8A06A", "#CC6536", "#8E3A1C"],
    },
    "dark": {
        "ink": "#E6EDF3", "sub": "#9DA7B3", "muted": "#7D8590", "grid": "#21262D", "axis": "#30363D",
        "accent": "#E07C48", "context": "#4A4F57", "band": "#E6EDF3",
        "ramp": ["#F3BE8E", "#E07C48", "#B4482A"],
    },
}

FONT_CANDIDATES = ["Apple SD Gothic Neo", "Pretendard", "Malgun Gothic", "NanumGothic", "Nanum Gothic",
                   "Noto Sans CJK KR", "Noto Sans KR"]

W_IN = 8.0          # 그림 폭(인치). README 표시폭 ~800px 에서 0.5배로 줄어든다
DPI = 200
BAR_PX = 44         # 표시 기준 22px 두께 (막대 두께 상한 24px)
ROUND_PX = 8        # 표시 기준 4px 둥근 끝


def _register_ttc_weights() -> None:
    """macOS 의 Apple SD Gothic Neo 는 한 .ttc 에 굵기 9종이 들어 있는데 matplotlib 은 첫 면(Regular)만
    등록한다. 그러면 제목·값 라벨의 굵기가 전부 무시된다. 필요한 굵기만 꺼내 임시 TTF 로 등록한다."""
    ttc_path = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if not ttc_path.exists():
        return
    try:
        from fontTools.ttLib import TTCollection
    except ImportError:
        return
    tmp = Path(tempfile.mkdtemp(prefix="readme_charts_"))
    for face in TTCollection(str(ttc_path)).fonts:
        name = face["name"]
        if name.getDebugName(1) != "Apple SD Gothic Neo" or name.getDebugName(2) not in ("SemiBold", "Bold"):
            continue
        path = tmp / f"AppleSDGothicNeo-{name.getDebugName(2)}.ttf"
        face.save(str(path))
        font_manager.fontManager.addfont(str(path))


def _font() -> str:
    _register_ttc_weights()
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_CANDIDATES:
        if name in have:
            return name
    return "sans-serif"


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _px_per_unit(ax) -> tuple[float, float]:
    """축의 데이터 1단위가 이미지에서 몇 px 인지 (x, y)."""
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return bbox.width / (x1 - x0), bbox.height / (y1 - y0)


def _bar(ax, *, base: float, value: float, center: float, color: str, horizontal: bool) -> None:
    """값 쪽 끝만 둥글고 기준선 쪽은 각진 막대. 두께는 px 로 고정한다."""
    ppx, ppy = _px_per_unit(ax)
    thick = BAR_PX / (ppy if horizontal else ppx)
    r_len = ROUND_PX / (ppx if horizontal else ppy)        # 길이 방향 반경
    r_thk = ROUND_PX / (ppy if horizontal else ppx)        # 두께 방향 반경
    length = value - base
    if horizontal:
        x, y, w, h = base, center - thick / 2, length, thick
    else:
        x, y, w, h = center - thick / 2, base, thick, length
    if length <= 2 * r_len:                                 # 너무 짧으면 각진 막대
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", zorder=3))
        return
    rx, ry = (r_len, r_thk) if horizontal else (r_thk, r_len)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=ry / rx, facecolor=color, edgecolor="none", zorder=3))
    # 기준선 쪽 모서리를 다시 각지게 덮는다
    if horizontal:
        ax.add_patch(Rectangle((x, y), 2 * r_len, h, facecolor=color, edgecolor="none", zorder=3))
    else:
        ax.add_patch(Rectangle((x, y), w, 2 * r_len, facecolor=color, edgecolor="none", zorder=3))


def _frame(ax, t: dict, *, grid_axis: str) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def _titles(fig, t: dict, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.965, title, color=t["ink"], fontsize=12.5, fontweight="semibold", va="top")
    fig.text(0.02, 0.885, subtitle, color=t["sub"], fontsize=8.6, va="top")


def _foot(fig, t: dict, text: str) -> None:
    fig.text(0.02, 0.03, text, color=t["muted"], fontsize=7.6, va="bottom")


def brand_correlation(t: dict) -> plt.Figure:
    c = _load("brand_correlation.json")
    between = c["between_brand"]
    rows = [  # (라벨, 값, 하한, 상한, 색)
        ("같은 브랜드 가맹점끼리", c["rho_asset"], c["rho_asset_ci_lo"], c["rho_asset_ci_hi"], t["accent"]),
        ("서로 다른 브랜드끼리", between["rho_between"], between["rho_between_ci_lo"], between["rho_between_ci_hi"],
         t["context"]),
    ]
    lo_b, hi_b = c["basel_corporate_R"]

    fig = plt.figure(figsize=(W_IN, 3.3), dpi=DPI)
    ax = fig.add_axes([0.25, 0.2, 0.7, 0.52])
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.6, 1.6)
    _frame(ax, t, grid_axis="x")
    ax.set_yticks([1, 0], [r[0] for r in rows], color=t["ink"], fontsize=10)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}")

    ax.axvspan(lo_b, hi_b, color=t["band"], alpha=0.07, zorder=1, linewidth=0)
    ax.text((lo_b + hi_b) / 2, 1.5, f"바젤 IRB 기업여신 가정 {lo_b:.2f}–{hi_b:.2f}",
            ha="center", va="center", color=t["sub"], fontsize=8.2)

    fig.canvas.draw()
    for y, (_, v, lo, hi, color) in zip([1, 0], rows, strict=True):
        _bar(ax, base=0, value=v, center=y, color=color, horizontal=True)
        ax.plot([lo, hi], [y, y], color=t["sub"], linewidth=1.2, zorder=4, solid_capstyle="butt")
        ax.text(max(v, hi) + 0.008, y, f"{v:.3f}", va="center", ha="left", color=t["ink"],
                fontsize=10.5, fontweight="bold")

    ratio = c["rho_asset"] / hi_b
    _titles(fig, t, "같은 브랜드 가맹점은 함께 흔들린다",
            f"지역별 점포 감소의 동시성으로 잰 자산상관 ρ (연도·업종 통제) — 바젤 기업여신 가정 상한의 {ratio:.1f}배")
    _foot(fig, t, f"브랜드-연도 {c['n_brand_years']:,} · 지역쌍 {c['n_region_pairs']:,} · "
                  f"가는 선 = 브랜드 블록 부트스트랩 95% 구간 · outputs/brand_correlation.json")
    return fig


def grade_validation(t: dict) -> plt.Figure:
    g = _load("grade_bands.json")
    oot = g["out_of_time"]
    rows = oot["by_grade"]
    base_rate = oot["events"] / oot["n"]

    fig = plt.figure(figsize=(W_IN, 4.1), dpi=DPI)
    ax = fig.add_axes([0.08, 0.25, 0.88, 0.5])
    top = max(r["ci_hi"] for r in rows) * 1.18
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, top)
    _frame(ax, t, grid_axis="y")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    labels = [f"{r['grade']} {r['grade_kr']}" for r in rows]
    ax.set_xticks(range(len(rows)), labels, color=t["ink"], fontsize=10)

    ax.axhline(base_rate, color=t["sub"], linewidth=1, zorder=2)
    ax.text(len(rows) - 0.42, base_rate, f"전체 {base_rate:.1%}", ha="right", va="bottom",
            color=t["sub"], fontsize=8.2)

    fig.canvas.draw()
    under_ticks = blended_transform_factory(ax.transData, ax.transAxes)
    for i, (r, color) in enumerate(zip(rows, t["ramp"], strict=True)):
        _bar(ax, base=0, value=r["rate"], center=i, color=color, horizontal=False)
        ax.plot([i, i], [r["ci_lo"], r["ci_hi"]], color=t["sub"], linewidth=1.2, zorder=4)
        # 값은 구간 위에 — 기준선·구간선과 겹치지 않는 유일한 자리
        ax.text(i, r["ci_hi"] + top * 0.025, f"{r['rate']:.1%}", ha="center", va="bottom", color=t["ink"],
                fontsize=11, fontweight="bold")
        ax.text(i, -0.17, f"악화 {r['events']} / {r['n']}개", transform=under_ticks, ha="center", va="top",
                color=t["muted"], fontsize=8)

    fit, test = oot["fit_years"][0], oot["test_years"][0]
    spread = rows[-1]["rate"] / rows[0]["rate"]
    _titles(fig, t, "등급이 실제 위험을 가른다",
            f"{fit}년 자료로 맞춘 등급을 {test}년 브랜드에 그대로 적용 — 주의 등급 악화율이 안정 등급의 {spread:.1f}배")
    _foot(fig, t, "악화 = 다음 해 공시 지표가 업종×연도 하위 구간에 진입 · 가는 선 = Clopper-Pearson 95% 구간 · "
                  "outputs/grade_bands.json")
    return fig


def tail_loss(t: dict) -> plt.Figure:
    m = _load("correlation_impact.json")
    levels = [("95%", "p95"), ("99%", "p99"), ("99.9%", "p999")]
    ind = [m[f"independent_{k}_mkrw"] / 100 for _, k in levels]      # 백만원 → 억원
    cor = [m[f"brand_correlated_{k}_mkrw"] / 100 for _, k in levels]

    fig = plt.figure(figsize=(W_IN, 3.9), dpi=DPI)
    ax = fig.add_axes([0.08, 0.2, 0.88, 0.55])
    top = max(cor) * 1.18
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_ylim(0, top)
    _frame(ax, t, grid_axis="y")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}억")
    ax.set_xticks(range(len(levels)), [f"손실 분위수 {lv}" for lv, _ in levels], color=t["ink"], fontsize=10)

    fig.canvas.draw()
    ppx, _ = _px_per_unit(ax)
    off = (BAR_PX + 4) / ppx / 2                                    # 막대 사이 2px 간격 (표시 기준)
    for i, (a, b) in enumerate(zip(ind, cor, strict=True)):
        _bar(ax, base=0, value=a, center=i - off, color=t["context"], horizontal=False)
        _bar(ax, base=0, value=b, center=i + off, color=t["accent"], horizontal=False)
        ax.text(i - off, a + top * 0.02, f"{a:.1f}", ha="center", va="bottom", color=t["sub"], fontsize=8.6)
        ax.text(i + off, b + top * 0.02, f"{b:.1f}", ha="center", va="bottom", color=t["ink"], fontsize=9.5,
                fontweight="bold")

    # 범례 (2계열이므로 항상 둔다)
    for j, (name, color) in enumerate([("차주 독립 가정", t["context"]), ("브랜드 상관 반영", t["accent"])]):
        fig.patches.append(Rectangle((0.64 + j * 0.17, 0.815), 0.012, 0.03, transform=fig.transFigure,
                                     facecolor=color, edgecolor="none"))
        fig.text(0.656 + j * 0.17, 0.83, name, color=t["sub"], fontsize=8.4, va="center")

    ul_i, ul_c = m["independent_ul99_mkrw"] / 100, m["brand_correlated_ul99_mkrw"] / 100
    _titles(fig, t, "브랜드를 무시하면 꼬리손실을 과소추정한다",
            f"비예상손실(99%) {ul_i:.1f}억 → {ul_c:.1f}억, {m['ul99_multiple']:.2f}배")
    _foot(fig, t, f"{m['n_brands']}개 브랜드 · 차주 {m['n_franchisees']:,}명 · 몬테카를로 {m['n_sims']:,}회 · "
                  f"LGD {m['lgd']:.0%} · 상관은 대리지표, 익스포저는 공시 창업비용 기반 추정 · "
                  "outputs/correlation_impact.json")
    return fig


def main() -> int:
    plt.rcParams["font.family"] = _font()
    plt.rcParams["axes.unicode_minus"] = False
    DEST.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        for name, fn in (("brand_correlation", brand_correlation), ("grade_validation", grade_validation),
                         ("tail_loss", tail_loss)):
            fig = fn(t)
            path = DEST / f"{name}-{mode}.png"
            fig.savefig(path, dpi=DPI, transparent=True)
            plt.close(fig)
            print(f"  → {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
