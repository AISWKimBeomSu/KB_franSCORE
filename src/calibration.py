"""확률 보정 — 등온회귀 + 계단별 경험적 베이즈 축소.

왜 새로 만드는가 (검증 실패 추적 결과)
    기존 보정은 **워크포워드 fold 예측**으로 적합했다. 그런데 워크포워드는 fold 마다
    모형을 재학습하므로 원점수 척도가 fold 마다 다르다 — 실측 평균 0.2826 / 0.3306 /
    0.1698 로 최대 44.8% 벌어진다. 높은 척도에서 적합한 보정기를 낮은 척도에 적용하면
    등온회귀 곡선의 아래쪽으로 매핑돼 **체계적으로 과소예측**한다.
    실제로 그랬다: 예측 7.47% vs 실제 9.87%, Spiegelhalter Z=+3.41, HL p=0.00027(기각).

    기저율 이동은 원인이 아니었다. 2023 fold 는 선행 기저율(9.90%)이 실제(8.11%)보다
    **높았는데도** 과소예측했다. 척도 이동이 그것을 뒤집을 만큼 컸다.

무엇을 바꿨나
    1. **운영 모형 점수로 보정한다.** 운영은 모형이 하나이므로 척도가 안정적이다
       (실측 연도별 평균 0.1752 / 0.1853 / 0.1698). 대신 학습 연도의 점수는 in-sample
       이라 쓰지 않는다 — 써 봤더니 Z=+17.8 로 훨씬 나빠졌다(과적합된 분리력이 그대로
       보정 곡선에 들어간다).
    2. **계단별 경험적 베이즈 축소.** 등온회귀의 극단 계단은 소표본에서 불안정하다.
       실측: 최하위 구간 예측 0.46% vs 실제 2.10%. 계단 j 의 (n_j, k_j) 를
       `(k_j + α·B) / (n_j + α)` 로 당긴다(B = 보정표본 기저율).
    3. **α 는 보정표본 안에서만 K-겹 교차검증으로 고른다.** 검증 연도를 보고 α 를
       고르면 그건 검정에 하이퍼파라미터를 맞추는 것이다.

왜 순위 기반 보정을 쓰지 않았나
    fold 척도 문제는 순위(백분위)로 보정하면 깨끗이 사라진다. 그러나 그러면
    `p = g(코호트 내 순위)` 가 되어 **등급 컷이 다시 순위 컷이 된다** — 절대 등급으로
    바꾼 이유가 통째로 무효화된다. 그래서 값 기반을 유지하고 척도 쪽을 고쳤다.

결과 (2022 로 적합 → 2023 검증, 둘 다 학습연도 밖)
    예측 10.00% vs 실제 8.11% · HL p=0.158 통과 · Spiegelhalter Z=−1.109 통과
    남은 편차는 **과대예측** 방향이다 — 은행에 안전한 쪽이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

ALPHAS: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
CV_FOLDS = 5
CV_SEED = 42
_EPS = 1e-6


class ShrunkIsotonic:
    """등온회귀 + 계단별 축소. joblib 으로 저장·복원된다.

    ⚠️ 클래스 참조가 pickle 에 들어가므로 모듈 경로를 옮기면 기존 산출물을 못 읽는다.
       위치를 바꿀 때는 보정기를 다시 만들 것.
    """

    def __init__(self, iso: IsotonicRegression, mapping: dict[float, float],
                 base: float, alpha: float, fitted_on: str, n: int, events: int):
        self.iso = iso
        self.mapping = mapping
        self.base = float(base)
        self.alpha = float(alpha)
        self.fitted_on = str(fitted_on)
        self.n = int(n)
        self.events = int(events)

    def _levels(self) -> tuple[np.ndarray, np.ndarray]:
        """정렬된 (계단값, 축소확률). 옛 pickle 에도 없는 필드라 지연 계산한다."""
        cache = getattr(self, "_lv_cache", None)
        if cache is None:
            k = np.array(sorted(self.mapping), dtype=float)
            v = np.array([self.mapping[x] for x in k], dtype=float)
            cache = (k, v)
            self._lv_cache = cache
        return cache

    def predict(self, x) -> np.ndarray:
        """원점수 → 보정확률. **계단 함수로** 평가한다.

        ⚠️ 예전에는 `self.mapping.get(lv, self.base)` 로 조회했다. 그런데
           `IsotonicRegression.predict()` 는 학습 knot 사이의 입력에 대해 **선형
           보간값**을 돌려준다 — 그 값은 계단 키에 없다. 그래서 조회가 실패하고
           조용히 기저율(base)로 대체됐다.

           실측(2024 코호트 1,442행): 14행이 이 경로를 탔고, 인접 계단으로
           올바르게 스냅하면 **12건의 등급이 바뀐다 — 11건이 FS2→FS3**, 즉 위험을
           낮게 표시하고 있었다. floor·ceil 두 규약이 동일한 12건을 내므로 규약
           선택과 무관한 오류다.

           더 무거운 문제는 단조성이었다. 보간 구간의 브랜드만 평균값을 받으므로
           원점수가 더 나쁜데 등급이 더 좋은 구간이 실재했다(FS3→FS2→FS3).
           계단 함수는 정의상 단조라 이 문제가 사라진다.

           floor 를 쓰는 이유: 등온회귀의 계단은 오른쪽 연속 계단함수이고, 각 계단은
           **실현율을 실제로 관측한 표본 집단**이다. 보간값을 그대로 쓰면 어떤 집단의
           실적으로도 뒷받침되지 않는 확률이 나온다. 아래 계단으로 내리면 모든 출력이
           검증 가능한 계단값 위에 남는다.
        """
        lv = np.round(self.iso.predict(np.asarray(x, dtype=float)), 10)
        keys, vals = self._levels()
        idx = np.clip(np.searchsorted(keys, lv, side="right") - 1, 0, len(keys) - 1)
        return np.clip(vals[idx], _EPS, 1.0 - _EPS)

    def describe(self) -> dict:
        return {"method": "isotonic+eb_shrinkage", "alpha": self.alpha,
                "fitted_on": self.fitted_on, "n": self.n, "events": self.events,
                "base_rate": round(self.base, 6), "n_levels": len(self.mapping)}


def _pav(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """가중 PAV — 오름차순을 깨는 인접 구간을 표본수 가중평균으로 합친다.

    축소는 계단마다 따로 하므로 **순서를 보장하지 않는다**. 표본이 적은 계단은
    기저율 쪽으로 세게 끌려가는데, 그 계단이 낮은 쪽에 있으면 위로 끌려 올라가
    바로 위 계단을 추월한다.

    실측(배포 보정기): 계단 0.000000 → 축소 0.013156, 계단 0.005814 → 축소 0.010916.
    원점수가 **더 나쁜** 브랜드가 **더 낮은** 확률을 받는다. 보정 함수가 단조가
    아니면 '보정'이라는 이름이 성립하지 않는다.

    단순히 누적최대(np.maximum.accumulate)로 밀어 올리지 않는다 — 그건 위반한
    쪽만 일방적으로 올려 확률의 총합(기대 사건수)을 부풀린다. 가중 PAV 는 합친
    구간의 가중평균을 쓰므로 표본 전체의 기대 사건수가 보존된다.
    """
    v = np.asarray(v, dtype=float).copy()
    w = np.asarray(w, dtype=float).copy()
    # (값, 가중치, 구간길이) 스택을 쌓으며 위반이 생기면 뒤에서부터 합친다
    vals: list[float] = []
    wts: list[float] = []
    lens: list[int] = []
    for vi, wi in zip(v, w, strict=True):
        vals.append(vi)
        wts.append(wi)
        lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, l2 = vals.pop(), wts.pop(), lens.pop()
            v1, w1, l1 = vals.pop(), wts.pop(), lens.pop()
            tot = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / tot if tot > 0 else (v1 + v2) / 2)
            wts.append(tot)
            lens.append(l1 + l2)
    out = np.concatenate([np.full(n, val) for val, n in zip(vals, lens, strict=True)])
    return out


def _fit_one(x: np.ndarray, y: np.ndarray, alpha: float) -> ShrunkIsotonic:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)
    base = float(np.mean(y))
    t = (pd.DataFrame({"lv": np.round(iso.predict(x), 10), "y": y})
         .groupby("lv").agg(n=("y", "size"), k=("y", "sum")))
    shrunk = ((t["k"] + alpha * base) / (t["n"] + alpha)).to_numpy()
    # 계단은 오름차순(groupby 색인)이므로 그 순서 그대로 단조로 사영한다
    shrunk = _pav(shrunk, t["n"].to_numpy())
    mp = dict(zip(t.index, shrunk, strict=True))
    return ShrunkIsotonic(iso, mp, base, alpha, "", len(y), int(np.sum(y)))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def choose_alpha(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[float, float]]:
    """보정표본 **안에서만** K-겹 교차검증으로 축소 강도를 고른다.

    검증 연도를 보고 고르면 검정에 하이퍼파라미터를 맞추는 것이 된다.
    """
    kf = KFold(CV_FOLDS, shuffle=True, random_state=CV_SEED)
    scores: dict[float, list[float]] = {a: [] for a in ALPHAS}
    for tr, te in kf.split(x):
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        for a in ALPHAS:
            scores[a].append(_logloss(y[te], _fit_one(x[tr], y[tr], a).predict(x[te])))
    mean = {a: float(np.mean(v)) for a, v in scores.items() if v}
    if not mean:
        return 10.0, {}
    return min(mean, key=lambda k: mean[k]), mean


def fit(x, y, *, fitted_on: str = "") -> tuple[ShrunkIsotonic, dict]:
    """보정기 적합. (보정기, CV 로그손실표) 반환."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    alpha, cv = choose_alpha(x, y)
    cal = _fit_one(x, y, alpha)
    cal.fitted_on = fitted_on
    return cal, {str(k): round(v, 6) for k, v in cv.items()}
