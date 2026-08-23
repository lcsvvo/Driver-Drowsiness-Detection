"""하품 확률을 시간축으로 누적한다 — "입을 벌리기만 해도 하품" 을 막는 장치.

프레임 한 장으로는 하품과 말하기를 가를 수 없다. 어느 한 순간만 보면 둘 다 그냥
"입이 벌어진 얼굴" 이다. 실제로 CNN 은 말하기 세션의 68% 에서 최소 한 번 하품 경보를
낸다(YawDD val+test 41세션 기준). 재학습으로 풀려면 시퀀스 모델이 필요한데, 하품
사건이 180개뿐이라 지금 데이터로는 과적합한다.

대신 **이미 매 프레임 공짜로 나오는 두 신호** 를 시간축으로 묶는다.

    p_yawn      CNN 출력 (class 0)
    open_ratio  입 벌림 게이트가 어차피 계산하는 값 (src/mouth_gate.py)

근거가 되는 실측 (YawDD 세션 단위 중앙값):

    조건       입 열린 비율   |기울기|/초   열린 구간(s)   최대 open_ratio
    normal        0.000        0.004        0.00           0.048
    talking       0.653        0.056        1.75           0.195
    yawning       0.367        0.018        2.47           0.453

말하기는 입이 3배 빠르게 움직이고 하품은 2배 넓게 벌어진다. 그래서 두 가지를 본다.

1. **얼마나 오래 확신했나** — 누수 적분기. `p_yawn` 이 판정선을 넘는 동안 시간을 쌓고,
   못 넘으면 더 빨리 깎는다. 말하기의 짧은 스파이크는 쌓이기 전에 식는다.
2. **얼마나 크게 벌렸나** — 최근 창의 최대 `open_ratio`. 말하기는 0.20 을 잘 못 넘는다.
   이 조건에 걸리면 **아예 쌓지 않는다.** 뒤에서 점수만 누르면 acc 는 차 있는데
   표시만 판정선 바로 아래에 붙어 멈춘 것처럼 보인다.

**`score` 는 0~1 연속값이다.** 쌓인 증거(초)를 상한으로 나눈 것이라 그대로 "얼마나
하품 같은가" 로 읽힌다. 실측 (세션별 최댓값):

    조건            p25     중앙값    p75     p90
    talking       0.000   0.000   0.000   0.404
    yawning       0.559   1.000   1.000   1.000
    DMD 하품(손X)   1.000   1.000   1.000   1.000

입을 안 벌리면 정확히 0, 말하기는 0 근처, 실제 하품은 1 에 붙는다.

발화 판정(`fired`)은 `acc >= fire` 로 따로 내고 히스테리시스를 건다 - 하품 한 번이
문턱 근처에서 깜빡이며 여러 번 경보로 세어지지 않게. DROWSY EMA 에 넣을 값은
`alarm` 으로 따로 낸다(그쪽은 0.5 가 판정선인 규칙을 쓴다).

**프레임이 아니라 시간을 센다.** 추론 루프 속도가 입을 벌리면 22Hz, 다물면 30Hz 로
변하고 PC 마다도 다르기 때문이다. 같은 이유로 `PerclosTracker` 도 프레임 대신 dt 를
더한다.

실측 (기본값 fire=0.15 / peak_min=0.20):

                        YawDD 하품   YawDD 말하기   DMD 사건 Recall   DMD 오경보
    현재 (누적 없음)        0.977        0.683          0.941         0.0032
    누적 적용              0.860        0.195          0.824         0.0016

말하기 오경보가 71% 줄어든다. 대가로 하품 검출이 0.977 -> 0.860 으로 내려간다.
split 을 나눠 봐도 방향은 같다(val 말하기 0.650->0.300, test 0.714->0.095). 다만
split 당 세션이 20개 남짓이라 소수점 둘째 자리는 믿을 값이 아니다.

놓치는 쪽이 더 급하면 `peak_min=0.0` 으로 두면 된다 — 말하기 0.439 로 덜 줄지만
DMD 사건 Recall 이 0.941 로 손실이 없다(특히 손으로 가린 하품은 입이 안 보여
open_ratio 가 낮으므로 peak_min 이 그쪽을 깎는다).
"""
from __future__ import annotations

from collections import deque

#: p_yawn 이 이 값을 넘으면 "증거" 로 친다. 하품 모델의 판정 임계값과 같게 둔다.
DEFAULT_EVIDENCE = 0.50

#: 누적값이 이 초수를 넘으면 하품으로 판정한다. 0.15 는 "확신 0.8 로 0.5초" 정도다.
DEFAULT_FIRE = 0.15

#: 누적 상한(초). 없으면 긴 하품 뒤에 경보가 몇 초씩 남는다(실측: 여운으로 뜬
#: 오경보의 100% 가 실제 하품 5초 이내였다).
DEFAULT_CAP = 1.0

#: 증거가 없을 때 몇 배로 빨리 깎을지. 1.0 이면 쌓일 때와 같은 속도.
DEFAULT_DECAY = 3.0

#: 최근 이 창(초)의 최대 open_ratio 가 peak_min 이상이어야 판정한다.
DEFAULT_PEAK_WINDOW = 3.0
DEFAULT_PEAK_MIN = 0.20

#: 한 번 발화하면 acc 가 fire*RELEASE 밑으로 떨어질 때까지 유지한다(히스테리시스).
#: 하품 한 번이 판정선 근처에서 깜빡이며 여러 번 경보로 세어지는 것을 막는다.
DEFAULT_RELEASE = 0.4

#: 프레임 간격이 이보다 벌어지면 잘라 쓴다. 셀을 멈췄다 재개하거나 카메라가
#: 스톨하면 dt 가 수 초로 튀는데, 그 한 프레임이 누적을 통째로 채우면 안 된다.
DEFAULT_MAX_GAP = 1.0


class YawnAccumulator:
    """(시각, p_yawn, open_ratio) 를 계속 받아 시간 누적된 하품 점수를 낸다.

    반환값
        score   0~1 연속값. "얼마나 하품 같은가". 표시용
        alarm   0.5 가 판정선인 값. DROWSY EMA 용 (eye_score 와 같은 규칙)
        fired   히스테리시스가 걸린 발화 여부. 실제 판정은 이것을 쓴다
        acc     쌓인 증거(초).  peak  최근 창의 최대 open_ratio
    """

    def __init__(self, evidence: float = DEFAULT_EVIDENCE,
                 fire: float = DEFAULT_FIRE,
                 cap: float = DEFAULT_CAP,
                 decay: float = DEFAULT_DECAY,
                 peak_window: float = DEFAULT_PEAK_WINDOW,
                 peak_min: float = DEFAULT_PEAK_MIN,
                 release: float = DEFAULT_RELEASE,
                 max_gap: float = DEFAULT_MAX_GAP):
        self.evidence = float(evidence)
        self.fire = float(fire)
        self.cap = float(cap)
        self.decay = float(decay)
        self.peak_window = float(peak_window)
        self.peak_min = float(peak_min)
        self.release = float(release)
        self.max_gap = float(max_gap)
        self.reset()

    def reset(self) -> None:
        self.acc = 0.0
        self._on = False
        self._prev_t = None
        self._peaks: deque = deque()      # (t, open_ratio)

    # -----------------------------------------------------------------
    def update(self, t: float, p_yawn: float, open_ratio: float | None) -> dict:
        """프레임 하나를 넣는다. t 는 초(단조 증가), open_ratio 는 못 쟀으면 None."""
        dt = 0.0 if self._prev_t is None else min(max(t - self._prev_t, 0.0), self.max_gap)
        self._prev_t = t

        # ---- 1. 최근 창의 최대 벌림 ----
        # 못 쟀으면(None) 창에 넣지 않는다. 넣으면 0 으로 들어가 최댓값을 흐린다.
        if open_ratio is not None:
            self._peaks.append((t, float(open_ratio)))
        while self._peaks and self._peaks[0][0] < t - self.peak_window:
            self._peaks.popleft()
        peak = max((r for _, r in self._peaks), default=0.0)

        # ---- 2. 누수 적분 ----
        # 최근에 크게 벌린 적이 없으면 **증거로 치지 않는다.** 말하기는 입을 크게
        # 벌리지 않으므로(최대 벌림 중앙값 0.195 대 하품 0.453) 여기서 걸러진다.
        #
        # 이 조건을 뒤에서 점수만 누르는 식으로 두면, acc 는 계속 차 있는데 표시만
        # 판정선 바로 아래(0.49)에 붙어 멈춘 것처럼 보인다. 애초에 안 쌓는 편이
        # 동작도 표시도 정직하다.
        gated = peak < self.peak_min
        p_eff = 0.0 if gated else float(p_yawn)

        step = (p_eff - self.evidence) * dt
        if p_eff < self.evidence:
            step *= self.decay                      # 증거가 없으면 빨리 식는다
        self.acc = min(self.cap, max(0.0, self.acc + step))

        # ---- 3. 발화 판정 (히스테리시스) ----
        # 켜지는 문턱과 꺼지는 문턱을 다르게 둔다. 같으면 하품 한 번이 판정선
        # 근처에서 깜빡이며 여러 번 경보로 세어진다.
        if self._on:
            self._on = self.acc > self.fire * self.release
        else:
            self._on = self.acc >= self.fire

        # ---- 4. 점수 ----
        # **연속값 0~1.** 쌓인 증거(초)를 상한으로 나눈 것이라 그대로 "얼마나 하품
        # 같은가" 로 읽힌다.
        #
        #   입을 안 벌림 / 크게 벌린 적 없음  ->  정확히 0   (아예 안 쌓인다)
        #   말하다 잠깐 튄 것               ->  0 근처     (쌓이기 전에 식는다)
        #   실제 하품                       ->  1 에 근접   (2.2초면 상한에 닿는다)
        score = min(max(self.acc / self.cap, 0.0), 1.0)

        # DROWSY EMA 용 값은 따로 낸다. 그쪽은 "0.5 가 판정선" 규칙을 쓰는데
        # (eye_score / yawn_score 와 같은 규칙), score 는 그 규칙을 따르지 않는다.
        # 발화 지점(fire)이 상한의 15% 라 score 를 그대로 넣으면 하품인데도
        # DROWSY 가 안 오른다.
        if not self._on:
            alarm = 0.5 * min(self.acc / max(self.fire, 1e-6), 1.0)
        else:
            alarm = 0.5 + 0.5 * (self.acc - self.fire) / max(self.cap - self.fire, 1e-6)
            alarm = min(max(alarm, 0.5), 1.0)

        return {"score": score, "alarm": alarm, "acc": self.acc, "peak": peak,
                "fired": self._on, "peak_gated": gated}
