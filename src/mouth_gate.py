"""입 벌림 게이트 — 학습 데이터 필터와 실시간 추론이 **같은 자**를 쓰게 하는 단일 출처.

하품 모델은 "입은 벌어졌다, 이게 하품인가 말하기인가"만 푼다. 입을 다물었는지는
CNN 이 아니라 랜드마크로 먼저 가른다.

    open_ratio <= threshold   ->  하품 아님. CNN 을 부르지 않고 p(yawn)=0
    open_ratio >  threshold   ->  CNN 에 물어본다

이렇게 나누는 이유는 두 가지다.

1. **학습과 조건을 맞춘다.** 데이터셋(`scripts/build_yawn_mouthopen_dataset.py`)이
   두 클래스 모두에서 입 다문 프레임을 빼고 만들어졌다. 그 모델에 입 다문 프레임을
   넣으면 학습에서 본 적 없는 입력이라 출력이 아무 의미가 없다.
2. **가장 흔한 오경보를 원천에서 막는다.** 입을 다물고 있는데 하품으로 뜨는 경우가
   사라진다. 그리고 다문 프레임에서는 CNN 을 건너뛰므로 그만큼 빨라진다.

대가: 게이트에 걸린 하품은 모델이 아무리 좋아도 못 잡는다. **임계값이 곧 Recall
상한이다.** DMD 하품 프레임(n=1,148, 프레임 단위 GT)으로 잰 값:

    게이트   하품인데 걸리는 비율   Recall 상한
    0.03            7.8%              0.922
    0.05           10.5%              0.895
    0.08           14.4%              0.856
    0.10           18.3%              0.817

**데이터셋을 만든 임계값과 추론 게이트의 임계값은 같아야 한다.** 다르면 학습에서
본 적 없는 구간이 추론에 들어오거나(게이트가 더 낮을 때), 학습 데이터의 일부가
추론에서 영영 안 쓰인다(더 높을 때).

지표
    gap = 안쪽 입술 위(13) - 아래(14) 거리
    eye = 눈 바깥 끝(33) - (263) 거리          <- 얼굴 크기 기준자
    open_ratio = gap / eye

`eye` 로 나누므로 얼굴이 크게 찍혔든 작게 찍혔든 같은 값이 나온다. 입 너비로 나누는
고전적 MAR 은 하품할 때 입 너비 자체가 변해서 기준자로 삼기에 불안정하다.

**얼굴 crop 에 대고 잰다.** 프레임 전체가 아니다. 학습 데이터의 측정도 crop 에
대고 했고, crop 쪽이 얼굴이 크게 잡혀 랜드마크가 안정적이다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# =====================================================================
# 상수 — 데이터셋 빌더와 추론이 공유한다
# =====================================================================
#: 랜드마크 번호 (mediapipe FaceLandmarker 478점 기준)
IDX_LIP_UPPER_IN, IDX_LIP_LOWER_IN = 13, 14
IDX_LIP_UPPER_OUT, IDX_LIP_LOWER_OUT = 0, 17
IDX_MOUTH_L, IDX_MOUTH_R = 61, 291
IDX_EYE_L, IDX_EYE_R = 33, 263

#: 랜드마크를 잴 때 이 크기로 줄인다. crop 은 353~579px 인데 랜드마크 정확도는
#: 256px 면 충분하고, 줄이면 측정이 2배 이상 빨라진다.
MEASURE_SIZE = 256

#: 기본 임계값. 데이터셋 빌더의 DEFAULT_THRESHOLD 와 같은 값이어야 한다.
DEFAULT_THRESHOLD = 0.05

#: 랜드마커 모델의 관례 위치 (YuNet 과 같은 자리).
#: 받는 곳: https://storage.googleapis.com/mediapipe-models/face_landmarker/
#:         face_landmarker/float16/1/face_landmarker.task
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "model" / "detectors" / "face_landmarker.task"


class MouthMeter:
    """crop 한 장 -> 입 벌림 지표. 얼굴/랜드마크를 못 찾으면 None.

    데이터셋 빌더가 쓰는 클래스다. 실시간에는 아래 MouthGate 를 쓴다.
    """

    def __init__(self, model: Path | str | None = None):
        # import 를 여기서 하는 이유: 이 모듈을 import 만 하고 측정을 안 하는
        # 경로(예: 상수만 쓰는 곳)에서 mediapipe 를 요구하지 않기 위해서다.
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        model = Path(model) if model else DEFAULT_MODEL
        if not model.exists():
            raise SystemExit(
                f"랜드마커 모델이 없습니다: {model}\n"
                "받는 곳: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        self._mp = mp
        self.lmk = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(model)),
                num_faces=1, running_mode=vision.RunningMode.IMAGE))

    def measure(self, img) -> dict | None:
        h, w = img.shape[:2]
        if max(h, w) > MEASURE_SIZE:
            s = MEASURE_SIZE / max(h, w)
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = self.lmk.detect(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not out.face_landmarks:
            return None
        p = np.array([[l.x * w, l.y * h] for l in out.face_landmarks[0]])
        gap = float(np.linalg.norm(p[IDX_LIP_UPPER_IN] - p[IDX_LIP_LOWER_IN]))
        lip = float(np.linalg.norm(p[IDX_LIP_UPPER_OUT] - p[IDX_LIP_LOWER_OUT]))
        mouth_w = float(np.linalg.norm(p[IDX_MOUTH_L] - p[IDX_MOUTH_R]))
        eye = float(np.linalg.norm(p[IDX_EYE_L] - p[IDX_EYE_R]))
        if eye <= 0 or mouth_w <= 0:
            return None
        return {
            "open_ratio": round(gap / eye, 5),        # 주 지표
            "mar": round(gap / mouth_w, 5),           # 고전적 MAR (참고용)
            "lip_ratio": round(lip / eye, 5),         # 바깥 입술 두께 포함
            "mouth_w_over_eye": round(mouth_w / eye, 5),
        }


class MouthGate:
    """실시간용 게이트. 얼굴 crop 을 받아 "CNN 을 부를지" 를 답한다.

    MouthMeter 와 같은 계산을 쓰되, 실시간에서 필요한 두 가지가 더 있다.

    - **랜드마크 실패를 열어 둔다(fail-open).** 고개를 돌리거나 손으로 가려서
      랜드마크가 안 잡히는 순간이 있다. 그때 게이트를 닫으면(=하품 아님) 손으로
      입을 가린 하품이 통째로 사라진다 - 하품에서 가장 흔한 자세다. 그래서 못
      쟀을 때는 CNN 에게 넘긴다. 대신 open_ratio 는 None 으로 보고한다.
    - **시간축으로 눌러 준다.** 프레임 단위 랜드마크는 흔들려서 임계값 근처에서
      게이트가 깜빡인다. 최근 몇 프레임의 최댓값을 쓰면(hold) 하품이 시작되는
      순간을 놓치지 않으면서 깜빡임이 없어진다.
    """

    def __init__(self, model=None, threshold: float = DEFAULT_THRESHOLD,
                 hold: int = 3):
        self.meter = MouthMeter(model)
        self.threshold = float(threshold)
        self.hold = max(int(hold), 1)
        self._recent: list[float] = []

    def reset(self) -> None:
        self._recent.clear()

    def measure(self, crop) -> float | None:
        """얼굴 crop -> open_ratio. 못 재면 None."""
        if crop is None or crop.size == 0:
            return None
        m = self.meter.measure(crop)
        return None if m is None else float(m["open_ratio"])

    def check(self, crop) -> tuple[bool, float | None]:
        """(CNN 을 부를까?, open_ratio) 를 답한다.

        open_ratio 가 None 이면 못 잰 것이고, 그때는 True(=CNN 에 넘김)를 돌린다.
        """
        r = self.measure(crop)
        if r is None:
            # 못 쟀다. hold 창을 건드리지 않고 그대로 통과시킨다 - 실패한 프레임이
            # 창에 0 으로 들어가 다음 몇 프레임까지 게이트를 닫아 버리면 안 된다.
            return True, None

        self._recent.append(r)
        if len(self._recent) > self.hold:
            del self._recent[:-self.hold]

        return max(self._recent) > self.threshold, r
