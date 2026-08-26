"""입 벌림 게이트 — 학습 데이터 필터와 실시간 추론이 같은 기준을 사용한다.

하품 모델은 "입은 벌어졌다, 이게 하품인가 말하기인가"를 판별한다.
입을 다물었는지는 CNN이 아니라 랜드마크 기반 게이트에서 먼저 판단한다.

    open_ratio <= threshold
        -> 하품 아님
        -> CNN을 호출하지 않고 p(yawn)=0

    open_ratio > threshold
        -> Yawn CNN으로 전달

학습 데이터셋과 실시간 추론에서 반드시 같은 threshold를 사용해야 한다.

지표
    gap = 안쪽 입술 위(13) - 아래(14) 거리
    eye = 눈 바깥 끝(33) - (263) 거리
    open_ratio = gap / eye

얼굴 크기의 영향을 줄이기 위해 입 벌림 거리를 눈 사이 거리로 정규화한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# =====================================================================
# 상수
# =====================================================================

# MediaPipe FaceLandmarker 478점 기준
IDX_LIP_UPPER_IN = 13
IDX_LIP_LOWER_IN = 14

IDX_LIP_UPPER_OUT = 0
IDX_LIP_LOWER_OUT = 17

IDX_MOUTH_L = 61
IDX_MOUTH_R = 291

IDX_EYE_L = 33
IDX_EYE_R = 263


# 랜드마크 측정 시 최대 이미지 크기
MEASURE_SIZE = 256


# 데이터셋 생성 시 사용한 임계값과 동일해야 한다.
DEFAULT_THRESHOLD = 0.05


# FaceLandmarker 모델 기본 위치
DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "detectors"
    / "face_landmarker.task"
)


# =====================================================================
# MouthMeter
# =====================================================================

class MouthMeter:
    """얼굴 crop 한 장에서 입 벌림 정도를 측정한다.

    반환값:
        {
            "open_ratio": ...,
            "mar": ...,
            "lip_ratio": ...,
            "mouth_w_over_eye": ...
        }

    얼굴 또는 랜드마크를 찾지 못하면 None을 반환한다.
    """

    def __init__(
        self,
        model: Path | str | None = None
    ):

        # 이 모듈을 import만 하는 경우에는 mediapipe가 필요하지 않도록
        # 실제 MouthMeter 객체가 생성될 때 import한다.
        import mediapipe as mp

        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        model = Path(model) if model else DEFAULT_MODEL

        if not model.exists():

            raise SystemExit(
                f"랜드마커 모델이 없습니다: {model}\n"
                "받는 곳: "
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/"
                "face_landmarker.task"
            )

        self._mp = mp

        # MediaPipe FaceLandmarker 생성
        self.lmk = vision.FaceLandmarker.create_from_options(

            vision.FaceLandmarkerOptions(

                base_options=mpp.BaseOptions(
                    model_asset_path=str(model)
                ),

                num_faces=1,

                running_mode=vision.RunningMode.IMAGE,
            )
        )


    def measure(self, img) -> dict | None:
        """얼굴 crop에서 입 벌림 지표를 계산한다."""

        # -------------------------------------------------------------
        # 잘못된 입력 방어
        # -------------------------------------------------------------

        if img is None:
            return None

        if img.size == 0:
            return None


        # -------------------------------------------------------------
        # 이미지 크기 축소
        # -------------------------------------------------------------

        h, w = img.shape[:2]

        if max(h, w) > MEASURE_SIZE:

            scale = MEASURE_SIZE / max(h, w)

            new_w = int(round(w * scale))
            new_h = int(round(h * scale))

            img = cv2.resize(
                img,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            h, w = img.shape[:2]


        # -------------------------------------------------------------
        # BGR -> RGB
        # -------------------------------------------------------------

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        # -------------------------------------------------------------
        # MediaPipe FaceLandmarker
        # -------------------------------------------------------------

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb
        )

        out = self.lmk.detect(mp_image)


        # 얼굴 랜드마크를 찾지 못한 경우
        if not out.face_landmarks:
            return None


        # -------------------------------------------------------------
        # 랜드마크 좌표
        # -------------------------------------------------------------

        landmarks = out.face_landmarks[0]

        p = np.array(
            [
                [
                    landmark.x * w,
                    landmark.y * h
                ]
                for landmark in landmarks
            ],
            dtype=np.float32
        )


        # -------------------------------------------------------------
        # 입 벌림 거리
        # -------------------------------------------------------------

        # 안쪽 입술 위-아래 거리
        gap = float(
            np.linalg.norm(
                p[IDX_LIP_UPPER_IN]
                - p[IDX_LIP_LOWER_IN]
            )
        )


        # 바깥쪽 입술 위-아래 거리
        lip = float(
            np.linalg.norm(
                p[IDX_LIP_UPPER_OUT]
                - p[IDX_LIP_LOWER_OUT]
            )
        )


        # 입 좌우 너비
        mouth_w = float(
            np.linalg.norm(
                p[IDX_MOUTH_L]
                - p[IDX_MOUTH_R]
            )
        )


        # 눈 바깥쪽 사이 거리
        eye = float(
            np.linalg.norm(
                p[IDX_EYE_L]
                - p[IDX_EYE_R]
            )
        )


        # -------------------------------------------------------------
        # 잘못된 거리 방어
        # -------------------------------------------------------------

        if eye <= 0:
            return None

        if mouth_w <= 0:
            return None


        # -------------------------------------------------------------
        # 지표 계산
        # -------------------------------------------------------------

        open_ratio = gap / eye

        mar = gap / mouth_w

        lip_ratio = lip / eye

        mouth_w_over_eye = mouth_w / eye


        return {

            # 실제 게이트에서 사용하는 주 지표
            "open_ratio": round(
                open_ratio,
                5
            ),

            # 고전적 MAR — 참고용
            "mar": round(
                mar,
                5
            ),

            # 바깥 입술 두께 포함
            "lip_ratio": round(
                lip_ratio,
                5
            ),

            # 입 너비 / 눈 거리
            "mouth_w_over_eye": round(
                mouth_w_over_eye,
                5
            ),
        }


# =====================================================================
# MouthGate
# =====================================================================

class MouthGate:
    """실시간 Yawn CNN 앞에서 입 벌림 여부를 판단한다.

    check(crop)의 반환값:

        (gate_open, open_ratio)

    gate_open == False
        -> 입을 다문 상태
        -> Yawn CNN 호출하지 않음

    gate_open == True
        -> 입이 충분히 벌어진 상태
        -> Yawn CNN 호출

    랜드마크 측정에 실패한 경우에는 fail-open 방식으로 True를 반환한다.
    즉, 측정 실패 때문에 실제 하품을 놓치는 것을 방지하기 위해 CNN으로 넘긴다.

    hold는 최근 N개의 open_ratio 중 최댓값을 사용한다.
    임계값 주변에서 게이트가 프레임마다 열렸다 닫혔다 하는 현상을 줄인다.
    """

    def __init__(
        self,
        model=None,
        threshold: float = DEFAULT_THRESHOLD,
        hold: int = 3
    ):

        self.meter = MouthMeter(model)

        self.threshold = float(
            threshold
        )

        self.hold = max(
            int(hold),
            1
        )

        self._recent: list[float] = []


    # -----------------------------------------------------------------
    # 게이트 상태 초기화
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """최근 open_ratio 기록을 초기화한다."""

        self._recent.clear()


    # -----------------------------------------------------------------
    # open_ratio 측정
    # -----------------------------------------------------------------

    def measure(
        self,
        crop
    ) -> float | None:
        """얼굴 crop에서 open_ratio를 측정한다.

        측정할 수 없는 경우 None을 반환한다.
        """

        if crop is None:
            return None

        if crop.size == 0:
            return None


        result = self.meter.measure(
            crop
        )


        if result is None:
            return None


        return float(
            result["open_ratio"]
        )


    # -----------------------------------------------------------------
    # 게이트 판정
    # -----------------------------------------------------------------

    def check(
        self,
        crop
    ) -> tuple[bool, float | None]:
        """Yawn CNN을 호출할지 결정한다.

        반환:
            (gate_open, open_ratio)

        gate_open:
            True  -> Yawn CNN 실행
            False -> Yawn CNN 실행하지 않음

        open_ratio:
            측정 성공 -> float
            측정 실패 -> None
        """

        r = self.measure(
            crop
        )


        # -------------------------------------------------------------
        # 랜드마크 측정 실패
        # -------------------------------------------------------------
        #
        # fail-open:
        #
        # 측정 실패를 "입 다물음"으로 간주하면
        # 얼굴 각도/가림 때문에 실제 하품을 놓칠 수 있다.
        #
        # 따라서 CNN으로 그대로 넘긴다.
        # -------------------------------------------------------------

        if r is None:
            return True, None


        # -------------------------------------------------------------
        # 최근 open_ratio 기록
        # -------------------------------------------------------------

        self._recent.append(r)


        # hold 개수만 유지
        if len(self._recent) > self.hold:

            self._recent = (
                self._recent[-self.hold:]
            )


        # -------------------------------------------------------------
        # 게이트 판정
        # -------------------------------------------------------------

        gate_open = (
            max(self._recent)
            > self.threshold
        )


        return gate_open, r