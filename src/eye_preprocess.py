"""
eye_preprocess.py — 학습과 추론이 '똑같이' 쓰는 눈 전처리 함수.

설계 원칙
- MRL 학습(그레이스케일 PNG)과 DMD 추론(YuNet BGR crop)이 서로 다른 전처리를
  거치면 도메인 갭이 생긴다. 그래서 두 경로가 이 함수 '하나'를 통과하게 한다.
- 실험용 스위치(size / grayscale / sharpen)를 인자로 노출한다. 어떤 조합이
  좋은지는 데이터로 판단하며, 학습·추론은 반드시 같은 값을 써야 한다.
- 현재 02_INFER_YuNet.ipynb 의 추론 순서(resize -> sharpen)를 그대로 맞춘다.
- 출력은 0~255 범위 float32. 모델 첫 레이어의 Rescaling(1/255) 이 정규화하므로
  여기서 스케일링하지 않는다.

반환 shape
- grayscale=True  -> (size, size, 1)
- grayscale=False -> (size, size, 3)  (RGB 순서)

주의: MRL 원본은 1채널 gray, 추론 crop 은 3채널 BGR 로 들어온다. 두 입력을
     모두 받아 동일 형식으로 내보내는 것이 이 함수의 핵심 역할이다.
"""

from __future__ import annotations
import cv2
import numpy as np

# 원본 레포와 동일한 sharpen 커널 (detector.py / 02_INFER 와 일치)
SHARPEN_KERNEL = np.array(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]],
    dtype=np.float32,
)


def sharpen(img: np.ndarray) -> np.ndarray:
    """원본 레포와 동일한 3x3 sharpen."""
    return cv2.filter2D(src=img, ddepth=-1, kernel=SHARPEN_KERNEL)


def preprocess_eye(
    img: np.ndarray,
    size: int = 128,
    grayscale: bool = True,
    do_sharpen: bool = False,
    input_is_bgr: bool = True,
) -> np.ndarray:
    """눈 이미지 한 장을 모델 입력 텐서로 변환한다.

    Parameters
    ----------
    img : np.ndarray
        (H,W) 또는 (H,W,1) 그레이스케일(MRL), 혹은 (H,W,3) 컬러(추론 crop).
    size : int
        정사각 리사이즈 한 변. 실험 대상(예: 96 / 128 / 256).
    grayscale : bool
        True 면 1채널로, False 면 3채널 RGB 로 출력. 실험 대상.
    do_sharpen : bool
        sharpen 적용 여부. 실험 대상. 학습·추론에서 동일해야 한다.
    input_is_bgr : bool
        3채널 입력이 BGR(OpenCV) 인지 여부. cv2.imread / YuNet crop 은 True.
        RGB 로 들어오면 False.

    Returns
    -------
    np.ndarray
        (size, size, C) float32, 값 범위 0~255.
    """
    a = np.asarray(img)

    # 1) 색 공간을 BGR 3채널 작업본으로 통일 (색 연산을 한 곳에서만 하기 위함)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.shape[2] == 1:
        work = cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif a.shape[2] == 3:
        work = a if input_is_bgr else cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"지원하지 않는 채널 수: {a.shape}")

    if work.dtype != np.uint8:
        work = np.clip(work, 0, 255).astype(np.uint8)

    # 2) 추론(02_INFER)과 동일한 순서: resize -> sharpen
    work = cv2.resize(work, (size, size), interpolation=cv2.INTER_AREA)
    if do_sharpen:
        work = sharpen(work)

    # 3) 출력 채널 선택
    if grayscale:
        g = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        out = g[:, :, None]
    else:
        out = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

    return out.astype(np.float32)


def channels(grayscale: bool) -> int:
    """모델 Input 채널 수 헬퍼."""
    return 1 if grayscale else 3
