"""
yawn_models.py — 하품 분류 모델 zoo. "다양한 파라미터 크기의 모델"을 한 곳에 등록한다.

설계 원칙
- 모든 모델은 항상 softmax 2-class 확률을 출력한다(class 0 = yawn). train_yawn.py 의
  평가 하네스(yawn_prob, metrics_at, _evaluate)가 이 출력 형태를 그대로 기대하므로,
  zoo 의 어떤 아키텍처를 넣어도 평가·distillation 코드를 바꿀 필요가 없다.
- 커스텀 CNN 계열은 train_yawn.build_yawn_model() 과 같은 증강 스택
  (Rescaling -> RandomFlip -> RandomRotation) 을 그대로 재사용한다. 구조 차이가 아니라
  "얼마나 크고 깊은가"만 실험 변수로 남기기 위함이다. cnn_small 은 아예 그 함수를
  그대로 호출해 기존 배포 구조와 100% 동일하게 만든다.
- 사전학습 backbone 계열(ImageNet)은 gray128 대신 rgb128 입력을 쓴다. eye_preprocess.
  preprocess_eye 가 이미 0~255 float(RGB) 를 내주므로, 각 backbone이 기대하는 스케일로
  바꾸는 Rescaling 레이어를 모델 첫 층에 넣어 흡수한다 — yawn_dataset.make_dataset
  파이프라인 자체는 손대지 않는다.

레지스트리 사용
    from yawn_models import ZOO
    spec = ZOO["cnn_base"]
    model = spec.build()
    # spec.size, spec.grayscale, spec.channels, spec.kind, spec.note
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import tensorflow as tf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train_yawn import build_yawn_model  # cnn_small 을 기존 배포 구조와 동일하게 재사용

SIZE_GRAY = 128
SIZE_RGB = 128


# =====================================================================
# 1. 커스텀 CNN 계열
# =====================================================================
def build_cnn(size: int, ch: int, conv_filters: tuple[int, ...], dense: int,
              batchnorm: bool = False, global_pool: bool = False,
              dropout: float = 0.0) -> tf.keras.Model:
    """build_yawn_model() 과 같은 증강 스택을 쓰는 커스텀 CNN. conv_filters 의 길이·값만
    바꿔 tiny~large 를 전부 이 함수 하나로 만든다.
    """
    layers: list = [
        tf.keras.layers.Input(shape=(size, size, ch)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08, fill_mode="reflect"),
    ]
    for f in conv_filters:
        layers.append(tf.keras.layers.Conv2D(f, 3, use_bias=not batchnorm))
        if batchnorm:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation("relu"))
        layers.append(tf.keras.layers.MaxPooling2D())
    layers.append(tf.keras.layers.GlobalAveragePooling2D() if global_pool
                  else tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(dense, activation="relu"))
    if dropout:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(2, activation="softmax"))
    return tf.keras.Sequential(layers)


# =====================================================================
# 2. 사전학습 backbone 계열
# =====================================================================
def _backbone_rescale(family: str) -> tf.keras.layers.Layer:
    """0~255 float(RGB) 입력을 backbone 이 기대하는 스케일로 바꾼다.

    keras.applications.*.preprocess_input 과 같은 선형 변환이라 Rescaling 하나로
    정확히 재현된다(둘 다 -1~1, 혹은 무변환).
    """
    if family == "mobilenetv2":
        return tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)
    if family == "efficientnet":
        # EfficientNet 은 preprocess_input 이 스케일링을 하지 않고(0~255 그대로),
        # 정규화가 모델 내부 Rescaling/Normalization 층에 이미 포함돼 있다.
        return tf.keras.layers.Rescaling(1.0)
    raise ValueError(f"알 수 없는 backbone family: {family!r}")


_APPLICATIONS: dict[str, Callable[..., tf.keras.Model]] = {
    "mobilenetv2": tf.keras.applications.MobileNetV2,
    "efficientnet": tf.keras.applications.EfficientNetB0,
}


def build_backbone(size: int, family: str, alpha: float = 1.0,
                   freeze: bool = True, dense: int = 128) -> tf.keras.Model:
    """ImageNet 사전학습 backbone + 커스텀 head. include_top=False, pooling='avg'."""
    kwargs = dict(input_shape=(size, size, 3), include_top=False, weights="imagenet",
                  pooling="avg")
    if family == "mobilenetv2":
        kwargs["alpha"] = alpha
    base = _APPLICATIONS[family](**kwargs)
    base.trainable = not freeze

    inputs = tf.keras.layers.Input(shape=(size, size, 3))
    x = _backbone_rescale(family)(inputs)
    # BatchNorm 통계는 freeze 여부와 무관하게 추론 모드로 고정한다(작은 데이터셋에서
    # 파인튜닝 시 BN 이 흔들리는 것을 막는 표준적인 처방).
    x = base(x, training=False)
    x = tf.keras.layers.Dense(dense, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=f"{family}_a{alpha}_{'frozen' if freeze else 'ft'}")


# =====================================================================
# 3. CNN + backbone 하이브리드
# =====================================================================
def build_hybrid(size: int, alpha: float = 0.35,
                 cut_layer: str = "block_3_expand_relu", dense: int = 128) -> tf.keras.Model:
    """MobileNetV2 얕은 block 까지만 frozen stem 으로 쓰고, 그 위에 커스텀 conv/dense
    head 를 처음부터 학습한다. "저수준 특징은 사전학습, 고수준 판단은 이 데이터로" 조합.

    cut_layer 는 TensorFlow의 MobileNetV2 특징추출 튜토리얼(pix2pix 기반 U-Net 예제)이
    쓰는 표준 skip-connection 층 이름이다. 128 입력 기준 block_3_expand_relu 출력은
    32x32 해상도라 그 위에 conv 2블록을 더 쌓기에 적당하다.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights="imagenet", alpha=alpha)
    stem = tf.keras.Model(base.input, base.get_layer(cut_layer).output, name="mbnet_stem")
    stem.trainable = False

    inputs = tf.keras.layers.Input(shape=(size, size, 3))
    x = _backbone_rescale("mobilenetv2")(inputs)
    x = stem(x, training=False)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="hybrid_mbnet_stem")


# =====================================================================
# 4. 레지스트리
# =====================================================================
@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                              # "cnn" | "backbone" | "hybrid"
    size: int
    grayscale: bool
    build: Callable[[], tf.keras.Model]
    note: str = ""

    @property
    def channels(self) -> int:
        return 1 if self.grayscale else 3


ZOO: dict[str, ModelSpec] = {
    "cnn_tiny": ModelSpec(
        "cnn_tiny", "cnn", SIZE_GRAY, True,
        lambda: build_cnn(SIZE_GRAY, 1, (16, 16), dense=64, global_pool=True),
        "student 후보. Conv 2블록(16,16) + GAP + Dense64. zoo 에서 가장 작다."),
    "cnn_small": ModelSpec(
        "cnn_small", "cnn", SIZE_GRAY, True,
        lambda: build_yawn_model(SIZE_GRAY, 1),
        "기존 배포 구조와 동일(train_yawn.build_yawn_model, Flatten 기반). student 후보이자"
        " 구조 기준점. Flatten 특성상 GAP 기반인 cnn_base/cnn_large 보다 파라미터가 많다."),
    "cnn_base": ModelSpec(
        "cnn_base", "cnn", SIZE_GRAY, True,
        lambda: build_cnn(SIZE_GRAY, 1, (32, 64, 64, 128), dense=256,
                          batchnorm=True, global_pool=True, dropout=0.3),
        "teacher 후보. Conv 4블록 + BatchNorm + GAP."),
    "cnn_large": ModelSpec(
        "cnn_large", "cnn", SIZE_GRAY, True,
        lambda: build_cnn(SIZE_GRAY, 1, (64, 64, 128, 128, 256), dense=256,
                          batchnorm=True, global_pool=True, dropout=0.4),
        "teacher 후보. Conv 5블록 + BatchNorm + GAP."),
    "mbnetv2_035": ModelSpec(
        "mbnetv2_035", "backbone", SIZE_RGB, False,
        lambda: build_backbone(SIZE_RGB, "mobilenetv2", alpha=0.35),
        "가벼운 사전학습 teacher 후보. ImageNet 가중치, 헤드만 학습(freeze)."),
    "mbnetv2_100": ModelSpec(
        "mbnetv2_100", "backbone", SIZE_RGB, False,
        lambda: build_backbone(SIZE_RGB, "mobilenetv2", alpha=1.0),
        "무거운 사전학습 teacher 후보."),
    "efficientnet_b0": ModelSpec(
        "efficientnet_b0", "backbone", SIZE_RGB, False,
        lambda: build_backbone(SIZE_RGB, "efficientnet"),
        "가장 무거운 사전학습 teacher 후보."),
    "hybrid_mbnet_stem": ModelSpec(
        "hybrid_mbnet_stem", "hybrid", SIZE_RGB, False,
        lambda: build_hybrid(SIZE_RGB),
        "MobileNetV2 얕은 block(frozen stem) + 커스텀 conv/dense head."),
}


def describe() -> None:
    """zoo 전체를 만들어보고 파라미터 수를 출력한다. 등록 오류를 빨리 잡기 위한 점검용."""
    for name, spec in ZOO.items():
        m = spec.build()
        print(f"[{spec.kind:<8}] {name:<20} size={spec.size} ch={spec.channels} "
              f"params={m.count_params():,}  {spec.note}")


if __name__ == "__main__":
    describe()
