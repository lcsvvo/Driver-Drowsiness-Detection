"""
train_eye_mrl.py — MRL 로 Eye CNN 재학습 + 정직한 평가.

이 스크립트 하나로 설계의 핵심 실험을 돌린다. 상단 CONFIG 만 바꾸면 된다.

실험 스위치
    SPLIT_MODE : "subject" | "random"   -> leakage 영향 비교 (필수 비교 실험)
    GRAYSCALE  : True | False            -> 도메인 갭 실험
    DO_SHARPEN : True | False            -> sharpen on/off 실험 (학습·추론 동일값 필수)
    INPUT_SIZE : 96 | 128 | 256          -> 입력 크기/속도 실험

평가 (필수)
    val 로 모델 선택(ModelCheckpoint) + EarlyStopping
    test(subject 독립)로 최종 1회 평가:
      Accuracy, Precision, Recall, F1, Confusion Matrix
      + Closed 클래스 별도 Recall/Precision/F1
      (졸음 감지에서 '감긴 눈을 놓치는' Closed false-negative 가 가장 위험)

전처리는 preprocessing/eye_preprocess.py 하나를 학습·추론이 공유한다.

사용 예
    python train_eye_mrl.py
    (사전: preprocessing/mrl_split.py 로 manifest CSV 를 먼저 생성)
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---- 경로: preprocessing 모듈 import ----
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "preprocessing"))

import tensorflow as tf
from eye_preprocess import channels
from mrl_dataset import make_dataset, class_counts, resolve_mrl_root

# =====================================================================
# CONFIG — 실험은 여기만 바꾼다
# =====================================================================
REPO_ROOT    = HERE.parent                    # .../Driver-Drowsiness-Detection
MRL_ROOT     = resolve_mrl_root(REPO_ROOT)    # .../data/MRL Eye/data
MANIFEST_DIR = HERE / "preprocessing"       # manifest CSV 위치
SPLIT_MODE   = "subject"                      # "subject" | "random"
GRAYSCALE    = True
DO_SHARPEN   = False
INPUT_SIZE   = 128
BATCH_SIZE   = 64
EPOCHS       = 20
SEED         = 42

ARTIFACT_DIR = HERE / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

MANIFEST = str(MANIFEST_DIR / f"mrl_manifest_{SPLIT_MODE}.csv")
TAG = f"{SPLIT_MODE}_{'gray' if GRAYSCALE else 'rgb'}_" \
      f"{'sharp' if DO_SHARPEN else 'nosharp'}_{INPUT_SIZE}"
MODEL_PATH = ARTIFACT_DIR / f"eye_mrl_{TAG}.keras"
METRICS_PATH = ARTIFACT_DIR / f"eye_mrl_{TAG}_metrics.json"


def build_eye_model(size: int, ch: int) -> tf.keras.Model:
    """기존 구조 유지 + 입력 파라미터화 + 회전 축소(눈 crop 에 과하지 않게)."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(size, size, ch)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.12, fill_mode="reflect"),  # 0.4->0.12
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),  # class0=Closed
    ])


def evaluate(model, test_ds) -> dict:
    """test 예측 -> 지표 계산. Closed(=0) 를 positive 로 별도 보고."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix, classification_report)
    y_true, y_prob = [], []
    for xb, yb in test_ds:
        p = model(xb, training=False).numpy()
        y_prob.append(p)
        y_true.append(yb.numpy())
    y_prob = np.concatenate(y_prob)
    y_true = np.argmax(np.concatenate(y_true), axis=1)
    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    # macro
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    # Closed 클래스(=0) 를 positive 로
    p_c, r_c, f_c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0], average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()  # [[CC,CO],[OC,OO]]

    print("\n===== TEST 평가 =====")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["Closed", "Open"], digits=4,
                                zero_division=0))
    print("Confusion Matrix  (행=실제, 열=예측) [Closed, Open]:")
    print(f"  실제 Closed: {cm[0]}")
    print(f"  실제 Open  : {cm[1]}")
    print(f"\n[졸음 관점] Closed-Recall={r_c[0]:.4f}  "
          f"Closed-Precision={p_c[0]:.4f}  Closed-F1={f_c[0]:.4f}")
    print("  * Closed-Recall 이 낮으면 '감긴 눈을 Open 으로 놓침' = 가장 위험")

    return {
        "tag": TAG,
        "accuracy": acc,
        "macro": {"precision": float(p_m), "recall": float(r_m), "f1": float(f_m)},
        "closed": {"precision": float(p_c[0]), "recall": float(r_c[0]),
                   "f1": float(f_c[0])},
        "confusion_matrix": cm,  # [[Closed->Closed, Closed->Open],[Open->Closed, Open->Open]]
    }


def main():
    print(f"CONFIG: {TAG}")
    print("manifest:", MANIFEST)
    for s in ("train", "val", "test"):
        print(f"  {s}:", class_counts(MANIFEST, s))

    ch = channels(GRAYSCALE)
    train_ds, n_tr = make_dataset(MANIFEST, "train", INPUT_SIZE, GRAYSCALE,
                                  DO_SHARPEN, BATCH_SIZE, seed=SEED, mrl_root=MRL_ROOT)
    val_ds, _ = make_dataset(MANIFEST, "val", INPUT_SIZE, GRAYSCALE,
                             DO_SHARPEN, BATCH_SIZE, mrl_root=MRL_ROOT)
    test_ds, _ = make_dataset(MANIFEST, "test", INPUT_SIZE, GRAYSCALE,
                              DO_SHARPEN, BATCH_SIZE, mrl_root=MRL_ROOT)

    model = build_eye_model(INPUT_SIZE, ch)
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy",
                                           save_best_only=True, mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                         mode="max", restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # 최종 평가는 best 모델로
    best = tf.keras.models.load_model(MODEL_PATH)
    metrics = evaluate(best, test_ds)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("\n저장:", MODEL_PATH)
    print("저장:", METRICS_PATH)


if __name__ == "__main__":
    main()
