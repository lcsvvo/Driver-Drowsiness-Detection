"""
yawn_dataset.py — DMD 하품 manifest CSV -> tf.data.Dataset.

mrl_dataset.py 와 같은 구조다. 다른 점만 적는다.

- 전처리는 eye_preprocess.preprocess_eye() 를 그대로 쓴다. 이름은 eye 지만 하는 일은
  "resize -> (선택)sharpen -> 채널 선택" 이라 뷰에 무관하다. 눈과 하품이 같은 함수를
  통과해야 나중에 두 모델을 한 추론 루프에 넣을 때 규격이 어긋나지 않는다.
- 라벨은 3-class(class_idx) 가 아니라 이진이다. **yawn=0, no_yawn=1** 로 둔다.
  01_TRAIN 과 02_INFER 가 `prediction[0]` 을 하품 확률로 읽고 있어서, 순서를 뒤집으면
  기존 추론 코드가 조용히 반대로 동작한다. 이 컬럼(`yawn_class`)은 train_yawn.py 가
  부분집합 CSV 를 쓸 때 만들어 넣는다.
- manifest 의 `path` 는 Dataset 폴더 기준 상대경로다. dataset_root 를 붙여야 열린다.

주의: 원본 crop 은 리사이즈되지 않은 정사각형(한 변 184~346px)이다. 리사이즈는 여기서
     한 번만 일어나야 하며, 학습과 추론이 같은 size/grayscale 값을 써야 한다.
"""

from __future__ import annotations
import csv
import os

import numpy as np
import tensorflow as tf

from eye_preprocess import preprocess_eye, channels

#: 이진 라벨 이름. 인덱스가 곧 softmax 출력 위치다.
CLASS_NAMES = ("yawn", "no_yawn")


def _read_manifest(csv_path: str, split: str, dataset_root: str = "",
                   label_col: str = "yawn_class"):
    """manifest 의 상대경로를 dataset_root 와 합쳐 실제 경로로 만든다.

    CSV 행 순서를 그대로 유지한다. 뒤에서 예측값과 manifest 행을 '위치'로 맞추기
    때문에 순서가 바뀌면 라벨이 어긋난다.
    """
    paths, labels = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if label_col not in (reader.fieldnames or []):
            raise ValueError(
                f"'{label_col}' 컬럼이 없습니다: {csv_path}\n"
                f"  있는 컬럼: {reader.fieldnames}")
        for row in reader:
            if row["split"] != split:
                continue
            p = row["path"]
            paths.append(os.path.join(dataset_root, p) if dataset_root else p)
            labels.append(int(row[label_col]))
    return paths, labels


def make_dataset(
    csv_path: str,
    split: str,
    size: int = 128,
    grayscale: bool = True,
    do_sharpen: bool = False,
    batch_size: int = 64,
    shuffle: bool = None,
    seed: int = 42,
    dataset_root: str = "",
    label_col: str = "yawn_class",
):
    """manifest 의 특정 split 을 tf.data 로. shuffle 기본값은 train 일 때만 True."""
    if shuffle is None:
        shuffle = (split == "train")

    paths, labels = _read_manifest(csv_path, split, dataset_root, label_col)
    if not paths:
        raise ValueError(f"'{split}' split 에 해당하는 행이 없습니다: {csv_path}")
    c = channels(grayscale)

    def _load(path, label):
        def _py(p):
            p = p.numpy().decode("utf-8")
            import cv2
            img = cv2.imread(p, cv2.IMREAD_COLOR)      # JPEG 컬러 -> BGR 3채널
            if img is None:
                raise FileNotFoundError(f"이미지를 열 수 없습니다: {p}")
            arr = preprocess_eye(img, size=size, grayscale=grayscale,
                                 do_sharpen=do_sharpen, input_is_bgr=True)
            return arr.astype(np.float32)

        img = tf.py_function(_py, [path], tf.float32)
        img.set_shape((size, size, c))
        label = tf.one_hot(label, 2)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 4096), seed=seed,
                        reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(paths)


# =====================================================================
# Distillation 용 — teacher 소프트 라벨을 이미지·정답 라벨과 같은 튜플로 싣는다
# =====================================================================
def _read_manifest_extra(csv_path: str, split: str, dataset_root: str,
                         label_col: str, extra_col: str):
    """_read_manifest() 와 같은 CSV 읽기 로직에 float 컬럼(extra_col) 하나를 더 얹는다.

    shuffle 은 tf.data.Dataset.from_tensor_slices 의 튜플 전체에 같이 적용되므로,
    여기서 세 리스트를 같은 인덱스로 만들어 두면 이후 shuffle 을 걸어도 image/label/
    teacher_prob 정렬이 깨지지 않는다.
    """
    paths, labels, extras = [], [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for col in (label_col, extra_col):
            if col not in fields:
                raise ValueError(
                    f"'{col}' 컬럼이 없습니다: {csv_path}\n  있는 컬럼: {fields}")
        for row in reader:
            if row["split"] != split:
                continue
            p = row["path"]
            paths.append(os.path.join(dataset_root, p) if dataset_root else p)
            labels.append(int(row[label_col]))
            extras.append(float(row[extra_col]))
    return paths, labels, extras


def make_distill_dataset(
    csv_path: str,
    split: str,
    size: int = 128,
    grayscale: bool = True,
    do_sharpen: bool = False,
    batch_size: int = 64,
    shuffle: bool = None,
    seed: int = 42,
    dataset_root: str = "",
    label_col: str = "yawn_class",
    teacher_col: str = "teacher_p_yawn",
):
    """teacher 소프트 확률(teacher_col, class0=yawn 확률)이 이미 채워진 CSV 를 받아
    (image, y_onehot, teacher_probs_2class) 3-튜플 tf.data.Dataset 을 만든다.

    csv_path 는 distill_yawn.cache_teacher_probs() 가 만든, teacher_col 이 추가된
    subset CSV 여야 한다. make_dataset() 과 전처리 경로는 동일하고(preprocess_eye),
    teacher 확률만 [p_yawn, 1-p_yawn] 형태로 같이 실어 나른다.
    """
    if shuffle is None:
        shuffle = (split == "train")

    paths, labels, tprobs = _read_manifest_extra(csv_path, split, dataset_root,
                                                  label_col, teacher_col)
    if not paths:
        raise ValueError(f"'{split}' split 에 해당하는 행이 없습니다: {csv_path}")
    c = channels(grayscale)

    def _load(path, label, teacher_p):
        def _py(p):
            p = p.numpy().decode("utf-8")
            import cv2
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"이미지를 열 수 없습니다: {p}")
            arr = preprocess_eye(img, size=size, grayscale=grayscale,
                                 do_sharpen=do_sharpen, input_is_bgr=True)
            return arr.astype(np.float32)

        img = tf.py_function(_py, [path], tf.float32)
        img.set_shape((size, size, c))
        y = tf.one_hot(label, 2)
        t = tf.stack([teacher_p, 1.0 - teacher_p])   # class0=yawn, 학생과 같은 순서
        return img, y, t

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, tprobs))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 4096), seed=seed,
                        reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(paths)


# =====================================================================
# 추론용 crop — 학습 데이터가 잘린 방식과 같게 자른다
# =====================================================================
#: Dataset/scripts/build_dmd_yawn_dataset.py 의 MARGIN 과 같은 값이어야 한다.
#: 아래쪽을 크게 잡는 이유: 하품하면 턱이 내려가고 입을 가리는 손은 얼굴 아래에서
#: 올라오는데, 얼굴 검출 박스는 손을 포함하지 않는다.
YAWN_MARGIN = {"x": 0.18, "top": 0.15, "bottom": 0.32}


def yawn_crop_box(box, img_w: int, img_h: int, margin: dict | None = None):
    """YuNet 얼굴박스 하나 -> 여백 포함 정사각형 crop 박스 (x1, y1, x2, y2).

    학습 데이터를 만든 build_dmd_yawn_dataset.square_box() 를 표본 1개짜리로 줄인
    것과 같은 계산이다. 그쪽은 UNION_PCT 분위수를 쓰지만 박스가 하나면 분위수가
    그 값 자신이 되므로 여백·정사각형화·클램프만 남는다.

    **학습 때와 완전히 같지는 않다.** 데이터셋은 세션 전체의 검출 박스를 합쳐 만든
    '고정 박스' 하나를 그 세션 내내 썼고(프레임마다 박스가 흔들리면 CNN 이 입 모양이
    아니라 crop 위치 변화를 학습하므로), 여기서는 프레임마다 새로 계산한다.
    그래서 추론 crop 이 학습 crop 보다 조금 더 타이트하고 프레임마다 흔들린다.
    호출부에서 박스를 시간축으로 눌러 주는 편이 좋다.

    box : (x, y, w, h)
    """
    margin = margin or YAWN_MARGIN
    x, y, w, h = (float(v) for v in box)
    x1, y1, x2, y2 = x, y, x + w, y + h

    x1 -= w * margin["x"]
    x2 += w * margin["x"]
    y1 -= h * margin["top"]
    y2 += h * margin["bottom"]

    # 정사각형화 - 짧은 축을 중심 기준으로 늘린다.
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(x2 - x1, y2 - y1), img_w, img_h)
    x1, y1 = cx - side / 2, cy - side / 2

    # 화면 밖으로 나가면 크기를 줄이지 않고 안쪽으로 민다(스케일 일관성 유지).
    x1 = min(max(x1, 0), img_w - side)
    y1 = min(max(y1, 0), img_h - side)
    return int(round(x1)), int(round(y1)), int(round(x1 + side)), int(round(y1 + side))


#: 추론용 crop 기하. train/val 12개 세션에서 '세션 고정 박스 vs 그 세션의 YuNet 얼굴박스
#: 중앙값'을 재서 얻은 값이다(test 세션은 보지 않았다).
#:
#:   R  = 고정박스 한 변 / 얼굴 높이            중앙값 1.82  (세션별 1.66 ~ 2.10)
#:   dx, dy = (고정박스 중심 - 얼굴 중심) / 얼굴 높이
#:
#: 위의 yawn_crop_box(여백식)를 프레임 하나에 그대로 쓰면 R 이 1.47 밖에 안 된다.
#: 학습 crop 은 세션 전체 검출의 '합집합' 이라 머리가 움직인 범위만큼 넓은데, 프레임
#: 하나로는 그 여유가 안 생기기 때문이다. 그 상태로 추론하면 얼굴이 학습 때보다 19%
#: 크게 찍혀서 모델이 하품 쪽으로 쏠린다(실측: 정확도 0.630 -> 0.485).
INFER_CROP = {"R": 1.82, "dx": 0.01, "dy": 0.09}


def yawn_infer_box(box, img_w: int, img_h: int, geom: dict | None = None):
    """YuNet 얼굴박스 -> **학습 crop 과 같은 스케일**의 정사각 crop 박스.

    yawn_crop_box() 는 데이터셋 생성 스크립트를 그대로 옮긴 것이라 '세션 고정 박스'
    를 만들 때만 학습과 일치한다. 실시간 추론에는 프레임이 하나뿐이므로 이 함수를 쓴다.

    box : (x, y, w, h)
    """
    geom = geom or INFER_CROP
    x, y, w, h = (float(v) for v in box)

    fcx, fcy = x + w / 2, y + h / 2
    side = min(h * geom["R"], img_w, img_h)

    x1 = fcx + geom["dx"] * h - side / 2
    y1 = fcy + geom["dy"] * h - side / 2

    # 화면 밖으로 나가면 크기를 줄이지 않고 안쪽으로 민다(스케일 일관성 유지).
    x1 = min(max(x1, 0), img_w - side)
    y1 = min(max(y1, 0), img_h - side)
    return int(round(x1)), int(round(y1)), int(round(x1 + side)), int(round(y1 + side))
