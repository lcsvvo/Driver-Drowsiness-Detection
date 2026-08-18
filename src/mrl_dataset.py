"""
mrl_dataset.py — manifest CSV -> tf.data.Dataset.

핵심: 이미지 로드 후 eye_preprocess.preprocess_eye() 를 tf.py_function 으로 적용한다.
     학습과 추론이 '완전히 같은' numpy 전처리를 통과하게 만들어 parity 를 보장한다.
     (tf 내장 resize 를 따로 쓰면 추론(cv2)과 미묘하게 달라질 수 있어 일부러 cv2 경유)

라벨: class_idx (Closed=0, Open=1) -> one-hot 2. 모델 Dense(2, softmax) 의 class0 이
      Closed 확률이 되어 기존 추론/EMA/PERCLOS 와 그대로 연결된다.
"""

from __future__ import annotations
import csv
import os
import numpy as np
import tensorflow as tf

from eye_preprocess import preprocess_eye, channels


def _read_manifest(csv_path: str, split: str, mrl_root: str = ""):
    """manifest 의 상대경로를 mrl_root 와 합쳐 실제 경로로 만든다."""
    paths, labels = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            p = row["path"]
            paths.append(os.path.join(mrl_root, p) if mrl_root else p)
            labels.append(int(row["class_idx"]))
    return paths, labels


def make_dataset(
    csv_path: str,
    split: str,
    size: int = 128,
    grayscale: bool = True,
    do_sharpen: bool = False,
    batch_size: int = 32,
    shuffle: bool = None,
    seed: int = 42,
    mrl_root: str = "",
):
    """manifest 의 특정 split 을 tf.data 로. shuffle 기본값은 train 일 때만 True.

    mrl_root : manifest 의 상대경로를 붙일 MRL 루트(.../MRL Eye/data).
    """
    if shuffle is None:
        shuffle = (split == "train")

    paths, labels = _read_manifest(csv_path, split, mrl_root)
    if not paths:
        raise ValueError(f"'{split}' split 에 해당하는 행이 없습니다: {csv_path}")
    c = channels(grayscale)

    def _load(path, label):
        def _py(p):
            p = p.numpy().decode("utf-8")
            # 디스크의 MRL 은 그레이스케일. cv2 로 읽어 preprocess 공유.
            import cv2
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
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


def class_counts(csv_path: str, split: str) -> dict:
    _, labels = _read_manifest(csv_path, split)
    return {"Closed": labels.count(0), "Open": labels.count(1), "n": len(labels)}


def resolve_mrl_root(repo_root) -> str:
    """관례 경로 'data/MRL Eye/data' 를 repo_root 기준으로 반환."""
    return os.path.join(str(repo_root), "data", "MRL Eye", "data")
