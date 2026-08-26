"""
train_yawn_zoo.py — 모델 zoo(yawn_models.ZOO) 아키텍처를 하품 데이터셋으로 학습·평가한다.

train_yawn.py 의 데이터/평가 하네스(resolve_dataset, load_manifest, subset, dataset_of,
_evaluate)를 그대로 재사용하고, 모델 생성부만 zoo 에서 가져온다. 아티팩트 저장 경로·이름
규칙(model/artifacts/yawn_{tag}.keras)도 train_yawn.run_experiment() 와 같다 —
compare_yawn_zoo.py 가 한 곳에서 전부 읽을 수 있도록.

_evaluate() 는 train_yawn.py 안에서 앞에 밑줄이 붙어 있지만("A/B/C 실행 중 학습을 뺀
평가만 담당") 모듈 프라이빗은 아니다 — 이 파일처럼 같은 저장소 안에서 같은 평가 로직을
재사용하려는 의도로 분리해 둔 함수다.

사용 (저장소 루트에서)
    python src/train_yawn_zoo.py --arch cnn_base \\
        --manifest ../yawn_mouthopen/yawn_mouthopen/metadata/manifest.csv
    python src/train_yawn_zoo.py --arch mbnetv2_100 --manifest ... --limit 200   # 스모크
    python src/train_yawn_zoo.py --arch cnn_tiny --manifest ...                  # student 베이스라인
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
import train_yawn as ty
from yawn_models import ZOO

SEED = ty.SEED


def run(arch: str, eval_view: str = "face", epochs: int = 20, batch: int = 64,
        lr: float | None = None, target_recall: float = 0.90,
        limit: int | None = None, root=None, manifest=None, verbose: int = 1) -> dict:
    """zoo 아키텍처 하나를 학습하고 지표 dict 를 반환한다. train_yawn.run_experiment() 와
    같은 흐름이고, 모델 생성만 다르다.
    """
    if arch not in ZOO:
        raise SystemExit(f"알 수 없는 arch: {arch!r}. 후보: {sorted(ZOO)}")
    spec = ZOO[arch]
    train = ["face"]   # yawn_mouthopen 등 신규 데이터셋은 face 뷰만 있다

    root, mp = ty.resolve_dataset(manifest, root)
    dataset_pkg = mp.parent.parent.name
    tag_prefix = f"{dataset_pkg}__" if dataset_pkg != "processed" else ""
    tag = (f"{tag_prefix}zoo-{arch}__eval-{eval_view}__"
           f"{'gray' if spec.grayscale else 'rgb'}{spec.size}"
           f"{'__smoke' if limit else ''}")
    ty.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ty.ARTIFACT_DIR / f"yawn_{tag}.keras"
    metrics_path = ty.ARTIFACT_DIR / f"yawn_{tag}_metrics.json"

    df = ty.load_manifest(mp)
    tr_csv, tr_sub = ty.subset(df, train, "train", limit, tag_prefix)
    va_csv, va_sub = ty.subset(df, train, "val", limit, tag_prefix)
    te_csv, te_sub = ty.subset(df, [eval_view], "test", limit, tag_prefix)

    train_ds = ty.dataset_of(tr_csv, "train", spec.size, spec.grayscale, False,
                             batch, True, root)[0]
    val_ds = ty.dataset_of(va_csv, "val", spec.size, spec.grayscale, False,
                           batch, False, root)[0]
    test_ds = ty.dataset_of(te_csv, "test", spec.size, spec.grayscale, False,
                            batch, False, root)[0]

    print(f"TAG  {tag}  (zoo: {arch}, {spec.kind})")
    print(f"데이터셋 {root}")
    print(f"train {len(tr_sub):,} (Yawn {(tr_sub.is_yawn == 1).mean()*100:.1f}%) | "
          f"val {len(va_sub):,} | test {len(te_sub):,} "
          f"(Yawn {(te_sub.is_yawn == 1).mean()*100:.1f}%)")

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = spec.build()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr) if lr else "adam",
                  loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=verbose,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy",
                                                     save_best_only=True, mode="max"),
                  tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                                   mode="max", restore_best_weights=True),
              ])

    best = tf.keras.models.load_model(model_path)
    out = ty._evaluate(best, df, tag, train, eval_view, spec.size, spec.grayscale, False,
                       batch, target_recall, limit, va_sub, val_ds, te_sub, test_ds,
                       n_train=len(tr_sub), model_path=model_path, root=root,
                       tag_prefix=tag_prefix, dataset_pkg=dataset_pkg)
    out["arch"] = arch
    out["zoo_kind"] = spec.kind
    out["training"] = "scratch"
    out["params"] = int(best.count_params())
    out["zoo_note"] = spec.note
    metrics_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n저장:", config._rel(model_path))
    print("저장:", config._rel(metrics_path))
    print(f"파라미터 수: {out['params']:,}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ZOO))
    ap.add_argument("--eval", dest="eval_view", default="face", choices=list(ty.VIEWS))
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--manifest", default=None,
                    help="manifest.csv 직접 지정. 예: yawn_mouthopen/metadata/manifest.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None,
                    help="스모크 테스트: split 당 이 개수로 줄여 파이프라인만 점검")
    a = ap.parse_args()

    run(a.arch, a.eval_view, a.epochs, a.batch, a.lr, a.target_recall,
        limit=a.limit, root=a.dataset_root, manifest=a.manifest)


if __name__ == "__main__":
    main()
