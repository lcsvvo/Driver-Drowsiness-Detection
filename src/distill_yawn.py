"""
distill_yawn.py — teacher(들)의 소프트 라벨로 student CNN 을 distillation 학습한다.

핵심 설계
- teacher/student 모두 표준 softmax 2-class 출력(class0=yawn)을 그대로 쓴다. 온도(T)
  소프트닝은 별도 logits 헤드 없이 수학적으로 정확한 항등식

      softmax(logits / T) == softmax(log(probs) / T)      (probs = softmax(logits))

  (같은 샘플 안에서 logsumexp 상수가 재정규화 시 소거되므로 항상 성립)을 이용한다.
  그래서 teacher/student 가 저장한 .keras(softmax 출력)를 그대로 로드해 쓸 수 있고,
  zoo 의 어떤 아키텍처든 손대지 않고 teacher/student 로 쓸 수 있다.
- teacher 소프트 확률은 **학습 전에 한 번만 계산해 CSV 에 캐시**한다. teacher 는
  training=False 로 호출돼 내부 RandomFlip/RandomRotation(train_yawn.build_yawn_model
  계열의 증강)이 꺼지므로 결정적이라 매 스텝 다시 돌릴 필요가 없다. 2-teacher 앙상블은
  각 teacher 확률의 평균을 쓴다.
- teacher 소프트 라벨을 이미지·정답 라벨과 같은 tf.data 튜플로 묶는다
  (yawn_dataset.make_distill_dataset) — shuffle 이 튜플 전체에 적용되므로 정렬이
  깨지지 않는다.
- student 학습은 tf.keras.Model 서브클래싱(train_step 오버라이드, keras.io Knowledge
  Distillation 튜토리얼과 동일한 패턴)으로 구현한다.

teacher/student 는 model/artifacts 의 **태그**로 지정한다(train_yawn.py, train_yawn_zoo.py
가 저장한 것과 같은 tag, 즉 yawn_<TAG>.keras 의 <TAG> 부분). "student-scratch" 베이스라인은
이 스크립트가 아니라 train_yawn_zoo.py 를 KD 없이 그대로 돌리면 된다(같은 arch, teacher 없이) —
같은 tag 규칙을 쓰므로 compare_yawn_zoo.py 가 한 표에서 바로 비교한다.

사용 (저장소 루트에서)
    # 단일 teacher
    python src/distill_yawn.py --student cnn_tiny \\
        --teacher yawn_mouthopen__zoo-cnn_large__eval-face__gray128 \\
        --manifest ../yawn_mouthopen/yawn_mouthopen/metadata/manifest.csv

    # 2-teacher 앙상블 (커스텀 CNN + 사전학습 backbone)
    python src/distill_yawn.py --student cnn_small \\
        --teacher yawn_mouthopen__zoo-cnn_large__eval-face__gray128 \\
                  yawn_mouthopen__zoo-mbnetv2_100__eval-face__rgb128 \\
        --manifest ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
import train_yawn as ty
from yawn_dataset import make_distill_dataset
from yawn_models import ZOO

SEED = ty.SEED
EPS = 1e-8


# =====================================================================
# 1. teacher 소프트 라벨 캐시
# =====================================================================
def _teacher_meta(tag: str) -> dict:
    p = ty.ARTIFACT_DIR / f"yawn_{tag}_metrics.json"
    if not p.exists():
        raise SystemExit(f"teacher metrics 가 없습니다: {config._rel(p)} (먼저 학습하세요)")
    return json.loads(p.read_text(encoding="utf-8"))


def _teacher_label(tags: list[str]) -> str:
    """metrics json 의 arch 필드(zoo 로 학습했으면 있음)로 짧은 이름을 만든다. 없으면
    태그 자체를 파일명-안전 문자열로 줄인다(train_yawn.py 로 학습한 teacher 대응).

    같은 arch라도 scratch 로 학습했는지 distillation 으로 학습했는지에 따라 다른
    teacher 다(가중치가 다르다). 여기서 구분해 주지 않으면 --student cnn_tiny
    --teacher <scratch cnn_large> 와 --student cnn_tiny --teacher <distilled cnn_large>
    가 똑같은 태그("...from-cnn_large...")로 떨어져 서로 덮어써 버린다(실제로 한 번
    발생했다). training=="distilled" 인 teacher 는 "-kd" 를 붙여 구분한다.
    """
    parts = []
    for t in tags:
        meta = _teacher_meta(t)
        name = meta.get("arch") or t.replace("/", "-").replace("__", "-")
        if meta.get("training") == "distilled":
            name += "-kd"
        parts.append(name)
    return "+".join(parts)


def cache_teacher_probs(tags: list[str], tr_csv: Path, root: Path, batch: int,
                        out_csv: Path) -> Path:
    """train subset CSV(tr_csv, 행 순서 고정)에 대해 teacher(들)의 yawn 확률을 재고
    평균해 새 컬럼(teacher_p_yawn)으로 저장한 CSV 를 만든다.

    teacher 마다 입력 규격(size/grayscale)이 다를 수 있어(커스텀 CNN=gray128,
    backbone=rgb128) 각 teacher 의 metrics json 에 저장된 규격으로 따로 데이터셋을
    만든다 — 이미지 파일과 행 순서는 tr_csv 로 전부 같다.
    """
    sub = pd.read_csv(tr_csv)
    probs = []
    for t in tags:
        meta = _teacher_meta(t)
        model = tf.keras.models.load_model(ty.ARTIFACT_DIR / f"yawn_{t}.keras")
        ds = ty.dataset_of(tr_csv, "train", meta["size"], meta["gray"],
                           meta.get("sharpen", False), batch, False, root)[0]
        p = ty.yawn_prob(model, ds)   # class0=yawn, 행 순서 = tr_csv 순서(shuffle=False)
        if len(p) != len(sub):
            raise RuntimeError(
                f"teacher({t}) 확률 개수({len(p)})가 CSV 행수({len(sub)})와 다릅니다.")
        probs.append(p)
        print(f"  teacher {t}: mean p_yawn={p.mean():.4f}")

    sub["teacher_p_yawn"] = np.mean(probs, axis=0)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    return out_csv


# =====================================================================
# 2. Distiller — KD 손실로 student 를 학습하는 tf.keras.Model
# =====================================================================
class Distiller(tf.keras.Model):
    """train_step 만 오버라이드한다. 이 래퍼 자체는 저장하지 않는다 — 학습이 끝나면
    self.student(표준 Sequential/Functional 모델)를 직접 .save() 한다.
    EarlyStopping(restore_best_weights=True) 가 최적 가중치를 이미 student 에
    복원해 두므로 별도 체크포인트 파일 왕복이 필요 없다.
    """

    def __init__(self, student: tf.keras.Model, alpha: float = 0.3,
                temperature: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.alpha = alpha
        self.T = temperature
        self.hard_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.kd_loss_fn = tf.keras.losses.KLDivergence()
        self.acc_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")

    def call(self, x, training=False):
        return self.student(x, training=training)

    @property
    def metrics(self):
        return [self.acc_metric]

    def _soft(self, probs: tf.Tensor) -> tf.Tensor:
        # softmax(logits/T) == softmax(log(probs)/T) — 모듈 docstring 참고.
        return tf.nn.softmax(tf.math.log(probs + EPS) / self.T)

    def train_step(self, data):
        x, y, teacher_probs = data
        with tf.GradientTape() as tape:
            student_probs = self.student(x, training=True)
            hard = self.hard_loss_fn(y, student_probs)
            kd = self.kd_loss_fn(self._soft(teacher_probs), self._soft(student_probs))
            loss = self.alpha * hard + (1.0 - self.alpha) * (self.T ** 2) * kd
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.acc_metric.update_state(y, student_probs)
        return {"loss": loss, "hard_loss": hard, "kd_loss": kd,
                "accuracy": self.acc_metric.result()}

    def test_step(self, data):
        x, y = data   # val_ds 는 표준 make_dataset() 산출물이라 (x, y) 2-튜플이다
        student_probs = self.student(x, training=False)
        hard = self.hard_loss_fn(y, student_probs)
        self.acc_metric.update_state(y, student_probs)
        return {"loss": hard, "accuracy": self.acc_metric.result()}


# =====================================================================
# 3. 실험 1회
# =====================================================================
def run(student_arch: str, teacher_tags: list[str], eval_view: str = "face",
        alpha: float = 0.3, temperature: float = 4.0, epochs: int = 30, batch: int = 64,
        lr: float | None = None, target_recall: float = 0.90, limit: int | None = None,
        root=None, manifest=None, verbose: int = 1) -> dict:
    """teacher(들) -> student(zoo 아키텍처) distillation 1회 실행."""
    if student_arch not in ZOO:
        raise SystemExit(f"알 수 없는 student arch: {student_arch!r}. 후보: {sorted(ZOO)}")
    if not (1 <= len(teacher_tags) <= 2):
        raise SystemExit("teacher 는 1개 또는 2개(앙상블)만 지원한다.")
    spec = ZOO[student_arch]
    train = ["face"]

    root, mp = ty.resolve_dataset(manifest, root)
    dataset_pkg = mp.parent.parent.name
    tag_prefix = f"{dataset_pkg}__" if dataset_pkg != "processed" else ""

    teacher_label = _teacher_label(teacher_tags)
    tag = (f"{tag_prefix}distill-{student_arch}__from-{teacher_label}"
           f"__T{temperature:g}-a{alpha:g}__eval-{eval_view}__"
           f"{'gray' if spec.grayscale else 'rgb'}{spec.size}"
           f"{'__smoke' if limit else ''}")
    ty.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ty.ARTIFACT_DIR / f"yawn_{tag}.keras"
    metrics_path = ty.ARTIFACT_DIR / f"yawn_{tag}_metrics.json"

    df = ty.load_manifest(mp)
    tr_csv, tr_sub = ty.subset(df, train, "train", limit, tag_prefix)
    va_csv, va_sub = ty.subset(df, train, "val", limit, tag_prefix)
    te_csv, te_sub = ty.subset(df, [eval_view], "test", limit, tag_prefix)

    print(f"TAG  {tag}")
    print(f"teacher(s): {teacher_tags}  ->  label={teacher_label}")
    print("teacher 소프트 라벨 캐시 중...")
    distill_csv = ty.SUBSET_DIR / f"{tag_prefix}distill-train__from-{teacher_label}.csv"
    cache_teacher_probs(teacher_tags, tr_csv, root, batch, distill_csv)

    train_ds = make_distill_dataset(str(distill_csv), "train", size=spec.size,
                                    grayscale=spec.grayscale, batch_size=batch,
                                    shuffle=True, seed=SEED, dataset_root=str(root))[0]
    val_ds = ty.dataset_of(va_csv, "val", spec.size, spec.grayscale, False,
                           batch, False, root)[0]
    test_ds = ty.dataset_of(te_csv, "test", spec.size, spec.grayscale, False,
                            batch, False, root)[0]

    print(f"train {len(tr_sub):,} (Yawn {(tr_sub.is_yawn == 1).mean()*100:.1f}%) | "
          f"val {len(va_sub):,} | test {len(te_sub):,}  "
          f"alpha={alpha} T={temperature}")

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    student = spec.build()
    distiller = Distiller(student, alpha=alpha, temperature=temperature)
    distiller.compile(optimizer=tf.keras.optimizers.Adam(lr) if lr
                      else tf.keras.optimizers.Adam())

    distiller.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=verbose,
                 callbacks=[
                     tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                                      mode="max", restore_best_weights=True),
                 ])
    student.save(model_path)

    out = ty._evaluate(student, df, tag, train, eval_view, spec.size, spec.grayscale,
                       False, batch, target_recall, limit, va_sub, val_ds, te_sub, test_ds,
                       n_train=len(tr_sub), model_path=model_path, root=root,
                       tag_prefix=tag_prefix, dataset_pkg=dataset_pkg)
    # zoo_kind 는 아키텍처 계열(cnn/backbone/hybrid) — train_yawn_zoo.py 와 같은 의미로
    # 맞춘다. distillation 여부는 별도 training 필드로 구분한다(compare_yawn_zoo.py 가
    # 이 두 필드로 "무엇을(arch) 어떻게 학습했나(training)"를 가른다).
    out.update(arch=student_arch, zoo_kind=spec.kind, training="distilled",
              params=int(student.count_params()), teacher_tags=teacher_tags,
              teacher_label=teacher_label, temperature=temperature, alpha=alpha)
    metrics_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n저장:", config._rel(model_path))
    print("저장:", config._rel(metrics_path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, choices=sorted(ZOO))
    ap.add_argument("--teacher", nargs="+", required=True,
                    help="teacher 태그 1~2개 (model/artifacts 의 yawn_<TAG>.keras 기준)")
    ap.add_argument("--eval", dest="eval_view", default="face", choices=list(ty.VIEWS))
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="hard-label CE 가중치 (1-alpha 가 KD 손실 가중치)")
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None,
                    help="스모크 테스트: split 당 이 개수로 줄여 파이프라인만 점검")
    a = ap.parse_args()

    run(a.student, a.teacher, a.eval_view, a.alpha, a.temperature, a.epochs, a.batch,
        a.lr, a.target_recall, limit=a.limit, root=a.dataset_root, manifest=a.manifest)


if __name__ == "__main__":
    main()
