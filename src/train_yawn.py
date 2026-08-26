"""Yawn CNN 학습·평가 — 뷰(정면/측면)만 바꿔 A/B/C 를 비교한다.

Dataset/processed 가 만든 `metadata/manifest.csv` 를 유일한 입력으로 쓴다.
split 은 manifest 에 이미 박혀 있고(피험자 단위), 눈 데이터셋의 DMD_SPLIT 과 같은
매핑이므로 이 스크립트는 split 을 만들지 않는다. 바꾸는 것은 `view` 필터뿐이다.

    A   face only        train=["face"]      정면 얼굴 카메라
    B   body only        train=["body"]      우측 측면 상반신 카메라
    C   face + body 합본  train=["face","body"]

평가 규칙
    - 배포 대상은 정면 웹캠이므로 주 지표는 **face test** 다. body test 는 대조군으로
      함께 보고한다(정면을 올리려다 측면을 깎았는지 확인).
    - 주 지표는 Yawn-Recall. 다만 하품은 눈 감김만큼 즉각적인 위험 신호가 아니라
      PERCLOS 를 보조하는 지표라, Precision 도 같은 비중으로 읽어야 한다.
    - 판정 임계값은 val 에서 고르고 test 에 그대로 적용한다. test 에서 고르면 낙관 편향.
    - **test 의 하품 비율(약 45%)은 실제 운전 중 비율(약 14%)보다 훨씬 높다.**
      샘플링 stride 가 하품 5 / 하품아님 30 이라서다(Dataset README §6). 그래서
      test Precision 을 그대로 믿으면 안 된다. 실제 비율로 환산한 Precision 을
      `precision_at_real_prior` 로 함께 보고한다.
    - 손으로 입을 가린 하품(`yawn_with_hand`)은 따로 Recall 을 낸다. 가장 어려운
      표본이고, 이 데이터셋이 굳이 클래스를 나눠 놓은 이유이기도 하다.

라벨
    이진 분류다. **class 0 = yawn, class 1 = no_yawn** 으로 둔다. 01_TRAIN 과
    02_INFER_YuNet 이 `prediction[0]` 을 하품 확률로 읽고 있어서, 순서를 뒤집으면
    기존 추론 코드가 에러 없이 반대로 동작한다.

입력 형식
    본 실험: size=128, gray=True  (눈 모델의 MAIN_SPEC 과 같은 규격)

사용 (저장소 루트에서)
    python src/train_yawn.py --train face
    python src/train_yawn.py --train body
    python src/train_yawn.py --train face body
    python src/train_yawn.py --train face --limit 400      # 스모크

    다른 데이터셋(예: yawn_mouthopen)을 쓰려면 --manifest 로 manifest.csv 를 직접 준다.
    dataset_root 는 <manifest>/../../.. 로 자동 추정되고, 아티팩트 태그에
    데이터셋 이름이 접두어로 붙어 DMD 가중치를 덮어쓰지 않는다.
    python src/train_yawn.py --train face \\
        --manifest ../yawn_mouthopen/yawn_mouthopen/metadata/manifest.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent            # src/
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
import pandas as pd
import tensorflow as tf
from eye_preprocess import channels
from yawn_dataset import make_dataset

VIEWS = ("face", "body")
SEED = 42

#: 하품 데이터셋 루트를 찾을 후보. Dataset 폴더는 이 저장소 밖에 있어서
#: (팀원마다 다른 위치에 clone 할 수 있다) 고정 경로 하나로 두지 않는다.
#: build_dmd_eye_dataset.py 의 data_subdir() 과 같은 방식이다.
DATASET_CANDIDATES = (
    config.PROJECT_ROOT / "data" / "Dataset",
    config.PROJECT_ROOT.parent / "Dataset",
    config.PROJECT_ROOT / "Dataset",
)

#: DMD(processed) 관례 위치. <dataset_root>/<pkg>/metadata/manifest.csv 규칙이고,
#: yawn_mouthopen 등 다른 데이터셋도 pkg 이름만 다를 뿐 같은 규칙을 따른다
#: (resolve_dataset() 의 --manifest 경로 추정이 이 규칙을 거꾸로 푼다).
MANIFEST_REL = Path("processed") / "metadata" / "manifest.csv"

#: 소스 필터를 적용한 부분집합 CSV 를 두는 곳. tempdir 대신 outputs 아래 고정 이름으로
#: 쓴다. 실행마다 이름이 바뀌지 않아 재현이 되고, 무엇이 학습에 들어갔는지 열어 볼 수 있다.
SUBSET_DIR = config.OUTPUTS_DIR / "yawn_dataset" / "subsets"

ARTIFACT_DIR = getattr(config, "ARTIFACT_DIR", config.PROJECT_ROOT / "artifacts")

#: 실제 운전 중 하품 프레임 비율. test 의 45% 는 샘플링 때문에 부풀려진 값이라
#: Precision 을 이 비율로 환산해 함께 보고한다. (Dataset README §6)
REAL_YAWN_PRIOR = 0.14


# =====================================================================
# 1. 데이터셋 위치
# =====================================================================
def resolve_dataset(manifest: str | None = None,
                    root: str | None = None) -> tuple[Path, Path]:
    """(dataset_root, manifest_path) 를 정한다.

    --manifest 를 직접 주면 그 경로를 쓰고, dataset_root 는 <root>/<pkg>/metadata/
    manifest.csv 규칙을 거꾸로 풀어 3단계 위로 추정한다(--dataset-root 를 같이 주면
    그 값이 우선한다). 둘 다 실측 규칙 — DMD manifest 는
    "<root>/processed/metadata/manifest.csv", yawn_mouthopen manifest 는
    "<root>/yawn_mouthopen/metadata/manifest.csv" — 를 그대로 따른다.

    --manifest 를 안 주면 기존 DATASET_CANDIDATES + MANIFEST_REL 관례(DMD/processed)
    로 찾는다.
    """
    if manifest:
        mp = Path(manifest).expanduser().resolve()
        if not mp.exists():
            raise SystemExit(f"manifest 가 없습니다: {mp}")
        r = Path(root).expanduser().resolve() if root else mp.parent.parent.parent
        return r, mp

    if root:
        p = Path(root).expanduser().resolve()
        mp = p / MANIFEST_REL
        if not mp.exists():
            raise SystemExit(f"manifest 가 없습니다: {mp}")
        return p, mp

    tried = []
    for c in DATASET_CANDIDATES:
        mp = c / MANIFEST_REL
        if mp.exists():
            return c.resolve(), mp
        tried.append(str(c))

    raise SystemExit(
        "하품 데이터셋을 찾지 못했습니다. 확인한 위치:\n  "
        + "\n  ".join(tried)
        + "\n--dataset-root 또는 --manifest 로 직접 지정하세요.")


# =====================================================================
# 2. 모델
# =====================================================================
def build_yawn_model(size: int, ch: int) -> tf.keras.Model:
    """train_eye.py 의 Conv 3개 + Dense 구조를 그대로 쓴다.

    같은 구조를 쓰는 이유는 성능이 최고라서가 아니라, 두 모델의 차이가 '데이터'
    에서만 오게 하려는 것이다. 구조까지 같이 바꾸면 무엇이 효과를 냈는지 알 수 없다.

    RandomRotation 은 0.12 -> 0.08 로 줄였다. crop 박스가 세션마다 고정이라
    (Dataset README §5) 실제로는 큰 회전이 일어나지 않는다.
    class0 = yawn 으로 두어 기존 추론 코드와 그대로 연결된다.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(size, size, ch)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08, fill_mode="reflect"),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])


# =====================================================================
# 3. manifest 부분집합
# =====================================================================
REQUIRED_COLS = {"path", "view", "session", "subject", "frame", "split",
                 "label", "class_idx", "is_yawn", "glasses"}


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f"manifest 컬럼 누락: {sorted(missing)}")

    # yawn=0, no_yawn=1. 여기 한 곳에서만 만든다.
    df["yawn_class"] = 1 - df["is_yawn"].astype(int)
    return df


def subset(df: pd.DataFrame, views, split: str,
           limit: int | None = None, tag_prefix: str = "") -> tuple[Path, pd.DataFrame]:
    """view·split 로 걸러 부분집합 CSV 를 쓰고 (경로, 데이터프레임) 반환.

    limit 을 주면 클래스 비율을 유지한 채 그만큼으로 줄인다. 스모크 테스트용이다.
    tag_prefix 는 데이터셋별 파일명 충돌을 막는다(예: "yawn_mouthopen__").
    """
    sub = df[(df.view.isin(views)) & (df.split == split)].reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"'{split}' 에 해당하는 행이 없습니다 (view={list(views)})")

    if limit and len(sub) > limit:
        frac = limit / len(sub)
        keep = []
        for _, g in sub.groupby("yawn_class"):
            keep.extend(g.sample(max(1, round(len(g) * frac)), random_state=SEED).index)
        sub = sub.loc[sorted(keep)].reset_index(drop=True)   # 원래 순서 복원

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBSET_DIR / f"{tag_prefix}{'+'.join(views)}__{split}{f'__n{limit}' if limit else ''}.csv"
    sub.to_csv(out, index=False)
    return out, sub


def dataset_of(csv_path: Path, split: str, size: int, gray: bool,
               sharpen: bool, batch: int, shuffle: bool, root: Path):
    return make_dataset(str(csv_path), split, size=size, grayscale=gray,
                        do_sharpen=sharpen, batch_size=batch, shuffle=shuffle,
                        seed=SEED, dataset_root=str(root))


# =====================================================================
# 4. 평가
# =====================================================================
def yawn_prob(model, ds) -> np.ndarray:
    """Yawn(class 0) 확률을 manifest 행 순서 그대로 반환."""
    out = []
    for xb, _ in ds:
        out.append(model(xb, training=False).numpy()[:, 0])
    return np.concatenate(out)


def precision_at_prior(recall: float, fpr: float, prior: float) -> float:
    """하품 비율이 prior 일 때의 Precision.

    test 는 하품 5프레임/하품아님 30프레임 stride 로 뽑혀 하품이 45% 나 된다.
    실제 운전에서는 14% 안팎이므로, 같은 모델이라도 실전 Precision 은 훨씬 낮다.
    Recall 과 FPR 은 클래스 비율에 영향받지 않으므로 이렇게 환산할 수 있다.
    """
    tp = recall * prior
    fp = fpr * (1.0 - prior)
    return float(tp / (tp + fp)) if tp + fp else 0.0


def metrics_at(y_yawn: np.ndarray, p_yawn: np.ndarray, thr: float) -> dict:
    pred = (p_yawn >= thr).astype(int)
    tp = int(((pred == 1) & (y_yawn == 1)).sum())
    fp = int(((pred == 1) & (y_yawn == 0)).sum())
    fn = int(((pred == 0) & (y_yawn == 1)).sum())
    tn = int(((pred == 0) & (y_yawn == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    pre = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return dict(threshold=round(float(thr), 4), n=len(y_yawn),
                accuracy=round((tp + tn) / len(y_yawn), 4),
                yawn_recall=round(rec, 4), yawn_precision=round(pre, 4),
                yawn_f1=round(f1, 4), false_positive_rate=round(fpr, 4),
                precision_at_real_prior=round(
                    precision_at_prior(rec, fpr, REAL_YAWN_PRIOR), 4),
                real_prior=REAL_YAWN_PRIOR,
                confusion=dict(yawn_as_yawn=tp, yawn_as_no=fn,
                               no_as_yawn=fp, no_as_no=tn))


#: 임계값 탐색 그리드. train_eye.py 와 같다. softmax 가 포화하면 최적 임계값이
#: 0.99 를 넘는 일이 흔해서 위쪽을 로그 간격으로 촘촘히 둔다.
THRESHOLD_GRID = np.unique(np.concatenate([
    np.arange(0.01, 0.99, 0.01),
    1.0 - np.logspace(-2, -5, 25),          # 0.99 ~ 0.99999
]))


def pick_threshold(y_yawn: np.ndarray, p_yawn: np.ndarray,
                   target_recall: float) -> tuple[float, bool, bool]:
    """val 에서 Yawn-Recall >= target 을 만족하는 가장 높은 임계값.

    임계값이 높아질수록 하품 판정이 줄어 Recall 이 내려간다. 제약을 만족하는 중
    가장 높은 값을 고르면 Recall 을 지키면서 Precision 이 최대가 된다.

    반환: (임계값, 목표달성여부, 그리드상한도달여부)
    """
    ok = [t for t in THRESHOLD_GRID
          if metrics_at(y_yawn, p_yawn, t)["yawn_recall"] >= target_recall]
    if not ok:
        return 0.5, False, False
    best = float(max(ok))
    return best, True, bool(best >= THRESHOLD_GRID[-1] - 1e-12)


def report(name: str, m: dict) -> None:
    c = m["confusion"]
    print(f"  [{name}] n={m['n']:,}  thr={m['threshold']:.2f}  "
          f"Yawn-Recall={m['yawn_recall']:.4f}  "
          f"Precision={m['yawn_precision']:.4f}  F1={m['yawn_f1']:.4f}  "
          f"Acc={m['accuracy']:.4f}")
    print(f"          실제Yawn -> Yawn {c['yawn_as_yawn']} / No {c['yawn_as_no']}"
          f"   |   실제No -> Yawn {c['no_as_yawn']} / No {c['no_as_no']}"
          f"   |   실전비율({m['real_prior']:.0%}) Precision "
          f"{m['precision_at_real_prior']:.4f}")


def recall_by_yawn_type(sub: pd.DataFrame, p_yawn: np.ndarray, thr: float) -> dict:
    """손 가림 여부별 Recall.

    이 부분집합에는 양성만 들어 있어 Precision 은 정의되지 않는다. Recall 만 낸다.
    """
    out = {}
    d = sub.assign(p_yawn=p_yawn)
    for lab in ("yawn_without_hand", "yawn_with_hand"):
        g = d[d.label == lab]
        if len(g) < 30:
            continue
        hit = int((g.p_yawn >= thr).sum())
        out[lab] = dict(n=len(g), recall=round(hit / len(g), 4),
                        missed=len(g) - hit)
    return out


# =====================================================================
# 5. 실험 1회
# =====================================================================
def run_experiment(train, eval_view="face", size=128, gray=True, sharpen=False,
                   epochs=20, batch=64, lr=None, target_recall=0.90,
                   verbose=1, limit=None, root=None, manifest=None) -> dict:
    """A/B/C 중 하나를 실행하고 지표 dict 를 반환한다."""
    train = list(train)
    bad = set(train) - set(VIEWS)
    if bad:
        raise SystemExit(f"알 수 없는 view: {sorted(bad)}")

    root, mp = resolve_dataset(manifest, root)
    #: manifest 가 들어 있는 패키지 폴더 이름. DMD 관례("processed")면 기존 파일명을
    #: 그대로 유지해 기존 아티팩트를 덮어쓰지 않고, 다른 데이터셋이면 태그에 접두어를
    #: 붙여 구분한다(예: "yawn_mouthopen__face__eval-face__gray128").
    dataset_pkg = mp.parent.parent.name
    tag_prefix = "" if dataset_pkg == "processed" else f"{dataset_pkg}__"

    tag = (f"{tag_prefix}{'+'.join(train)}__eval-{eval_view}__"
           f"{'gray' if gray else 'rgb'}{size}"
           f"{'__sharp' if sharpen else ''}{'__smoke' if limit else ''}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / f"yawn_{tag}.keras"
    metrics_path = ARTIFACT_DIR / f"yawn_{tag}_metrics.json"

    df = load_manifest(mp)
    tr_csv, tr_sub = subset(df, train, "train", limit, tag_prefix)
    va_csv, va_sub = subset(df, train, "val", limit, tag_prefix)   # 모델 선택은 학습과 같은 도메인에서
    te_csv, te_sub = subset(df, [eval_view], "test", limit, tag_prefix)

    train_ds = dataset_of(tr_csv, "train", size, gray, sharpen, batch, True, root)[0]
    val_ds = dataset_of(va_csv, "val", size, gray, sharpen, batch, False, root)[0]
    test_ds = dataset_of(te_csv, "test", size, gray, sharpen, batch, False, root)[0]

    print(f"TAG  {tag}")
    print(f"데이터셋 {root}")
    print(f"train {len(tr_sub):,} (Yawn {(tr_sub.is_yawn == 1).mean()*100:.1f}%) | "
          f"val {len(va_sub):,} | test {len(te_sub):,} "
          f"(Yawn {(te_sub.is_yawn == 1).mean()*100:.1f}%)")

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = build_yawn_model(size, channels(gray))
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

    out = _evaluate(best, df, tag, train, eval_view, size, gray, sharpen, batch,
                    target_recall, limit, va_sub, val_ds, te_sub, test_ds,
                    n_train=len(tr_sub), model_path=model_path, root=root,
                    tag_prefix=tag_prefix, dataset_pkg=dataset_pkg)
    metrics_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n저장:", config._rel(model_path))
    print("저장:", config._rel(metrics_path))
    return out


def _evaluate(model, df, tag, train, eval_view, size, gray, sharpen, batch,
              target_recall, limit, va_sub, val_ds, te_sub, test_ds,
              n_train, model_path, root, tag_prefix: str = "",
              dataset_pkg: str = "processed") -> dict:
    """임계값 선택 + test 평가 + 대조군 + 조건별 분해. 학습은 하지 않는다."""
    # --- 임계값은 val 에서 고른다 (test 에서 고르면 낙관 편향) ---
    va_p = yawn_prob(model, val_ds)
    thr, reached, saturated = pick_threshold(
        va_sub.is_yawn.values, va_p, target_recall)
    if not reached:
        print(f"[!] val 에서 Yawn-Recall {target_recall:.2f} 를 만족하는 임계값이 없어 "
              f"0.5 로 둡니다.")
    if saturated:
        print(f"[!] 선택된 임계값이 탐색 그리드 상한({THRESHOLD_GRID[-1]:.5f})에 걸렸습니다.")

    # --- 주 평가: eval_view 의 test ---
    te_p = yawn_prob(model, test_ds)
    y_te = te_sub.is_yawn.values

    print("\n===== TEST =====")
    m_main = metrics_at(y_te, te_p, thr)
    report(f"{eval_view} test @val-thr", m_main)
    m_half = metrics_at(y_te, te_p, 0.5)
    report(f"{eval_view} test @0.50", m_half)

    # --- 대조군: 반대편 뷰의 test ---
    other = next((v for v in VIEWS if v != eval_view), None)
    m_other = None
    if other and not df[(df.view == other) & (df.split == "test")].empty:
        ot_csv, ot_sub = subset(df, [other], "test", limit, tag_prefix)
        ot_ds = dataset_of(ot_csv, "test", size, gray, sharpen, batch, False, root)[0]
        m_other = metrics_at(ot_sub.is_yawn.values, yawn_prob(model, ot_ds), thr)
        report(f"{other} test @val-thr (대조군)", m_other)

    # --- 손 가림 여부별 Recall (가장 어려운 표본) ---
    by_type = recall_by_yawn_type(te_sub, te_p, thr)
    if by_type:
        print("\n  -- 하품 종류별 Recall --")
        for k, v in by_type.items():
            print(f"  [{k}] n={v['n']:,}  Recall={v['recall']:.4f}  "
                  f"(놓침 {v['missed']})")

    # --- 안경 조건별 (표본이 적어 참고용) ---
    by_glasses = {}
    d = te_sub.assign(p_yawn=te_p)
    for g in sorted(d.glasses.dropna().unique()):
        gm = d[d.glasses == g]
        if len(gm) >= 50:
            by_glasses[f"glasses={bool(g)}"] = metrics_at(
                gm.is_yawn.values, gm.p_yawn.values, thr)
    if by_glasses:
        print("\n  -- 안경 조건별 (표본 적음, 참고용) --")
        for k, v in by_glasses.items():
            report(k, v)

    # --- 피험자별 (@0.50) ---
    # test 피험자가 4명뿐이라 합계 지표는 '평균적으로 이만큼 한다'가 아니라
    # '이 4명에서 이랬다'는 뜻이다. 사람마다 편차가 크면 합계는 의미가 약하므로
    # 반드시 펼쳐서 함께 보고한다.
    by_subject = {}
    for s, g in d.groupby("subject"):
        by_subject[str(s)] = metrics_at(g.is_yawn.values, g.p_yawn.values, 0.5)
    print("\n  -- test 피험자별 (@0.50) --")
    accs = []
    for k, v in by_subject.items():
        accs.append(v["accuracy"])
        print(f"  [{k}] n={v['n']:,}  Recall={v['yawn_recall']:.4f}  "
              f"Precision={v['yawn_precision']:.4f}  Acc={v['accuracy']:.4f}")
    subject_spread = dict(
        n_subjects=len(accs),
        accuracy_min=round(float(np.min(accs)), 4),
        accuracy_max=round(float(np.max(accs)), 4),
        accuracy_std=round(float(np.std(accs)), 4),
    )
    print(f"  -> 피험자별 정확도 {subject_spread['accuracy_min']:.3f} ~ "
          f"{subject_spread['accuracy_max']:.3f} (std {subject_spread['accuracy_std']:.3f})")

    return dict(tag=tag, train=train, eval_view=eval_view, size=size,
                gray=gray, sharpen=sharpen, dataset=dataset_pkg,
                classes={"0": "yawn", "1": "no_yawn"},
                threshold=thr, threshold_reached_target=reached,
                threshold_grid_saturated=saturated, target_recall=target_recall,
                test_main=m_main, test_main_at_050=m_half, test_other=m_other,
                by_yawn_type=by_type, by_glasses=by_glasses,
                by_subject=by_subject, subject_spread=subject_spread,
                n_train=n_train, n_val=len(va_sub), n_test=len(te_sub),
                model=str(config._rel(model_path)))


def reevaluate(tag: str, target_recall: float | None = None, batch: int = 64,
               root=None, manifest=None) -> dict:
    """저장된 모델로 임계값 선택과 평가만 다시 한다. **재학습하지 않는다.**

    평가 항목을 추가한 뒤 학습을 다시 돌리지 않고 metrics JSON 을 갱신하기 위한
    함수다. 학습 조건(train/eval/size/gray)은 기존 JSON 에서 읽으므로 인자로 받지 않는다.
    데이터셋(dataset/manifest)도 기존 JSON 의 "dataset" 필드로 추정하되, 옛 JSON(필드
    없음)은 DMD("processed") 관례로 가정한다. 다른 데이터셋을 재평가하려면 --manifest
    로 명시한다.
    """
    model_path = ARTIFACT_DIR / f"yawn_{tag}.keras"
    metrics_path = ARTIFACT_DIR / f"yawn_{tag}_metrics.json"
    for p in (model_path, metrics_path):
        if not p.exists():
            raise SystemExit(f"{config._rel(p)} 가 없습니다. 먼저 학습하세요.")

    old = json.loads(metrics_path.read_text(encoding="utf-8"))
    train, eval_view = old["train"], old["eval_view"]
    size, gray, sharpen = old["size"], old["gray"], old["sharpen"]
    target = old["target_recall"] if target_recall is None else target_recall
    dataset_pkg = old.get("dataset", "processed")
    tag_prefix = "" if dataset_pkg == "processed" else f"{dataset_pkg}__"

    root, mp = resolve_dataset(manifest, root)
    df = load_manifest(mp)
    va_csv, va_sub = subset(df, train, "val", tag_prefix=tag_prefix)
    te_csv, te_sub = subset(df, [eval_view], "test", tag_prefix=tag_prefix)
    val_ds = dataset_of(va_csv, "val", size, gray, sharpen, batch, False, root)[0]
    test_ds = dataset_of(te_csv, "test", size, gray, sharpen, batch, False, root)[0]

    model = tf.keras.models.load_model(model_path)
    print(f"TAG  {tag}  (재평가 — 학습 없음)")

    out = _evaluate(model, df, tag, train, eval_view, size, gray, sharpen, batch,
                    target, None, va_sub, val_ds, te_sub, test_ds,
                    n_train=old["n_train"], model_path=model_path, root=root,
                    tag_prefix=tag_prefix, dataset_pkg=dataset_pkg)
    metrics_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n갱신:", config._rel(metrics_path))
    return out


# =====================================================================
# 6. 조건 비교
# =====================================================================
def cond_label(meta: dict) -> str:
    train = "+".join(meta.get("train", []))
    return {"face": "A face only", "body": "B body only",
            "face+body": "C face+body"}.get(train, train)


def load_all_metrics(include_smoke: bool = False) -> dict:
    """artifacts 의 yawn metrics JSON 을 전부 읽는다.

    메모리 변수가 아니라 디스크를 정본으로 둔다. 노트북 셀을 따로 돌렸거나 커널을
    재시작해도 결과를 모을 수 있다.
    """
    out = {}
    for p in sorted(ARTIFACT_DIR.glob("yawn_*_metrics.json")):
        tag = p.name[len("yawn_"):-len("_metrics.json")]
        if not include_smoke and tag.endswith("__smoke"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[건너뜀] {p.name}: 읽기 실패 ({e})")
            continue
        d.setdefault("tag", tag)
        out[tag] = d
    return out


def compare(results=None) -> pd.DataFrame:
    """조건 비교표. 인자를 주지 않으면 디스크의 metrics 를 전부 읽는다."""
    if results is None:
        results = load_all_metrics()
    items = list(results.values()) if isinstance(results, dict) else list(results)

    rows = []
    for r in items:
        m = (r or {}).get("test_main")
        if not isinstance(m, dict) or "yawn_recall" not in m:
            print(f"[건너뜀] {(r or {}).get('tag', '?')}: test_main 형식이 맞지 않습니다.")
            continue
        bt = r.get("by_yawn_type") or {}
        # @0.50 도 함께 낸다. metrics 의 threshold 는 val 에서 Recall 0.90 을 맞추도록
        # 고른 값인데 하품에서는 역효과여서 실제 추론은 0.50 을 쓴다
        # (docs/yawn_model.md §3). 두 숫자를 같이 보여야 문서와 어긋나지 않는다.
        h = r.get("test_main_at_050") or {}
        rows.append({
            "cond": cond_label(r), "tag": r["tag"], "thr": r.get("threshold"),
            "yawn_recall": m["yawn_recall"], "yawn_precision": m["yawn_precision"],
            "prec_real_prior": m.get("precision_at_real_prior"),
            "accuracy": m["accuracy"],
            "recall@0.50": h.get("yawn_recall"), "prec@0.50": h.get("yawn_precision"),
            "acc@0.50": h.get("accuracy"),
            "prec_real@0.50": h.get("precision_at_real_prior"),
            "recall_with_hand": (bt.get("yawn_with_hand") or {}).get("recall"),
            "recall_no_hand": (bt.get("yawn_without_hand") or {}).get("recall"),
            "other_view_recall": (r.get("test_other") or {}).get("yawn_recall"),
            "subj_acc_min": (r.get("subject_spread") or {}).get("accuracy_min"),
            "subj_acc_max": (r.get("subject_spread") or {}).get("accuracy_max"),
            "n_test": r.get("n_test"),
        })
    t = pd.DataFrame(rows)
    return t.sort_values("yawn_recall", ascending=False) if not t.empty else t


# =====================================================================
# 7. CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["face"], choices=list(VIEWS))
    ap.add_argument("--eval", dest="eval_view", default="face", choices=list(VIEWS))
    ap.add_argument("--dataset-root", default=None,
                    help="Dataset 폴더. 생략하면 관례 위치를 순서대로 찾는다")
    ap.add_argument("--manifest", default=None,
                    help="manifest.csv 직접 지정 (예: yawn_mouthopen/metadata/manifest.csv). "
                         "주면 --dataset-root 는 3단계 위로 자동 추정한다")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--rgb", action="store_true", help="컬러 입력 부록 실험용")
    ap.add_argument("--sharpen", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None,
                    help="스모크 테스트: split 당 이 개수로 줄여 파이프라인만 점검")
    a = ap.parse_args()

    run_experiment(a.train, a.eval_view, a.size, not a.rgb, a.sharpen,
                   a.epochs, a.batch, a.lr, a.target_recall,
                   limit=a.limit, root=a.dataset_root, manifest=a.manifest)


if __name__ == "__main__":
    main()
