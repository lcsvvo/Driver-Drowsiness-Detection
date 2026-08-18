"""Eye CNN 학습·평가 — 통합 manifest 하나에서 학습 소스만 바꿔 A/B/C/C' 를 비교한다.

STEP11 이 만든 `outputs/eye_dataset/eye_manifest.csv` 를 유일한 입력으로 쓴다.
split 은 manifest 에 이미 박혀 있으므로 이 스크립트는 split 을 만들지 않는다.
바꾸는 것은 `source` 필터뿐이다.

    A   MRL only          train=["mrl"]
    B   DMD only          train=["dmd"]
    C   MRL + DMD 합본     train=["mrl","dmd"]
    C'  MRL -> DMD 미세조정 train=["dmd"], pretrained=<A 모델>, lr=1e-4

평가 규칙
    - 주 지표는 Closed-Recall. Closed 를 Open 으로 놓치는 것이 가장 위험하다.
    - DMD 는 같은 프레임의 좌·우 눈이 독립 표본이 아니므로 **프레임 단위로 집계**한다
      (좌우 Closed 확률의 max. 02_INFER 의 '한쪽이라도 감기면 Closed' 규칙과 같다).
    - 판정 임계값은 val 에서 고르고 test 에 그대로 적용한다. test 에서 고르면 낙관 편향이 생긴다.
    - test 는 DMD hold-out 을 주로 보고, MRL test 도 함께 보고한다. DMD 를 올리려다
      MRL 성능을 깎았는지 확인하는 대조군이다.

입력 형식
    본 실험  : size=128, gray=True   (MRL 이 IR 그레이스케일이라 채널을 통일)
    부록 실험: size=256, gray=False  (01/02 의 eye_model.keras 와 같은 입력 규격)

사용 (저장소 루트에서)
    python src/train_eye.py --train mrl            --eval dmd
    python src/train_eye.py --train dmd            --eval dmd
    python src/train_eye.py --train mrl dmd        --eval dmd
    python src/train_eye.py --train dmd --eval dmd --pretrained <A.keras> --lr 1e-4
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
from mrl_dataset import make_dataset

SOURCES = ("mrl", "dmd")
SEED = 42

#: 통합 manifest. STEP11 산출물.
MANIFEST = config.OUTPUTS_DIR / "eye_dataset" / "eye_manifest.csv"

#: 소스 필터를 적용한 부분집합 CSV 를 두는 곳.
#: tempdir 대신 outputs 아래 고정 이름으로 쓴다. 실행마다 이름이 바뀌지 않아
#: 재현이 되고, 무엇이 학습에 들어갔는지 나중에 열어 볼 수 있다.
SUBSET_DIR = config.OUTPUTS_DIR / "eye_dataset" / "subsets"

ARTIFACT_DIR = getattr(config, "ARTIFACT_DIR", config.PROJECT_ROOT / "artifacts")


# =====================================================================
# 1. 모델 (01_TRAIN 구조 유지 + 입력 파라미터화)
# =====================================================================
def build_eye_model(size: int, ch: int) -> tf.keras.Model:
    """01_TRAIN 의 Conv 3개 + Dense 구조를 그대로 두고 입력 크기·채널만 인자화.

    RandomRotation 만 0.4 -> 0.12 로 줄였다. 0.4 는 최대 +-72도로 눈 crop 에 과하다.
    class0 = Closed 로 두어 기존 추론·PERCLOS 코드와 그대로 연결된다.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(size, size, ch)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.12, fill_mode="reflect"),
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
# 2. manifest 부분집합
# =====================================================================
def load_manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        raise SystemExit(f"통합 manifest 가 없습니다: {config._rel(MANIFEST)}\n"
                         "  11_eye_dataset_build.ipynb 를 먼저 실행하세요.")
    df = pd.read_csv(MANIFEST)
    missing = {"path", "source", "subject", "video", "frame", "side",
               "label", "class_idx", "split"} - set(df.columns)
    if missing:
        raise SystemExit(f"manifest 컬럼 누락: {sorted(missing)}")
    return df


def subset(df: pd.DataFrame, sources, split: str,
           limit: int | None = None) -> tuple[Path, pd.DataFrame]:
    """source·split 로 걸러 부분집합 CSV 를 쓰고 (경로, 데이터프레임) 반환.

    manifest 행 순서를 그대로 유지한다. 뒤에서 예측값과 행을 위치로 맞추기 때문에
    순서가 바뀌면 라벨이 어긋난다.

    limit 을 주면 클래스 비율을 유지한 채 그만큼으로 줄인다. 스모크 테스트
    (전체 학습 전에 파이프라인 오류만 확인) 용도다.
    """
    sub = df[(df.source.isin(sources)) & (df.split == split)].reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"'{split}' 에 해당하는 행이 없습니다 (source={list(sources)})")

    if limit and len(sub) > limit:
        frac = limit / len(sub)
        keep = []
        for _, g in sub.groupby("class_idx"):
            keep.extend(g.sample(max(1, round(len(g) * frac)), random_state=SEED).index)
        sub = sub.loc[sorted(keep)].reset_index(drop=True)   # 원래 순서 복원

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBSET_DIR / f"{'+'.join(sources)}__{split}{f'__n{limit}' if limit else ''}.csv"
    sub.to_csv(out, index=False)
    return out, sub


def dataset_of(csv_path: Path, split: str, size: int, gray: bool,
               sharpen: bool, batch: int, shuffle: bool):
    # manifest path 가 저장소 루트 기준 상대경로라 root 를 PROJECT_ROOT 로 준다.
    return make_dataset(str(csv_path), split, size=size, grayscale=gray,
                        do_sharpen=sharpen, batch_size=batch, shuffle=shuffle,
                        seed=SEED, mrl_root=str(config.PROJECT_ROOT))


# =====================================================================
# 3. 평가
# =====================================================================
def closed_prob(model, ds) -> np.ndarray:
    """Closed(class 0) 확률을 manifest 순서 그대로 반환."""
    out = []
    for xb, _ in ds:
        out.append(model(xb, training=False).numpy()[:, 0])
    return np.concatenate(out)


def to_frame_level(sub: pd.DataFrame, p_closed: np.ndarray) -> pd.DataFrame:
    """같은 프레임의 좌·우 눈을 한 표본으로 합친다 (Closed 확률 max).

    DMD 는 프레임당 crop 2장이라 crop 단위로 세면 독립 표본 수가 두 배로 과대 계산된다.
    MRL 은 side='-' 이라 그룹 크기가 1 이므로 값이 그대로 유지된다.
    """
    d = sub.assign(p_closed=p_closed)
    key = ["source", "video", "frame", "path"]
    d["_key"] = np.where(d.source == "dmd",
                         d.video.astype(str) + "#" + d.frame.astype(str),
                         d.path.astype(str))
    g = d.groupby("_key", sort=False).agg(
        p_closed=("p_closed", "max"),        # 한쪽이라도 감기면 Closed (02_INFER 규칙)
        y_closed=("class_idx", lambda s: int((s == 0).all())),
        source=("source", "first"),
        subject=("subject", "first"),
        glasses=("glasses", "first"),
    ).reset_index(drop=True)
    return g


def metrics_at(y_closed: np.ndarray, p_closed: np.ndarray, thr: float) -> dict:
    pred = (p_closed >= thr).astype(int)
    tp = int(((pred == 1) & (y_closed == 1)).sum())
    fp = int(((pred == 1) & (y_closed == 0)).sum())
    fn = int(((pred == 0) & (y_closed == 1)).sum())
    tn = int(((pred == 0) & (y_closed == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    pre = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    return dict(threshold=round(float(thr), 4), n=len(y_closed),
                accuracy=round((tp + tn) / len(y_closed), 4),
                closed_recall=round(rec, 4), closed_precision=round(pre, 4),
                closed_f1=round(f1, 4),
                confusion=dict(closed_as_closed=tp, closed_as_open=fn,
                               open_as_closed=fp, open_as_open=tn))


#: 임계값 탐색 그리드.
#: 0.99 위쪽을 로그 간격으로 촘촘히 둔다. softmax 가 포화하면 최적 임계값이
#: 0.99 를 넘는 일이 흔한데, 상한이 0.95 였을 때 A·C\' 조건이 상한에 걸려
#: 그보다 높은 값을 시험해 보지 못했다.
THRESHOLD_GRID = np.unique(np.concatenate([
    np.arange(0.01, 0.99, 0.01),
    1.0 - np.logspace(-2, -5, 25),          # 0.99 ~ 0.99999
]))


def pick_threshold(y_closed: np.ndarray, p_closed: np.ndarray,
                   target_recall: float) -> tuple[float, bool, bool]:
    """val 에서 Closed-Recall >= target 을 만족하는 가장 높은 임계값.

    임계값이 높아질수록 Closed 판정이 줄어 Recall 이 내려간다. 제약을 만족하는 중
    가장 높은 값을 고르면 Recall 을 지키면서 Precision 이 최대가 된다.

    반환: (임계값, 목표달성여부, 그리드상한도달여부)
    상한에 걸리면 더 높은 임계값에서 Precision 이 더 올랐을 수 있으므로 보고한다.
    """
    ok = [t for t in THRESHOLD_GRID
          if metrics_at(y_closed, p_closed, t)["closed_recall"] >= target_recall]
    if not ok:
        return 0.5, False, False
    best = float(max(ok))
    return best, True, bool(best >= THRESHOLD_GRID[-1] - 1e-12)


def report(name: str, m: dict) -> None:
    c = m["confusion"]
    print(f"  [{name}] n={m['n']:,}  thr={m['threshold']:.2f}  "
          f"Closed-Recall={m['closed_recall']:.4f}  "
          f"Precision={m['closed_precision']:.4f}  F1={m['closed_f1']:.4f}  "
          f"Acc={m['accuracy']:.4f}")
    print(f"          실제Closed -> Closed {c['closed_as_closed']} / Open {c['closed_as_open']}"
          f"   |   실제Open -> Closed {c['open_as_closed']} / Open {c['open_as_open']}")


# =====================================================================
# 4. 실험 1회
# =====================================================================
def run_experiment(train, eval_source="dmd", pretrained=None, size=128, gray=True,
                   sharpen=False, epochs=20, batch=64, lr=None,
                   target_recall=0.90, verbose=1, limit=None) -> dict:
    """A/B/C/C' 중 하나를 실행하고 지표 dict 를 반환한다.

    limit 을 주면 train/val/test 를 그 크기로 줄여 파이프라인만 점검한다(스모크).
    산출물 이름에 __smoke 가 붙어 본 실험 결과를 덮어쓰지 않는다.
    """
    train = list(train)
    bad = set(train) - set(SOURCES)
    if bad:
        raise SystemExit(f"알 수 없는 source: {sorted(bad)}")

    tag = (f"{'+'.join(train)}__eval-{eval_source}__"
           f"{'gray' if gray else 'rgb'}{size}"
           f"{'__sharp' if sharpen else ''}{'__ft' if pretrained else ''}"
           f"{'__smoke' if limit else ''}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / f"eye_{tag}.keras"
    metrics_path = ARTIFACT_DIR / f"eye_{tag}_metrics.json"

    df = load_manifest()
    tr_csv, tr_sub = subset(df, train, "train", limit)
    va_csv, va_sub = subset(df, train, "val", limit)    # 학습 소스와 같은 도메인에서 모델 선택
    te_csv, te_sub = subset(df, [eval_source], "test", limit)

    train_ds = dataset_of(tr_csv, "train", size, gray, sharpen, batch, True)[0]
    val_ds = dataset_of(va_csv, "val", size, gray, sharpen, batch, False)[0]
    test_ds = dataset_of(te_csv, "test", size, gray, sharpen, batch, False)[0]

    print(f"TAG  {tag}")
    print(f"train {len(tr_sub):,} (Closed {(tr_sub.class_idx==0).mean()*100:.1f}%) | "
          f"val {len(va_sub):,} | test {len(te_sub):,} "
          f"(Closed {(te_sub.class_idx==0).mean()*100:.1f}%)")

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    if pretrained:
        model = tf.keras.models.load_model(pretrained)
        print("사전학습 로드:", Path(pretrained).name)
    else:
        model = build_eye_model(size, channels(gray))
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

    out = _evaluate(best, df, tag, train, eval_source, size, gray, sharpen, batch,
                    target_recall, limit, va_sub, val_ds, te_sub, test_ds,
                    n_train=len(tr_sub), model_path=model_path,
                    pretrained=Path(pretrained).name if pretrained else None)
    metrics_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n저장:", config._rel(model_path))
    print("저장:", config._rel(metrics_path))
    return out


def _evaluate(model, df, tag, train, eval_source, size, gray, sharpen, batch,
              target_recall, limit, va_sub, val_ds, te_sub, test_ds,
              n_train, model_path, pretrained=None) -> dict:
    """임계값 선택 + test 평가 + 대조군 + 안경 조건별. 학습은 하지 않는다.

    run_experiment 와 reevaluate 가 같은 평가 경로를 쓰도록 분리했다.
    """
    # --- 임계값은 val 에서 고른다 (test 에서 고르면 낙관 편향) ---
    va_frame = to_frame_level(va_sub, closed_prob(model, val_ds))
    thr, reached, saturated = pick_threshold(
        va_frame.y_closed.values, va_frame.p_closed.values, target_recall)
    if not reached:
        print(f"[!] val 에서 Closed-Recall {target_recall:.2f} 를 만족하는 임계값이 없어 "
              f"0.5 로 둡니다.")
    if saturated:
        print(f"[!] 선택된 임계값이 탐색 그리드 상한({THRESHOLD_GRID[-1]:.5f})에 걸렸습니다. "
              f"더 높은 값에서 Precision 이 더 올랐을 수 있습니다.")

    # --- 주 평가: eval_source 의 test ---
    te_frame = to_frame_level(te_sub, closed_prob(model, test_ds))
    print("\n===== TEST =====")
    m_main = metrics_at(te_frame.y_closed.values, te_frame.p_closed.values, thr)
    report(f"{eval_source} test @val-thr", m_main)
    m_half = metrics_at(te_frame.y_closed.values, te_frame.p_closed.values, 0.5)
    report(f"{eval_source} test @0.50", m_half)

    # --- 대조군: 반대편 소스의 test. DMD 를 올리려다 MRL 을 깎았는지 본다 ---
    other = next((s for s in SOURCES if s != eval_source), None)
    m_other = None
    if other and not df[(df.source == other) & (df.split == "test")].empty:
        ot_csv, ot_sub = subset(df, [other], "test", limit)
        ot_ds = dataset_of(ot_csv, "test", size, gray, sharpen, batch, False)[0]
        ot_frame = to_frame_level(ot_sub, closed_prob(model, ot_ds))
        m_other = metrics_at(ot_frame.y_closed.values, ot_frame.p_closed.values, thr)
        report(f"{other} test @val-thr (대조군)", m_other)

    # --- 안경 조건별 (표본이 적어 참고용) ---
    by_glasses = {}
    for g in sorted(te_frame.glasses.dropna().unique()):
        gm = te_frame[te_frame.glasses == g]
        if len(gm) >= 50:
            by_glasses[f"glasses={int(g)}"] = metrics_at(
                gm.y_closed.values, gm.p_closed.values, thr)
    if by_glasses:
        print("\n  -- 안경 조건별 (표본 적음, 참고용) --")
        for k, v in by_glasses.items():
            report(k, v)

    return dict(tag=tag, train=train, eval_source=eval_source, size=size,
                gray=gray, sharpen=sharpen, pretrained=pretrained,
                threshold=thr, threshold_reached_target=reached,
                threshold_grid_saturated=saturated, target_recall=target_recall,
                test_main=m_main, test_main_at_050=m_half, test_other=m_other,
                by_glasses=by_glasses,
                n_train=n_train, n_val=len(va_sub), n_test_frames=len(te_frame),
                model=str(config._rel(model_path)))


def reevaluate(tag: str, target_recall: float | None = None, batch: int = 64) -> dict:
    """저장된 모델로 임계값 선택과 평가만 다시 한다. **재학습하지 않는다.**

    임계값 탐색 그리드를 넓힌 뒤, 학습을 다시 돌리지 않고 결과를 갱신하기 위한 함수다.
    학습 조건(train/eval/size/gray)은 기존 metrics JSON 에서 읽으므로 인자로 받지 않는다.
    """
    model_path = ARTIFACT_DIR / f"eye_{tag}.keras"
    metrics_path = ARTIFACT_DIR / f"eye_{tag}_metrics.json"
    for p in (model_path, metrics_path):
        if not p.exists():
            raise SystemExit(f"{config._rel(p)} 가 없습니다. 먼저 학습하세요.")

    old = json.loads(metrics_path.read_text(encoding="utf-8"))
    train, eval_source = old["train"], old["eval_source"]
    size, gray, sharpen = old["size"], old["gray"], old["sharpen"]
    target = old["target_recall"] if target_recall is None else target_recall

    df = load_manifest()
    va_csv, va_sub = subset(df, train, "val")
    te_csv, te_sub = subset(df, [eval_source], "test")
    val_ds = dataset_of(va_csv, "val", size, gray, sharpen, batch, False)[0]
    test_ds = dataset_of(te_csv, "test", size, gray, sharpen, batch, False)[0]

    model = tf.keras.models.load_model(model_path)
    print(f"TAG  {tag}  (재평가 — 학습 없음)")
    print(f"이전 임계값 {old['threshold']} -> 새 그리드로 재선택")

    out = _evaluate(model, df, tag, train, eval_source, size, gray, sharpen, batch,
                    target, None, va_sub, val_ds, te_sub, test_ds,
                    n_train=old["n_train"], model_path=model_path,
                    pretrained=old.get("pretrained"))
    metrics_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n갱신:", config._rel(metrics_path))
    return out


#: 본 실험 규격. 여기서 벗어나면 부록으로 분류한다.
MAIN_SPEC = dict(size=128, gray=True)


def is_appendix(meta: dict) -> bool:
    """본 실험(128 gray)이 아니면 부록. 입력 규격이 다른 결과를 같은 표에 두지 않기 위함."""
    return not (meta.get("size") == MAIN_SPEC["size"]
                and bool(meta.get("gray", True)) == MAIN_SPEC["gray"])


def cond_label(meta: dict) -> str:
    """metrics dict -> 사람이 읽는 조건 이름."""
    train = "+".join(meta.get("train", []))
    if is_appendix(meta):
        return f"부록 {train} {'gray' if meta.get('gray', True) else 'rgb'}{meta.get('size')}"
    if meta.get("pretrained"):
        return "C' MRL->DMD ft"
    return {"mrl": "A MRL only", "dmd": "B DMD only",
            "mrl+dmd": "C MRL+DMD"}.get(train, train)


def load_all_metrics(include_smoke: bool = False) -> dict:
    """artifacts 의 metrics JSON 을 전부 읽는다.

    노트북 셀을 따로 돌렸거나 커널을 재시작해도 결과를 모을 수 있게, 메모리 변수가
    아니라 디스크를 정본으로 둔다. 형식이 맞지 않는 파일은 건너뛰고 이유를 알린다.
    """
    out = {}
    for p in sorted(ARTIFACT_DIR.glob("eye_*_metrics.json")):
        tag = p.name[len("eye_"):-len("_metrics.json")]
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


def _usable(r: dict) -> bool:
    """compare 가 쓸 수 있는 형식인지. 아니면 이유를 출력한다."""
    tag = (r or {}).get("tag", "?")
    m = (r or {}).get("test_main")
    if not isinstance(m, dict) or "closed_recall" not in m:
        print(f"[건너뜀] {tag}: test_main 에 closed_recall 이 없습니다(예전 형식). "
              f"T.reevaluate('{tag}') 로 갱신하세요.")
        return False
    return True


def compare(results) -> pd.DataFrame:
    """조건 비교표. dict(tag->meta) 또는 list 를 받는다.

    형식이 안 맞는 항목은 KeyError 로 죽지 않고 건너뛴다. 조건 이름(cond)도 여기서 붙여
    호출부가 메모리 변수 이름에 의존하지 않게 한다.
    """
    items = list(results.values()) if isinstance(results, dict) else list(results)
    rows = []
    for r in items:
        if not r or not _usable(r):
            continue
        m = r["test_main"]
        o = r.get("test_other") or None
        rows.append({
            "cond": cond_label(r), "tag": r["tag"], "thr": r.get("threshold"),
            "closed_recall": m["closed_recall"], "closed_precision": m["closed_precision"],
            "closed_f1": m["closed_f1"], "accuracy": m["accuracy"],
            "other_closed_recall": (o or {}).get("closed_recall"),
            "n_test_frames": r.get("n_test_frames"),
        })
    t = pd.DataFrame(rows)
    return t.sort_values("closed_recall", ascending=False) if not t.empty else t


# =====================================================================
# 5. CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["mrl"], choices=list(SOURCES))
    ap.add_argument("--eval", dest="eval_source", default="dmd", choices=list(SOURCES))
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--rgb", action="store_true", help="256 RGB 부록 실험용")
    ap.add_argument("--sharpen", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--target_recall", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None,
                    help="스모크 테스트: split 당 이 개수로 줄여 파이프라인만 점검")
    a = ap.parse_args()
    run_experiment(a.train, a.eval_source, a.pretrained, a.size, not a.rgb,
                   a.sharpen, a.epochs, a.batch, a.lr, a.target_recall, limit=a.limit)


if __name__ == "__main__":
    main()
