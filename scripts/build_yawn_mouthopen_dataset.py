"""입을 벌린 프레임만 남긴 하품 분류 데이터셋. (DMD + YawDD 합본)

목표는 하나다. **"입은 벌어졌는데 하품이 아닌" 프레임을 negative 로 쓰는 것.**

원본을 그대로 학습하면 모델이 하품을 배우는 게 아니라 "입이 벌어졌는가"를 배운다.
negative(`no_yawn`) 의 대부분이 입을 다물고 있기 때문이다. 그 모델은 말하거나
노래하는 운전자를 그대로 하품으로 판정한다 - 옛 하품 모델의 Precision 이 0.203 이던
이유이기도 하다(`docs/yawn_model.md`).

그래서 두 클래스 **양쪽에서 입을 다문 프레임을 빼고** 남긴다. 남은 문제는
"입은 벌어졌다. 이게 하품인가 말하기인가" 하나로 좁혀진다. 추론에서도 같은 문(gate)을
앞에 두면 - 입을 다물었으면 CNN 을 부르지 않고 p(yawn)=0 - 학습과 추론이 같은
조건에서 돌아간다.

출처 두 개를 합친다
-------------------
    dmd    Dataset/processed          DMD 정면 얼굴 crop. **프레임 단위 어노테이션**
             yawn    <- yawn_without_hand
             no_yawn <- no_yawn
             제외    <- yawn_with_hand (손이 입을 가려 벌림을 잴 수 없다)

    yawdd  Dataset/processed_yawdd    YawDD mirror 얼굴 crop. 영상 단위 라벨
             yawn    <- yawning 영상에서 입이 벌어진 프레임
             no_yawn <- normal/talking 영상에서 입이 벌어진 프레임
             제외    <- talking_yawning (한 영상에 말하기와 하품이 섞여 라벨 불가)

**두 출처는 성격이 정반대라 서로를 메운다.** 실측(open_ratio 중앙값):

    DMD   no_yawn 0.003   yawn_without_hand 0.261
    YawDD normal  0.005   talking 0.073        yawning 0.011

DMD 의 no_yawn 은 정상 주행이라 거의 다 입을 다물고 있다 - 임계값 0.05 를 넘는 것이
5.0% 뿐이라, 합본에서 DMD 는 사실상 **positive 공급원**이다. 대신 그 positive 는
프레임 단위 GT 라 라벨이 정확하다. 반대로 "입은 벌렸는데 하품이 아닌" negative 는
거의 전부 YawDD 의 talking 에서 온다. 한쪽만 써서는 이 데이터셋이 성립하지 않는다.

입 벌림 측정
------------
mediapipe FaceLandmarker(478점)로 잰다.

    gap = 안쪽 입술 위(13) - 아래(14) 거리
    eye = 눈 바깥 끝(33) - (263) 거리          <- 얼굴 크기 기준자
    open_ratio = gap / eye

`eye` 로 나누는 이유: 얼굴이 크게 찍혔든 작게 찍혔든 같은 값이 나와야 한다. 입 너비로
나누는 고전적 MAR 은 하품할 때 입 너비 자체가 변해서 기준자로 삼기에 불안정하다.
두 출처의 crop 기하가 같고(R=1.82) open_ratio 가 얼굴 크기에 무관하므로 두 출처의
측정값을 같은 자로 비교할 수 있다.

**두 클래스에 같은 임계값을 쓴다.** positive 만 크게 벌어진 것으로 고르고 negative 를
작게 벌어진 것으로 고르면, 모델은 다시 "입 벌린 정도"만 보면 되는 문제를 풀게 된다.
그래서 같은 문을 통과시키고, 그 다음의 판단은 모델에 맡긴다.

임계값이 곧 Recall 상한이다
---------------------------
게이트에 걸린 프레임은 추론에서 p(yawn)=0 이 되므로, 하품인데 입이 덜 벌어진 프레임은
모델이 아무리 좋아도 못 잡는다. DMD 하품 프레임(n=1,148, 프레임 단위 GT)으로 잰
게이트별 상한:

    게이트   하품인데 걸리는 비율   Recall 상한
    0.03            7.8%              0.922
    0.05           10.5%              0.895
    0.08           14.4%              0.856
    0.10           18.3%              0.817

기본값 0.05 는 DMD no_yawn 의 95%(정상 주행 = 입 다뭄)를 걸러내면서 Recall 상한을
0.895 로 남기는 지점이다. 옛 YawDD 전용 빌드는 0.10 을 썼는데, 그때는 게이트가
데이터셋 구성용이었고 추론에는 쓰지 않아서 상한이 문제되지 않았다. 추론에 같은 문을
달기로 한 이상 상한을 봐야 한다.

출력
    <출처>/metadata/mouth_metrics.csv     그 출처 전부의 측정값 (거른 것 포함)
    <out>/face/<split>/<yawn|no_yawn>/<출처>__*.jpg
    <out>/metadata/manifest.csv           학습용
    <out>/metadata/dataset.json           구성·임계값·수치 요약

이미지는 복사하지 않고 **하드링크**한다(같은 드라이브라 용량이 늘지 않는다).
독립된 사본이 필요하면 --copy.

사용
    python scripts/build_yawn_mouthopen_dataset.py --measure --sources dmd
    python scripts/build_yawn_mouthopen_dataset.py --stats
    python scripts/build_yawn_mouthopen_dataset.py --build --out yawn_mouthopen_v2
    python scripts/build_yawn_mouthopen_dataset.py --build --threshold 0.10 --match-openness

    # 옛 YawDD 전용 데이터셋을 그대로 다시 만들기
    python scripts/build_yawn_mouthopen_dataset.py --build --sources yawdd \
        --threshold 0.10 --out yawn_mouthopen
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# =====================================================================
# 1. 경로 · 출처
# =====================================================================
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent

# 입 벌림 측정은 src/mouth_gate.py 한 곳에만 둔다. 추론 게이트가 같은 함수를 쓰므로
# 여기서 다시 구현하면 두 값이 조용히 갈라진다.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mouth_gate import (          # noqa: E402
    DEFAULT_MODEL as LANDMARKER,
    DEFAULT_THRESHOLD,
    MEASURE_SIZE,
    MouthMeter,
)

DATASET_CANDIDATES = (
    REPO_ROOT / "data" / "Dataset",
    REPO_ROOT.parent / "Dataset",
    REPO_ROOT / "Dataset",
    _HERE.parent,
)

DEFAULT_OUT = "yawn_mouthopen_v2"

#: 출처 정의. 새 출처를 붙이려면 여기에 한 항목만 추가하면 된다.
#:
#:   pkg          Dataset 아래 폴더 이름 (= manifest 가 있는 곳)
#:   view         manifest 의 view 컬럼을 이 값으로 거른다. None 이면 안 거른다
#:   cond_col     "조건"으로 볼 컬럼. 이 값이 class_of 의 키가 된다
#:   class_of     조건 -> 출력 클래스. 여기 없는 조건은 통째로 제외된다
#:   quality      출력 클래스별 라벨 신뢰도. 학습에서 가중치를 다르게 줄 때 쓴다
SOURCES: dict[str, dict] = {
    "dmd": {
        "pkg": "processed",
        "view": "face",
        "cond_col": "label",
        "class_of": {"no_yawn": "no_yawn", "yawn_without_hand": "yawn"},
        # 프레임 단위 어노테이션(scripts/dmd_yawn_gt.py)이라 두 클래스 다 확실하다.
        "quality": {"no_yawn": "strong", "yawn": "strong"},
        "excluded_note": "yawn_with_hand(손이 입을 가려 벌림 측정 불가)",
    },
    "yawdd": {
        "pkg": "processed_yawdd",
        "view": None,
        "cond_col": "video_condition",
        "class_of": {"normal": "no_yawn", "talking": "no_yawn", "yawning": "yawn"},
        # 영상 단위 라벨이다. no_yawn 영상은 하품이 없는 것이 확실하지만,
        # yawning 영상에는 하품이 아닌 구간도 들어 있다.
        "quality": {"no_yawn": "strong", "yawn": "filtered_weak"},
        "excluded_note": "talking_yawning(한 영상에 말하기와 하품이 섞여 라벨 불가)",
    },
}

MANIFEST_REL = Path("metadata") / "manifest.csv"
METRICS_REL = Path("metadata") / "mouth_metrics.csv"

CLASS_IDX = {"no_yawn": 0, "yawn": 1}

#: 임계값·측정 규격·랜드마커 경로는 전부 src/mouth_gate.py 에서 가져온다(위 import).
#: 추론 게이트와 같은 값을 쓰기 위해서다.

METRIC_COLUMNS = [
    "path", "session", "video_condition", "landmark_ok",
    "open_ratio", "mar", "lip_ratio", "mouth_w_over_eye",
]

MANIFEST_COLUMNS = [
    "path", "src_path", "source", "view", "session", "subject", "video", "frame",
    "split", "label", "class_idx", "is_yawn", "open_ratio", "mar",
    "video_condition", "label_quality", "face_detected", "face_conf",
    "gender", "glasses", "eyewear", "facial_hair",
]

#: manifest 에서 그대로 옮기는 컬럼. 출처에 없는 것은 빈 값으로 둔다
#: (DMD 에는 eyewear/facial_hair 가 없고, YawDD 에는 age 가 없다).
PASSTHROUGH = ("view", "session", "subject", "video", "frame", "split",
               "face_detected", "face_conf", "gender", "glasses",
               "eyewear", "facial_hair")


# =====================================================================
# 3. 헬퍼
# =====================================================================
def dataset_dir(explicit: str | None, sources: list[str]) -> Path:
    """Dataset 폴더. 요청한 출처의 manifest 가 모두 있어야 한다."""
    need = [SOURCES[s]["pkg"] for s in sources]

    def ok(root: Path) -> bool:
        return all((root / pkg / MANIFEST_REL).exists() for pkg in need)

    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not ok(p):
            missing = [pkg for pkg in need if not (p / pkg / MANIFEST_REL).exists()]
            raise SystemExit(f"manifest 가 없습니다: "
                             + ", ".join(str(p / m / MANIFEST_REL) for m in missing))
        return p
    for c in DATASET_CANDIDATES:
        if ok(c):
            return c.resolve()
    raise SystemExit(
        f"출처 {sources} 를 모두 가진 Dataset 폴더를 찾지 못했습니다. 확인한 위치:\n  "
        + "\n  ".join(str(c) for c in DATASET_CANDIDATES)
        + "\n--dataset-root 로 직접 지정하거나, 빠진 출처를 먼저 만드세요.")


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def quantile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    a = sorted(values)
    return a[min(int(p * (len(a) - 1)), len(a) - 1)]


def source_rows(ds: Path, name: str) -> list[dict]:
    """출처 하나의 manifest 를 읽어 view 로 거른다. 조건 컬럼은 video_condition 으로 통일."""
    src = SOURCES[name]
    rows = read_csv(ds / src["pkg"] / MANIFEST_REL)
    if src["view"]:
        rows = [r for r in rows if r.get("view") == src["view"]]
    col = src["cond_col"]
    for r in rows:
        # DMD 는 label 이 곧 조건이다. 아래 로직이 출처를 가리지 않게 이름을 맞춘다.
        r["video_condition"] = r[col]
    return rows


# =====================================================================
# 4. --measure : 입 벌림 측정 (측정기는 src/mouth_gate.MouthMeter)
# =====================================================================
def measure_source(ds: Path, name: str, rows: list[dict]) -> list[dict]:
    meter = MouthMeter(LANDMARKER)
    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        img = cv2.imread(str(ds / r["path"]))
        m = meter.measure(img) if img is not None else None
        rec = {"path": r["path"], "session": r["session"],
               "video_condition": r["video_condition"],
               "landmark_ok": int(m is not None)}
        rec.update(m or {"open_ratio": "", "mar": "", "lip_ratio": "",
                         "mouth_w_over_eye": ""})
        out.append(rec)
        if i % 2000 == 0 or i == len(rows):
            print(f"  {i:,}/{len(rows):,}  {time.time() - t0:.0f}s")
    dst = ds / SOURCES[name]["pkg"] / METRICS_REL
    write_csv(dst, out, METRIC_COLUMNS)
    fail = sum(1 for r in out if not r["landmark_ok"])
    print(f"  측정 {len(out):,}장 (랜드마크 실패 {fail}장, {fail/len(out):.1%})")
    print(f"  기록: {dst}")
    return out


def load_metrics(ds: Path, name: str) -> list[dict]:
    p = ds / SOURCES[name]["pkg"] / METRICS_REL
    if not p.exists():
        raise SystemExit(
            f"'{name}' 의 측정 결과가 없습니다: {p}\n"
            f"먼저 실행하세요: --measure --sources {name}")
    return read_csv(p)


# =====================================================================
# 5. --stats : 임계값을 고르기 위한 분포
# =====================================================================
def show_stats(per_source: dict[str, list[dict]]) -> None:
    print("\nopen_ratio 분위수 (gap / eye)")
    print(f"{'출처 · 조건':<30}{'n':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p90':>8}{'실패':>8}")

    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for name, metrics in per_source.items():
        fail = Counter()
        for m in metrics:
            if str(m["landmark_ok"]) == "1":
                by_key[(name, m["video_condition"])].append(float(m["open_ratio"]))
            else:
                fail[m["video_condition"]] += 1
        for cond in SOURCES[name]["class_of"]:
            v = by_key.get((name, cond))
            if not v:
                continue
            cls = SOURCES[name]["class_of"][cond]
            print(f"{name + ' · ' + cond + f' [{cls}]':<30}{len(v):>8,}"
                  + "".join(f"{quantile(v, p):>8.3f}" for p in (.25, .5, .75, .9))
                  + f"{fail[cond]:>8,}")

    print("\n임계값별 표본 수 (두 클래스에 같은 임계값)")
    print(f"{'임계값':<8}{'no_yawn':>10}{'yawn':>10}{'합계':>10}{'yawn 비율':>11}"
          f"{'  (no_yawn 중 YawDD talking 기여)':<0}")
    for th in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        neg = pos = talk = 0
        for (name, cond), v in by_key.items():
            cls = SOURCES[name]["class_of"].get(cond)
            if cls is None:
                continue
            n = sum(1 for x in v if x > th)
            if cls == "yawn":
                pos += n
            else:
                neg += n
                if name == "yawdd" and cond == "talking":
                    talk += n
        tot = neg + pos
        print(f"{th:<8.2f}{neg:>10,}{pos:>10,}{tot:>10,}"
              f"{(pos / tot if tot else 0):>11.1%}"
              f"   {(talk / neg if neg else 0):>5.1%}")


# =====================================================================
# 6. --build : 필터링된 데이터셋
# =====================================================================
def match_openness(rows: list[dict], bin_width: float = 0.02,
                   seed: int = 42) -> tuple[list[dict], Counter]:
    """두 클래스의 open_ratio 분포를 같게 맞춘다(split 별로).

    입만 벌리면 되는 문(threshold)을 통과시켜도, 하품은 원래 더 크게 벌어져서
    두 클래스의 중앙값이 크게 벌어진다. 모델이 입 모양이나 눈 대신 '얼마나
    벌어졌나'만 보고도 상당히 맞출 수 있다는 뜻이다.

    open_ratio 를 좁은 구간으로 나눠 구간마다 두 클래스를 같은 수로 맞추면, 그
    지름길이 막힌다. 대신 표본이 줄고(구간마다 적은 쪽에 맞춘다) 하품이 아주 크게
    벌어진 구간은 통째로 빠진다 - 그 구간에는 짝지을 no_yawn 이 없기 때문이다.
    """
    rng = random.Random(seed)
    kept, dropped = [], Counter()
    groups: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        b = int(float(r["open_ratio"]) / bin_width)
        groups[(r["split"], b)][r["label"]].append(r)

    for key in sorted(groups):
        g = groups[key]
        n = min(len(g.get("no_yawn", [])), len(g.get("yawn", [])))
        for label, items in g.items():
            items = sorted(items, key=lambda r: (r["source"], r["session"], int(r["frame"])))
            if n == 0:
                dropped[f"{label}(짝 없는 구간)"] += len(items)
                continue
            pick = items if len(items) <= n else rng.sample(items, n)
            kept.extend(pick)
            dropped[f"{label}(구간 수 맞춤)"] += len(items) - len(pick)
    return kept, dropped


def collect(ds: Path, name: str, rows: list[dict], metrics: list[dict],
            threshold: float, out_name: str,
            skipped: Counter) -> list[dict]:
    """출처 하나 -> 게이트를 통과한 manifest 행들."""
    src = SOURCES[name]
    by_path = {m["path"]: m for m in metrics}
    out = []

    for r in rows:
        cls = src["class_of"].get(r["video_condition"])
        if cls is None:
            skipped[f"{name}: {src['excluded_note']}"] += 1
            continue
        m = by_path.get(r["path"])
        if m is None or not int(m["landmark_ok"]):
            skipped[f"{name}: 랜드마크 실패"] += 1
            continue
        open_ratio = float(m["open_ratio"])
        if open_ratio <= threshold:
            skipped[f"{name}: 입 다뭄"] += 1
            continue

        # 파일 이름에 출처를 박는다. 두 출처의 이름 규칙이 달라 충돌하지는 않지만,
        # 섞인 폴더에서 어느 데이터셋에서 왔는지 눈으로 보이는 편이 낫다.
        fname = f"{name}__{Path(r['path']).name}"
        out.append({
            **{k: r.get(k, "") for k in PASSTHROUGH},
            "path": f"{out_name}/face/{r['split']}/{cls}/{fname}",
            "src_path": r["path"],
            "source": name,
            "label": cls,
            "class_idx": CLASS_IDX[cls],
            "is_yawn": CLASS_IDX[cls],
            "open_ratio": open_ratio,
            "mar": m["mar"],
            "video_condition": r["video_condition"],
            "label_quality": src["quality"][cls],
        })
    return out


def build(ds: Path, sources: list[str], per_rows: dict[str, list[dict]],
          per_metrics: dict[str, list[dict]], threshold: float, out_name: str,
          copy: bool, match: bool = False) -> None:
    out_dir = ds / out_name
    if (out_dir / "face").exists():
        shutil.rmtree(out_dir / "face")      # 임계값을 바꿔 다시 만들 때 섞이지 않게

    skipped: Counter = Counter()
    rows: list[dict] = []
    for name in sources:
        rows += collect(ds, name, per_rows[name], per_metrics[name],
                        threshold, out_name, skipped)

    if match:
        rows, dropped = match_openness(rows)
        skipped.update(dropped)

    for r in rows:
        target = ds / r["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        src = ds / r["src_path"]
        if copy:
            shutil.copy2(src, target)
        else:
            try:
                target.hardlink_to(src)
            except OSError:
                shutil.copy2(src, target)      # 드라이브가 다르면 링크가 안 된다

    rows.sort(key=lambda r: (r["source"], r["session"], int(r["frame"])))
    write_csv(out_dir / "metadata" / "manifest.csv", rows, MANIFEST_COLUMNS)
    summarize(out_dir, sources, rows, threshold, skipped, copy, match)


def summarize(out_dir: Path, sources: list[str], rows: list[dict], threshold: float,
              skipped: Counter, copy: bool, match: bool = False) -> None:
    by_split_label = defaultdict(Counter)
    by_source_label = defaultdict(Counter)
    for r in rows:
        by_split_label[r["split"]][r["label"]] += 1
        by_source_label[r["source"]][r["label"]] += 1
    by_cond = Counter(f"{r['source']}/{r['video_condition']}" for r in rows)
    subjects = defaultdict(set)
    for r in rows:
        subjects[r["split"]].add(f"{r['source']}/{r['subject']}")
    ratios = defaultdict(list)
    for r in rows:
        ratios[r["label"]].append(float(r["open_ratio"]))
    quality = Counter(r["label_quality"] for r in rows)

    doc = {
        "name": "mouth-open yawning frames (DMD + YawDD)",
        "sources": {n: {"pkg": SOURCES[n]["pkg"], "class_of": SOURCES[n]["class_of"],
                        "excluded": SOURCES[n]["excluded_note"],
                        "label_quality": SOURCES[n]["quality"]} for n in sources},
        "task": "frame-level yawning classification with mouth-open negatives",
        "why": ("negative 가 대부분 입을 다물고 있으면 모델이 하품이 아니라 '입이 "
                "벌어졌는가'를 배운다. 그 모델은 말하는 운전자를 하품으로 오경보한다. "
                "그래서 두 클래스 모두 입을 벌린 프레임만 남겼다."),
        "classes": ["no_yawn", "yawn"],
        "class_idx": CLASS_IDX,
        "binary": {"column": "is_yawn", "0": "no_yawn (입은 벌렸지만 하품 아님)",
                   "1": "yawn (하품 프레임)"},
        "mouth_open": {
            "metric": "open_ratio = |lip13 - lip14| / |eye33 - eye263|",
            "landmarker": "mediapipe FaceLandmarker (478점, float16)",
            "measure_size": MEASURE_SIZE,
            "threshold": threshold,
            "applied_to": "두 출처·두 클래스 모두 같은 임계값",
            "inference_gate": ("추론에서도 같은 임계값으로 게이트를 걸어야 학습과 "
                               "조건이 같다. open_ratio <= threshold 이면 CNN 을 "
                               "부르지 않고 p(yawn)=0 으로 둔다."),
            "recall_ceiling_note": ("게이트에 걸린 하품 프레임은 모델이 못 잡는다. "
                                    "DMD 프레임 단위 GT 기준 상한: 0.03->0.922, "
                                    "0.05->0.895, 0.08->0.856, 0.10->0.817"),
            "match_openness": match,
            "match_note": ("True 면 두 클래스의 open_ratio 분포를 구간별로 같게 맞췄다. "
                           "'얼마나 벌어졌나'로는 못 맞히게 되지만 표본이 줄어든다"),
        },
        "label_quality": {
            "strong": ("DMD 양쪽 + YawDD no_yawn. DMD 는 프레임 단위 어노테이션이고, "
                       "YawDD no_yawn 은 normal/talking 영상이라 하품이 없는 것이 확실하다"),
            "filtered_weak": ("YawDD yawn - 하품 영상에서 입이 벌어진 프레임. 프레임 단위 "
                              "어노테이션이 아니라 여전히 약한 라벨이지만, 필터 전보다 "
                              "하품일 확률이 훨씬 높다"),
            "counts": dict(quality),
        },
        "excluded": dict(skipped),
        "counts": {
            "total": len(rows),
            "by_label": dict(Counter(r["label"] for r in rows)),
            "by_source": {k: dict(v) for k, v in sorted(by_source_label.items())},
            "by_split": {k: dict(v) for k, v in by_split_label.items()},
            "by_source_condition": dict(by_cond),
            "subjects": {k: len(v) for k, v in sorted(subjects.items())},
        },
        "open_ratio_median": {k: round(float(np.median(v)), 4)
                              for k, v in sorted(ratios.items())},
        "storage": "복사" if copy else "하드링크 (원본과 같은 파일, 용량 증가 없음)",
        "manifest": f"{out_dir.name}/metadata/manifest.csv",
        "columns": MANIFEST_COLUMNS,
    }
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata" / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"\n입 벌림 임계값 {threshold} -> 표본 {len(rows):,}장")
    print(f"{'split':<8}{'no_yawn':>10}{'yawn':>10}{'합계':>10}{'피험자':>8}")
    for split in ("train", "val", "test"):
        c = by_split_label[split]
        print(f"{split:<8}{c['no_yawn']:>10,}{c['yawn']:>10,}"
              f"{sum(c.values()):>10,}{len(subjects[split]):>8}")
    print(f"\n{'출처':<8}{'no_yawn':>10}{'yawn':>10}{'합계':>10}")
    for name in sources:
        c = by_source_label[name]
        print(f"{name:<8}{c['no_yawn']:>10,}{c['yawn']:>10,}{sum(c.values()):>10,}")
    print(f"\n  라벨 신뢰도: {dict(quality)}")
    print(f"  제외:       {dict(skipped)}")
    print(f"  open_ratio 중앙값: {doc['open_ratio_median']}")
    print(f"  기록: {out_dir / 'metadata' / 'manifest.csv'}")
    print(f"        {out_dir / 'metadata' / 'dataset.json'}")


# =====================================================================
# 7. main
# =====================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true",
                    help="crop 전부의 입 벌림을 재서 mouth_metrics.csv 를 만든다")
    ap.add_argument("--stats", action="store_true", help="조건별 분포와 임계값 표")
    ap.add_argument("--build", action="store_true", help="필터링된 데이터셋 생성")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES),
                    choices=list(SOURCES),
                    help=f"쓸 출처 (기본 {' '.join(SOURCES)})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"출력 폴더 이름 (기본 {DEFAULT_OUT})")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"입 벌림 임계값 open_ratio (기본 {DEFAULT_THRESHOLD}). "
                         "추론 게이트와 같은 값을 쓸 것")
    ap.add_argument("--match-openness", action="store_true",
                    help="두 클래스의 입 벌림 분포를 같게 맞춘다(지름길 차단, 표본 감소)")
    ap.add_argument("--copy", action="store_true",
                    help="하드링크 대신 복사 (용량이 그만큼 늘어난다)")
    ap.add_argument("--dataset-root", default=None, help="Dataset 폴더 경로")
    a = ap.parse_args()

    if not (a.measure or a.stats or a.build):
        ap.error("--measure / --stats / --build 중 하나를 주세요.")

    ds = dataset_dir(a.dataset_root, a.sources)
    print(f"Dataset : {ds}")

    per_rows = {n: source_rows(ds, n) for n in a.sources}
    for n in a.sources:
        print(f"원본    : {len(per_rows[n]):>7,}장  {n} ({SOURCES[n]['pkg']})")

    per_metrics: dict[str, list[dict]] = {}
    for n in a.sources:
        if a.measure:
            print(f"\n[측정] {n} 입 벌림")
            per_metrics[n] = measure_source(ds, n, per_rows[n])
        else:
            per_metrics[n] = load_metrics(ds, n)

    if a.stats:
        show_stats(per_metrics)

    if a.build:
        print(f"\n[생성] {a.out} - 입 벌림 > {a.threshold} 인 프레임만"
              f"{' (분포 맞춤)' if a.match_openness else ''}")
        build(ds, a.sources, per_rows, per_metrics, a.threshold, a.out,
              a.copy, a.match_openness)


if __name__ == "__main__":
    main()
