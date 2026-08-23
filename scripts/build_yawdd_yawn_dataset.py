"""정규화된 YawDD mirror 영상 -> 얼굴 crop 하품 데이터셋 + manifest.

전제: 먼저 `python scripts/prepare_yawdd.py` 를 돌려 raw/YawDD/metadata/videos.csv
가 있어야 한다. split·피험자·조건은 전부 그 파일에서 읽는다.

**라벨의 성질을 먼저 알아야 한다.** YawDD 의 라벨은 파일명에 붙은 *영상 단위* 라벨이고,
프레임 단위 어노테이션이 없다. DMD 하품 데이터셋(`Dataset/processed`)과 결정적으로
다른 점이다.

    normal / talking      -> 이 영상에는 하품이 없다. **모든 프레임이 no_yawn** 이다.
                             프레임 라벨로 그대로 써도 된다 (label_quality=strong).
                             talking 은 "입은 열리지만 하품은 아닌" 어려운 음성 표본이라
                             오경보를 줄이는 데 특히 값이 있다.
    yawning               -> 하품이 **들어 있다**. 그러나 영상 대부분은 하품 전후의
                             평범한 프레임이다. 여기서 뽑은 프레임에 붙는 yawn 라벨은
                             틀릴 수 있다 (label_quality=weak).
    talking&yawning       -> 말하기와 하품이 한 영상에 섞여 있다. 가장 약한 라벨이다.

그래서 이 스크립트는 라벨을 3종으로 나눠 기록한다: `no_yawn`(강), `yawn_video`(약),
`mixed_video`(약). **약한 라벨을 그대로 학습에 넣으면 안 된다.** 쓰는 법은
`Dataset/processed_yawdd/README.md` §3 에 적어 두었다.

crop 규격 - 영상마다 고정 박스 하나
    영상 전체에서 얼굴을 검출해 중앙값 박스를 구하고, 거기에 학습·추론이 쓰는
    기하(R=1.82, dx=0.01, dy=0.09)를 적용해 정사각형 박스 하나로 고정한다.
    - 고정 박스라 손이 입을 가려 검출이 끊긴 프레임도 버리지 않는다.
    - 박스가 프레임마다 흔들리지 않아 CNN 이 crop 위치 변화를 학습하지 않는다.
    - R 을 DMD 와 같은 값으로 두는 것이 핵심이다. YawDD 얼굴은 화면에서 크게 잡혀
      여백식(build_dmd_yawn_dataset.MARGIN)을 그대로 쓰면 R=1.5 근처가 나오고,
      DMD 로 학습한 모델·추론 경로(src/yawn_dataset.INFER_CROP)와 스케일이 어긋난다.
    - R 을 지키려면 박스가 화면(640x480)을 넘는 영상이 있다. 한 변을 깎는 대신
      가장자리를 복제해 채운다. 그래야 모든 영상에서 얼굴 크기가 같다.

출력
    processed_yawdd/face/<split>/<label>/<session>_f<frame:06d>.jpg
    processed_yawdd/metadata/crop_boxes.json   영상별 고정 박스 + 검출 통계
    processed_yawdd/metadata/manifest.csv      표본 1장 = 1행
    processed_yawdd/metadata/dataset.json      클래스/split/전처리 설정 요약

사용
    python scripts/build_yawdd_yawn_dataset.py --plan    # 영상을 읽지 않고 표본 수만
    python scripts/build_yawdd_yawn_dataset.py --boxes   # 1회차: 고정 박스
    python scripts/build_yawdd_yawn_dataset.py --build   # 2회차: crop 저장 + manifest
    python scripts/build_yawdd_yawn_dataset.py --all
    python scripts/build_yawdd_yawn_dataset.py --all --one   # 영상 1개만 (점검용)
"""
from __future__ import annotations

import argparse
import csv
import json
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
# 1. 경로
# =====================================================================
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent

DATASET_CANDIDATES = (
    REPO_ROOT / "data" / "Dataset",
    REPO_ROOT.parent / "Dataset",
    REPO_ROOT / "Dataset",
    _HERE.parent,
)

RAW_REL = Path("raw") / "YawDD"
VIDEOS_CSV_REL = RAW_REL / "metadata" / "videos.csv"
OUT_NAME = "processed_yawdd"

YUNET_CANDIDATES = (
    REPO_ROOT / "model" / "detectors" / "face_detection_yunet_2023mar.onnx",
    REPO_ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx",
)

# =====================================================================
# 2. 파라미터
# =====================================================================
#: YuNet 설정. 02_INFER_YuNet.ipynb / build_dmd_*_dataset.py 와 같은 값.
YUNET_SCORE, YUNET_NMS, YUNET_TOPK = 0.6, 0.3, 5000

#: crop 기하. src/yawn_dataset.py 의 INFER_CROP 과 **같은 값이어야 한다**.
#: R  = crop 한 변 / 얼굴 높이,  dx,dy = (crop 중심 - 얼굴 중심) / 얼굴 높이.
#: DMD 학습 crop 에서 잰 값이라, 이 값을 지켜야 두 데이터셋의 얼굴 크기가 맞는다.
CROP_GEOM = {"R": 1.82, "dx": 0.01, "dy": 0.09}

#: 이상치 판정 - 오검출 하나가 중앙값을 흔드는 것을 막는다.
#: (build_dmd_yawn_dataset.py 와 같은 기준)
OUTLIER_DIST = 1.5
OUTLIER_SIZE = (0.5, 2.0)

#: 1회차에서 얼굴을 찾을 때의 프레임 간격. 박스 중앙값만 구하면 되므로 성기게 본다.
BOX_STRIDE = 15

#: 2회차 표본 간격(프레임). 조건마다 다르게 둔다.
#: - normal/talking 은 어느 프레임을 뽑아도 라벨이 같다. 성기게 뽑아 중복을 줄인다.
#: - yawning 은 영상 안에서 하품 구간이 일부뿐이라 성기게 뽑으면 하품이 통째로
#:   빠질 수 있다. 촘촘히 뽑되, 그만큼 약한 라벨의 오염도 늘어난다는 점을 감안한다.
STRIDE = {"normal": 15, "talking": 15, "yawning": 8, "talking_yawning": 10}

#: 검출률이 이보다 낮으면 고정 박스를 믿을 수 없다고 보고 그 영상을 건너뛴다.
MIN_DETECT_RATE = 0.3
MIN_DETECTIONS = 5

JPEG_QUALITY = 95

#: 프레임 픽셀 표준편차가 이보다 낮으면 내용이 없는 프레임으로 보고 버린다.
BLANK_STD = 3.0

#: 영상 조건 -> (프레임 라벨, class_idx, is_yawn, 라벨 신뢰도)
LABEL_OF = {
    "normal":          ("no_yawn",     0, 0, "strong"),
    "talking":         ("no_yawn",     0, 0, "strong"),
    "yawning":         ("yawn_video",  1, 1, "weak"),
    "talking_yawning": ("mixed_video", 2, 1, "weak"),
}
CLASS_NAMES = ("no_yawn", "yawn_video", "mixed_video")

#: DMD manifest 와 컬럼 이름을 맞춘다. `glasses` 는 src/train_yawn.py 의
#: REQUIRED_COLS 에 들어 있어서, 없으면 그 스크립트가 컬럼 누락으로 멈춘다.
#: 안경 종류(glasses/sunglasses)의 구분은 `eyewear` 에 따로 남긴다.
MANIFEST_COLUMNS = [
    "path", "view", "session", "subject", "video", "frame", "split",
    "label", "class_idx", "is_yawn", "label_quality", "video_condition",
    "face_detected", "face_conf", "gender", "glasses", "eyewear",
    "facial_hair",
]


# =====================================================================
# 3. 헬퍼
# =====================================================================
def dataset_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not (p / VIDEOS_CSV_REL).exists():
            raise SystemExit(f"videos.csv 가 없습니다: {p / VIDEOS_CSV_REL}")
        return p
    for c in DATASET_CANDIDATES:
        if (c / VIDEOS_CSV_REL).exists():
            return c.resolve()
    raise SystemExit(
        "YawDD videos.csv 를 찾지 못했습니다. 확인한 위치:\n  "
        + "\n  ".join(str(c / VIDEOS_CSV_REL) for c in DATASET_CANDIDATES)
        + "\n먼저 scripts/prepare_yawdd.py 를 실행하세요.")


def yunet_path() -> Path:
    for p in YUNET_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "YuNet onnx 를 찾지 못했습니다. 확인한 위치:\n  "
        + "\n  ".join(str(p) for p in YUNET_CANDIDATES))


def load_videos(ds: Path, mirror_only: bool = True) -> list[dict]:
    with open(ds / VIDEOS_CSV_REL, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if mirror_only and r["dataset"] != "mirror":
            continue          # Dash 는 라벨이 없어 학습 데이터로 쓰지 않는다
        if not int(r["readable"]):
            continue
        r["session"] = Path(r["path"]).stem
        out.append(r)
    return sorted(out, key=lambda r: r["session"])


class YuNet:
    """build_dmd_yawn_dataset.YuNet 과 같다. 같은 검출기 = 같은 얼굴 기준."""

    def __init__(self, path: Path):
        self._sz = (640, 480)
        self.det = cv2.FaceDetectorYN.create(
            str(path), "", self._sz, YUNET_SCORE, YUNET_NMS, YUNET_TOPK)

    def largest(self, img):
        h, w = img.shape[:2]
        if (w, h) != self._sz:
            self.det.setInputSize((w, h))
            self._sz = (w, h)
        _, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda f: f[2] * f[3])
        return float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])


def reject_outliers(boxes: np.ndarray) -> tuple[np.ndarray, int]:
    """운전자 얼굴이 아닌 검출을 버린다. (남은 박스, 버린 개수)"""
    if len(boxes) < 5:
        return boxes, 0
    w_med = float(np.median(boxes[:, 2]))
    h_med = float(np.median(boxes[:, 3]))
    cx, cy = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
    dist = np.hypot(cx - np.median(cx), cy - np.median(cy))
    keep = (
        (dist <= OUTLIER_DIST * h_med)
        & (boxes[:, 2] >= OUTLIER_SIZE[0] * w_med)
        & (boxes[:, 2] <= OUTLIER_SIZE[1] * w_med)
    )
    if keep.sum() < max(5, 0.5 * len(boxes)):
        return boxes, 0
    return boxes[keep], int((~keep).sum())


def fixed_box(boxes: np.ndarray, img_w: int, img_h: int,
              geom: dict | None = None) -> tuple[int, int, int, int]:
    """검출 박스들 -> 영상 내내 쓸 정사각형 고정 박스 (x1, y1, x2, y2).

    중앙값 박스에 CROP_GEOM 을 적용한다. src/yawn_dataset.yawn_infer_box() 를
    '박스 여러 개의 중앙값'에 적용한 것과 같은 계산이다.

    **한 변을 화면 크기로 자르지 않는다.** YawDD 는 얼굴이 크게 잡혀서 R=1.82 를
    지키려면 640x480 을 넘는 영상이 절반쯤 된다. 거기서 한 변을 480 으로 깎으면
    그 영상만 얼굴이 최대 20% 크게 찍히고(R 1.51~1.82), 그 차이가 피험자와
    붙어 있어서 CNN 이 입 모양 대신 얼굴 크기를 배울 수 있다. 그래서 박스는
    화면 밖으로 나가게 두고, 자를 때 가장자리를 복제해 채운다(crop_with_pad).
    화면 안으로 넣을 수 있으면 밀어 넣으므로, 실제로 채워지는 경우는 많지 않다.
    """
    geom = geom or CROP_GEOM
    x = float(np.median(boxes[:, 0]))
    y = float(np.median(boxes[:, 1]))
    w = float(np.median(boxes[:, 2]))
    h = float(np.median(boxes[:, 3]))

    fcx, fcy = x + w / 2, y + h / 2
    side = h * geom["R"]
    x1 = fcx + geom["dx"] * h - side / 2
    y1 = fcy + geom["dy"] * h - side / 2

    # 들어갈 수 있으면 화면 안으로 민다. 못 들어가면(side > 화면) 중앙에 둔다.
    x1 = min(max(x1, 0), img_w - side) if side <= img_w else (img_w - side) / 2
    y1 = min(max(y1, 0), img_h - side) if side <= img_h else (img_h - side) / 2
    return int(round(x1)), int(round(y1)), int(round(x1 + side)), int(round(y1 + side))


def crop_with_pad(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """박스가 화면 밖으로 나가면 가장자리를 복제해 채운 뒤 자른다.

    검은색으로 채우지 않는 이유: 검은 띠는 '없는 픽셀'이라는 신호가 너무 강해서
    CNN 이 그것으로 영상을 구분할 수 있다. 복제는 차 천장·문짝이 조금 늘어나 보이는
    정도로 끝난다. 얼마나 채웠는지는 crop_boxes.json 의 pad 에 남긴다.
    """
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    l, t = max(0, -x1), max(0, -y1)
    r, b = max(0, x2 - w), max(0, y2 - h)
    if l or t or r or b:
        img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_REPLICATE)
        x1, x2, y1, y2 = x1 + l, x2 + l, y1 + t, y2 + t
    return img[y1:y2, x1:x2]


def iter_frames(video: Path, stride: int, limit: int | None = None):
    """stride 간격 프레임만 디코드해 (frame_idx, image) 를 내놓는다.

    seek 하지 않는다. mpeg4 는 키프레임 간격이 커서 프레임 단위 seek 이 부정확하고
    느리다. grab() 은 디코드를 건너뛰므로 훑는 비용이 싸다.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video}")
    n = 0
    try:
        while True:
            if not cap.grab():
                break
            if n % stride == 0:
                ok, img = cap.retrieve()
                if ok and img is not None:
                    yield n, img
                    if limit and (n // stride) + 1 >= limit:
                        break
            n += 1
    finally:
        cap.release()


def _rel(p: Path, ds: Path) -> str:
    """Dataset 폴더 기준 상대경로. manifest 에 개인 PC 절대경로를 남기지 않는다."""
    return Path(p).resolve().relative_to(ds).as_posix()


# =====================================================================
# 4. --plan : 영상을 읽지 않고 표본 수만
# =====================================================================
def plan(videos: list[dict]) -> None:
    per_split = defaultdict(Counter)
    total = 0
    for v in videos:
        label = LABEL_OF[v["condition"]][0]
        n = int(v["frames_meta"]) // STRIDE[v["condition"]]
        per_split[v["split"]][label] += n
        total += n
    print(f"영상 {len(videos)}개 -> 예상 표본 {total:,}장 (frames_meta 기준 추정)")
    head = f"{'split':<6}" + "".join(f"{c:>14}" for c in CLASS_NAMES) + f"{'합계':>10}"
    print(head)
    for split in ("train", "val", "test"):
        c = per_split[split]
        print(f"{split:<6}" + "".join(f"{c[k]:>14,}" for k in CLASS_NAMES)
              + f"{sum(c.values()):>10,}")


# =====================================================================
# 5. --boxes : 영상별 고정 crop 박스
# =====================================================================
def compute_boxes(videos: list[dict], ds: Path, out_dir: Path) -> dict:
    det = YuNet(yunet_path())
    boxes_out: dict[str, dict] = {}
    t0 = time.time()

    for i, v in enumerate(videos, 1):
        path = ds / v["path"]
        found, confs = [], []
        tried = 0
        img_w = img_h = 0
        for _, img in iter_frames(path, BOX_STRIDE):
            img_h, img_w = img.shape[:2]
            tried += 1
            f = det.largest(img)
            if f:
                found.append(f[:4])
                confs.append(f[4])

        rate = len(found) / tried if tried else 0.0
        rec = {"detections": len(found), "sampled": tried,
               "detect_rate": round(rate, 4),
               "conf_median": round(float(np.median(confs)), 4) if confs else 0.0,
               "frame_size": [img_w, img_h]}

        if len(found) < MIN_DETECTIONS or rate < MIN_DETECT_RATE:
            rec["skipped"] = "얼굴 검출이 너무 적어 고정 박스를 정할 수 없음"
            boxes_out[v["session"]] = rec
            print(f"  [건너뜀] {v['session']} 검출 {len(found)}/{tried}")
            continue

        arr = np.array(found, dtype=np.float64)
        kept, dropped = reject_outliers(arr)
        x1, y1, x2, y2 = fixed_box(kept, img_w, img_h)
        face_h = float(np.median(kept[:, 3]))
        pad = [max(0, -x1), max(0, -y1), max(0, x2 - img_w), max(0, y2 - img_h)]
        rec.update({
            "box": [x1, y1, x2, y2],
            "side": x2 - x1,
            "pad": pad,                       # 좌 상 우 하, 가장자리 복제로 채운 px
            "pad_ratio": round(max(pad) / (x2 - x1), 3),
            "face_w_median": round(float(np.median(kept[:, 2])), 1),
            "face_h_median": round(face_h, 1),
            "side_over_face_h": round((x2 - x1) / face_h, 3),
            "outliers_dropped": dropped,
        })
        boxes_out[v["session"]] = rec

        if i % 40 == 0 or i == len(videos):
            print(f"  {i}/{len(videos)}  {time.time() - t0:.0f}s")

    meta = out_dir / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    with open(meta / "crop_boxes.json", "w", encoding="utf-8") as f:
        json.dump(boxes_out, f, ensure_ascii=False, indent=2)

    ok = [r for r in boxes_out.values() if "box" in r]
    sides = [r["side_over_face_h"] for r in ok]
    rates = [r["detect_rate"] for r in ok]
    pads = [r["pad_ratio"] for r in ok]
    print(f"\n고정 박스 {len(ok)}/{len(boxes_out)}개 확정")
    if ok:
        print(f"  side/face_h : {min(sides):.2f} ~ {max(sides):.2f} "
              f"(중앙값 {float(np.median(sides)):.2f}, 목표 {CROP_GEOM['R']})")
        print(f"  검출률      : {min(rates):.0%} ~ {max(rates):.0%} "
              f"(중앙값 {float(np.median(rates)):.0%})")
        print(f"  가장자리 채움: {sum(1 for p in pads if p > 0)}개 영상 "
              f"(한 변 대비 최대 {max(pads):.0%})")
    print(f"  기록: {meta / 'crop_boxes.json'}")
    return boxes_out


# =====================================================================
# 6. --build : crop 저장 + manifest
# =====================================================================
def build(videos: list[dict], ds: Path, out_dir: Path, boxes: dict,
          png: bool = False) -> list[dict]:
    det = YuNet(yunet_path())
    ext = ".png" if png else ".jpg"
    params = [] if png else [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    rows: list[dict] = []
    blank = 0
    t0 = time.time()

    for i, v in enumerate(videos, 1):
        rec = boxes.get(v["session"], {})
        if "box" not in rec:
            continue
        x1, y1, x2, y2 = rec["box"]
        label, class_idx, is_yawn, quality = LABEL_OF[v["condition"]]
        split = v["split"]
        dest = out_dir / "face" / split / label
        dest.mkdir(parents=True, exist_ok=True)

        for frame_idx, img in iter_frames(ds / v["path"], STRIDE[v["condition"]]):
            crop = crop_with_pad(img, (x1, y1, x2, y2))
            if crop.size == 0 or float(crop.std()) < BLANK_STD:
                blank += 1
                continue
            f = det.largest(img)
            name = f"{v['session']}_f{frame_idx:06d}{ext}"
            cv2.imwrite(str(dest / name), crop, params)
            rows.append({
                "path": _rel(dest / name, ds),
                "view": "face",
                "session": v["session"],
                "subject": v["subject"],
                "video": v["original_path"] or v["session"],
                "frame": frame_idx,
                "split": split,
                "label": label,
                "class_idx": class_idx,
                "is_yawn": is_yawn,
                "label_quality": quality,
                "video_condition": v["condition"],
                "face_detected": int(f is not None),
                "face_conf": round(f[4], 4) if f else 0.0,
                "gender": v["gender"],
                "glasses": v["eyewear"] != "noglasses",
                "eyewear": v["eyewear"],
                "facial_hair": v["facial_hair"],
            })

        if i % 40 == 0 or i == len(videos):
            print(f"  {i}/{len(videos)}  누적 {len(rows):,}장  {time.time() - t0:.0f}s")

    meta = out_dir / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["session"], r["frame"]))
    with open(meta / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\ncrop {len(rows):,}장 (빈 프레임 {blank}장 제외)")
    print(f"  기록: {meta / 'manifest.csv'}")
    return rows


def summarize(rows: list[dict], videos: list[dict], boxes: dict,
              out_dir: Path) -> None:
    by_split = Counter(r["split"] for r in rows)
    by_label = Counter(r["label"] for r in rows)
    by_split_label = defaultdict(Counter)
    for r in rows:
        by_split_label[r["split"]][r["label"]] += 1
    subjects = defaultdict(set)
    for v in videos:
        subjects[v["split"]].add(v["subject"])

    doc = {
        "name": "YawDD yawning frames (mirror set)",
        "source": "YawDD (Yawning Detection Dataset), mirror camera set",
        "task": "frame-level yawning classification",
        "label_source": "파일명의 영상 단위 라벨. 프레임 단위 어노테이션은 원본에 없다",
        "label_warning": (
            "yawn_video / mixed_video 는 약한 라벨이다. 그 영상에 하품이 들어 있다는 "
            "뜻이지 그 프레임이 하품이라는 뜻이 아니다. README §3 을 읽을 것."),
        "classes": list(CLASS_NAMES),
        "class_idx": {n: i for i, n in enumerate(CLASS_NAMES)},
        "label_quality": {
            "strong": "normal/talking 영상 - 모든 프레임이 no_yawn 이다",
            "weak": "yawning/talking&yawning 영상 - 하품 프레임과 아닌 프레임이 섞여 있다",
        },
        "views": {"face": "차량 룸미러 위치 카메라의 전체 화면에서 얼굴만 crop"},
        "sampling": {"stride": STRIDE,
                     "note": "stride 는 프레임 인덱스 기준. 조건마다 다르다"},
        "crop": {
            "detector": f"YuNet(score={YUNET_SCORE}, nms={YUNET_NMS})",
            "method": "영상별 검출 박스 중앙값 + 고정 기하 -> 정사각형 고정 박스",
            "geom": CROP_GEOM,
            "geom_note": ("src/yawn_dataset.INFER_CROP 과 같은 값. DMD 학습 crop 과 "
                          "얼굴 스케일을 맞추기 위한 것이다"),
            "pad": ("박스가 화면 밖으로 나가면 가장자리 복제(BORDER_REPLICATE)로 채운다. "
                    "영상별 채운 양은 crop_boxes.json 의 pad / pad_ratio"),
            "resize": "하지 않음 - 학습 시점에 처리",
        },
        "format": "PNG" if any(r["path"].endswith(".png") for r in rows) else "JPEG q95",
        "split": {
            "by": "subject",
            "note": "같은 사람의 다른 안경 조건(glasses/sunglasses)은 같은 split 에 둔다",
            "subjects": {k: sorted(v) for k, v in sorted(subjects.items())},
        },
        "counts": {
            "total": len(rows),
            "videos_used": len({r["session"] for r in rows}),
            "videos_skipped": sum(1 for r in boxes.values() if "box" not in r),
            "by_split": dict(by_split),
            "by_label": dict(by_label),
            "by_split_label": {k: dict(v) for k, v in by_split_label.items()},
            "strong_labeled": sum(1 for r in rows if r["label_quality"] == "strong"),
            "weak_labeled": sum(1 for r in rows if r["label_quality"] == "weak"),
            "face_not_detected": sum(1 for r in rows if not r["face_detected"]),
        },
        "manifest": f"{OUT_NAME}/metadata/manifest.csv",
        "columns": MANIFEST_COLUMNS,
    }
    with open(out_dir / "metadata" / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print("\n표본 수")
    head = f"{'split':<6}" + "".join(f"{c:>14}" for c in CLASS_NAMES) + f"{'합계':>10}"
    print(head)
    for split in ("train", "val", "test"):
        c = by_split_label[split]
        print(f"{split:<6}" + "".join(f"{c[k]:>14,}" for k in CLASS_NAMES)
              + f"{sum(c.values()):>10,}")
    print(f"  강한 라벨 {doc['counts']['strong_labeled']:,}장 / "
          f"약한 라벨 {doc['counts']['weak_labeled']:,}장")
    print(f"  얼굴 검출 실패 {doc['counts']['face_not_detected']}장 "
          f"(라벨이 아니라 품질 메타다)")
    print(f"  기록: {out_dir / 'metadata' / 'dataset.json'}")


# =====================================================================
# 7. main
# =====================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="영상을 읽지 않고 표본 수만")
    ap.add_argument("--boxes", action="store_true", help="1회차: 고정 crop 박스")
    ap.add_argument("--build", action="store_true", help="2회차: crop 저장 + manifest")
    ap.add_argument("--all", action="store_true", help="--boxes 후 --build")
    ap.add_argument("--one", action="store_true", help="영상 1개만 (점검용)")
    ap.add_argument("--limit", type=int, default=0, help="영상 N개만 (점검용)")
    ap.add_argument("--png", action="store_true", help="PNG 무손실 저장 (기본 JPEG q95)")
    ap.add_argument("--dataset-root", default=None, help="Dataset 폴더 경로")
    a = ap.parse_args()

    if not (a.plan or a.boxes or a.build or a.all):
        ap.error("--plan / --boxes / --build / --all 중 하나를 주세요.")

    ds = dataset_dir(a.dataset_root)
    out_dir = ds / OUT_NAME
    videos = load_videos(ds)
    if a.one:
        videos = videos[:1]
    elif a.limit:
        videos = videos[:a.limit]
    print(f"Dataset : {ds}\n영상    : {len(videos)}개 (mirror, 읽기 가능)")

    if a.plan:
        plan(videos)
        return

    boxes_path = out_dir / "metadata" / "crop_boxes.json"
    if a.boxes or a.all:
        print("\n[1회차] 고정 crop 박스")
        boxes = compute_boxes(videos, ds, out_dir)
    else:
        if not boxes_path.exists():
            raise SystemExit(f"crop_boxes.json 이 없습니다. 먼저 --boxes 를 실행하세요: {boxes_path}")
        with open(boxes_path, encoding="utf-8") as f:
            boxes = json.load(f)

    if a.build or a.all:
        print("\n[2회차] crop 저장")
        rows = build(videos, ds, out_dir, boxes, png=a.png)
        summarize(rows, videos, boxes, out_dir)


if __name__ == "__main__":
    main()
