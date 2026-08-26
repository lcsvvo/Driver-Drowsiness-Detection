"""DMD mosaic 영상 -> 라벨된 얼굴 crop 하품 데이터셋 + manifest.

mosaic(1280x720)은 4분할이다. 좌상=hands, 우상=오버레이 텍스트,
좌하=face(정면), 우하=body(측면). 이 중 face / body 두 셀만 쓴다.
mosaic 프레임 인덱스 = 어노테이션 프레임 인덱스다(오버레이의 Frame 값으로 확인했다).

crop 영역 결정 방식 - 세션 x 뷰마다 고정 박스 하나
    프레임마다 YuNet 으로 얼굴을 찾고, 그 박스들의 합집합에 여백을 더해
    정사각형 하나로 고정한 뒤 모든 프레임에 같은 박스를 쓴다.
    프레임마다 검출 박스를 그대로 쓰지 않는 이유는 두 가지다.
      1. 검출 실패 프레임(측면 뷰에서 발생)을 버리지 않아도 된다. 고정 박스는
         검출 결과와 무관하게 잘리므로, 하품 도중 손에 얼굴이 가려 검출이
         끊긴 프레임 - 바로 우리가 가장 원하는 프레임 - 이 살아남는다.
      2. 박스가 프레임마다 흔들리면 CNN 이 입 모양 변화 대신 크롭 위치 변화를
         학습한다. 고정 박스에서는 화면 안에서 실제로 움직이는 것만 변한다.

2단계로 나눈 이유: 합집합은 그 세션의 모든 표본을 본 뒤에야 정해진다.
    --boxes  영상 1회차. 표본 프레임에서 얼굴만 검출해 박스 통계를 남긴다.
    --build  영상 2회차. 확정된 고정 박스로 잘라 저장하고 manifest 를 쓴다.

출력
    processed/<view>/<split>/<label>/<session>_f<frame:06d>.jpg
    processed/metadata/crop_boxes.json   세션 x 뷰별 고정 박스 + 검출 통계
    processed/metadata/manifest.csv      표본 1장 = 1행
    processed/metadata/dataset.json      클래스/split/전처리 설정 요약

사용 (Dataset 폴더 기준 어디서든)
    python scripts/build_dmd_yawn_dataset.py --plan     # 영상을 읽지 않고 표본 수만
    python scripts/build_dmd_yawn_dataset.py --boxes    # 1회차: 고정 박스 계산
    python scripts/build_dmd_yawn_dataset.py --build    # 2회차: crop 저장
    python scripts/build_dmd_yawn_dataset.py --all      # boxes + build
    python scripts/build_dmd_yawn_dataset.py --all --one   # 첫 영상 1개만 (점검용)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Windows 콘솔 기본 코드페이지(cp949)에서 한글 출력이 깨지지 않게 한다.
# 파일 출력은 항상 encoding="utf-8" 로 명시하므로 여기서 손댈 필요가 없다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import dmd_yawn_gt as gt

# =====================================================================
# 1. 경로
# =====================================================================
DATASET_DIR = _HERE.parent                       # .../Dataset
RAW_DIR = DATASET_DIR / "raw"
VIDEO_DIR = RAW_DIR / "DMD"
ANN_DIR = RAW_DIR / "DMD-annotation"
OUT_DIR = DATASET_DIR / "processed"
META_DIR = OUT_DIR / "metadata"

#: YuNet onnx. 팀 저장소에 이미 있는 것을 그대로 쓴다(같은 검출기 = 같은 얼굴 기준).
YUNET_CANDIDATES = [
    DATASET_DIR.parent / "Driver-Drowsiness-Detection1" / "model" / "detectors" / "face_detection_yunet_2023mar.onnx",
    DATASET_DIR.parent / "Driver-Drowsiness-Detection" / "model" / "detectors" / "face_detection_yunet_2023mar.onnx",
]

# =====================================================================
# 2. mosaic 셀 + crop 파라미터
# =====================================================================
#: (x, y, w, h) - 1280x720 mosaic 기준. 좌하 face / 우하 body.
QUADRANTS = {"face": (0, 360, 640, 360), "body": (640, 360, 640, 360)}

#: YuNet 설정. 02_INFER_YuNet.ipynb / build_dmd_eye_dataset.py 와 같은 값을 쓴다.
YUNET_SCORE, YUNET_NMS, YUNET_TOPK = 0.6, 0.3, 5000

#: 합집합을 낼 때 버릴 극단 분위수(%). 0 이면 순수 합집합.
#: 이상치 제거(reject_outliers) 뒤에 남는 잔여 흔들림을 다듬는 용도라 약하게 둔다.
UNION_PCT = 0.5

#: 이상치 판정 기준 - 오검출 하나가 합집합 전체를 망치는 것을 막는다.
#: 실제로 gB_6 의 body 뷰에서 271개 검출 중 3개가 운전자 얼굴 중앙값에서
#: 229~271px 떨어진 곳(창밖 배경)에 conf 0.65~0.80 으로 잡혔다. 나머지 90%는
#: 17px 이내였는데, 이 3개 때문에 박스가 셀 전체(360x360)로 부풀어 얼굴이
#: 손톱만하게 찍혔다. 분위수만으로는 1% 남짓한 이런 오검출을 걸러내지 못한다.
#:
#: 운전석에 앉은 사람의 머리는 얼굴 높이의 1.5배 넘게 움직이지 않는다 -
#: 그보다 멀리서 잡힌 얼굴은 운전자가 아니다. 크기 기준도 같은 논리다.
OUTLIER_DIST = 1.5      # 중심이 median 에서 (얼굴 높이 median x 이 값) 넘게 떨어지면 버림
OUTLIER_SIZE = (0.5, 2.0)   # 얼굴 폭이 median 대비 이 범위 밖이면 버림

#: 합집합 박스에 더할 여백(박스 크기 대비 비율).
#: 아래쪽을 크게 잡는 이유: 하품하면 턱이 내려가고, 입을 가리는 손은 얼굴 아래에서
#: 올라온다. 얼굴 검출 박스는 손을 포함하지 않으므로 여백으로 확보해야 한다.
MARGIN = {"x": 0.18, "top": 0.15, "bottom": 0.32}

#: 저장 포맷. JPEG q95 는 육안/CNN 모두에서 PNG 와 차이가 없고 용량이 1/5 이다.
#: 압축 아티팩트를 아예 배제하고 싶으면 --png.
JPEG_QUALITY = 95

#: 이보다 픽셀 표준편차가 낮으면 '내용 없는 셀'로 보고 버린다.
#: dmd_yawn_gt.view_ranges 로 이미 걸러지지만, 어떤 세션의 stream 메타가 실제와
#: 다르더라도 검은 사각형이 no_yawn 으로 새어 들어가지 않게 하는 2차 방어다.
BLANK_STD = 3.0

MANIFEST_COLUMNS = [
    "path", "view", "session", "subject", "video", "frame", "split",
    "label", "class_idx", "is_yawn", "eye_state", "is_blink",
    "face_detected", "face_conf", "gender", "age", "glasses",
]


# =====================================================================
# 3. 헬퍼
# =====================================================================
def yunet_path() -> Path:
    for p in YUNET_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "YuNet onnx 를 찾지 못했습니다. 확인한 위치:\n  " +
        "\n  ".join(str(p) for p in YUNET_CANDIDATES))


def ann_files() -> list[Path]:
    fs = sorted(ANN_DIR.glob("*" + gt.JSON_SUFFIX))
    if not fs:
        raise FileNotFoundError(f"어노테이션이 없습니다: {ANN_DIR}")
    return fs


def video_of(base: str) -> Path:
    p = VIDEO_DIR / (base + gt.AVI_SUFFIX)
    if not p.exists():
        raise FileNotFoundError(f"영상이 없습니다: {p}")
    return p


def _rel(p: Path) -> str:
    """Dataset 폴더 기준 상대경로. manifest 에 개인 PC 절대경로를 남기지 않는다."""
    return Path(p).resolve().relative_to(DATASET_DIR).as_posix()


def frames_for_view(sel: list[tuple[int, str]], ranges: dict, view: str) -> dict[int, str]:
    """그 뷰에서 실제로 쓸 수 있는 (frame -> label). 셀이 검은 구간은 뺀다."""
    lo, hi = ranges.get(view, (None, None))
    if lo is None:
        return dict(sel)
    return {f: lab for f, lab in sel if lo <= f <= hi}


def reject_outliers(boxes: np.ndarray) -> tuple[np.ndarray, int]:
    """운전자 얼굴이 아닌 검출을 버린다. (남은 박스, 버린 개수)

    boxes: (N, 4) = x, y, w, h
    중앙값은 이상치가 소수일 때 흔들리지 않으므로, 중앙값을 기준으로 삼는다.
    """
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
    # 전부 걸러지는 일은 없어야 한다. 그런 상황이면 기준이 틀린 것이므로 원본을 쓴다.
    if keep.sum() < max(5, 0.5 * len(boxes)):
        return boxes, 0
    return boxes[keep], int((~keep).sum())


def square_box(boxes: np.ndarray, img_w: int, img_h: int,
               pct: float = UNION_PCT, margin: dict | None = None) -> tuple[int, int, int, int]:
    """검출 박스들 -> 여백 포함 정사각형 고정 박스 (x1, y1, x2, y2).

    boxes: (N, 4) = x, y, w, h. 이상치는 이미 걸러진 상태로 들어온다.
    """
    margin = margin or MARGIN
    x1s, y1s = boxes[:, 0], boxes[:, 1]
    x2s, y2s = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    if pct > 0:
        x1, y1 = np.percentile(x1s, pct), np.percentile(y1s, pct)
        x2, y2 = np.percentile(x2s, 100 - pct), np.percentile(y2s, 100 - pct)
    else:
        x1, y1, x2, y2 = x1s.min(), y1s.min(), x2s.max(), y2s.max()

    w, h = x2 - x1, y2 - y1
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


class YuNet:
    def __init__(self, path: Path):
        self._sz = (640, 360)
        self.det = cv2.FaceDetectorYN.create(
            str(path), "", self._sz, YUNET_SCORE, YUNET_NMS, YUNET_TOPK)

    def largest(self, img):
        """가장 큰 얼굴 1개 (x, y, w, h, conf). 없으면 None."""
        h, w = img.shape[:2]
        if (w, h) != self._sz:
            self.det.setInputSize((w, h))
            self._sz = (w, h)
        _, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda f: f[2] * f[3])
        return float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])


def iter_frames(video: Path, wanted: set[int]):
    """필요한 프레임만 디코드해서 (frame_idx, image) 를 순서대로 내놓는다.

    seek 하지 않고 순차로 훑는 이유: mpeg4 는 키프레임 간격이 커서 프레임 단위
    seek 이 부정확하고 느리다. grab()은 디코드를 건너뛰므로 훑는 비용이 싸다.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video}")
    last = max(wanted) if wanted else -1
    n = 0
    try:
        while n <= last:
            if not cap.grab():
                break
            if n in wanted:
                ok, img = cap.retrieve()
                if ok:
                    yield n, img
            n += 1
    finally:
        cap.release()


# =====================================================================
# 4. --plan : 영상을 읽지 않고 표본 수만
# =====================================================================
def plan(guard: int = gt.GUARD, stride: dict | None = None) -> list[dict]:
    rows = []
    for jp in ann_files():
        base = gt.video_base(jp)
        ol = gt.load_openlabel(jp)
        yawn, _, _ = gt.frame_labels(ol)
        meta = gt.video_meta(ol)
        sel = gt.select_frames(yawn, guard, stride)
        cnt = {c: 0 for c in gt.CLASSES}
        for _, lab in sel:
            cnt[lab] += 1
        gtc = {c: yawn.count(c) for c in gt.CLASSES}
        rows.append(dict(
            video=base, session=gt.safe_id(base), subject=gt.subject_of(base),
            split=gt.split_of(base), frames=meta["ann_frames"],
            glasses=meta["glasses"], gender=meta["gender"],
            gt_no=gtc[gt.NO_YAWN], gt_wo=gtc[gt.YAWN_WITHOUT_HAND], gt_wh=gtc[gt.YAWN_WITH_HAND],
            n_no=cnt[gt.NO_YAWN], n_wo=cnt[gt.YAWN_WITHOUT_HAND], n_wh=cnt[gt.YAWN_WITH_HAND],
        ))
    return rows


def print_plan(rows: list[dict], views: list[str]):
    print(f"{'session':<26}{'split':<7}{'frames':>7}   |"
          f"{'no':>7}{'w/o':>6}{'w/h':>6}   ->{'no':>6}{'w/o':>5}{'w/h':>5}")
    print("-" * 88)
    for r in rows:
        print(f"{r['session']:<26}{r['split']:<7}{r['frames']:>7}   |"
              f"{r['gt_no']:>7}{r['gt_wo']:>6}{r['gt_wh']:>6}   ->"
              f"{r['n_no']:>6}{r['n_wo']:>5}{r['n_wh']:>5}")
    print("-" * 88)
    for sp in ("train", "val", "test", "TOTAL"):
        sel = rows if sp == "TOTAL" else [r for r in rows if r["split"] == sp]
        if not sel:
            continue
        no = sum(r["n_no"] for r in sel)
        wo = sum(r["n_wo"] for r in sel)
        wh = sum(r["n_wh"] for r in sel)
        y = wo + wh
        print(f"{sp:<7} 영상 {len(sel):>2}개  no_yawn {no:>5}  without_hand {wo:>5}  "
              f"with_hand {wh:>5}  | yawn:no_yawn = {y / max(no, 1):.2f}:1  "
              f"뷰당 {no + y:>5}장")
    tot = sum(r["n_no"] + r["n_wo"] + r["n_wh"] for r in rows)
    print(f"\n뷰 {len(views)}개({', '.join(views)}) 합계 예상 {tot * len(views):,}장 "
          f"(고정 박스로 자르므로 검출 실패와 무관하게 이 수가 그대로 나온다)")


# =====================================================================
# 5. --boxes : 1회차, 고정 crop 박스 계산
# =====================================================================
def compute_boxes(anns: list[Path], views: list[str], guard: int,
                  stride: dict | None, verbose: bool = True) -> dict:
    det = YuNet(yunet_path())
    out: dict = {}
    for i, jp in enumerate(anns, 1):
        base = gt.video_base(jp)
        sid = gt.safe_id(base)
        ol = gt.load_openlabel(jp)
        yawn, _, _ = gt.frame_labels(ol)
        ranges = gt.view_ranges(ol)
        sel = gt.select_frames(yawn, guard, stride)
        per_view = {v: frames_for_view(sel, ranges, v) for v in views}
        wanted = set().union(*per_view.values()) if per_view else set()
        t0 = time.time()

        found = {v: [] for v in views}
        for fr, img in iter_frames(video_of(base), wanted):
            for v in views:
                if fr not in per_view[v]:
                    continue          # 그 뷰에서는 셀이 비어 있는 프레임
                x, y, w, h = QUADRANTS[v]
                r = det.largest(img[y:y + h, x:x + w])
                if r is not None:
                    found[v].append(r)

        entry = {}
        for v in views:
            _, _, qw, qh = QUADRANTS[v]
            arr = np.array(found[v], dtype=float) if found[v] else np.zeros((0, 5))
            n_use = len(per_view[v])
            hit = len(arr) / max(n_use, 1)
            if len(arr) < 10:
                raise RuntimeError(
                    f"{sid}/{v}: 얼굴 검출 {len(arr)}건. 고정 박스를 정할 수 없습니다.")
            kept, n_out = reject_outliers(arr[:, :4])
            box = square_box(kept, qw, qh)
            entry[v] = {
                "box": list(box),
                "side": box[2] - box[0],
                "valid_range": list(ranges.get(v, [])),
                "n_sampled": n_use,
                "n_skipped_blank_cell": len(sel) - n_use,
                "n_detected": len(arr),
                "n_outlier_rejected": n_out,
                "hit_rate": round(hit, 4),
                "conf_mean": round(float(arr[:, 4].mean()), 4),
                "face_w_median": round(float(np.median(kept[:, 2])), 1),
                "face_h_median": round(float(np.median(kept[:, 3])), 1),
                # 얼굴이 crop 안에서 차지하는 비율. 1.7~2.2 가 정상 범위다.
                "side_over_face_h": round(float((box[2] - box[0]) / np.median(kept[:, 3])), 2),
            }
        out[sid] = {"video": base, "subject": gt.subject_of(base),
                    "split": gt.split_of(base), "views": entry}
        if verbose:
            s = "  ".join(
                f"{v} side{entry[v]['side']:>3} (얼굴대비 {entry[v]['side_over_face_h']:.1f}x) "
                f"hit{entry[v]['hit_rate'] * 100:.0f}% 이상치{entry[v]['n_outlier_rejected']}"
                for v in views)
            print(f"[{i}/{len(anns)}] {sid:<26} {len(wanted):>4}표본 "
                  f"{time.time() - t0:5.1f}s  {s}")
    return out


# =====================================================================
# 6. --build : 2회차, crop 저장 + manifest
# =====================================================================
def build(anns: list[Path], views: list[str], boxes: dict, guard: int,
          stride: dict | None, png: bool = False, verbose: bool = True) -> Path:
    det = YuNet(yunet_path())
    ext = ".png" if png else ".jpg"
    enc = [] if png else [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    rows: list[dict] = []

    for i, jp in enumerate(anns, 1):
        base = gt.video_base(jp)
        sid = gt.safe_id(base)
        if sid not in boxes:
            raise KeyError(f"{sid} 의 crop 박스가 없습니다. --boxes 를 먼저 실행하세요.")
        ol = gt.load_openlabel(jp)
        yawn, eye, blink = gt.frame_labels(ol)
        meta = gt.video_meta(ol)
        sp = gt.split_of(base)
        sel = gt.select_frames(yawn, guard, stride)
        ranges = gt.view_ranges(ol)
        per_view = {v: frames_for_view(sel, ranges, v) for v in views}
        label_of = dict(sel)
        t0 = time.time()

        # 폴더는 실제로 쓸 것만 만든다.
        for v in views:
            for lab in set(per_view[v].values()):
                (OUT_DIR / v / sp / lab).mkdir(parents=True, exist_ok=True)

        n_saved = 0
        n_blank = 0
        wanted = set().union(*per_view.values()) if per_view else set()
        for fr, img in iter_frames(video_of(base), wanted):
            lab = label_of[fr]
            for v in views:
                if fr not in per_view[v]:
                    continue          # 그 뷰에서는 셀이 비어 있는 프레임
                qx, qy, qw, qh = QUADRANTS[v]
                sub = img[qy:qy + qh, qx:qx + qw]
                x1, y1, x2, y2 = boxes[sid]["views"][v]["box"]
                crop = sub[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                if float(crop.std()) < BLANK_STD:
                    n_blank += 1      # 메타가 놓친 빈 셀. 저장하지 않는다.
                    continue
                # 검출 여부/신뢰도는 라벨이 아니라 품질 메타로만 남긴다.
                # 고정 박스로 자르므로 검출에 실패해도 이미지는 저장된다.
                d = det.largest(sub)
                p = OUT_DIR / v / sp / lab / f"{sid}_f{fr:06d}{ext}"
                cv2.imwrite(str(p), crop, enc)
                rows.append(dict(
                    path=_rel(p), view=v, session=sid, subject=gt.subject_of(base),
                    video=base, frame=fr, split=sp, label=lab,
                    class_idx=gt.CLASS_IDX[lab], is_yawn=gt.IS_YAWN[lab],
                    eye_state=eye[fr], is_blink=blink[fr],
                    face_detected=int(d is not None),
                    face_conf=round(d[4], 4) if d else "",
                    gender=meta["gender"], age=meta["age"], glasses=meta["glasses"],
                ))
                n_saved += 1
        if verbose:
            skipped = sum(len(sel) - len(per_view[v]) for v in views) + n_blank
            note = f"  (빈 셀 {skipped}장 제외)" if skipped else ""
            print(f"[{i}/{len(anns)}] {sid:<26} {len(sel):>4}프레임 x {len(views)}뷰 "
                  f"= {n_saved:>5}장  {time.time() - t0:5.1f}s{note}")

    META_DIR.mkdir(parents=True, exist_ok=True)
    mp = META_DIR / "manifest.csv"
    with open(mp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    with open(META_DIR / "dataset.json", "w", encoding="utf-8") as f:
        json.dump({
            "name": "DMD yawning frames",
            "source": "DMD (Driver Monitoring Dataset) mosaic + OpenLABEL drowsiness annotation",
            "task": "frame-level yawning classification",
            "label_source": "DMD 공식 어노테이션의 yawning/* action (수동 판정이 아니다)",
            "classes": list(gt.CLASSES),
            "class_idx": gt.CLASS_IDX,
            "binary": {"column": "is_yawn", "0": gt.NO_YAWN,
                       "1": gt.YAWN_WITHOUT_HAND + " + " + gt.YAWN_WITH_HAND},
            "views": {v: {"quadrant": list(QUADRANTS[v]),
                          "desc": "좌하 정면 얼굴" if v == "face" else "우하 측면 상반신"}
                      for v in views},
            "sampling": {
                "stride": stride or gt.STRIDE, "guard_frames": guard,
                "note": "stride 는 프레임 인덱스가 아니라 클래스 내 등장 순번 기준",
                "guard_note": "하품 구간 경계 +-guard 프레임은 어느 클래스로도 뽑지 않는다",
                "blank_cell": "카메라별 frame_shift 때문에 mosaic 앞뒤로 셀이 검은 "
                              "구간이 있다. 뷰별 유효 구간 밖 프레임은 그 뷰에서 제외했다",
            },
            "crop": {"detector": f"YuNet(score={YUNET_SCORE}, nms={YUNET_NMS})",
                     "method": "세션x뷰별 검출 박스 합집합 + 여백 -> 정사각형 고정 박스",
                     "union_percentile": UNION_PCT, "margin": MARGIN,
                     "resize": "하지 않음 - 학습 시점에 처리"},
            "format": "PNG" if png else f"JPEG q{JPEG_QUALITY}",
            "split": {"by": "subject", "map": gt.DMD_SPLIT},
            "counts": summary,
            "manifest": _rel(mp),
            "columns": MANIFEST_COLUMNS,
        }, f, ensure_ascii=False, indent=2)
    return mp


def summarize(rows: list[dict]) -> dict:
    out: dict = {"total": len(rows)}
    for key in ("view", "split", "label"):
        d: dict = {}
        for r in rows:
            d[r[key]] = d.get(r[key], 0) + 1
        out["by_" + key] = d
    grid: dict = {}
    for r in rows:
        grid.setdefault(r["split"], {}).setdefault(r["label"], 0)
        grid[r["split"]][r["label"]] += 1
    out["by_split_label"] = grid
    out["face_not_detected"] = sum(1 for r in rows if not r["face_detected"])
    return out


# =====================================================================
# 7. CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="영상을 읽지 않고 표본 수만 계산")
    ap.add_argument("--boxes", action="store_true", help="1회차: 고정 crop 박스 계산")
    ap.add_argument("--build", action="store_true", help="2회차: crop 저장 + manifest")
    ap.add_argument("--all", action="store_true", help="--boxes 후 --build")
    ap.add_argument("--one", action="store_true", help="첫 영상 1개만 (점검용)")
    ap.add_argument("--views", default="face,body", help="쓸 뷰 (기본 face,body)")
    ap.add_argument("--guard", type=int, default=gt.GUARD,
                    help=f"경계 제외 프레임 (기본 {gt.GUARD})")
    ap.add_argument("--stride-yawn", type=int, default=gt.STRIDE[gt.YAWN_WITH_HAND])
    ap.add_argument("--stride-no", type=int, default=gt.STRIDE[gt.NO_YAWN])
    ap.add_argument("--png", action="store_true", help="PNG 무손실 저장 (기본 JPEG q95)")
    a = ap.parse_args()

    if not (a.plan or a.boxes or a.build or a.all):
        ap.error("--plan / --boxes / --build / --all 중 하나를 지정하세요.")

    views = [v.strip() for v in a.views.split(",") if v.strip()]
    bad = [v for v in views if v not in QUADRANTS]
    if bad:
        ap.error(f"알 수 없는 뷰: {bad} (가능: {list(QUADRANTS)})")
    stride = {gt.NO_YAWN: a.stride_no,
              gt.YAWN_WITHOUT_HAND: a.stride_yawn,
              gt.YAWN_WITH_HAND: a.stride_yawn}

    anns = ann_files()
    if a.one:
        anns = anns[:1]

    if a.plan:
        print_plan(plan(a.guard, stride), views)
        return

    bp = META_DIR / "crop_boxes.json"
    if a.boxes or a.all:
        print(f"== 1회차: 고정 crop 박스 계산 ({len(anns)}개 영상) ==")
        boxes = compute_boxes(anns, views, a.guard, stride)
        META_DIR.mkdir(parents=True, exist_ok=True)
        # 일부만 돌렸을 때 기존 결과를 지우지 않도록 병합한다.
        old = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else {}
        old.update(boxes)
        bp.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
        print("-> " + _rel(bp))

    if a.build or a.all:
        if not bp.exists():
            ap.error(f"{_rel(bp)} 가 없습니다. --boxes 를 먼저 실행하세요.")
        boxes = json.loads(bp.read_text(encoding="utf-8"))
        print(f"\n== 2회차: crop 저장 ({len(anns)}개 영상 x {len(views)}뷰) ==")
        mp = build(anns, views, boxes, a.guard, stride, a.png)
        print("-> " + _rel(mp))
        s = json.loads((META_DIR / "dataset.json").read_text(encoding="utf-8"))["counts"]
        print(f"\n총 {s['total']:,}장  {s['by_label']}")
        print(f"split별: {s['by_split_label']}")


if __name__ == "__main__":
    main()
