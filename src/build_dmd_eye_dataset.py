"""DMD mosaic 영상 -> 라벨된 눈 crop 데이터셋 + DMD manifest.

역할은 하나다: 프레임 단위 GT(10_eye_gt_dmd 산출물)를 이용해 DMD 영상에서
눈 crop 을 뽑고, MRL manifest 와 같은 스키마의 CSV 를 만든다.
학습·평가·통합은 여기서 하지 않는다(11/12 노트북 담당).

흐름 (프레임마다):
    mosaic 프레임 -> 좌하 셀(정면 얼굴 타일) -> YuNet 얼굴검출 -> 양쪽 눈 crop
    -> 그 프레임의 eye_gt_binary 로 라벨 (close=1 -> Closed, open=0 -> Open)
    -> eye_gt_binary == -1 (opening/closing/undefined/none) 은 제외

설계 결정 3개 (근거는 11_eye_dataset_build 의 결정 박스에 수치와 함께 기록)
    1. split 은 subject 단위 고정 매핑(DMD_SPLIT). 같은 사람의 2세션은 같은 split.
    2. train/val 은 '클래스별 비대칭 stride'. Closed 는 희소해서 촘촘히(2),
       Open 은 풍부해서 듬성듬성(12) 뽑아 클래스 균형을 맞춘다.
    3. test 는 두 클래스에 '같은' stride(4). 실전 Closed 비율(약 11%)을 보존해야
       Precision 과 PERCLOS 가 왜곡되지 않는다.

출력
    outputs/dmd_eye/<split>/<Closed|Open>/<video>_<frame:06d>_<L|R>.png
        원본 눈 crop(가변 크기). resize/gray/sharpen 을 하지 않는다 —
        MRL 과 똑같이 학습 시점에 preprocess_eye 가 처리해야 학습·추론이 일치한다.
    outputs/dmd_eye/dmd_eye_manifest.csv
        columns: path, source, subject, video, frame, side, label, class_idx,
                 glasses, split, face_conf
        path 는 '저장소 루트 기준 상대경로'. 절대경로를 쓰지 않는 이유는 팀원 4명의
        PC 경로가 다르고, CSV 에 개인 계정명이 남기 때문이다.

YuNet 설정은 02_INFER_YuNet.ipynb 의 Drowsiness_Detector 와 동일하다
(score 0.6 / nms 0.3 / top_k 5000, 가장 큰 얼굴, eye_size = max(face_w*0.22, 20)).
02 는 수정 금지 대상이므로 값을 옮겨 적었고, 바꾸지 않는다.

사용 (저장소 루트에서)
    python src/build_dmd_eye_dataset.py --plan          # 영상을 읽지 않고 예상 표본 수만 계산
    python src/build_dmd_eye_dataset.py --one           # 첫 영상 1개만 (동작 점검)
    python src/build_dmd_eye_dataset.py --all           # 16개 전부
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent             # src/
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))                  # dmd_crop 등 같은 폴더 모듈
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))           # 저장소 루트의 config

import config
from dmd_crop import bottomleft_cell

# =====================================================================
# 1. split 정의 — subject 단위 고정
# =====================================================================
#: DMD 13명. 같은 사람의 2세션(gB_10, gF_23, gZ_33)은 반드시 같은 split.
#: 안경 착용자는 gA_1, gE_29, gZ_36 3명뿐이라 train 2 / test 1 로 나눈다.
#: (기존 TEST_SUBJECTS 는 test 에 안경 착용자가 0명이어서 안경 일반화를 측정할 수 없었다)
DMD_SPLIT: dict[str, str] = {
    # train — 7명 / 9영상
    "gA_1":  "train",   # 안경
    "gB_7":  "train",
    "gB_9":  "train",
    "gB_10": "train",   # 2세션
    "gC_14": "train",
    "gZ_33": "train",   # 2세션
    "gZ_36": "train",   # 안경
    # val — 2명 / 3영상 (모델 선택 전용)
    "gA_5":  "val",
    "gF_23": "val",     # 2세션
    # test — 4명 / 4영상 (학습에 쓰지 않음, 성별 2:2 / 안경 1명)
    "gE_29": "test",    # 안경
    "gC_13": "test",
    "gB_6":  "test",
    "gZ_37": "test",
}

#: split x 클래스별 stride. 값이 n 이면 그 클래스에서 n 프레임마다 1장 취한다.
#: 프레임 인덱스가 아니라 '클래스별 등장 순서'로 센다 — 인덱스 기준으로 세면
#: Closed 구간이 짧아 특정 구간만 통째로 빠지거나 남는다.
STRIDE: dict[str, dict[str, int]] = {
    "train": {"Closed": 2, "Open": 12},
    "val":   {"Closed": 2, "Open": 12},
    "test":  {"Closed": 4, "Open": 4},   # 균일 — 실전 클래스 비율 보존
}

LABEL_OF = {1: "Closed", 0: "Open"}
CLASS_IDX = {"Closed": 0, "Open": 1}     # MRL·EyeCNN 과 동일 (class0 = Closed)

MANIFEST_COLUMNS = ["path", "source", "subject", "video", "frame", "side",
                    "label", "class_idx", "glasses", "split", "face_conf"]

JSON_SUFFIX = "_rgb_ann_drowsiness.json"
AVI_SUFFIX = "_rgb_mosaic.avi"


# =====================================================================
# 2. 경로·메타 헬퍼
# =====================================================================
#: 데이터셋 폴더를 찾을 후보 위치. 앞쪽이 우선.
#: config.py 에는 DATA_DIR 까지만 있고 데이터셋 단위 경로 변수가 없다. 그래서
#: 레이아웃(data/raw/<name> 과 data/<name>)을 둘 다 받아들여, 폴더를 옮겨도
#: 노트북마다 경로를 고쳐 다니지 않게 한다. 해결책이 아니라 완충 장치다 —
#: 항구적으로는 config.py 에 데이터셋 경로 변수를 두는 편이 맞다.
_DATA_LAYOUTS = ("raw/{name}", "{name}")


def data_subdir(name: str) -> Path:
    """`data` 아래에서 데이터셋 폴더를 찾아 반환. 없으면 후보를 모두 적어 예외."""
    tried = []
    for layout in _DATA_LAYOUTS:
        p = config.DATA_DIR / layout.format(name=name)
        if p.is_dir():
            return p
        tried.append(config._rel(p))
    raise FileNotFoundError(
        f"'{name}' 데이터 폴더를 찾지 못했습니다. 확인한 위치: {tried}")


def yunet_path() -> Path:
    """YuNet onnx 경로.

    config.YUNET_MODEL 이 있으면 그것을 쓴다. RealTimeDetection(w.YuNet) 브랜치를
    아직 병합하지 않은 상태에서도 노트북이 AttributeError 로 죽지 않게 fallback 을 둔다.
    """
    p = getattr(config, "YUNET_MODEL", None)
    if p is None:
        p = config.PROJECT_ROOT / "model" / "detectors" / "face_detection_yunet_2023mar.onnx"
    return Path(p)


def dmd_dir() -> Path:
    return data_subdir("DMD")


def dmd_jsons() -> list[Path]:
    """DMD annotation JSON 16개를 이름순으로."""
    return sorted(dmd_dir().glob(f"*{JSON_SUFFIX}"))


def video_base(json_path: Path | str) -> str:
    return Path(json_path).name.replace(JSON_SUFFIX, "")


def subject_of(json_path: Path | str) -> str:
    return video_base(json_path).split("_s5")[0]


def split_of(json_path: Path | str) -> str:
    su = subject_of(json_path)
    if su not in DMD_SPLIT:
        raise KeyError(f"DMD_SPLIT 에 없는 subject: {su}")
    return DMD_SPLIT[su]


def gt_dir() -> Path:
    return config.OUTPUTS_DIR / "dmd_gt"


def load_glasses() -> dict[str, bool]:
    """영상별 안경 여부를 _summary.csv 에서 읽는다.

    JSON 을 다시 파싱하지 않는 이유: 10_eye_gt_dmd 가 이미 만든 표를 재사용하면
    두 노트북이 같은 값을 두 번 계산하는 중복이 사라진다.
    """
    p = gt_dir() / "_summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} 가 없습니다. 10_eye_gt_dmd 를 먼저 실행하세요.")
    with open(p, encoding="utf-8") as f:
        return {r["video"]: str(r["glasses"]).strip().lower() == "true"
                for r in csv.DictReader(f)}


def load_frame_gt(base: str) -> dict[int, int]:
    """<video>_frame_gt.csv -> {frame: eye_gt_binary}."""
    p = gt_dir() / f"{base}_frame_gt.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} 가 없습니다. 10_eye_gt_dmd 를 먼저 실행하세요.")
    with open(p, encoding="utf-8") as f:
        return {int(r["frame"]): int(r["eye_gt_binary"]) for r in csv.DictReader(f)}


def _rel_to_root(p: Path) -> str:
    """저장소 루트 기준 POSIX 상대경로. CSV 에 개인 PC 절대경로를 남기지 않는다."""
    return Path(p).resolve().relative_to(config.PROJECT_ROOT).as_posix()


# =====================================================================
# 3. 예상 표본 수 (영상을 읽지 않는다)
# =====================================================================
def plan(jsons: list[Path] | None = None) -> list[dict]:
    """GT CSV 만 읽어 split x 클래스별 '상한' 표본 수를 계산한다.

    무거운 영상 디코딩 전에 표본 수와 클래스 균형을 먼저 확인하기 위한 함수다.
    실제 결과는 YuNet 얼굴검출 실패분만큼 이보다 적다.
    """
    jsons = jsons or dmd_jsons()
    rows = []
    for jp in jsons:
        base = video_base(jp)
        sp = split_of(jp)
        gt = load_frame_gt(base)
        n_closed = sum(1 for b in gt.values() if b == 1)
        n_open = sum(1 for b in gt.values() if b == 0)
        sc, so = STRIDE[sp]["Closed"], STRIDE[sp]["Open"]
        rows.append(dict(
            video=base, subject=subject_of(jp), split=sp,
            gt_closed=n_closed, gt_open=n_open,
            gt_excluded=sum(1 for b in gt.values() if b == -1),
            # 프레임 1개당 눈 2개(L, R)
            crops_closed=-(-n_closed // sc) * 2,
            crops_open=-(-n_open // so) * 2,
        ))
    return rows


# =====================================================================
# 4. YuNet — 02_INFER_YuNet 과 동일 설정
# =====================================================================
class YuNetDetector:
    """얼굴 검출 전용. 눈 상태 판정은 하지 않는다(그건 Eye CNN 의 역할).

    02_INFER_YuNet.ipynb 의 Drowsiness_Detector 에서 검출·눈추출만 떼어낸 것이다.
    눈 crop 은 resize/sharpen 하지 않은 원본(raw BGR)으로 반환한다.
    """

    def __init__(self, yunet=None, score=0.6, nms=0.3, top_k=5000):
        path = Path(yunet) if yunet else yunet_path()
        if not path.exists():
            raise FileNotFoundError(
                f"YuNet 모델이 없습니다: {config._rel(path)}\n"
                "  RealTimeDetection(w.YuNet) 브랜치를 병합하면 "
                "model/detectors/face_detection_yunet_2023mar.onnx 와 "
                "config.YUNET_MODEL 이 함께 들어옵니다."
            )
        self._sz = (320, 240)
        self.det = cv2.FaceDetectorYN.create(str(path), "", self._sz, score, nms, top_k)

    def detect_face(self, img):
        """가장 큰 얼굴 1개의 YuNet 원시 출력(15값). 없으면 None."""
        h, w = img.shape[:2]
        if (w, h) != self._sz:
            self.det.setInputSize((w, h))            # 입력 크기가 바뀌면 알려줘야 한다
            self._sz = (w, h)
        _, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def eye_crops(self, img, f):
        """눈 키포인트 중심 정사각 crop 2개(raw BGR) + face confidence.

        f[4:6] = 인물 기준 오른쪽 눈 = 이미지 왼쪽. 이미지 좌표계 기준으로
        L/R 이름을 붙인다(02 의 _to_mtcnn_format 과 같은 규칙).
        """
        size = max(int(f[2] * 0.22), 20)             # 02 와 동일: 얼굴 폭의 22%
        h, w = img.shape[:2]
        out = []
        for side, ex, ey in (("L", int(f[4]), int(f[5])), ("R", int(f[6]), int(f[7]))):
            x1, y1 = max(ex - size, 0), max(ey - size, 0)
            x2, y2 = min(ex + size, w), min(ey + size, h)
            c = img[y1:y2, x1:x2]
            if c.size == 0:
                return None
            out.append((side, c))
        return out, float(f[14])


# =====================================================================
# 5. 데이터셋 생성
# =====================================================================
def build(videos, det: YuNetDetector, out_dir: Path, verbose: bool = True) -> Path | None:
    """videos(JSON 경로 리스트)에서 눈 crop 을 뽑고 manifest CSV 를 쓴다.

    반환: manifest 경로. crop 이 하나도 안 나오면 None.
    """
    out_dir = Path(out_dir)
    glasses = load_glasses()
    rows: list[dict] = []
    stats: list[dict] = []

    for jp in videos:
        base = video_base(jp)
        subject = subject_of(jp)
        sp = split_of(jp)
        gt = load_frame_gt(base)
        stride = STRIDE[sp]

        avi = Path(str(jp).replace(JSON_SUFFIX, AVI_SUFFIX))
        cap = cv2.VideoCapture(str(avi))
        if not cap.isOpened():
            print(f"[skip] 영상을 열 수 없습니다: {_rel_to_root(avi)}")
            continue

        seen = {"Closed": 0, "Open": 0}      # 클래스별 등장 순서 (stride 기준)
        kept = {"Closed": 0, "Open": 0}      # 실제 저장한 crop 수
        no_face = 0
        i = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            b = gt.get(i, -1)
            if b == -1:                       # 전이(opening/closing)·미라벨 제외
                continue
            label = LABEL_OF[b]
            if seen[label] % stride[label]:   # 클래스별 stride
                seen[label] += 1
                continue
            seen[label] += 1

            tile = bottomleft_cell(frame)     # mosaic 좌하 = 정면 얼굴 타일
            f = det.detect_face(tile)
            if f is None:
                no_face += 1
                continue
            res = det.eye_crops(tile, f)
            if res is None:
                no_face += 1
                continue
            crops, conf = res

            sub = out_dir / sp / label
            sub.mkdir(parents=True, exist_ok=True)
            for side, crop in crops:
                fn = f"{base}_{i:06d}_{side}.png"
                cv2.imwrite(str(sub / fn), crop)
                rows.append(dict(
                    path=_rel_to_root(sub / fn), source="dmd", subject=subject,
                    video=base, frame=i, side=side, label=label,
                    class_idx=CLASS_IDX[label], glasses=int(glasses.get(base, False)),
                    split=sp, face_conf=round(conf, 3),
                ))
                kept[label] += 1
        cap.release()

        stats.append(dict(video=base, split=sp, closed=kept["Closed"],
                          open=kept["Open"], no_face=no_face))
        if verbose:
            print(f"{base}  [{sp:5}] Closed {kept['Closed']:5d} / Open {kept['Open']:5d}"
                  f"  (검출실패 {no_face})")

    if not rows:
        print("생성된 crop 이 없습니다.")
        return None

    man = out_dir / "dmd_eye_manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    if verbose:
        print("\n=== dmd_eye 요약 ===")
        for sp in ("train", "val", "test"):
            c = sum(s["closed"] for s in stats if s["split"] == sp)
            o = sum(s["open"] for s in stats if s["split"] == sp)
            n = c + o
            nsub = len({r["subject"] for r in rows if r["split"] == sp})
            if n:
                print(f"  {sp:5}: {n:6d} crops | Closed {c:5d} ({c/n*100:4.1f}%) "
                      f"/ Open {o:5d} | subjects {nsub}")
        print("저장:", _rel_to_root(man))
    return man


# =====================================================================
# 6. CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--yunet", default=None, help="기본값은 config.YUNET_MODEL")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="영상을 읽지 않고 예상 표본 수만")
    g.add_argument("--one", action="store_true", help="첫 영상 1개만")
    g.add_argument("--all", action="store_true", help="16개 전부")
    args = ap.parse_args()

    jsons = dmd_jsons()
    if not jsons:
        raise SystemExit(f"DMD annotation JSON 을 찾지 못했습니다: "
                         f"{config._rel(dmd_dir())}")

    if args.plan:
        for r in plan(jsons):
            print(r)
        return

    out_dir = Path(args.out_dir) if args.out_dir else config.OUTPUTS_DIR / "dmd_eye"
    videos = jsons[:1] if args.one else jsons
    build(videos, YuNetDetector(args.yunet), out_dir)


if __name__ == "__main__":
    main()
