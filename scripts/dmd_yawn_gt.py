"""DMD OpenLABEL(VCD 4.x) 어노테이션 -> 하품 프레임 GT + 샘플 프레임 선정.

이 모듈은 영상을 열지 않는다. JSON 만 읽어서 "어느 프레임이 어떤 클래스인가"와
"그 중 어떤 프레임을 데이터셋에 담을 것인가"만 결정한다. 영상 디코딩(무거움)과
분리해 두면 --plan 으로 표본 수와 클래스 균형을 몇 초 만에 확인할 수 있다.

Driver-Drowsiness-Detection/src/dmd_annotation.py 의 파싱 규칙과 같은 결과를 낸다
(같은 JSON, 같은 action type 이름). 눈(eye) 데이터셋과 하품 데이터셋이 같은
프레임 인덱스 체계를 쓰므로, 나중에 frame 컬럼으로 두 데이터셋을 조인할 수 있다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# =====================================================================
# 1. 라벨 정의
# =====================================================================
#: 3-class. 폴더 이름이자 manifest 의 label 값.
#: with_hand 를 따로 두는 이유: 손이 입을 가리면 화면상 '입 벌림'이 보이지 않아
#: 시각적으로 전혀 다른 문제다. 합쳐서 학습할지는 is_yawn(binary)으로 선택한다.
NO_YAWN = "no_yawn"
YAWN_WITH_HAND = "yawn_with_hand"
YAWN_WITHOUT_HAND = "yawn_without_hand"
CLASSES = (NO_YAWN, YAWN_WITHOUT_HAND, YAWN_WITH_HAND)
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

#: label -> 이진 라벨. 2-class 로 쓰고 싶을 때 manifest 의 is_yawn 컬럼만 쓰면 된다.
IS_YAWN = {NO_YAWN: 0, YAWN_WITHOUT_HAND: 1, YAWN_WITH_HAND: 1}

EYE_SHORT = {
    "eyes_state/open": "open",
    "eyes_state/close": "close",
    "eyes_state/opening": "opening",
    "eyes_state/closing": "closing",
    "eyes_state/undefined": "undefined",
}

JSON_SUFFIX = "_rgb_ann_drowsiness.json"
AVI_SUFFIX = "_rgb_mosaic.avi"

# =====================================================================
# 2. split — subject 단위 고정 매핑
# =====================================================================
#: Driver-Drowsiness-Detection/src/build_dmd_eye_dataset.py 의 DMD_SPLIT 과 동일하다.
#: 같은 값을 쓰는 이유: 눈 데이터셋에서 test 였던 사람이 하품 데이터셋에서 train 이면
#: 두 모델을 합쳐 평가할 때 test 피험자가 이미 학습에 노출된 셈이 된다.
#: 같은 사람의 2세션(gB_10, gF_23, gZ_33)은 반드시 같은 split.
DMD_SPLIT: dict[str, str] = {
    # train — 7명 / 9영상
    "gA_1": "train",    # 안경
    "gB_7": "train",
    "gB_9": "train",
    "gB_10": "train",   # 2세션
    "gC_14": "train",
    "gZ_33": "train",   # 2세션
    "gZ_36": "train",   # 안경
    # val — 2명 / 3영상
    "gA_5": "val",
    "gF_23": "val",     # 2세션
    # test — 4명 / 4영상
    "gE_29": "test",    # 안경
    "gC_13": "test",
    "gB_6": "test",
    "gZ_37": "test",
}

# =====================================================================
# 3. 샘플링 규칙
# =====================================================================
#: 클래스별 stride. n 이면 그 클래스에서 n 프레임마다 1장.
#: 하품 12,427 프레임 / 하품 아님 75,783 프레임으로 약 1:6 불균형이라, 하품을
#: 촘촘히(5) 비하품을 듬성듬성(30) 뽑아 최종 1:1 에 가깝게 맞춘다.
STRIDE = {NO_YAWN: 30, YAWN_WITHOUT_HAND: 5, YAWN_WITH_HAND: 5}

#: 하품 구간 경계에서 no_yawn 으로 뽑지 않을 여유 프레임 수.
#: 하품 시작 직전/직후 프레임은 입이 이미 열리는 중이라 "하품 아님"으로 학습시키면
#: 정답이 모순된다. 경계 ±GUARD 프레임은 어느 클래스로도 뽑지 않는다.
GUARD = 5


# =====================================================================
# 4. 파싱
# =====================================================================
def load_openlabel(json_path: Path | str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)["openlabel"]


def video_base(path: Path | str) -> str:
    """어노테이션/영상 경로 -> 공통 basename (gA_1_s5_2019-03-14T14;26;17+01;00)."""
    return Path(path).name.replace(JSON_SUFFIX, "").replace(AVI_SUFFIX, "")


def safe_id(base: str) -> str:
    """파일명에 쓸 수 있는 세션 ID. 콜론/세미콜론/타임존 기호를 없앤다.

    gA_1_s5_2019-03-14T14;26;17+01;00 -> gA_1_s5_20190314T142617
    """
    m = re.match(r"(.+?_s\d+)_(\d{4})-(\d{2})-(\d{2})T(\d{2});(\d{2});(\d{2})", base)
    if not m:
        return re.sub(r"[^A-Za-z0-9_]", "", base)
    g = m.groups()
    return f"{g[0]}_{g[1]}{g[2]}{g[3]}T{g[4]}{g[5]}{g[6]}"


def subject_of(base: str) -> str:
    return base.split("_s5")[0]


def split_of(base: str) -> str:
    su = subject_of(base)
    if su not in DMD_SPLIT:
        raise KeyError(f"DMD_SPLIT 에 없는 subject: {su}")
    return DMD_SPLIT[su]


def video_meta(ol: dict) -> dict:
    """피험자 속성. 안경 여부는 나중에 오분류 분석에 쓴다."""
    obj = ol["objects"]["0"]["object_data"]

    def _get(kind, name):
        return next((x["val"] for x in obj.get(kind, []) if x["name"] == name), None)

    ctx = next(iter(ol.get("contexts", {}).values()), {}).get("context_data", {})

    def _c(name):
        return next((x["val"] for x in ctx.get("text", []) if x["name"] == name), None)

    return {
        "ann_frames": ol["frame_intervals"][0]["frame_end"] + 1,
        "gender": _get("text", "gender"),
        "age": _get("num", "age"),
        "glasses": _get("boolean", "glasses"),
        "weather": _c("weather"),
        "setup": _c("setup"),
    }


def frame_labels(ol: dict) -> tuple[list[str], list[str], list[int]]:
    """프레임별 (하품 라벨, 눈 상태, 깜빡임) 배열을 만든다.

    반환 길이는 모두 ann_frames. 하품 라벨은 CLASSES 중 하나다.
    """
    n = ol["frame_intervals"][0]["frame_end"] + 1
    yawn = [NO_YAWN] * n
    eye = ["none"] * n
    blink = [0] * n

    for a in ol["actions"].values():
        t = a["type"]
        for fi in a["frame_intervals"]:
            s, e = fi["frame_start"], min(fi["frame_end"], n - 1)
            if t.startswith("yawning/"):
                lab = YAWN_WITH_HAND if "with hand" in t else YAWN_WITHOUT_HAND
                for fr in range(s, e + 1):
                    yawn[fr] = lab
            elif t in EYE_SHORT:
                for fr in range(s, e + 1):
                    eye[fr] = EYE_SHORT[t]
            elif t == "blinks/blinking":
                for fr in range(s, e + 1):
                    blink[fr] = 1
    return yawn, eye, blink


def guard_mask(yawn: list[str], guard: int = GUARD) -> list[bool]:
    """클래스가 바뀌는 경계 ±guard 프레임을 True 로 표시(=제외 대상)."""
    n = len(yawn)
    mask = [False] * n
    if guard <= 0:
        return mask
    for i in range(1, n):
        if yawn[i] != yawn[i - 1]:
            for j in range(max(0, i - guard), min(n, i + guard)):
                mask[j] = True
    return mask


def select_frames(yawn: list[str], guard: int = GUARD,
                  stride: dict[str, int] | None = None) -> list[tuple[int, str]]:
    """(frame, label) 샘플 목록. 클래스별 '등장 순서' 기준으로 stride 를 센다.

    프레임 인덱스로 세지 않는 이유: 하품 구간은 5초 남짓으로 짧아서 인덱스 기준
    stride 를 쓰면 어떤 하품은 통째로 빠지고 어떤 하품만 여러 장 뽑힌다.
    클래스 내 순번으로 세면 모든 하품 구간에서 고르게 뽑힌다.
    """
    stride = stride or STRIDE
    skip = guard_mask(yawn, guard)
    seen = {c: 0 for c in CLASSES}
    out = []
    for fr, lab in enumerate(yawn):
        if skip[fr]:
            continue
        if seen[lab] % stride[lab] == 0:
            out.append((fr, lab))
        seen[lab] += 1
    return out


# =====================================================================
# 5. 뷰별 유효 프레임 구간
# =====================================================================
#: 뷰 이름 -> OpenLABEL streams 키.
VIEW_STREAM = {"face": "face_camera", "body": "body_camera", "hands": "hands_camera"}


def view_ranges(ol: dict) -> dict[str, tuple[int, int]]:
    """뷰별로 실제 영상이 들어 있는 mosaic 프레임 구간 (lo, hi) 을 반환한다(양끝 포함).

    카메라 3대는 녹화 시작 시각이 조금씩 다르다. mosaic 은 이 차이를 frame_shift 로
    맞춰 합성했기 때문에, 늦게 시작한 카메라의 셀은 앞부분이 '완전한 검은 화면'이다.
    예: gA_1 의 body 는 shift 54 -> mosaic 프레임 0~53 의 우하 셀이 새까맣다.
    (hands 는 74, face 는 0. 실제로 픽셀 std<3 인 구간과 정확히 일치하는 것을 확인했다)

    끝쪽도 마찬가지다. face 는 total_frames 5480 이라 mosaic 마지막 프레임 5480 이
    범위를 벗어나 검다. 그래서 hi = shift + total_frames - 1 로 잡는다.

    이 구간을 벗어난 프레임을 그대로 저장하면 '검은 사각형'이 no_yawn 으로 학습된다.
    """
    out = {}
    for view, key in VIEW_STREAM.items():
        sp = ol.get("streams", {}).get(key, {}).get("stream_properties", {})
        total = sp.get("total_frames")
        shift = sp.get("sync", {}).get("frame_shift", 0) or 0
        if total is None:
            continue
        out[view] = (int(shift), int(shift) + int(total) - 1)
    return out
