"""DMD OpenLABEL(VCD 4.x) -> 프레임 단위 GT 테이블 + mosaic 정렬 검증. (검증용 스크립트)"""
from __future__ import annotations
import json, os, glob, subprocess
from collections import defaultdict

EYE_TYPES = ["eyes_state/open", "eyes_state/close",
             "eyes_state/opening", "eyes_state/closing", "eyes_state/undefined"]
EYE_SHORT = {"eyes_state/open": "open", "eyes_state/close": "close",
             "eyes_state/opening": "opening", "eyes_state/closing": "closing",
             "eyes_state/undefined": "undefined"}


def load_openlabel(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)["openlabel"]


def video_meta(ol: dict) -> dict:
    face = ol["streams"]["face_camera"]["stream_properties"]
    obj = ol["objects"]["0"]["object_data"]
    def _t(name):
        return next((x["val"] for x in obj.get("text", []) if x["name"] == name), None)
    def _b(name):
        return next((x["val"] for x in obj.get("boolean", []) if x["name"] == name), None)
    def _n(name):
        return next((x["val"] for x in obj.get("num", []) if x["name"] == name), None)
    ctx = next(iter(ol.get("contexts", {}).values()), {}).get("context_data", {})
    def _c(name):
        return next((x["val"] for x in ctx.get("text", []) if x["name"] == name), None)
    return {
        "ann_frames": ol["frame_intervals"][0]["frame_end"] + 1,
        "face_total_frames": face.get("total_frames"),
        "face_frame_shift": face.get("sync", {}).get("frame_shift"),
        "gender": _t("gender"), "age": _n("age"), "glasses": _b("glasses"),
        "experience": _t("experience"), "drive_freq": _t("drive_freq"),
        "weather": _c("weather"), "setup": _c("setup"), "recordTime": _c("recordTime"),
    }


def build_frame_table(ol: dict) -> list[dict]:
    """프레임별 GT 행 리스트. eye_state / eye_closed / eye_gt_binary / is_blink / is_yawn / yawn_type."""
    N = ol["frame_intervals"][0]["frame_end"] + 1
    eye = [None] * N
    blink = [0] * N
    yawn = [None] * N  # None / 'with_hand' / 'without_hand'
    for a in ol["actions"].values():
        t = a["type"]
        for fi in a["frame_intervals"]:
            s, e = fi["frame_start"], fi["frame_end"]
            if t in EYE_TYPES:
                for fr in range(s, e + 1):
                    eye[fr] = EYE_SHORT[t]
            elif t == "blinks/blinking":
                for fr in range(s, e + 1):
                    blink[fr] = 1
            elif t.startswith("yawning/"):
                yt = "with_hand" if "with hand" in t else "without_hand"
                for fr in range(s, e + 1):
                    yawn[fr] = yt
    rows = []
    for fr in range(N):
        es = eye[fr] if eye[fr] is not None else "none"
        # EyeCNN Closed 판정과 비교할 이진 GT: close=1, open=0, 전이/미정/none=-1(평가 제외)
        if es == "close":
            b = 1
        elif es == "open":
            b = 0
        else:
            b = -1
        rows.append({
            "frame": fr,
            "eye_state": es,
            "eye_closed": 1 if es == "close" else 0,
            "eye_gt_binary": b,
            "is_blink": blink[fr],
            "is_yawn": 1 if yawn[fr] is not None else 0,
            "yawn_type": yawn[fr] if yawn[fr] is not None else "none",
        })
    return rows


def mosaic_frame_count(avi_path: str) -> int | None:
    """ffprobe(빠름, 헤더 nb_frames) -> cv2 fallback."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of",
             "default=noprint_wrappers=1:nokey=1", avi_path],
            capture_output=True, text=True, timeout=30)
        v = out.stdout.strip()
        if v.isdigit():
            return int(v)
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(avi_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        return n if n > 0 else None
    except Exception:
        return None
