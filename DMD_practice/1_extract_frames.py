#!/usr/bin/env python3
"""Extract frames and produce timestamp mapping CSV.
Saves frames to out_dir/{video_id}/frames/frame_%06d.png and timestamps CSV.
"""
import argparse
import os
import subprocess
import cv2
import csv
from tqdm import tqdm


def get_fps_ffprobe(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    if "/" in out:
        num, den = out.split("/")
        try:
            return float(num) / float(den)
        except Exception:
            return 0.0
    try:
        return float(out)
    except Exception:
        return 0.0


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def extract(path, out_dir):
    video_id = os.path.splitext(os.path.basename(path))[0]
    vd_out = os.path.join(out_dir, video_id)
    frames_dir = os.path.join(vd_out, "frames")
    ensure_dir(frames_dir)
    mapping_csv = os.path.join(vd_out, "frames_timestamps.csv")
    if os.path.exists(mapping_csv):
        print(f"Skipping {video_id}: frames already extracted")
        return

    cap = cv2.VideoCapture(path)
    fps = get_fps_ffprobe(path) or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    idx = 0
    rows = []
    pbar = tqdm(total=total, desc=f"Extract {video_id}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = f"frame_{idx:06d}.png"
        out_path = os.path.join(frames_dir, fname)
        cv2.imwrite(out_path, frame)
        ts = idx / fps if fps else 0.0
        rows.append((idx, f"{ts:.6f}", fname))
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    with open(mapping_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp_sec", "filename"])
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith('.avi')])
    for path in files:
        try:
            extract(path, args.out_dir)
        except Exception as e:
            print(f"Error {path}: {e}")


if __name__ == '__main__':
    main()
