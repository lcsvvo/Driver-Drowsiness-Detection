#!/usr/bin/env python3
"""Interactive labeling tool for yawn timestamps.

Controls:
 - Space: play/pause
 - z: record current time as yawn peak (press once at max mouth opening)
 - Left/Right arrows: decrease/increase playback speed (0.5x / 2x)
 - + / - : finer faster/slower playback (millisecond adjustment)
 - s: save annotations so far
 - q or ESC: quit

Saves CSV to out_dir/annotations/{video_id}.csv with columns:
 video_id,yawn_index,timestamp_sec,frame_idx
"""
import argparse
import os
import csv
import subprocess
import cv2


def get_fps_ffprobe(path):
    try:
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
            return float(num) / float(den)
        return float(out)
    except Exception:
        return 0.0


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def label_video(path, out_dir, max_yawns=3):
    video_id = os.path.splitext(os.path.basename(path))[0]
    ann_dir = os.path.join(out_dir, "annotations")
    ensure_dir(ann_dir)
    out_csv = os.path.join(ann_dir, f"{video_id}.csv")
    if os.path.exists(out_csv):
        print(f"[SKIP] {video_id} already labeled -> {out_csv}")
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] cannot open {path}")
        return

    fps = get_fps_ffprobe(path) or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    win = "Labeler"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    playing = True
    # delay between frames when playing (ms). Derived from fps and speed_multiplier.
    speed_multiplier = 1.0
    def compute_delay():
        return int(max(1, round(1000 / (fps * speed_multiplier))))
    delay = compute_delay()
    yawns = []

    print(f"Labeling {video_id}: fps={fps:.3f}, frames={total}")
    print("Controls: Space=play/pause, z=peak, Left/Right=slow/fast, +/-=fine adjust, s=save, q=quit")

    while True:
        if playing:
            ret, frame = cap.read()
            if not ret:
                playing = False
                continue
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

        disp = frame.copy()
        ts = frame_idx / fps if fps else 0.0
        cv2.putText(disp, f"{video_id} {frame_idx}/{total}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(disp, f"t={ts:.3f}s", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if yawns:
            cv2.putText(disp, f"Yawns: {len(yawns)}/{max_yawns}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, disp)

        raw = cv2.waitKey(delay if playing else 0)
        key = raw & 0xFF if raw != -1 else -1
        if key == 32:  # space
            playing = not playing
        elif key in (ord('q'), 27):
            break
        elif key == ord('z'):
            if len(yawns) < max_yawns:
                yawns.append((len(yawns), ts, frame_idx))
                print(f"Recorded yawn {len(yawns)} at {ts:.3f}s (frame {frame_idx})")
            else:
                print("Already recorded max yawns")
        elif key == ord('s'):
            if not yawns:
                print("No yawns to save")
            else:
                with open(out_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["video_id", "yawn_index", "timestamp_sec", "frame_idx"])
                    for idx, t, fidx in yawns:
                        writer.writerow([video_id, idx, f"{t:.6f}", fidx])
                print(f"Saved {out_csv}")
        elif raw in (81, 2424832, 65361) or key in (81, 65361):  # left arrow -> slow down (half speed)
            speed_multiplier = max(0.25, speed_multiplier / 2.0)
            delay = compute_delay()
            print(f"Playback speed={speed_multiplier}x, delay={delay}ms")
        elif raw in (83, 2555904, 65363) or key in (83, 65363):  # right arrow -> speed up (double)
            speed_multiplier = min(8.0, speed_multiplier * 2.0)
            delay = compute_delay()
            print(f"Playback speed={speed_multiplier}x, delay={delay}ms")
        elif key == ord('+') or key == ord('='):
            # fine-tune: decrease delay by 10ms (faster)
            delay = max(1, delay - 10)
            print(f"Playback delay={delay}ms")
        elif key == ord('-'):
            # fine-tune: increase delay by 10ms (slower)
            delay = delay + 10
            print(f"Playback delay={delay}ms")

        # show current speed and auto stop when collected max_yawns
        cv2.putText(disp, f"speed={speed_multiplier}x", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if len(yawns) >= max_yawns:
            cv2.putText(disp, "Collected max yawns. Press 's' to save.", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(win, disp)

    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_yawns", type=int, default=3)
    args = p.parse_args()

    files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith('.avi')])
    if not files:
        print("No .avi files found in input_dir")
        return
    for path in files:
        try:
            label_video(path, args.out_dir, args.max_yawns)
        except Exception as e:
            print(f"Error on {path}: {e}")


if __name__ == '__main__':
    main()
