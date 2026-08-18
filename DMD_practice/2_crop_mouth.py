#!/usr/bin/env python3
"""Crop mouth region from left-bottom face area using MediaPipe Face Mesh.
Input frames: out_dir/{video_id}/frames/*.png
Output mouth crops: out_dir/{video_id}/mouth_crops/frame_%06d.png
Logs missing detections to out_dir/{video_id}/missing_landmarks.csv
"""
import argparse
import os
import csv
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x
import cv2
try:
    import mediapipe as mp
    # ensure mediapipe has the expected API
    if hasattr(mp, 'solutions'):
        _HAS_MEDIAPIPE = True
    else:
        mp = None
        _HAS_MEDIAPIPE = False
except Exception:
    mp = None
    _HAS_MEDIAPIPE = False
# ensure Haar cascade is available when mediapipe is not used
if not _HAS_MEDIAPIPE:
    # Prefer using OpenCV's packaged face cascade and derive mouth region
    # from the detected face if MediaPipe is not available. This is more
    # robust than relying on a separate mouth-only cascade XML that may
    # not be present on the system or fail to download in restricted envs.
    try:
        face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(face_xml):
            _FACE_CASCADE = cv2.CascadeClassifier(face_xml)
        else:
            # fallback: download to ~/.cache/dmd_haarcascades
            import urllib.request
            _HAAR_PATH = os.path.join(os.path.expanduser('~'), '.cache', 'dmd_haarcascades')
            os.makedirs(_HAAR_PATH, exist_ok=True)
            cached_face = os.path.join(_HAAR_PATH, 'haarcascade_frontalface_default.xml')
            if not os.path.exists(cached_face):
                try:
                    urllib.request.urlretrieve(
                        'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
                        cached_face,
                    )
                except Exception:
                    try:
                        urllib.request.urlretrieve(
                            'https://raw.githubusercontent.com/opencv/opencv/master/master/data/haarcascades/haarcascade_frontalface_default.xml',
                            cached_face,
                        )
                    except Exception:
                        pass
            _FACE_CASCADE = cv2.CascadeClassifier(cached_face) if os.path.exists(cached_face) else None
    except Exception:
        _FACE_CASCADE = None
    # keep _MOUTH_CASCADE as None (we'll use face-based heuristic instead)
    _MOUTH_CASCADE = None


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def crop_mouth_for_video(video_id, vd_dir, out_dir, padding=0.35):
    frames_dir = os.path.join(vd_dir, 'frames')
    if not os.path.isdir(frames_dir):
        print(f"No frames for {video_id}, skipping")
        return
    out_mouth = os.path.join(out_dir, video_id, 'mouth_crops')
    ensure_dir(out_mouth)
    missing_csv = os.path.join(out_dir, video_id, 'missing_landmarks.csv')

    face_mesh = None
    mp_face = None
    if _HAS_MEDIAPIPE:
        mp_face = mp.solutions.face_mesh
        face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    missing_rows = []
    for fname in tqdm(frame_files, desc=f"Crop {video_id}"):
        fpath = os.path.join(frames_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]
        # left-bottom crop
        crop = img[int(h/2):h, 0:int(w/2)].copy()
        ch, cw = crop.shape[:2]
        mouth = None
        if face_mesh is not None:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                missing_rows.append((video_id, fname))
                continue
            lm = res.multi_face_landmarks[0]
            # collect mouth landmark indices from FACEMESH_LIPS
            idxs = set()
            for a, b in mp_face.FACEMESH_LIPS:
                idxs.add(a)
                idxs.add(b)
            xs = []
            ys = []
            for i in idxs:
                lmpt = lm.landmark[i]
                xs.append(lmpt.x * cw)
                ys.append(lmpt.y * ch)
            x0 = max(0, int(min(xs)))
            x1 = min(cw, int(max(xs)))
            y0 = max(0, int(min(ys)))
            y1 = min(ch, int(max(ys)))
            mouth = crop[y0:y1, x0:x1]
        else:
            # Use face detection on the crop and approximate mouth as the
            # lower portion of the face rectangle. This avoids relying on
            # a separate mouth cascade which may be missing or unreliable.
            mouth = None
            if '_FACE_CASCADE' in globals() and _FACE_CASCADE is not None:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) == 0:
                    missing_rows.append((video_id, fname))
                    continue
                fx, fy, fw, fh = max(faces, key=lambda r: r[2]*r[3])
                # approximate mouth area as lower 40% of face bbox
                mx = fx
                my = int(fy + fh * 0.6)
                mw = fw
                mh = int(fh * 0.35)
                x0, y0, x1, y1 = mx, my, mx + mw, my + mh
                # clamp to crop bounds
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(cw, x1)
                y1 = min(ch, y1)
                if x1 <= x0 or y1 <= y0:
                    missing_rows.append((video_id, fname))
                    continue
                mouth = crop[y0:y1, x0:x1]
            else:
                missing_rows.append((video_id, fname))
                continue
        # padding
        try:
            pw = int((x1 - x0) * padding)
            ph = int((y1 - y0) * padding)
            x0 = max(0, x0 - pw)
            x1 = min(cw, x1 + pw)
            y0 = max(0, y0 - ph)
            y1 = min(ch, y1 + ph)
            mouth = crop[y0:y1, x0:x1]
        except Exception:
            # if mouth already set (haar), skip padding
            pass
        out_name = os.path.join(out_mouth, fname)
        cv2.imwrite(out_name, mouth)

    if face_mesh is not None:
        face_mesh.close()
    if missing_rows:
        with open(missing_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'frame_file'])
            writer.writerows(missing_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True, help='out_dir from extraction')
    p.add_argument("--out_dir", required=True)
    p.add_argument("--padding", type=float, default=0.35)
    args = p.parse_args()

    vids = sorted([d for d in os.listdir(args.in_dir) if os.path.isdir(os.path.join(args.in_dir, d))])
    for vid in vids:
        try:
            crop_mouth_for_video(vid, os.path.join(args.in_dir, vid), args.out_dir, args.padding)
        except Exception as e:
            print(f"Error {vid}: {e}")


if __name__ == '__main__':
    main()
