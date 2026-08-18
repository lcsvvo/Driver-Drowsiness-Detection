#!/usr/bin/env python3
"""Extract face from the bottom-left cell of mosaic frames and save as AVI.

This crops the left-bottom cell (configurable), runs a Haar face detector on the
cell, and writes the detected face (or the last successful crop) as frames into
an AVI video.
"""
import os
import glob
import argparse
try:
    import cv2
except Exception:
    raise SystemExit('cv2 is required. Activate your venv and `pip install opencv-python`')
from tqdm import tqdm


def locate_cascade():
    # prefer cv2 bundled cascades
    base = getattr(cv2, 'data', None)
    if base:
        path = os.path.join(base.haarcascades, 'haarcascade_frontalface_default.xml')
        if os.path.exists(path):
            return path

    # fallback: user cache used earlier by other scripts
    cache_path = os.path.expanduser('~/.cache/dmd_haarcascades/haarcascade_frontalface_default.xml')
    if os.path.exists(cache_path):
        return cache_path

    return None


def extract_face_cell_to_avi(video_id, vd_dir, out_dir, rows=2, cols=2, cell_row=1, cell_col=0, margin=0.35, fps=None, pattern='frame_*.png'):
    frames_dir = os.path.join(vd_dir, 'frames')
    files = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    if not files:
        raise FileNotFoundError(f'No frames found in {frames_dir} with pattern {pattern}')

    first = cv2.imread(files[0])
    if first is None:
        raise RuntimeError(f'Unable to read first frame: {files[0]}')
    h, w = first.shape[:2]

    cell_w = w // cols
    cell_h = h // rows
    x0 = int(cell_col * cell_w)
    y0 = int(cell_row * cell_h)
    x1 = x0 + int(cell_w)
    y1 = y0 + int(cell_h)

    # fps inference as before
    if fps is None:
        avi_candidates = glob.glob(os.path.join(vd_dir, '*.avi')) + glob.glob(os.path.join(os.path.dirname(vd_dir), '*.avi'))
        if avi_candidates:
            cap = cv2.VideoCapture(avi_candidates[0])
            try:
                f = cap.get(cv2.CAP_PROP_FPS)
                fps = float(f) if f and f > 0 else 25.0
            finally:
                cap.release()
        else:
            fps = 25.0

    cascade_path = locate_cascade()
    if not cascade_path:
        raise RuntimeError('Could not find haarcascade_frontalface_default.xml; run prior setup or ensure OpenCV data is available')

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f'Failed to load cascade from {cascade_path}')

    out_path = os.path.join(out_dir, f'{video_id}_leftbottom_face.avi')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_size = (cell_w, cell_h)
    writer = cv2.VideoWriter(out_path, fourcc, fps, out_size)
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open VideoWriter for {out_path}')

    last_crop = None
    for fpath in tqdm(files, desc=f'Writing {os.path.basename(out_path)}'):
        img = cv2.imread(fpath)
        if img is None:
            # write last_crop or black
            if last_crop is None:
                writer.write(255 * np.zeros((out_size[1], out_size[0], 3), dtype='uint8'))
            else:
                writer.write(last_crop)
            continue

        cell = img[y0:y1, x0:x1]
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
        if len(faces) > 0:
            # pick largest face
            x,y,wf,hf = max(faces, key=lambda r: r[2]*r[3])
            # expand by margin
            mx = int(wf * margin)
            my = int(hf * margin)
            ax0 = max(0, x - mx)
            ay0 = max(0, y - my)
            ax1 = min(cell.shape[1], x + wf + mx)
            ay1 = min(cell.shape[0], y + hf + my)
            crop = cell[ay0:ay1, ax0:ax1]
            if crop.size == 0:
                # fallback to cell
                crop = cell
        else:
            # no face detected
            if last_crop is not None:
                crop = last_crop
            else:
                crop = cell

        # resize to out_size and write
        crop_resized = cv2.resize(crop, out_size)
        writer.write(crop_resized)
        last_crop = crop_resized

    writer.release()
    print('Saved:', out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video_id', required=True)
    p.add_argument('--vd_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--rows', type=int, default=2)
    p.add_argument('--cols', type=int, default=2)
    p.add_argument('--cell_row', type=int, default=1)
    p.add_argument('--cell_col', type=int, default=0)
    p.add_argument('--margin', type=float, default=0.35, help='Expand face bbox by this ratio')
    p.add_argument('--fps', type=float, default=None)
    p.add_argument('--pattern', default='frame_*.png')
    args = p.parse_args()

    extract_face_cell_to_avi(
        args.video_id,
        args.vd_dir,
        args.out_dir,
        rows=args.rows,
        cols=args.cols,
        cell_row=args.cell_row,
        cell_col=args.cell_col,
        margin=args.margin,
        fps=args.fps,
        pattern=args.pattern,
    )


if __name__ == '__main__':
    import numpy as np
    main()
