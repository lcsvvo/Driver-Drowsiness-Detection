#!/usr/bin/env python3
"""Extract the bottom-left cell from mosaic frames and save as an AVI video.

Usage example:
  python3 4_extract_bottomleft_avi.py \
    --video_id gB_10_s5_2019-03-12T10;35;20+01;00_rgb_mosaic \
    --vd_dir /Users/jeon-wonseog/Desktop/DMD_out/gB_10_s5_2019-03-12T10;35;20+01;00_rgb_mosaic \
    --out_dir /Users/jeon-wonseog/Desktop/DMD_out --rows 2 --cols 2 --cell_row 1 --cell_col 0
"""
import os
import glob
import argparse
try:
    import cv2
except Exception as e:
    raise SystemExit('cv2 is required. Activate your venv and `pip install opencv-python`')
from tqdm import tqdm


def extract_cell_from_frames(video_id, vd_dir, out_dir, rows=2, cols=2, cell_row=1, cell_col=0, fps=None, pattern='frame_*.png'):
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

    if fps is None:
        # try to infer from any avi in vd_dir or parent
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

    out_path = os.path.join(out_dir, f'{video_id}_leftbottom.avi')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (cell_w, cell_h))
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open VideoWriter for {out_path}')

    for fpath in tqdm(files, desc=f'Writing {os.path.basename(out_path)}'):
        img = cv2.imread(fpath)
        if img is None:
            continue
        crop = img[y0:y1, x0:x1]
        if crop.shape[1] != cell_w or crop.shape[0] != cell_h:
            crop = cv2.resize(crop, (cell_w, cell_h))
        writer.write(crop)

    writer.release()
    print('Saved:', out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video_id', required=True)
    p.add_argument('--vd_dir', required=True, help='Video directory containing `frames/`')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--rows', type=int, default=2)
    p.add_argument('--cols', type=int, default=2)
    p.add_argument('--cell_row', type=int, default=1, help='0-indexed row of cell (bottom row = 1 for rows=2)')
    p.add_argument('--cell_col', type=int, default=0, help='0-indexed column of cell (left column = 0)')
    p.add_argument('--fps', type=float, default=None)
    p.add_argument('--pattern', default='frame_*.png')
    args = p.parse_args()

    extract_cell_from_frames(
        args.video_id,
        args.vd_dir,
        args.out_dir,
        rows=args.rows,
        cols=args.cols,
        cell_row=args.cell_row,
        cell_col=args.cell_col,
        fps=args.fps,
        pattern=args.pattern,
    )


if __name__ == '__main__':
    main()
