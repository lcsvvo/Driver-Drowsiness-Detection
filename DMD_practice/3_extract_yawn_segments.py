#!/usr/bin/env python3
"""Extract yawn segments (±0.5s) from mouth_crops based on annotations.
Creates out_dir/{video_id}/yawn_segments/yawn_{n}/ with symlinks to mouth_crops frames.
"""
import argparse
import os
import csv
import shutil
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def read_mapping(vd_dir):
    mapping = {}
    path = os.path.join(vd_dir, 'frames_timestamps.csv')
    if not os.path.exists(path):
        return mapping
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            mapping[int(r['frame_idx'])] = (float(r['timestamp_sec']), r['filename'])
    return mapping


def extract_segments(out_dir, window=0.5):
    ann_dir = os.path.join(out_dir, 'annotations')
    if not os.path.isdir(ann_dir):
        print('No annotations dir, nothing to do')
        return

    for ann in sorted([f for f in os.listdir(ann_dir) if f.endswith('.csv')]):
        video_id = os.path.splitext(ann)[0]
        vd_dir = os.path.join(out_dir, video_id)
        mouth_dir = os.path.join(vd_dir, 'mouth_crops')
        if not os.path.isdir(mouth_dir):
            print(f"No mouth crops for {video_id}, skipping")
            continue
        mapping = read_mapping(vd_dir)
        if not mapping:
            print(f"No mapping for {video_id}, skipping")
            continue

        anns = []
        with open(os.path.join(ann_dir, ann), 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                anns.append((int(r['yawn_index']), float(r['timestamp_sec'])))

        if not anns:
            continue

        # invert mapping to list for search
        items = [(fi, ts_fn[0], ts_fn[1]) for fi, ts_fn in mapping.items()]
        items.sort()

        for idx, ts in anns:
            start = ts - window
            end = ts + window
            out_seg = os.path.join(vd_dir, 'yawn_segments', f'yawn_{idx}')
            ensure_dir(out_seg)
            for fi, t, fname in items:
                if start <= t <= end:
                    src = os.path.join(mouth_dir, fname)
                    dst = os.path.join(out_seg, fname)
                    try:
                        if not os.path.exists(dst):
                            os.symlink(os.path.abspath(src), dst)
                    except Exception:
                        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--window", type=float, default=0.5)
    args = p.parse_args()
    extract_segments(args.out_dir, args.window)


if __name__ == '__main__':
    main()
