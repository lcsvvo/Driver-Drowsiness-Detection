#!/usr/bin/env python3
"""Run entire pipeline: label -> extract frames -> crop mouth -> extract segments
This script shells out to the step scripts; you can also run steps individually.
"""
import argparse
import subprocess
import sys
import os


def cmd(prog, args):
    return [sys.executable, os.path.join(os.getcwd(), prog)] + args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--padding", type=float, default=0.35)
    args = p.parse_args()

    steps = [
        ("Labeling", '0_label_tool.py', ["--input_dir", args.input_dir, "--out_dir", args.out_dir]),
        ("Extract frames", '1_extract_frames.py', ["--input_dir", args.input_dir, "--out_dir", args.out_dir]),
        ("Crop mouth", '2_crop_mouth.py', ["--in_dir", args.out_dir, "--out_dir", args.out_dir, "--padding", str(args.padding)]),
        ("Extract segments", '3_extract_yawn_segments.py', ["--out_dir", args.out_dir]),
    ]

    for name, prog, pargs in steps:
        print(f"=== {name} ===")
        cp = cmd(prog, pargs)
        r = subprocess.run(cp)
        if r.returncode != 0:
            print(f"Step {name} failed: return {r.returncode}")
            break


if __name__ == '__main__':
    main()
