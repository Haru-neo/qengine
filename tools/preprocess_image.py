#!/usr/bin/env python3
"""Convert an image to the fp32 raw tensor that vision::VisionModel expects.

Output: little-endian fp32, shape [3, H, W] with H=W=image_size, contiguous
        in C/H/W order. Channels are normalized via (x/255 - mean) / std
        using the values stored in the mmproj GGUF (default 0.5/0.5 for
        Qwen3-VL).

Usage:
    python3 tools/preprocess_image.py <image> <out.raw> [--size 768] \\
            [--mean 0.5,0.5,0.5] [--std 0.5,0.5,0.5]
"""
import argparse
import sys
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out_raw")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--mean", default="0.5,0.5,0.5")
    ap.add_argument("--std", default="0.5,0.5,0.5")
    args = ap.parse_args()

    mean = np.array([float(x) for x in args.mean.split(",")], dtype=np.float32)
    std  = np.array([float(x) for x in args.std.split(",")], dtype=np.float32)

    img = Image.open(args.image).convert("RGB").resize(
        (args.size, args.size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0       # H, W, C
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1).copy()                    # C, H, W
    arr.astype("<f4").tofile(args.out_raw)
    print(f"wrote {arr.shape} -> {args.out_raw}", file=sys.stderr)


if __name__ == "__main__":
    main()
