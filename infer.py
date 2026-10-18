# python infer.py                        (all models, input/ -> outputs/)
# python infer.py --models esrgan        (only one model)

import argparse
import glob
import os

from models import MODELS, load_model
from utils import load_rgb, save_rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="input", help="folder with low-resolution images")
    p.add_argument("--out", default="outputs")
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "*.png")) + glob.glob(os.path.join(args.input, "*.jpg")))
    os.makedirs(args.out, exist_ok=True)
    for name in args.models:
        upscale = load_model(name)
        for f in files:
            img = load_rgb(f)
            sr = upscale(img)
            out = os.path.join(args.out, f"{os.path.splitext(os.path.basename(f))[0]}_{name}.png")
            save_rgb(sr, out)
            print(f"{name}: {f} {img.shape[1]}x{img.shape[0]} -> {sr.shape[1]}x{sr.shape[0]}  {out}")


if __name__ == "__main__":
    main()
