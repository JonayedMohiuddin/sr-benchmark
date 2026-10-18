import glob
import os

import numpy as np
from PIL import Image


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def save_rgb(img, path):
    Image.fromarray(img).save(path)


def resize(img, width, height):
    # Bicubic resize (PIL). When shrinking, PIL anti-aliases like MATLAB's imresize
    return np.array(Image.fromarray(img).resize((width, height), Image.BICUBIC))


def make_lr(hr, scale):
    # The degradation used by the benchmarks: crop to a multiple of `scale`, then bicubic downscale
    h, w = hr.shape[0] // scale * scale, hr.shape[1] // scale * scale
    hr = hr[:h, :w]
    return resize(hr, w // scale, h // scale), hr


def test_images(dataset, root="data/test"):
    # All HR images of a benchmark set (Set5 / Set14 / BSD100): list of (name, image)
    files = sorted(glob.glob(os.path.join(root, dataset, "**", "*.png"), recursive=True))
    assert files, f"no images in {root}/{dataset} - download the test sets listed in urls.txt"
    return [(os.path.splitext(os.path.basename(f))[0], load_rgb(f)) for f in files]
