# Super-resolution: bicubic vs EDSR vs ESRGAN

Comparing three ways to upscale images 4x on Set5, Set14 and BSD100, using pretrained models
(nothing is trained here).

- **bicubic**: plain interpolation, the baseline
- **EDSR**: CNN trained with a pixel loss (torchsr)
- **ESRGAN**: GAN-trained generator (official weights, loaded with spandrel)

## Setup

```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt
```

Download the test sets and ESRGAN weights from the links in `urls.txt`. The EDSR weights download automatically.

## Usage

```
python infer.py       # upscales everything in input/ into outputs/
python evaluate.py    # benchmark, results go to results/
```

## Results (x4)

PSNR/SSIM are computed on the Y channel with borders shaved. For LPIPS, lower is better.

| Model   | Set5 PSNR / SSIM / LPIPS | Set14 PSNR / SSIM / LPIPS | BSD100 PSNR / SSIM / LPIPS |
|---------|--------------------------|---------------------------|----------------------------|
| bicubic | 28.43 / 0.811 / 0.343    | 26.09 / 0.705 / 0.440     | 25.96 / 0.668 / 0.528      |
| EDSR    | 32.49 / 0.899 / 0.172    | 28.83 / 0.788 / 0.275     | 27.73 / 0.742 / 0.363      |
| ESRGAN  | 30.50 / 0.852 / 0.073    | 26.32 / 0.699 / 0.132     | 25.33 / 0.651 / 0.162      |

The bicubic and EDSR numbers match the published ones, so the evaluation is set up correctly.

## Takeaway

EDSR is the most accurate. ESRGAN scores worse on PSNR (on BSD100 it's even below bicubic),
but it looks far sharper, and its LPIPS is about 2x better. Part of that sharpness is made up:
in `results/compare_Set14_baboon.png` the fur looks real, but the error map is brightest right there.

So for anything quantitative, like satellite imagery, I'd use a pixel-loss model and keep GAN
output for visuals only. Without ground truth, one check you can still do is to downscale the
output and compare it with the input (LR-PSNR in `results/summary.md`). It catches some changes,
but not invented fine detail.
