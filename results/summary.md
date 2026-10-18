# Results (x4)

PSNR / SSIM: Y channel, border shaved. LPIPS: lower is better. LR-PSNR: SR downscaled again vs the LR input (consistency with the observation).

| Model | Set5 PSNR / SSIM / LPIPS | Set14 PSNR / SSIM / LPIPS | BSD100 PSNR / SSIM / LPIPS | LR-PSNR (BSD100) |
|---|---|---|---|---|
| bicubic | 28.43 / 0.8111 / 0.343 | 26.09 / 0.7048 / 0.440 | 25.96 / 0.6681 / 0.528 | 39.31 |
| edsr | 32.49 / 0.8989 / 0.172 | 28.83 / 0.7880 / 0.275 | 27.73 / 0.7422 / 0.363 | 53.62 |
| esrgan | 30.50 / 0.8523 / 0.073 | 26.32 / 0.6988 / 0.132 | 25.33 / 0.6509 / 0.162 | 43.88 |
