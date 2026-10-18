# Benchmark bicubic, EDSR and ESRGAN at x4 on Set5, Set14 and BSD100.
# For every image: HR --bicubic downscale x4--> LR --model--> SR, then compare SR with HR:
#   PSNR, SSIM (Y channel, border shaved), LPIPS (lower = more realistic) and
#   LR-PSNR: downscale SR again and compare with the LR input. Needs no ground truth (works on real
#            satellite data). Low = the model changed what the sensor saw. High does NOT prove the added
#            detail is real, because downscaling removes that detail again.
# Outputs (results/): metrics.csv, summary.md, perception_distortion.png, compare_*.png

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import LPIPS, psnr, ssim, y_channel
from models import DEVICE, MODELS, SCALE, load_model
from utils import make_lr, resize, test_images

DATASETS = ["Set5", "Set14", "BSD100"]
OUT = "results"
# zoomed regions for the comparison figures: (dataset, image, y, x, size) in HR pixels
SHOWCASE = [("Set14", "baboon", 40, 300, 120), ("Set5", "butterfly", 90, 90, 110),
            ("BSD100", "253027", 150, 250, 120), ("Set14", "comic", 90, 60, 110)]


def comparison_figure(title, hr, outputs, scores, box, path):
    # Top row: zoomed crop of HR and of every model. Bottom row: |model - HR| error map
    y, x, s = box
    fig, ax = plt.subplots(2, len(outputs) + 1, figsize=(2.8 * (len(outputs) + 1), 6))
    ax[0, 0].imshow(hr)
    ax[0, 0].add_patch(plt.Rectangle((x, y), s, s, fill=False, ec="yellow", lw=2))
    ax[0, 0].set_title(title, fontsize=9)
    ax[1, 0].imshow(hr[y:y + s, x:x + s])
    ax[1, 0].set_title("HR (ground truth)", fontsize=9)
    for i, (name, sr) in enumerate(outputs.items(), 1):
        p, l = scores[name]
        ax[0, i].imshow(sr[y:y + s, x:x + s])
        ax[0, i].set_title(f"{name}\nPSNR {p:.2f} | LPIPS {l:.3f}", fontsize=9)
        ax[1, i].imshow(np.abs(y_channel(sr) - y_channel(hr))[y:y + s, x:x + s], cmap="magma", vmin=0, vmax=60)
        ax[1, i].set_title("error vs HR", fontsize=9)
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    lpips = LPIPS(DEVICE)
    models = {name: load_model(name) for name in MODELS}
    rows = []
    for ds in DATASETS:
        for img_name, hr in test_images(ds):
            lr, hr = make_lr(hr, SCALE)
            outputs, scores = {}, {}
            for name, upscale in models.items():
                sr = upscale(lr)
                r = dict(dataset=ds, image=img_name, model=name,
                         psnr=psnr(sr, hr, SCALE), ssim=ssim(sr, hr, SCALE), lpips=lpips(sr, hr, SCALE),
                         lr_psnr=psnr(resize(sr, lr.shape[1], lr.shape[0]), lr))
                rows.append(r)
                outputs[name], scores[name] = sr, (r["psnr"], r["lpips"])
            for sds, simg, *box in SHOWCASE:
                if (sds, simg) == (ds, img_name):
                    comparison_figure(f"{ds}/{img_name} x{SCALE}", hr, outputs, scores, box,
                                      os.path.join(OUT, f"compare_{ds}_{img_name}.png"))
        print(f"{ds} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
    mean = df.groupby(["model", "dataset"], sort=False)[["psnr", "ssim", "lpips", "lr_psnr"]].mean()

    # summary table
    lines = [f"# Results (x{SCALE})", "", "PSNR / SSIM: Y channel, border shaved. LPIPS: lower is better. "
             "LR-PSNR: SR downscaled again vs the LR input (consistency with the observation).", "",
             "| Model | " + " | ".join(f"{d} PSNR / SSIM / LPIPS" for d in DATASETS) + " | LR-PSNR (BSD100) |",
             "|---|" + "---|" * (len(DATASETS) + 1)]
    for name in MODELS:
        m = lambda d: mean.loc[(name, d)]
        lines.append(f"| {name} | " + " | ".join(f"{m(d).psnr:.2f} / {m(d).ssim:.4f} / {m(d).lpips:.3f}"
                                                 for d in DATASETS) + f" | {m('BSD100').lr_psnr:.2f} |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # perception-distortion plot (BSD100): PSNR (x, fidelity) vs LPIPS (y, realism)
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    for i, name in enumerate(MODELS):
        m = mean.loc[(name, "BSD100")]
        ax.scatter(m.psnr, m.lpips, s=70, color=f"C{i}")
        ax.annotate(name, (m.psnr, m.lpips), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set(title=f"BSD100 x{SCALE}", xlabel="PSNR (dB)  ->  more faithful pixels",
           ylabel="LPIPS  (lower = more realistic)")
    ax.invert_yaxis()  # up = more realistic, right = more faithful
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "perception_distortion.png"), dpi=120)


if __name__ == "__main__":
    main()
