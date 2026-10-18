# Pretrained x4 super-resolution models, loaded from libraries (nothing is trained here).
# Every model is wrapped as a function  upscale(lr) -> sr  (uint8 RGB numpy in, uint8 RGB out).
#
# name      library    params   what it is
# bicubic   Pillow     -        plain interpolation (baseline)
# edsr      torchsr    43 M     deep ResNet trained with pixel (L1) loss -> the most faithful output
# esrgan    spandrel   16.7 M   GAN-trained (VGG + adversarial loss)     -> the sharpest-looking output

import os

import torch

from utils import resize

SCALE = 4
MODELS = ["bicubic", "edsr", "esrgan"]
ESRGAN_WEIGHTS = "pretrained/RRDB_ESRGAN_x4.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def torch_upscaler(net):
    # Wrap a PyTorch model (input/output: float RGB in [0, 1]) as an upscale(lr) function
    net = net.to(DEVICE).eval()

    @torch.no_grad()
    def upscale(lr):
        x = torch.from_numpy(lr).permute(2, 0, 1)[None].float().div(255).to(DEVICE)
        y = net(x).clamp(0, 1)
        return y[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return upscale


def load_model(name):
    if name == "bicubic":
        return lambda lr: resize(lr, lr.shape[1] * SCALE, lr.shape[0] * SCALE)

    if name == "edsr":
        from torchsr.models import edsr
        return torch_upscaler(edsr(scale=SCALE, pretrained=True))  # weights download automatically

    if name == "esrgan":
        import spandrel  # reads the official ESRGAN .pth file and builds the right network
        assert os.path.exists(ESRGAN_WEIGHTS), "download the weights listed in urls.txt into pretrained/"
        return torch_upscaler(spandrel.ModelLoader().load_from_file(ESRGAN_WEIGHTS).model)

    raise ValueError(f"unknown model {name}, choose from {MODELS}")
