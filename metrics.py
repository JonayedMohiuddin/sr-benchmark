import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def y_channel(img):
    # uint8 RGB -> Y of YCbCr (BT.601, range 16..235), same formula as MATLAB's rgb2ycbcr
    img = img.astype(np.float64) / 255
    return 16 + 65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]


def shave(img, b):
    return img[b:-b, b:-b] if b else img


# PSNR, pixel fidelity, higher = better
def psnr(sr, hr, border=0):
    return peak_signal_noise_ratio(shave(y_channel(hr), border), shave(y_channel(sr), border), data_range=255)


# SSIM, structural similarity, higher = better (max 1)
def ssim(sr, hr, border=0):
    # gaussian 11x11 window, sigma 1.5 = the original SSIM paper's settings
    return structural_similarity(shave(y_channel(hr), border), shave(y_channel(sr), border), data_range=255,
                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


# LPIPS, learned perceptual distance (deep features), lower = looks more real
class LPIPS:
    def __init__(self, device):
        import lpips
        self.net, self.device = lpips.LPIPS(net="alex", verbose=False).to(device).eval(), device

    @torch.no_grad()
    def __call__(self, sr, hr, border=0):
        t = lambda im: torch.from_numpy(shave(im, border).copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(self.device)
        return float(self.net(t(sr), t(hr)))
