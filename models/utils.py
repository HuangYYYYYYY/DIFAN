
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import math
import time

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if classname.find('KernelConv2D') == -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.04)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.04)
        torch.nn.init.constant_(m.bias.data, 0)

def adjust_learning_rate(optimizer, epoch, decay_rate, decay_every, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lrs = []
    for param_group in optimizer.param_groups:

        lr = param_group['lr_init'] * (decay_rate ** (epoch // decay_every))
        param_group['lr'] = lr
        lrs.append(lr)

    return lrs

#from DPDD code
def get_psnr2(img1, img2, PIXEL_MAX=1.0):
    mse_ = torch.mean( (img1 - img2) ** 2 )
    return 10 * torch.log10(PIXEL_MAX / mse_)

    # return calculate_psnr(img1, img2)

Backward_tensorGrid = {}
DPD_zero = {}
def DPD(tensorInput, tensorFlow, padding_mode = 'zeros', device='cuda'):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        DPD_zero[str(tensorFlow.size())] = torch.zeros_like(tensorFlow[:, 0:1, :, :]).to(device)
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).to(device)

    DPDM = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),  DPD_zero[str(tensorFlow.size())]], 1)
    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + DPDM).permute(0, 2, 3, 1), mode='bilinear', padding_mode=padding_mode, align_corners = True)

def upsample(inp, h = None, w = None, mode = 'bilinear'):
    # if h is None or w is None:
    return F.interpolate(input=inp, size=(int(h), int(w)), mode=mode)
    # elif scale_factor is not None:
    #     return F.interpolate(input=inp, scale_factor=scale_factor, mode='bilinear', align_corners=False)




import numpy as np
import cv2

def blur_metric(psf_patches):
    psf_patches_np = psf_patches.cpu().numpy().transpose(0, 2, 3, 1)
    blur_values = []

    for psf_patch_np in psf_patches_np:
        gray_patch = cv2.cvtColor(psf_patch_np, cv2.COLOR_RGB2GRAY)
        _, thresholded_patch = cv2.threshold(gray_patch, 0.1, 1, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresholded_patch = cv2.erode(thresholded_patch, kernel, iterations=1)
        thresholded_patch = cv2.dilate(thresholded_patch, kernel, iterations=1)

        blur_value = np.mean(gray_patch * thresholded_patch) if np.any(thresholded_patch != 0) else 0
        blur_values.append(blur_value)

    blur_values = np.array(blur_values)
    zero_indices = np.where(blur_values == 0)[0]

    max_search_range = 5
    padded_blur_values = np.pad(blur_values, (max_search_range, max_search_range), mode='constant', constant_values=0)

    for idx in zero_indices:
        for search_range in range(1, max_search_range + 1):
            neighbor_indices = np.arange(idx - search_range, idx + search_range + 1)
            neighbor_values = padded_blur_values[neighbor_indices + max_search_range]

            non_zero_neighbor_values = neighbor_values[neighbor_values != 0]
            if non_zero_neighbor_values.size > 0:
                blur_values[idx] = np.mean(non_zero_neighbor_values)
                break

    epsilon = 1e-8
    min_value = np.min(blur_values)
    max_value = np.max(blur_values)
    if max_value - min_value > epsilon:
        normalized_blur_values = (blur_values - min_value) / (max_value - min_value)
    else:
        normalized_blur_values = blur_values
    return normalized_blur_values




# ----------
# SSIM
# ----------
import torch
import torch.nn.functional as F

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True, C1=0.01**2, C2=0.03**2):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def get_ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    window = window.to(img1.device)
    return _ssim(img1, img2, window, window_size, channel, size_average)


import numpy as np

def compute_pmp(feature_map):
    batch_size, channels, height, width = feature_map.shape
    patch_size = height
    pmp = torch.zeros_like(feature_map)

    for i in range(batch_size):
        for c in range(channels):
            pmp[i, c] = find_min_pixels(feature_map[i, c],patch_size)

    return pmp.mean(dim=1)

import numpy as np

def find_min_pixels(I, patch_size):
    M, N  = I.shape
    Mp = int(np.ceil(M / patch_size))
    Np = int(np.ceil(N / patch_size))
    J = torch.zeros((M, N), dtype=I.dtype,device=I.device)

    for m in range(Mp):
        for n in range(Np):
            idx1 = [m * patch_size, min((m + 1) * patch_size, M)]
            idx2 = [n * patch_size, min((n + 1) * patch_size, N)]
            patch = I[idx1[0]:idx1[1], idx2[0]:idx2[1]]
            val, idx = torch.min(patch.view(-1), 0)  # Flatten the patch and get the min value and its index
            idx = unravel_index(idx, patch.shape)  # Convert the 1D index to 2D index using PyTorch instead of numpy
            cur_patch = torch.zeros(patch.shape, dtype=I.dtype, device=I.device)
            cur_patch[idx] = val
            J[idx1[0]:idx1[1], idx2[0]:idx2[1]] = cur_patch

    return J

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))




def gini(array):
    array = array.view(-1)
    array[array < 0] = 0
    array += 0.0000001
    array, _ = torch.sort(array)
    index = torch.arange(1, array.shape[0] + 1).to(array.device)
    n = array.shape[0]
    return ((torch.sum((2 * index - n  - 1) * array)) / (n * torch.sum(array)))
