"""Metric 함수 정의
"""

import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import lpips
import pytorch_fid_wrapper as pfw
SMOOTH = 1e-6

def get_metric_function(metric_function_str):
    """
    Add metrics, weights for weighted score
    """

    if metric_function_str == 'psnr':
        psnr = calculate_psnr()
        return psnr

    elif metric_function_str == 'ssim':
        ssim =calculate_ssim()
        return ssim 

    elif metric_function_str == 'mssim':
        mssim =calculate_mssim()
        return mssim

    elif metric_function_str == 'lpips':
        lpips =calculate_lpips()
        return lpips
    
    elif metric_function_str == 'fid':
        fid =calculate_fid()
        return fid

def calculate_psnr(img1, img2):
    loss_l2 = F.mse_loss(img1.float(), img2.float(), reduction="mean")
    PSNR_val = 20 * torch.log10(1.0/torch.sqrt(loss_l2))
    return PSNR_val.item()

def calculate_ssim(image1, image2):
    ssim_value = ssim(image1, image2, multichannel=True)
    return ssim_value

def calculate_mssim(img1, img2, scales=5):
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # 예시 가중치
    mssim_values = []

    # 각 색상 채널에 대해 SSIM 계산
    for channel in range(3):  # RGB 채널 반복
        mssim_per_channel = []
        img1_chan = img1[:, :, channel]
        img2_chan = img2[:, :, channel]

        for scale in range(scales):
            ssim_val = ssim(img1_chan, img2_chan, data_range=img1_chan.max() - img1_chan.min())
            mssim_per_channel.append(ssim_val)

            # 다운샘플링은 이미지 크기가 충분히 큰 경우에만 수행
            if img1_chan.shape[0] > 1 and img1_chan.shape[1] > 1:
                img1_chan = img1_chan[::2, ::2]
                img2_chan = img2_chan[::2, ::2]

        overall_mssim = np.prod(np.array(mssim_per_channel) ** np.array(weights[:len(mssim_per_channel)]))
        mssim_values.append(overall_mssim)

    # 각 채널의 결과를 평균 내어 최종 MSSIM 값을 반환
    return np.mean(mssim_values)

def calculate_lpips(img1, img2):
    loss_fn = lpips.LPIPS(net='alex')
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)()
    lpips_score = loss_fn(img1, img2)
    return lpips_score.item()

def calculate_fid(img1, img2):
	fid_val = pfw.fid(img2, img1, batch_size=img2.shape[0])
	return fid_val.item()