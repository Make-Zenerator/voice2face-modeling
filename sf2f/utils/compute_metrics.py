"""
Metric 함수 정의
"""

import torch
import numpy as np
import torch.nn.functional as F
# from skimage.metrics import structural_similarity as ssim
import lpips
import pytorch_fid_wrapper as pfw
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as mssim
from torchmetrics.image import PeakSignalNoiseRatio as psnr
SMOOTH = 1e-6

def get_metric_function(metric_function_str):
    """
    Add metrics, weights for weighted score
    """

    if metric_function_str == 'psnr':
        psnr = calculate_psnr()
        return psnr

    elif metric_function_str == 'ssim':
        ssim = calculate_ssim()
        return ssim 

    elif metric_function_str == 'mssim':
        mssim = calculate_mssim()
        return mssim

    elif metric_function_str == 'lpips':
        lpips = calculate_lpips()
        return lpips
    
    elif metric_function_str == 'fid':
        fid = calculate_fid()
        return fid
    
    elif metric_function_str == 'all':
        metrics = all_metrics
        return metrics

def calculate_psnr(img1, img2):
    loss_l2_list = []
    for i in range(len(img1)):
        loss_l2_list.append(F.mse_loss(img1[i].float(), img2[i].float(), reduction='mean'))
    # loss_l2 = F.mse_loss(img1.float(), img2.float(), reduction="mean")
    PSNR_val = [20 * torch.log10(255.0/torch.sqrt(loss_l2_list[i])) for i in range(len(loss_l2_list))]
    return PSNR_val

def calculate_ssim(image1, image2):
    # ssim_value = ssim(image1, image2)
    ssim_list = []
    for i in range(image1.shape[0]):
        ssim_list.append(ssim(image1[i].unsqueeze(0), image2[i].unsqueeze(0)))
    # return ssim_value
    return ssim_list

def calculate_mssim(img1, img2, device='cpu'):
    # weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # 예시 가중치
    mssim_values = []
    # # 각 색상 채널에 대해 SSIM 계산
    # for channel in range(3):  # RGB 채널 반복
    #     mssim_per_channel = []
    #     img1_chan = img1[:, channel, :, :]
    #     img2_chan = img2[:, channel, :, :]

    #     for scale in range(scales):
    #         ssim_val = ssim(img1_chan[scale], img2_chan[scale], data_range=img1_chan.max() - img1_chan.min())
    #         mssim_per_channel.append(ssim_val)

    #         # 다운샘플링은 이미지 크기가 충분히 큰 경우에만 수행
    #         if img1_chan.shape[0] > 1 and img1_chan.shape[1] > 1:
    #             img1_chan = img1_chan[::2, ::2]
    #             img2_chan = img2_chan[::2, ::2]

    #     overall_mssim = np.prod(np.array(mssim_per_channel) ** np.array(weights[:len(mssim_per_channel)]))
    #     mssim_values.append(overall_mssim)
    max_val = max(img1.max(), img2.max())
    min_val = min(img1.min(), img2.min())
    data_range = max_val - min_val
    compute_mssim = mssim(
        data_range=data_range.item(),
        kernel_size=3,  # 커널 크기를 줄임
        sigma=1.5,
        betas=(0.0448, 0.2856, 0.3001)
    ).to(device)

    for i in range(len(img1)):
        mssim_values.append(compute_mssim(img1[i].unsqueeze(0), img2[i].unsqueeze(0)).item())

    # 각 채널의 결과를 평균 내어 최종 MSSIM 값을 반환
    return mssim_values

def calculate_lpips(img1, img2, device):
    loss_fn = lpips.LPIPS(net='alex').to(device)
    # img1 = img1.permute(2, 0, 1).unsqueeze(0)
    # img2 = img2.permute(2, 0, 1).unsqueeze(0)()
    lpips_score = loss_fn(img1, img2)
    lpips_score = lpips_score.squeeze()
    if lpips_score.dim() == 0:
        return lpips_score.unsqueeze(dim=0)
    return lpips_score

def calculate_fid(img1, img2, device):
	# fid_val = pfw.fid(img2, img1, batch_size=img2.shape[0])
    fid_list = []
    for i in range(img2.shape[0]):
        fid_list.append(pfw.fid(img2[i].unsqueeze(0), img1[i].unsqueeze(0),batch_size=1))
	# return fid_val.item()
    return fid_list

def all_metrics(img1, img2, device):

    metrics_scores = {0:[], 1:[], 2:[], 3:[], 4: []}
    # metrics_map = {0: 'psnr', 1: 'ssim', 2: 'mssim', 3: 'lpips', 4: 'fid'}
    metrics_scores[0].append(calculate_psnr(img1, img2))
    metrics_scores[1].append(calculate_ssim(img1, img2))
    metrics_scores[2].append(calculate_mssim(img1, img2, device=device))
    metrics_scores[3].append(calculate_lpips(img1, img2, device))
    metrics_scores[4].append(calculate_fid(img1, img2, device))

    return metrics_scores
