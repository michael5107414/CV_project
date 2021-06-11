import cv2
import numpy as np
import math
from skimage.measure import compare_ssim
#from skimage.metrics import structural_similarity
import os
import glob
import os
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR

def ssim(img1, img2):
    return compare_ssim(img1.astype(np.float32)/255., img2.astype(np.float32)/255., gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--cmp', type=str, required=True)

    args = parser.parse_args()

    It = cv2.imread(args.cmp)
    gt = cv2.imread(args.gt)

    psnr_score = psnr(gt, It)
    ssim_score = ssim(gt, It)
    print(f'{args.gt[16:]:<36}| PSNR:{psnr_score:.3f} | SSIM:{ssim_score:.3f} |')