import torch
import torch.nn.functional as F
from torchmetrics.functional import *

def calc_bpp(rate, image):
	H, W = image.shape[-2:]
	return rate.mean() / (H * W)

def calc_mse(target, pred):
	mse = torch.mean((target - pred) ** 2, dim=(-1, -2, -3)) * 255.0 ** 2
	if mse.shape[0] == 1:
		mse = mse[0]
	return mse.mean()

def calc_psnr(mse, eps):
	mse = F.threshold(mse, eps, eps)
	psnr = 10. * torch.log10(255. ** 2 / mse)
	return psnr