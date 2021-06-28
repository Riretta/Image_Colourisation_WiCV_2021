from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from IPython import embed


def load_img(img_path):
    out_np = np.asarray(Image.open(img_path).convert('RGB'))
    if (out_np.ndim == 2):
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


def resize_img(img, HW=(256, 256), resample=3):
    # width, height = img.size  # Get dimensions
    #
    # left = (width - HW[1]) / 2
    # top = (height - HW[0]) / 2
    # right = (width + HW[1]) / 2
    # bottom = (height + HW[0]) / 2
    #
    # # Crop the center of the image
    # img = img.crop((left, top, right, bottom))
    # return np.asarray(img)
    return np.asarray(img.resize((HW[1], HW[0]), resample=resample))



def preprocess_img(img_rgb_orig, HW=(256, 256), resample=3, resize = True):
    # return original size L and resized L as torch Tensors
    if resize: img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    else: img_rgb_rs = np.asarray(img_rgb_orig)
    if len(img_rgb_rs.shape) <3:
        img_rgb_rs = np.stack((img_rgb_rs,)*3, axis=-1)

    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_ab_rs = img_lab_rs[:, :, 1:3]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_rs_ab = torch.Tensor(img_ab_rs).permute(2,0,1)
    tens_rs_l = torch.Tensor(img_l_rs)[None, :, :]
    tens_rxs_ab = F.interpolate(tens_rs_ab.unsqueeze(0), size=56).squeeze(0)

    return (tens_rs_ab, tens_rs_l, tens_rxs_ab )


def postprocess_tens(tens_orig_l, out_ab,j, mode='bilinear',HW_request=[224,224]):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W
    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]
    batch = list(tens_orig_l.shape)[0]
        #
    # call resize function if needed
    if (HW_orig[0] != HW[0] or HW_orig[1] != HW[1]) or (HW_orig[0] != HW_request[0] or HW_orig[1] != HW_request[1]):
        HW_orig = HW_request
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode=mode)
        tens_orig_l =  F.interpolate(tens_orig_l, size=HW_orig, mode=mode)
    else:
        out_ab_orig = out_ab

    # out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    if len(tens_orig_l.shape)>3 and int(batch) > 1: tens_orig_l = tens_orig_l[j,...]
    if len(tens_orig_l.shape)>3 and int(batch) ==1: tens_orig_l = tens_orig_l.squeeze(0)
    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig[j, ...]), dim=0)
    # return color.lab2rgb(out_lab_orig.data.cpu().numpy()[j, ...].transpose((1, 2, 0)))
    return color.lab2rgb(out_lab_orig.data.cpu().numpy().transpose((1, 2, 0)))

def postprocess_tens_old(tens_orig_l, out_ab,j, mode='bilinear'):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if (HW_orig[0] != HW[0] or HW_orig[1] != HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[j, ...].transpose((1, 2, 0)))


def original_l(data,index):
    path, _ = data.imgs[index]
    img = Image.open(path)
    img_rgb_ = img.convert('RGB')
    img_rgb_ = np.asarray(img_rgb_)
    img_lab_ = color.rgb2lab(img_rgb_)
    tens_orig = (torch.Tensor(img_lab_[:, :, 0])[None, :, :])

    return tens_orig

