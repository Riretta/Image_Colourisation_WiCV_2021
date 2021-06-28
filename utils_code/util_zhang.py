from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed
from Colour_utils.data_imagenet import  ValImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if (out_np.ndim == 2):
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


def resize_img(img, HW=(256, 256), resample=3):
    return np.asarray(img.resize((HW[1], HW[0]), resample=resample)) #Image.fromarray(


def preprocess_img(img_rgb_orig, HW=(256, 256), resample=3):
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_ab_rs = img_lab_rs[:, :, 1:3]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_rs_ab = torch.Tensor(img_ab_rs).permute(2,0,1)
    tens_rs_l = torch.Tensor(img_l_rs)[None, :, :]
    tens_rxs_ab = F.interpolate(tens_rs_ab.unsqueeze(0), size=56).squeeze(0)

    return (tens_rs_ab, tens_rs_l, tens_rxs_ab)


def postprocess_tens(tens_orig_l, out_ab,j, mode='bilinear'):
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

def load_dataset(dataset_path_train,dataset_path_val,batch_size):
    class RGB2LAB(object):
        def __init__(self):
            super(RGB2LAB, self).__init__()

        def __call__(self, img):
            (tens_rs_ab, tens_rs_l, tens_rxs_ab) = preprocess_img(img, HW=(224, 224), resample=3)
            return tens_rs_ab, tens_rs_l, tens_rxs_ab

    original_transform = transforms.Compose([RGB2LAB()])

    ImageDataset = {'train': ValImageFolder(dataset_path_train, transform=original_transform),
                    # datasets.ImageNet(dataset_path,split='train',transform=original_transform),
                    'val': ValImageFolder(dataset_path_val, transform=original_transform)}
    dataloaders = {'train': DataLoader(ImageDataset['train'], batch_size=batch_size, shuffle=True, num_workers=4,
                                       pin_memory=False),
                   'val': DataLoader(ImageDataset['val'], batch_size=batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)}
    return dataloaders

def load_Imagenet_dataset(dataset_path,batch_size):
    class RGB2LAB(object):
        def __init__(self):
            super(RGB2LAB, self).__init__()

        def __call__(self, img):
            (tens_rs_ab, tens_rs_l, tens_rxs_ab) = preprocess_img(img, HW=(224, 224), resample=3)
            return tens_rs_ab, tens_rs_l, tens_rxs_ab

    original_transform = transforms.Compose([RGB2LAB()])

    ImageDataset = {'train': datasets.ImageNet(dataset_path, split='train', transform=original_transform),
                    'val': datasets.ImageNet(dataset_path, split='val', transform=original_transform)}
    dataloaders = {'train': DataLoader(ImageDataset['train'], batch_size=batch_size, shuffle=False, num_workers=4,
                                       pin_memory=False),
                   'val': DataLoader(ImageDataset['val'], batch_size=batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)}
    return dataloaders
