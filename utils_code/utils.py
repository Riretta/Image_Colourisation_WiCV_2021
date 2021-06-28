import os
from torch import save, load, stack, from_numpy,FloatTensor
import numpy as np
import torch.nn.functional as F
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from Colour_utils.base_color import BaseColor


def load_image(image_tensor):
    images = []
    for i in range(len(image_tensor)):
        image_single = rgb2lab(image_tensor[i])[:, :, 0] - 50.
        image_single = from_numpy(image_single).unsqueeze(0)
        images.append(image_single)
    images = stack(images)
    return images

def resume_model(name_file, model, map_location):
    if os.path.isfile(name_file):
        print("=> loading checkpoint '{}'".format(name_file))
        checkpoint = load(name_file, map_location=map_location)
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(name_file, checkpoint['epoch']))
        epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(name_file))

    return model, epoch


def _decode(data_l, reconstructionQ, rebalance=1):
    bs = BaseColor()
    image_rgb_list = []
    for i in range(len(data_l)):
        data_l_single = data_l[i] #+ 50
        data_l_single = data_l_single.unsqueeze(2).cpu().data.numpy()#.transpose((1, 2, 0))
        reconstructionQ_single = reconstructionQ[i]
        reconstructionQ_rh = reconstructionQ_single * rebalance
        reconstructionQ_rh = F.softmax(reconstructionQ_rh, dim=0).cpu().data.numpy().transpose((1, 2, 0))
        class8 = np.argmax(reconstructionQ_rh, axis=-1)

        cc = np.load(os.path.join('./resources', 'pts_in_hull.npy'))
        data_ab = cc[class8[:][:]]

        #data_ab = bs.normalize_ab(from_numpy(data_ab)).permute(2, 0, 1).unsqueeze(0)
        data_ab = from_numpy(data_ab).permute(2, 0, 1).unsqueeze(0)
        data_ab = data_ab.type(FloatTensor)
        data_ab = F.interpolate(data_ab, size=224).float().squeeze(0).permute(1,2,0)
        img_lab = np.concatenate((data_l_single, data_ab.numpy()), axis=-1)
        try:
            img_rgb = lab2rgb(img_lab)
        except Exception as ex:
            print('_decode.lab2rgb exception: {}'.format(ex))
        image_rgb_list.append((img_rgb, img_lab))

    return image_rgb_list

def _decodeAB(data_l, reconstructionAB, rebalance=1):
    image_rgb_list = []
    for i in range(len(data_l)):
        data_l_single = data_l[i] + 50
        data_l_single = data_l_single.unsqueeze(0).cpu().data.numpy().transpose((1, 2, 0))
        reconstructionAB_single = reconstructionAB[i]
        reconstructionAB_rh = reconstructionAB_single * rebalance
        reconstructionAB_rh = reconstructionAB_rh.cpu().data.numpy().transpose((1, 2, 0))
        img_lab = np.concatenate((data_l_single, reconstructionAB_rh), axis=-1)
        img_rgb = lab2rgb(img_lab)
        image_rgb_list.append((img_rgb, img_lab))
    return image_rgb_list


def _decode_fake(data_l, AB):
    image_rgb_list = []
    for i in range(len(data_l)):
        data_l_single = data_l[i] + 50
        data_l_single = data_l_single.unsqueeze(0).cpu().data.numpy().transpose((1, 2, 0))
        AB_rh = AB[i].cpu().data.numpy()#.transpose((1, 2, 0))
        img_lab = np.concatenate((data_l_single, AB_rh), axis=-1)
        img_rgb = lab2rgb(img_lab)
        image_rgb_list.append((img_rgb, img_lab))
    return image_rgb_list


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def display_image(images, idx=None):
    if idx is None:
        plt.imshow(images)
    else:
        plt.imshow(images[idx])
    plt.show()

import shutil
def dir_mk(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)
