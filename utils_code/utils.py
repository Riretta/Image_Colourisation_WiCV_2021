import os
from torch import save, load, stack, from_numpy, nn
import numpy as np
import torch.nn.functional as F
from skimage.color import rgb2lab, lab2rgb

def rfolder(name_folder, ID, resume=False):
    if resume: name_folder = name_folder+'_Resume'
    if not os.path.exists(name_folder):
        os.mkdir(name_folder)

    folder_results = os.path.join(name_folder, "_" + str(ID))

    if not os.path.exists(folder_results):
        os.mkdir(folder_results)
    else:
        print("abspath", os.path.dirname(folder_results))
        counter = len(os.listdir(os.path.dirname(folder_results)))
        folder_results = folder_results + "_" + str(counter + 1)
        os.mkdir(folder_results)

    if not os.path.exists(os.path.join(folder_results, "model_log")) and os.path.exists(folder_results):
        os.mkdir(os.path.join(folder_results, "model_log"))
        os.mkdir(os.path.join(folder_results, "BACKUP_model_log"))

    print(" saved in  ", folder_results)
    return folder_results


def lr_decrease(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        init_lr = param_group['lr']
        param_group['lr'] = init_lr * lr_decay


def isnan(x):
    return x != x


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    save(state, filename)


def resume_model(name_file, model, optimizer, map_location):
    if os.path.isfile(name_file):
        print("=> loading checkpoint '{}'".format(name_file))
        checkpoint = load(name_file, map_location=map_location)
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(name_file, checkpoint['epoch']))
        epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(name_file))

    return model, optimizer, epoch



def load_image(image_tensor):
    images = []
    for i in range(len(image_tensor)):
        image_single = rgb2lab(image_tensor[i])[:, :, 0] - 50.
        image_single = from_numpy(image_single).unsqueeze(0)
        images.append(image_single)
    images = stack(images)
    return images


def _decode(data_l, reconstructionQ, rebalance=1):
    image_rgb_list = []
    for i in range(len(data_l)):
        data_l_single = data_l[i] + 50
        data_l_single = data_l_single.unsqueeze(0).cpu().data.numpy().transpose((1, 2, 0))
        reconstructionQ_single = reconstructionQ[i]
        reconstructionQ_rh = reconstructionQ_single * rebalance
        reconstructionQ_rh = F.softmax(reconstructionQ_rh, dim=0).cpu().data.numpy().transpose((1, 2, 0))
        class8 = np.argmax(reconstructionQ_rh, axis=-1)

        cc = np.load(os.path.join('../resources', 'pts_in_hull.npy'))
        data_ab = cc[class8[:][:]]
        img_lab = np.concatenate((data_l_single, data_ab), axis=-1)
        img_rgb = lab2rgb(img_lab)
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


def reconstruction_loss(data, reconstructions,criterion):
    reconstructions_A = reconstructions[:,0,:,:]
    data_A = data[:,0,:,:]
    reconstructions_B = reconstructions[:,1,:,:]
    data_B = data[:,1,:,:]
    loss_A = criterion(reconstructions_A.view(reconstructions.size(0), -1), data_A.view(reconstructions.size(0), -1))
    loss_B = criterion(reconstructions_B.view(reconstructions.size(0), -1), data_B.view(reconstructions.size(0), -1))

    loss = loss_A + loss_B
    return loss #* 0.001