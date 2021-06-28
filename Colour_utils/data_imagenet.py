from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
import torch
import numpy as np
from PIL import Image
import os
import utils_code.util_zhang as util_zhang

class RGB2LAB(object):
    def __init__(self):
        super(RGB2LAB,self).__init__()
    def __call__(self,img):
        (tens_rs_ab, tens_rs_l, tens_rxs_ab)= util_zhang.preprocess_img(img, HW=(224, 224), resample=3)
        return tens_rs_ab, tens_rs_l, tens_rxs_ab

original_transform = transforms.Compose([RGB2LAB()])

class ValImageFolder(datasets.ImageFolder):
#    def __init__(self,data_dir):
 #       self.file_list=os.listdir(data_dir)
  #      self.data_dir=data_dir

    def __init__(self, root, transform=None):
        super(ValImageFolder, self).__init__(root, transform=transform)

    def __getitem__(self, index):

        path,_ =self.imgs[index]
        name_file=path.split('/')[-1]
        target = path.split(os.sep)
        img= Image.open(path)
        img = img.convert('RGB')
        img_scale = self.transform(img)
        target = target[-2] #torch.tensor(int(target[-2]))

        return img_scale,index, target, name_file

    def __len__(self):
        return len(self.imgs)

nomalization_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
class ValImageFolder_generated(datasets.ImageFolder):
#    def __init__(self,data_dir):
 #       self.file_list=os.listdir(data_dir)
  #      self.data_dir=data_dir

    def __getitem__(self, index):

        path,_ =self.imgs[index]
        target = path.split(os.sep)
        img = Image.open(path)
        img = img.crop
        img = img.resize((224,224))
        img = img.convert('RGB')
        img_tensor = nomalization_transform(img)
        target = target[-2] #torch.tensor(int(target[-2]))
        return img_tensor, target

    def __len__(self):
        return len(self.imgs)

class ValTrainImageFolder(datasets.ImageFolder):
    def __getitem__(self,index):
        try:
            path,_=self.imgs[index] 
            img=self.loader(path)
            img = img.convert('RGB')
            if self.transform is not None: img_original = self.transform(img)
            else: img_original = img

            img_resize=transforms.Resize(56)(img_original)
            #train
            img_lab = rgb2lab(img_resize)
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
            img_ab = img_ab.type(torch.FloatTensor)
            
            img_grey = np.array(img_original)
            img_grey = rgb2lab(img_grey)[:,:,0]-50.
            img_grey = torch.from_numpy(img_grey)
            img_grey = img_grey.type(torch.FloatTensor)
            #val
            img_rescale = np.asarray(img_resize)
            img_rescale = rgb2lab(img_rescale)[:,:,0]-50
            img_rescale = torch.from_numpy(img_rescale)
            
            img_original = np.asarray(img_original)
            img_original = torch.from_numpy(img_original)

            return img_grey, img_ab, img_original, img_rescale

        except:
            print('exception')
            print(index)
            print(self.imgs[index])
            pass 
