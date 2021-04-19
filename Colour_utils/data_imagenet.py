
from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
from skimage import io
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

scale_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip()
    #transforms.ToTensor()
])


class TrainImageFolder(datasets.ImageFolder):
    #def __init__(self, data_dir, transform):
    #   self.file_list=os.listdir(data_dir)
    #   self.transform=transform
    #   self.data_dir=data_dir
    def __getitem__(self, index):
 #       try:

         avoiding = True
         while avoiding:
             try:
                 path,_=self.imgs[index]
                 img=self.loader(path)
                 avoiding = False 
             except Exception as e:
                 # print('lost ',path)
                 # print('EXCEPTION ',e)
                 index = index + 1
             
            #img=Image.open(self.data_dir+'/'+self.file_list[index])
         if self.transform is not None:

                img_original = self.transform(img)
                img_resize=transforms.Resize(56)(img_original)
                img_original = np.array(img_original)
                img_lab = rgb2lab(img_resize)
                img_ab = img_lab[:, :, 1:3]
                img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
                img_ab = img_ab.type(torch.FloatTensor)

                img_l = rgb2lab(img_original)[:,:,0]#-50.
                img_l = torch.from_numpy(img_l)
                img_l = img_l.type(torch.FloatTensor)

                img_lab = torch.from_numpy(img_lab)
                img_lab = img_lab.type(torch.FloatTensor)
                return img_l, img_ab, img_lab
         else:
                print('no transformation')
#        except:
#            print('exception') 
#            print(index)
#            print(self.imgs[index])
#            pass

    def __len__(self):
        return len(self.imgs)

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

        return img_scale, target

    def __len__(self):
        return len(self.imgs)


class ValTrainImageFolder(datasets.ImageFolder):
    def __getitem__(self,index):
        try:
            path,_=self.imgs[index] 
            img=self.loader(path)
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
