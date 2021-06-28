import os
import cv2


val_dir_name = "/media/TBData/Rita/Datasets/ImageNet/Copy_generated/10k_nocrop"

for class_folder in os.listdir(val_dir_name):
    for image in os.listdir(os.path.join(val_dir_name, class_folder)):
        original = cv2.imread(os.path.join(val_dir_name, class_folder, image))
        cv2.imwrite("/media/TBData/Rita/Datasets/ImageNet/data_together/"+image.replace('JPEG', 'png'), original)