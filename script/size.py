import SimpleITK as sitk
import numpy as np
import os
import scipy.ndimage
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F  
import torch
from PIL import Image
from scipy import ndimage 

def target_image_crop():
    img_dir = "./data/4_fake_nii/Test_cycle"
    img_names = os.listdir(img_dir)
    img_names = [item for item in img_names if "t2" in item]
    print(len(img_names))
    for img_name in img_names:
        img_obj = sitk.ReadImage(img_dir + '/' + img_name)
        img = sitk.GetArrayFromImage(img_obj)
        print(img.shape)
        a,b,c = img.shape
        if a < 24:
            print(img.shape)
            print(img_name)

target_image_crop()