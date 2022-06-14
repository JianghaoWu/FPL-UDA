import tensorflow as tf
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

def save_array_as_nii_volume(img_nii, filename, reference_name):
    sitk_img = sitk.ReadImage(img_nii)
    img = sitk.GetArrayFromImage(sitk_img)  # indexes are z,y,x (notice the ordering)
    print(img.shape,'old')
    zoom = [1.0, 14.0/16.0, 238.0/272.0]    # 12   204 for 213,,240;;;;  14    238 for other
    img = ndimage.zoom(img, zoom)
    print(img.shape,'new')
    #convert array to image object
    # out_img_obj = sitk.GetImageFromArray(img_sub)

    # im=Image.open("heibai.jpg",'r')
 
    img=torch.Tensor(np.asarray(img))
    # print("shape:",img.shape)
    
    # p3d = (88,88,144,144,10,30)     
    p3d = (105,105,179,129,10,30)   #最多的
    # p3d = (105,105,179,129,5,25)    #219,220
    # p3d = (90,90,153,111,5,5)    # 213,,,,240
    img = F.pad(img,p3d,'constant',0)
    X = sitk.GetImageFromArray(X)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        X.CopyInformation(img_ref)
    sitk.WriteImage(X, filename)

# for i in range(219,221):
for i in [211,212,214,215,216,217,218,221,222,223,224,225,226,227,228,230,231,232,233,234,235,236,237,239,241,242]:
# for i in [213,229,238,240]:
    print(i)
    img_nii = '/mnt/39E12DAE493BA6C1/wujianghao/0813/data/T2_TRAIN_2/seg_submit/crossmoda_'+str(i)+'_hrT2.nii.gz'
# save_array_as_nii_volume(img_nii, )
    save_array_as_nii_volume(img_nii, '/mnt/39E12DAE493BA6C1/wujianghao/0813/data/T2_TRAIN_2/seg_submit/0813_submit/crossmoda_'+str(i)+'_hrT2.nii.gz','/mnt/39E12DAE493BA6C1/wujianghao/data/target_valid_no_crop/crossmoda_'+str(i)+'_hrT2.nii.gz')
# print(np.shape(allImg))

# img = allImg[:, :, 55]
# img_nii = sitk.ReadImage(filename)
# zoom2shape(img_nii,'/data1/wujh/crossMoDA/cycleGAN/contrastive-unpaired-translation/results/VS_CUT/train_latest/nii/sped.nii.gz')