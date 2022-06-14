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

def save_array_as_nii_volume(img_nii, filename, reference_name):
    sitk_img = sitk.ReadImage(img_nii)
    img = sitk.GetArrayFromImage(sitk_img)  # indexes are z,y,x (notice the ordering)

    # im=Image.open("heibai.jpg",'r')
 
    img=torch.Tensor(np.asarray(img))
    print("shape:",img.shape)
    p3d = (88,88,144,144,20,20)
    # p3d = (56,56,112,112,0,0)
    img = F.pad(img,p3d,'constant',0)
    print('t3的矩阵大小为',img.shape)
    # dim=(10,10,10,10)
    # X=F.pad(img,dim,"constant",value=0)
 
    X=img.data.numpy()
    X = sitk.GetImageFromArray(X)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        X.CopyInformation(img_ref)
    sitk.WriteImage(X, filename)

for i in range(2,243): 
# for i in [213,229,238,240]:
    print(i)
    img_nii = '/mnt/39E12DAE493BA6C1/wujianghao/vs_seg2021/result/T1TOT2_0730_T2_crop/crossmoda_'+str(i)+'_hrT2.nii.gz'
# save_array_as_nii_volume(img_nii, )
    save_array_as_nii_volume(img_nii, '/mnt/39E12DAE493BA6C1/wujianghao/vs_seg2021/result/T1TOT2_0730_T2_crop/padding/crossmoda_'+str(i)+'_hrT2.nii.gz','/mnt/39E12DAE493BA6C1/wujianghao/vs_seg2021/result/T1TOT2_0730_T2_nocrop/crossmoda_'+str(i)+'_hrT2.nii.gz')
# print(np.shape(allImg))

# img = allImg[:, :, 55]
# img_nii = sitk.ReadImage(filename)
# zoom2shape(img_nii,'/data1/wujh/crossMoDA/cycleGAN/contrastive-unpaired-translation/results/VS_CUT/train_latest/nii/sped.nii.gz')
