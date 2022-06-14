import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt # plt 用于显示图片
import os
import csv
import pandas as pd
import random
from random import shuffle
 
 
def save_array_as_nii_volume(data, filename, reference_name):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    # print(img.shape)
    # img = img.rotate(90)  
    # print(img.shape)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)

'''image_path = '/data2/jianghao/data/VS/T1-T2-T1/T1toT1_PNG'
# glob.glob(r"/home/qiaoyunhao/*/*.png"
image_names = glob.glob(r"/data2/jianghao/data/VS/T1-T2-T1/T1toT1_PNG/*.png")
# print(len(image_names),'999')
filenames = []
# image_names = os.listdir(image_path)

# image_names = [item for item in image_names if "ceT1" in item]
image_names.sort()'''
for i in range(1,106):
# for i in [30,40,49,56,82,86,88,93]:
# def allimage(i):
    # image_path = '/mnt/39E12DAE493BA6C1/wujianghao/data/trainA_cycle_png'
    # image_path = '/data1/wujh/data/VS/valid/validPNG'
# glob.glob(r"/home/qiaoyunhao/*/*.png"
    image_names = glob.glob(r"/mnt/39E12DAE493BA6C1/wujianghao/data/trainA/T2_T1/fake_B/*fake.png")
# print(len(image_names),'999')
    filenames = []
# image_names = os.listdir(image_path)

# image_names = [item for item in image_names if "ceT1" in item]
    image_names.sort()
    image_names = [item for item in image_names if 'crossmoda_'+(str(i))+'_ce' in item]
    # print(image_names)
    image_arr = image_names
    # image_arr = glob.glob(str(image_names) + str("/*"))
    # print(image_arr,'666')
    # image_arr.sort()
#  '/data1/wujh/data/VS/valid/validPNG_subfile/crossmoda_242_hrT2.nii.gz/crossmoda_242_hrT2.ni_z018.png'
    print(image_arr, len(image_arr))
    allImg = []
    allImg = np.zeros([len(image_arr), 160,272], dtype='uint8') #40
    for c in range(len(image_arr)):
        single_image_name = image_arr[c]
    # img_as_img = Image.open(single_image_name)
        img_as_img = Image.open(single_image_name).convert('L')
    # img_as_img.show()
        img_as_np = np.asarray(img_as_img)
        img_as_np = img_as_np.T

    # img_as_np.SetSpacing(spacing)
    # spacing = np.array[0.4102,0.4102,1.5]
    # resample = sitk.ResampleImageFilter()
    # resample.SetOutputSpacing(spacing)



        allImg[c, :, :] = img_as_np
 
 
# np.transpose(allImg,[2,0,1])
# allImg = allImg*[[1,0,0],[0,0,1],[0,1,0]]
    print(str(i))
    save_array_as_nii_volume(allImg, '/mnt/39E12DAE493BA6C1/wujianghao/data/trainA/nii/crossmoda_'+str(i)+'_ceT1.nii.gz','/mnt/39E12DAE493BA6C1/wujianghao/data/data_nii/source_training_crop_real/crossmoda_'+str(i+300)+'_ceT1.nii.gz')
    # save_array_as_nii_volume(allImg, '/data1/wujh/data/VS/106-210-GR/crossmoda_'+str(i)+'_hrT2.nii.gz','/data1/wujh/data/VS/nii_real_no_label/T2/crossmoda_'+str(i)+'_hrT2.nii.gz')
    print(np.shape(allImg))
    # img = allImg[:, :, 55]
# plt.imshow(img, cmap='gray')
# plt.show()