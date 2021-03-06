"""Script for writing cvs files
"""

import os
import csv
import pandas as pd
import random
from random import shuffle

def create_csv_file(data_dir, output_file, fields):
    """
    create a csv file to store the paths of files for each patient
    """
    filenames = []
    image_names = os.listdir(data_dir)
    lab_dir = '/data2/jianghao/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task001_MulOrganPEVD_Starv2/labelsTs/'
    image_names = [item for item in image_names if "nii.gz" in item]
    image_names.sort()
    print('total number of images {0:}'.format(len(image_names)))
    for img_name_ in image_names:
        img_name = data_dir + img_name_
        # lab_name = img_name.replace("_z", "_seg_z")
        lab_name = lab_dir + img_name_
        filenames.append([img_name, lab_name])

    with open(output_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(fields)
        for item in filenames:
            csv_writer.writerow(item)

def random_split_dataset():
    random.seed(2021)
    input_file = 'vs_seg2021/config_best40/no2_all.csv'
    train_names_file = 'vs_seg2021/config_best40/no2_train.csv'
    valid_names_file = 'vs_seg2021/config_best40/no2_valid.csv'
    # test_names_file  = 'config/image_test.csv'
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    shuffle(data_lines)
    N = len(data_lines)
    n1 = int(N * 0.8)
    #n2 = int(N * 0.8)
    print('image number', N)
    print('training number', n1)
    print('validation number', N - n1)
    #print('testing number', N - n2)
    train_lines  = sorted(data_lines[:n1])
    valid_lines  = sorted(data_lines[n1:])
    #test_lines   = data_lines[n2:]
    with open(train_names_file, 'w') as f:
        f.writelines(lines[:1] + train_lines)
    with open(valid_names_file, 'w') as f:
        f.writelines(lines[:1] + valid_lines)
    #with open(test_names_file, 'w') as f:
    #    f.writelines(lines[:1] + test_lines)

def get_evaluation_image_pairs(test_csv, gt_seg_csv):
    with open(test_csv, 'r') as f:
        input_lines = f.readlines()[1:]
        output_lines = []
        for item in input_lines:
            gt_name = item.split(',')[1].rstrip()
            seg_name = item.split(',')[0].rstrip()
            output_lines.append([gt_name, seg_name])
    with open(gt_seg_csv, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["ground_truth", "segmentation"])
        for item in output_lines:
            csv_writer.writerow(item)


if __name__ == "__main__":
    # create cvs file for promise 2012
    fields      = ['image', 'label']
    data_dir    = '/data2/jianghao/nnUNetFrame/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task001_MulOrganPEVD_Starv2/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw_postprocessed/'
    output_file = './config_best40/mulorgan_ct_valid.csv'
    create_csv_file(data_dir, output_file, fields)

    # split the data into training, validation and testing
    # random_split_dataset()

    # # obtain ground truth and segmentation pairs for evaluation
    # test_csv    = "./vs_seg2021/config_best40/no2_valid.csv"
    # gt_seg_csv  = "./vs_seg2021/config_best40/no2_valid_seg.csv"
    # get_evaluation_image_pairs(test_csv, gt_seg_csv)

