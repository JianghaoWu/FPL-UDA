# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np

def get_soft_label(input_tensor, num_class, data_type = 'float'):
    """
        convert a label tensor to one-hot label 
        input_tensor: tensor with shae [B, 1, D, H, W] or [B, 1, H, W]
        output_tensor: shape [B, num_class, D, H, W] or [B, num_class, H, W]
    """

    shape = input_tensor.shape
    if len(shape) == 5:
        output_tensor = torch.nn.functional.one_hot(input_tensor[:, 0], num_classes = num_class).permute(0, 4, 1, 2, 3)
    elif len(shape) == 4:
        output_tensor = torch.nn.functional.one_hot(input_tensor[:, 0], num_classes = num_class).permute(0, 3, 1, 2)
    else:
        raise ValueError("dimention of data can only be 4 or 5: {0:}".format(len(shape)))
    
    if(data_type == 'float'):
        output_tensor = output_tensor.float()
    elif(data_type == 'double'):
        output_tensor = output_tensor.double()
    else:
        raise ValueError("data type can only be float and double: {0:}".format(data_type))

    return output_tensor

def one_hot_array(input_array, num_class=None):
    """
        convert a label array to one-hot label 
        input_array: tensor with shae [B, 1, D, H, W] or [B, 1, H, W]
        output_array: shape [B, num_class, D, H, W] or [B, num_class, H, W]
    """
    if not num_class:
        num_class = np.max(input_array)+1
    
    one_hot_list = []
    for cls in range(num_class):
        _vol = np.zeros(input_array.shape)
        _vol[input_array == cls] = 1
        one_hot_list.append(_vol)

    return np.concatenate(one_hot_list, axis=1)

def reshape_tensor_to_2D(x):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(x.size())
    num_class  = list(x.size())[1]
    if(tensor_dim == 5):
        x_perm  = x.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        x_perm  = x.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))
    
    y = torch.reshape(x_perm, (-1, num_class)) 
    return y 

def reshape_prediction_and_ground_truth(predict, soft_y):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(predict.size())
    num_class  = list(predict.size())[1]
    if(tensor_dim == 5):
        soft_y  = soft_y.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        soft_y  = soft_y.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))
    
    predict = torch.reshape(predict, (-1, num_class)) 
    soft_y  = torch.reshape(soft_y,  (-1, num_class))
      
    return predict, soft_y

def get_classwise_dice(predict, soft_y, pix_w = None):
    """
    get dice scores for each class in predict (after softmax) and soft_y
    """
    
    if(pix_w is None):
        y_vol = torch.sum(soft_y,  dim = 0)
        p_vol = torch.sum(predict, dim = 0)
        intersect = torch.sum(soft_y * predict, dim = 0)
    else:
        y_vol = torch.sum(soft_y * pix_w,  dim = 0)
        p_vol = torch.sum(predict * pix_w, dim = 0)
        intersect = torch.sum(soft_y * predict * pix_w, dim = 0)
    dice_score = (2.0 * intersect + 1e-5)/ (y_vol + p_vol + 1e-5)
    return dice_score 
