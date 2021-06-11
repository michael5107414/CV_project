"""
pwcnet_predict_from_img_pairs.py

Run inference on a list of images pairs.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
from copy import deepcopy
from skimage.io import imread
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import display_img_pairs_w_flows
from optflow import flow_write
import os
import numpy as np
import argparse

# TODO: Set the path to the trained model (make sure you've downloaded it first from http://bit.ly/tfoptflow)
ckpt_path = './models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

# Configure the model for inference, starting with the default options
# Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = ckpt_path
nn_opts['batch_size'] = 1
nn_opts['gpu_devices'] = ['/device:GPU:0']  
nn_opts['controller'] = '/device:GPU:0'

# We're running the PWC-Net-large model in quarter-resolution mode
# That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

# The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
# of 64. Hence, we need to crop the predicted flows to their original size
nn_opts['adapt_info'] = (1, 436, 1024, 2)

def flow_generate(data_dir, path1, path2):
    image1, image2 = imread(f'{data_dir}/{path1}'), imread(f'{data_dir}/{path2}')
    img_pairs = [(image1, image2), (image2, image1)]
    pred_labels = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)
    flow_write(pred_labels[0], f'{data_dir}/flow01.flo')
    flow_write(pred_labels[1], f'{data_dir}/flow10.flo')

# Instantiate the model in inference mode and display the model configuration
if __name__ == '__main__':
    nn = ModelPWCNet(mode='test', options=nn_opts)
    nn.print_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['validation', 'test'], required=True)
    args = parser.parse_args()

    if (args.task == 'validation'):
        dir_path = [[i for i in range(7)],[i for i in range(3)], [i for i in range(3)]]
    else:
        dir_path = [[i for i in range(7,12)],[i for i in range(3,5)], [i for i in range(3,5)]]

    print("produce task1 flow...")
    for num in dir_path[0]:
        data_dir = f'../data/{args.task}/0_center_frame/{num}/input'
        flow_generate(data_dir, 'frame10.png', 'frame11.png')

    print("produce task2 flow...")
    data_dir = f'../data/{args.task}/1_30fps_to_240fps'
    for outer in dir_path[1]:
        for inner in range(12):
            data_dir = f'../data/{args.task}/1_30fps_to_240fps/{outer}/{inner}/input'
            flow_generate(data_dir, f'{str(inner*8).zfill(5)}.jpg', f'{str(inner*8+8).zfill(5)}.jpg')

    print("produce task3 flow...")
    for outer in dir_path[2]:
        for inner in range(8):
            data_dir = f'../data/{args.task}/2_24fps_to_60fps/{outer}/{inner}/input'
            flow_generate(data_dir, f'{str(inner*10).zfill(5)}.jpg', f'{str(inner*10+10).zfill(5)}.jpg')