#!/usr/bin/env python
import torch

import cv2
import numpy
import os
from utils import softsplat, warp
from utils.io import read_pfm
from RRIN_master.model import Net
import argparse
assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()

    assert(numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=1, offset=0) == 202021.25)

    intWidth = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=4)[0]
    intHeight = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=8)[0]

    return numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=intHeight * intWidth * 2, offset=12).reshape([ intHeight, intWidth, 2 ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='result')
    parser.add_argument('--task', type=str, default='validation', choices=['validation', 'testing'])
    parser.add_argument('--skip-task0', action='store_true')
    parser.add_argument('--skip-task1', action='store_true')
    parser.add_argument('--skip-task2', action='store_true')
    args = parser.parse_args()

    if (args.task == 'validation'):
        dir_path = [[i for i in range(7)],[i for i in range(3)], [i for i in range(3)]]
    else:
        dir_path = [[i for i in range(7,17)],[i for i in range(3,5)], [i for i in range(3,5)]]

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.isdir(f'{args.save_dir}/{args.task}'):
        os.mkdir(f'{args.save_dir}/{args.task}')
    
    RRIN = Net()
    state = torch.load('pretrained_model.pth.tar')
    RRIN.load_state_dict(state,strict=True)
    RRIN = RRIN.cuda()
    RRIN.eval()
    
    #Task1
    ##################################
    if not args.skip_task0:
        print("task 0 start...")
        data_dir = f'data/{args.task}/0_center_frame'
        task_dir = f'{args.save_dir}/{args.task}/0_center_frame'
        if not os.path.isdir(task_dir):
            os.mkdir(task_dir)

        for num in dir_path[0]:
            if not os.path.isdir(f'{task_dir}/{num}'):
                os.mkdir(f'{task_dir}/{num}')

            info_path = f'{data_dir}/{num}/input'

            tenFirst = torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename=f'{info_path}/frame10.png', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
            tenSecond = torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename=f'{info_path}/frame11.png', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
            tenFlow01 = torch.FloatTensor(numpy.ascontiguousarray(read_flo(f'{info_path}/flow01.flo').transpose(2, 0, 1)[None, :, :, :])).cuda()
            tenFlow10 = torch.FloatTensor(numpy.ascontiguousarray(read_flo(f'{info_path}/flow10.flo').transpose(2, 0, 1)[None, :, :, :])).cuda()
            depth0,_ = read_pfm(f'{info_path}/frame10.pfm')
            depth1,_ = read_pfm(f'{info_path}/frame11.pfm')
            
            img = warp.image_generate(0.5, tenFirst, tenSecond, tenFlow01, tenFlow10, depth0, depth1, RRIN, hole_fill=True, frame_refinement= True)
            cv2.imwrite(f'{task_dir}/{num}/frame10i11.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print("task 0 finish.")

    if not args.skip_task1:
        print("task 1 start...")
        data_dir = f'data/{args.task}/1_30fps_to_240fps'
        task_dir = f'{args.save_dir}/{args.task}/1_30fps_to_240fps'
        if not os.path.isdir(task_dir):
            os.mkdir(task_dir)

        for outer in dir_path[1]:
            if not os.path.isdir(f'{task_dir}/{outer}'):
                os.mkdir(f'{task_dir}/{outer}')
            for inner in range(12):
                if not os.path.isdir(f'{task_dir}/{outer}/{inner}'):
                    os.mkdir(f'{task_dir}/{outer}/{inner}')

                info_path = f'{data_dir}/{outer}/{inner}/input'
                start = inner*8
                tenFirst = torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename=f'{info_path}/{str(start).zfill(5)}.jpg', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
                tenSecond = torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename=f'{info_path}/{str(start+8).zfill(5)}.jpg', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
                tenFlow01 = torch.FloatTensor(numpy.ascontiguousarray(read_flo(f'{info_path}/flow01.flo').transpose(2, 0, 1)[None, :, :, :])).cuda()
                tenFlow10 = torch.FloatTensor(numpy.ascontiguousarray(read_flo(f'{info_path}/flow10.flo').transpose(2, 0, 1)[None, :, :, :])).cuda()
                depth0,_ = read_pfm(f'{info_path}/{str(start).zfill(5)}.pfm')
                depth1,_ = read_pfm(f'{info_path}/{str(start+8).zfill(5)}.pfm')
                
                for stamp in range(1,8):
                    img = warp.image_generate(stamp/8, tenFirst, tenSecond, tenFlow01, tenFlow10, depth0, depth1, RRIN, hole_fill=True, frame_refinement= False)
                    cv2.imwrite(f'{task_dir}/{outer}/{inner}/{str(start+stamp).zfill(5)}.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print(f"subtask 1-{outer} finish.")
        print("task 1 finish.")

    if not args.skip_task2:
        data_dir = f'data/{args.task}/2_24fps_to_60fps'
        task_dir = f'{args.save_dir}/{args.task}/2_24fps_to_60fps'
        if not os.path.isdir(task_dir):
            os.mkdir(task_dir)

        for outer in dir_path[2]:
            if not os.path.isdir(f'{task_dir}/{outer}'):
                os.mkdir(f'{task_dir}/{outer}')
            for inner in range(8):
                if not os.path.isdir(f'{task_dir}/{outer}/{inner}'):
                    os.mkdir(f'{task_dir}/{outer}/{inner}')

                info_path = f'{data_dir}/{outer}/{inner}/input'
                start = inner*10
                tenFirst = torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename=f'{info_path}/{str(start).zfill(5)}.jpg', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
                tenSecond = torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename=f'{info_path}/{str(start+10).zfill(5)}.jpg', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
                tenFlow01 = torch.FloatTensor(numpy.ascontiguousarray(read_flo(f'{info_path}/flow01.flo').transpose(2, 0, 1)[None, :, :, :])).cuda()
                tenFlow10 = torch.FloatTensor(numpy.ascontiguousarray(read_flo(f'{info_path}/flow10.flo').transpose(2, 0, 1)[None, :, :, :])).cuda()
                depth0,_ = read_pfm(f'{info_path}/{str(start).zfill(5)}.pfm')
                depth1,_ = read_pfm(f'{info_path}/{str(start+10).zfill(5)}.pfm')
                
                for stamp in [4-start%4, 8-start%4]:
                    img = warp.image_generate(stamp/10, tenFirst, tenSecond, tenFlow01, tenFlow10, depth0, depth1, RRIN, hole_fill=True, frame_refinement= False)
                    cv2.imwrite(f'{task_dir}/{outer}/{inner}/{str(start+stamp).zfill(5)}.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print(f"subtask 2-{outer} finish.")
        print("task 2 finish.")