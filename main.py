#!/usr/bin/env python
import torch

import cv2
import numpy
import os
import softsplat
import argparse
assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()

    assert(numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=1, offset=0) == 202021.25)

    intWidth = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=4)[0]
    intHeight = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=8)[0]

    return numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=intHeight * intWidth * 2, offset=12).reshape([ intHeight, intWidth, 2 ])


backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

def backwarp_img_generate(flowt0, flowt1, tenFirst, tenSecond):
    hole = numpy.where((flowt0[0,:,:]==0)&(flowt0[1,:,:]==0), True, False)
    pad_flow = numpy.pad(flowt0, ((0,),(1,),(1,)), 'constant')
    weight = numpy.where((pad_flow[0,0:-2,1:-1]!=0)|(pad_flow[1,0:-2,1:-1]!=0), 1, 0) + \
             numpy.where((pad_flow[0,2:,1:-1]!=0)|(pad_flow[1,2:,1:-1]!=0), 1, 0) + \
             numpy.where((pad_flow[0,1:-1,0:-2]!=0)|(pad_flow[1,1:-1,0:-2]!=0), 1, 0) + \
             numpy.where((pad_flow[0,1:-1,2:]!=0)|(pad_flow[1,1:-1,2:]!=0), 1, 0)
    weight = numpy.where(weight==0, 40, weight)
    
    neighbor = pad_flow[:,0:-2,1:-1] + pad_flow[:,2:,1:-1] + pad_flow[:,1:-1,0:-2] + pad_flow[:,1:-1,2:]
    flowt0 = numpy.where(hole, neighbor/weight,flowt0)
    
    hole = numpy.where((flowt1[0,:,:]==0)&(flowt1[1,:,:]==0), True, False)
    pad_flow = numpy.pad(flowt1, ((0,),(1,),(1,)), 'constant')
    weight = numpy.where((pad_flow[0,0:-2,1:-1]!=0)|(pad_flow[1,0:-2,1:-1]!=0), 1, 0) + \
             numpy.where((pad_flow[0,2:,1:-1]!=0)|(pad_flow[1,2:,1:-1]!=0), 1, 0) + \
             numpy.where((pad_flow[0,1:-1,0:-2]!=0)|(pad_flow[1,1:-1,0:-2]!=0), 1, 0) + \
             numpy.where((pad_flow[0,1:-1,2:]!=0)|(pad_flow[1,1:-1,2:]!=0), 1, 0)
    weight = numpy.where(weight==0, 40, weight)
    
    neighbor = pad_flow[:,0:-2,1:-1] + pad_flow[:,2:,1:-1] + pad_flow[:,1:-1,0:-2] + pad_flow[:,1:-1,2:]
    flowt1 = numpy.where(hole, neighbor/weight,flowt1)
    
    flowt0[0,:,:] += numpy.arange(flowt0.shape[2])
    flowt0[1,:,:] += numpy.arange(flowt0.shape[1])[:,numpy.newaxis]
    backwardimgt0 =  cv2.remap(tenFirst.cpu().numpy().squeeze().transpose(1,2,0).astype(numpy.float32), flowt0.transpose(1,2,0).astype(numpy.float32), None, cv2.INTER_LINEAR).transpose(2,0,1)
    flowt1[0,:,:] += numpy.arange(flowt1.shape[2])
    flowt1[1,:,:] += numpy.arange(flowt1.shape[1])[:,numpy.newaxis]
    backwardimgt1 =  cv2.remap(tenSecond.cpu().numpy().squeeze().transpose(1,2,0).astype(numpy.float32), flowt1.transpose(1,2,0).astype(numpy.float32), None, cv2.INTER_LINEAR).transpose(2,0,1)
    return backwardimgt0, backwardimgt1


def image_generate(fltTime, tenFirst, tenSecond, tenFlow01, tenFlow10):
    tenMetric = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenInput=tenSecond, tenFlow=tenFlow01), reduction='none').mean(1, True)
    tenSoftmax01 = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow01 * fltTime, tenMetric=-20.0 * tenMetric, strType='softmax') # -20.0 is a hyperparameter, called 'alpha' in the paper, that could be learned using a torch.Parameter
    tenSoftmax01 = tenSoftmax01[0, :, :, :].cpu().numpy()
    ######################## Flowt0 ################################
    flowt0 = -softsplat.FunctionSoftsplat(tenInput=tenFlow01, tenFlow=tenFlow01 * fltTime, tenMetric=-20.0 * tenMetric, strType='softmax')[0, :, :, :].cpu().numpy()
    ################################################################
    
    tenMetric = torch.nn.functional.l1_loss(input=tenSecond, target=backwarp(tenInput=tenFirst, tenFlow=tenFlow10), reduction='none').mean(1, True)
    tenSoftmax10 = softsplat.FunctionSoftsplat(tenInput=tenSecond, tenFlow=tenFlow10 * (1-fltTime), tenMetric=-20.0 * tenMetric, strType='softmax')
    tenSoftmax10 = tenSoftmax10[0, :, :, :].cpu().numpy()
    ######################## Flowt1 ################################
    flowt1 = -softsplat.FunctionSoftsplat(tenInput=tenFlow10, tenFlow=tenFlow10 * (1-fltTime), tenMetric=-20.0 * tenMetric, strType='softmax')[0, :, :, :].cpu().numpy()
    ################################################################
    
    ################# Cal backwarp and hole ########################
    backwardimgt0, backwardimgt1 = backwarp_img_generate(flowt0, flowt1, tenFirst, tenSecond)
    Hole01 = numpy.where((tenSoftmax01[0,:,:]==0)&(tenSoftmax01[1,:,:]==0)&(tenSoftmax01[2,:,:]==0), True, False)
    Hole10 = numpy.where((tenSoftmax10[0,:,:]==0)&(tenSoftmax10[1,:,:]==0)&(tenSoftmax10[2,:,:]==0), True, False)
    Hole = Hole01 & Hole10
    ################################################################

    tenSoftmax01_mod = numpy.where(tenSoftmax01==0, tenSoftmax10, tenSoftmax01)
    w1, w2 = numpy.exp(1-fltTime), numpy.exp(fltTime) #1/fltTime, 1/(1-fltTime) 
    a1, a2 = w1/(w1+w2), w2/(w1+w2)
    tenSoftmax10_mod = numpy.where(tenSoftmax10==0, tenSoftmax01, (tenSoftmax01_mod*a1+tenSoftmax10*a2))
    ######################## Final Image ###########################
    tenSoftmax = numpy.where(Hole, (backwardimgt0*a1+backwardimgt1*a2), tenSoftmax10_mod)
    ################################################################
    return tenSoftmax.transpose(1, 2, 0)*255


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

    #Task1
    ##################################
    if not args.skip_task0:
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

            img = image_generate(0.5, tenFirst, tenSecond, tenFlow01, tenFlow10)
            cv2.imwrite(f'{task_dir}/{num}/frame10i11.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    if not args.skip_task1:
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

                for stamp in range(1,8):
                    img = image_generate(stamp/8, tenFirst, tenSecond, tenFlow01, tenFlow10)
                    cv2.imwrite(f'{task_dir}/{outer}/{inner}/{str(start+stamp).zfill(5)}.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

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

                for stamp in [4-start%4, 8-start%4]:
                    img = image_generate(stamp/10, tenFirst, tenSecond, tenFlow01, tenFlow10)
                    cv2.imwrite(f'{task_dir}/{outer}/{inner}/{str(start+stamp).zfill(5)}.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])