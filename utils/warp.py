import torch
from torchvision import transforms
import cv2
import numpy as np
import numpy
from PIL import Image
from . import softsplat

backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

def backwarp_map(flowt0, flowt1, tenFirst, tenSecond):
    img = []
    for flow, data in [(flowt0, tenFirst), (flowt1, tenSecond)]:
        hole = numpy.where((flow[0,:,:]==0)&(flow[1,:,:]==0), True, False)
        pad_flow = numpy.pad(flow, ((0,),(1,),(1,)), 'constant')
        weight = numpy.where((pad_flow[0,0:-2,1:-1]!=0)|(pad_flow[1,0:-2,1:-1]!=0), 1, 0) + \
             numpy.where((pad_flow[0,2:,1:-1]!=0)|(pad_flow[1,2:,1:-1]!=0), 1, 0) + \
             numpy.where((pad_flow[0,1:-1,0:-2]!=0)|(pad_flow[1,1:-1,0:-2]!=0), 1, 0) + \
             numpy.where((pad_flow[0,1:-1,2:]!=0)|(pad_flow[1,1:-1,2:]!=0), 1, 0)
        weight = numpy.where(weight==0, 40, weight)

        neighbor = pad_flow[:,0:-2,1:-1] + pad_flow[:,2:,1:-1] + pad_flow[:,1:-1,0:-2] + pad_flow[:,1:-1,2:]
        flow = numpy.where(hole, neighbor/weight, flow)

        flow[0,:,:] += numpy.arange(flow.shape[2])
        flow[1,:,:] += numpy.arange(flow.shape[1])[:,numpy.newaxis]
        img.append(cv2.remap(data.cpu().numpy().squeeze().transpose(1,2,0).astype(numpy.float32), flow.transpose(1,2,0).astype(numpy.float32), None, cv2.INTER_LINEAR).transpose(2,0,1))
        
    return img


def image_generate(fltTime, tenFirst, tenSecond, tenFlow01, tenFlow10, depth0, depth1, Net = None, hole_fill=True, frame_refinement = True):
    depth0 = torch.from_numpy(depth0.copy()).unsqueeze(0).unsqueeze(0).cuda()
    depth1 = torch.from_numpy(depth1.copy()).unsqueeze(0).unsqueeze(0).cuda()
    
    ###################### Range Map0 ##############################
    Occlusion_map0 = range_map(tenFlow10)
    tenFlow01_new = tenFlow01.cpu().numpy()
    tenFlow01_new =  torch.from_numpy(numpy.where(Occlusion_map0, 0, tenFlow01_new)).cuda()
    tenFirst_new = tenFirst.cpu().numpy()
    tenFirst_new =  torch.from_numpy(numpy.where(Occlusion_map0, 0, tenFirst_new)).cuda()
    #tenFlow01 = tenFlow01_new
    ################################################################
    ###################### Range Map1 ##############################
    Occlusion_map1 = range_map(tenFlow01)
    tenFlow10_new = tenFlow10.cpu().numpy()
    tenFlow10_new =  torch.from_numpy(numpy.where(Occlusion_map1, 0, tenFlow10_new)).cuda()
    tenSecond_new = tenSecond.cpu().numpy()
    tenSecond_new =  torch.from_numpy(numpy.where(Occlusion_map0, 0, tenSecond_new)).cuda()
    #tenFlow10 = tenFlow10_new
    ################################################################
    
    #################### Metric Calculate ##########################
    tenMetric = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenInput=tenSecond, tenFlow=tenFlow01), reduction='none').mean(1, True)
    ######################## Flowt0 ################################
    flowt0 = -softsplat.FunctionSoftsplat(tenInput=tenFlow01_new, tenFlow=tenFlow01_new * fltTime, tenMetric=1/depth0, strType='linear')[0, :, :, :].cpu().numpy()
    ################################################################
    ################## Forward Warping Image #######################
    tenSoftmax01 = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow01 * fltTime, tenMetric=-8.0 * tenMetric, strType='softmax') # -20.0 is a hyperparameter, called 'alpha' in the paper, that could be learned using a torch.Parameter
    tenSoftmax01 = tenSoftmax01[0, :, :, :].cpu().numpy()
    ################################################################
    
    #################### Metric Calculate ##########################
    tenMetric = torch.nn.functional.l1_loss(input=tenSecond, target=backwarp(tenInput=tenFirst, tenFlow=tenFlow10), reduction='none').mean(1, True)
    ######################## Flowt1 ################################
    flowt1 = -softsplat.FunctionSoftsplat(tenInput=tenFlow10_new, tenFlow=tenFlow10_new * (1-fltTime), tenMetric=1/depth1, strType='linear')[0, :, :, :].cpu().numpy()
    ################################################################
    ################## Forward Warping Image #######################
    tenSoftmax10 = softsplat.FunctionSoftsplat(tenInput=tenSecond, tenFlow=tenFlow10 * (1-fltTime), tenMetric=-8.0 * tenMetric, strType='softmax')
    tenSoftmax10 = tenSoftmax10[0, :, :, :].cpu().numpy()
    ################################################################

    ################# Cal backwarp and hole ########################
    backwardimgt0, backwardimgt1 = backwarp_map(flowt0, flowt1, tenFirst, tenSecond)
    Hole01 = numpy.where((tenSoftmax01[0,:,:]==0)|(tenSoftmax01[1,:,:]==0)|(tenSoftmax01[2,:,:]==0), 1, 0).astype(numpy.uint8)
    Hole10 = numpy.where((tenSoftmax10[0,:,:]==0)|(tenSoftmax10[1,:,:]==0)|(tenSoftmax10[2,:,:]==0), 1, 0).astype(numpy.uint8)
    kernel = numpy.ones((3,3), numpy.uint8)
    #Hole01 = cv2.dilate(Hole01, kernel, iterations = 1)
    Hole01 = cv2.morphologyEx(Hole01, cv2.MORPH_CLOSE, kernel)
    #Hole10 = cv2.dilate(Hole10, kernel, iterations = 1)
    Hole10 = cv2.morphologyEx(Hole10, cv2.MORPH_CLOSE, kernel)
    Hole = Hole01 & Hole10
    Hole = cv2.morphologyEx(Hole, cv2.MORPH_CLOSE, kernel)
    ################################################################

    tenSoftmax01_mod = numpy.where(Hole01==1, tenSoftmax10, tenSoftmax01)
    w1, w2 = numpy.exp(1-fltTime), numpy.exp(fltTime) #1/fltTime, 1/(1-fltTime) 
    a1, a2 = w1/(w1+w2), w2/(w1+w2)
    tenSoftmax = numpy.where(Hole10==1, tenSoftmax01, (tenSoftmax01_mod*a1+tenSoftmax10*a2))
    ######################## Final Image ###########################
    if hole_fill:
        tenSoftmax = numpy.where(Hole, (backwardimgt0*a1+backwardimgt1*a2), tenSoftmax)
    ################################################################
    if frame_refinement:
        H = tenSoftmax.shape[1]
        W = tenSoftmax.shape[2]

        img_0 = inter_frame_preprocess(tenFirst, isTensor = True)
        img_1 = inter_frame_preprocess(tenSecond, isTensor = True)
        intermediate = inter_frame_preprocess(tenSoftmax)

        output = Net(img_0, img_1, intermediate, fltTime)
        output = output[0,:,0:H,0:W].squeeze(0).cpu()
        output = transforms.functional.to_pil_image(output)
        output = np.array(output)
        return output
    ################################################################
    
    return tenSoftmax.transpose(1, 2, 0)*255


def range_map(flow):
    h, w = flow.shape[2], flow.shape[3]
    src_map = torch.ones((1,1,h,w)).cuda()
    dst_map = softsplat.FunctionSoftsplat(tenInput=src_map, tenFlow=flow, tenMetric=None, strType='average')[0,0,:,:].cpu().numpy()
    
    Occlusion_map = numpy.where(dst_map==0, True, False)
    return Occlusion_map

def inter_frame_preprocess(intermediate, isTensor = False):
    if isTensor == False:
        transform = transforms.ToTensor()
        img = transform(intermediate).unsqueeze(0).cuda()
        img = img.permute(0, 2, 3, 1)
        if img.size(1)==1:
            img = img1.expand(-1, 3,-1,-1)
    else:
        img = intermediate
    
    _,_,H,W = img.size()
    H_,W_ = int(np.ceil(H/32)*32),int(np.ceil(W/32)*32)
    pader = torch.nn.ReplicationPad2d([0, W_-W , 0, H_-H])
    img = pader(img)
    return img