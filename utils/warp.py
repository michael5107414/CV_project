import torch
import cv2
import numpy
import softsplat

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

        flow[0,:,:] = numpy.arange(flowt0.shape[2])
        flow[1,:,:] += numpy.arange(flow.shape[1])[:,numpy.newaxis]
        img.append(cv2.remap(tenFirst.cpu().numpy().squeeze().transpose(1,2,0).astype(numpy.float32), flowt0.transpose(1,2,0).astype(numpy.float32), None, cv2.INTER_LINEAR).transpose(2,0,1))
    
    return backwardimgt0, backwardimgt1


def image_generate(fltTime, tenFirst, tenSecond, tenFlow01, tenFlow10, hole_fill=False):
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
    tenSoftmax = numpy.where(tenSoftmax10==0, tenSoftmax01, (tenSoftmax01_mod*a1+tenSoftmax10*a2))
    ######################## Final Image ###########################
    if hole_fill:
        tenSoftmax = numpy.where(Hole, (backwardimgt0*a1+backwardimgt1*a2), tenSoftmax)
    ################################################################
    return tenSoftmax.transpose(1, 2, 0)*255
