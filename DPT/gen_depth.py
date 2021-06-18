"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse

import util.io

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def run(model, img_names, device, output_path):
    print("start processing")
    for ind, img_name in enumerate(img_names):
        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input

        img = util.io.read_image(img_name)

        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        util.io.write_depth(filename, prediction, bits=3, absolute_depth=False)
        
    print("finished")
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='validation', choices=['validation', 'testing'])
    parser.add_argument('--skip-task0', action='store_true')
    parser.add_argument('--skip-task1', action='store_true')
    parser.add_argument('--skip-task2', action='store_true')
    args = parser.parse_args()
    
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    net_w = net_h = 384
    model = DPTDepthModel(
        path="weights/dpt_large-midas-2f21e586.pt",
        backbone="vitl16_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose([
        Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        normalization,
        PrepareForNet(),
    ])

    model.eval()

    if device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    if (args.task == 'validation'):
        dir_path = [[i for i in range(7)],[i for i in range(3)], [i for i in range(3)]]
    else:
        dir_path = [[i for i in range(7,17)],[i for i in range(3,5)], [i for i in range(3,5)]]


    #Task1
    ##################################
    if not args.skip_task0:
        print("task 0 start...")
        data_dir = f'data/{args.task}/0_center_frame'

        for num in dir_path[0]:
            info_path = f'{data_dir}/{num}/input'

            # get input
            img_names = glob.glob(os.path.join(info_path, "*.png"))
            img_names += glob.glob(os.path.join(info_path, "*.jpg"))
            num_images = len(img_names)
            run(model, img_names, device, info_path)
            
        print("task 0 finish.")


    if not args.skip_task1:
        print("task 1 start...")
        data_dir = f'data/{args.task}/1_30fps_to_240fps'

        for outer in dir_path[1]:
            for inner in range(12):
                info_path = f'{data_dir}/{outer}/{inner}/input'

                # get input
                img_names = glob.glob(os.path.join(info_path, "*.png"))
                img_names += glob.glob(os.path.join(info_path, "*.jpg"))
                num_images = len(img_names)
                run(model, img_names, device, info_path)
            print(f"subtask 1-{outer} finish.")
        print("task 1 finish.")


    if not args.skip_task2:
        data_dir = f'data/{args.task}/2_24fps_to_60fps'

        for outer in dir_path[2]:
            for inner in range(8):
                info_path = f'{data_dir}/{outer}/{inner}/input'
                
                # get input
                img_names = glob.glob(os.path.join(info_path, "*.png"))
                img_names += glob.glob(os.path.join(info_path, "*.jpg"))
                num_images = len(img_names)
                run(model, img_names, device, info_path)
            print(f"subtask 2-{outer} finish.")
        print("task 2 finish.")