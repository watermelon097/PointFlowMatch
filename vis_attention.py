import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
sys.path.append("/home/jun.wang/projects/PointFlowMatch/dinov2/")
from dinov2.models.vision_transformer import vit_small


if __name__ == '__main__':
    image_size = (512, 512)
    output_dir = './attentions'
    patch_size = 14

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = vit_small(
            patch_size=14,
            img_size=526,
            init_values=1.0,
            #ffn_layer="mlp",
            block_chunks=0
    )

    model.load_state_dict(torch.load('ckpt/dinov2_vits14_pretrain.pth'))
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()

    img = Image.open('vis/rgb_obs_10_0.png')
    img = img.convert('RGB')
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    print(img.shape)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
    # plot img
    # plt.imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
    # plt.show()
    print(w, h)
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    print(w_featmap, h_featmap)
    print(img.shape)

    attentions = model.get_last_self_attention(img.to(device))

    nh = attentions.shape[1] # number of head
    print(nh)
    print(attentions.shape)
    # we keep only the output patch attention
    # for every patch
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    print(attentions.shape)
    # weird: one pixel gets high attention over all heads?
    print(torch.max(attentions, dim=1)) 
    # attentions[:, 283] = 0 
 
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="bilinear", align_corners=False)[0].cpu().numpy()
    print(attentions.shape)
    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)

    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")
