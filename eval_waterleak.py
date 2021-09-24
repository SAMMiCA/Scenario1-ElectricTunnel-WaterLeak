
import os
import argparse

import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms
from network.resnet import resnet50_ai
from ai_dataset import AI_Dataset_waterleak

import glob
from tqdm import tqdm
from PIL import Image
import pdb
import random
import shutil
import matplotlib.pyplot as plt



def create_cam(config):
    if not os.path.exists(config.result_path):
        os.mkdir(config.result_path)


def max_norm(cam_cp):
    N, C, H, W = cam_cp.size()
    cam_cp = F.relu(cam_cp)
    max_v = torch.max(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
    min_v = torch.min(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
    cam_cp = F.relu(cam_cp - min_v - 1e-5) / (max_v - min_v + 1e-5)

    return cam_cp


if __name__=='__main__':

    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    object_categories= ['abnormal','normal']

    transformations = transforms.Compose([
        # transforms.RandomResizedCrop(384,scale=(0.8,1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_img, std=std_img),
    ])


    inv_normalize = transforms.Normalize(
        mean=[-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
        std=[1 / std_img[0], 1 / std_img[1], 1 / std_img[2]]
    )

    dataset = AI_Dataset_waterleak(mode='val_all',transform=None)

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,  # config.batch_size
        num_workers=4,  #
        shuffle=False,
    )

    device = "cuda"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    resnet =resnet50_ai(pretrained=False).cuda()
    resnet = torch.nn.DataParallel(resnet)
    resnet.load_state_dict(torch.load("./model/wlmodel_best_resnet_v3.pth"))
    # resnet.load_state_dict(torch.load("./model/wlmodel_93.5.pth"))


    resnet.eval()

    root_dir = './output'

    save_root_dir = os.path.join(root_dir,'waterleak2')

    if os.path.isdir(save_root_dir):
        shutil.rmtree(save_root_dir + "/", ignore_errors=True)
    else:
        os.mkdir(save_root_dir)
        print("make directory")

    # cnt = 0

    for iteration,(image,label,image_dir) in enumerate(tqdm(val_loader)):
        if label[0][0]==1:
            batch,_,height,width = image.size()

            image= image.cuda()
            label = label.cuda()

            image_ori = image

            image_dir=image_dir[0]

            basename = os.path.basename(image_dir).split('.')[0]
            scales = [0.5,1,1.5,2.0]

            CAMs = torch.zeros(1,2,height,width).cuda()

            for scale in scales:

                if scale != 1:
                    image = F.interpolate(image,size=(int(height*scale),int(width*scale)),mode='bilinear',align_corners=True)

                with torch.no_grad():
                    resnet.eval()
                    outputs_v, cam = resnet(image)

                    h_x = F.softmax(outputs_v, dim=1).data.squeeze()
                    probs, idx = h_x.sort(0, True)

                    params = list(resnet.parameters())
                    weight_softmax = params[-2]

                    # cam = max_norm(cam)
                    cam = F.interpolate(cam,size=(height,width),mode='bilinear',align_corners=True)

                    CAMs += cam
            #+++++++++++++++++++++++++++++++++++++
            CAMs = max_norm(CAMs)

            cam_dict = {}

            for i in range(2):
                if label[0, 0] ==1:
                    cam_dict[i] = CAMs[0][i]

            np.save(os.path.join("./dict", basename + '.npy'), cam_dict)



            #
            #
            # fig = plt.figure(figsize=(8,12),dpi=200)
            # ax1 = fig.add_subplot(2, 1, 1)
            # ax2 = fig.add_subplot(2, 1, 2)
            #
            # ax1.set_xticks([])
            # ax1.set_yticks([])
            # ax2.set_xticks([])
            # ax2.set_yticks([])
            #
            # # ax1.tick_params(left=False, bottom=False)
            # # ax2.tick_params(left=False, bottom=False)
            #
            # # plt.subplot(1,2,1)
            #
            # # print()
            #
            # if label[0][0]==1:
            #     normal_img = np.uint8(255*inv_normalize(image_ori.squeeze(0)).permute(1,2,0).cpu())
            #
            #     ax1.imshow(normal_img)
            #
            #     ax2.imshow(np.uint8(255*inv_normalize(image_ori.squeeze(0)).permute(1,2,0).detach().cpu()))
            #     ax2.imshow(np.uint8(255*CAMs[0][int(label[0][1])].detach().cpu()), cmap='jet',alpha=0.6)
            #     #
            #     basename = basename.split('.')[0]+'_'+object_categories[int(label[0][1])]+'.jpg'
            #     # basename = basename.split('.')[0] + '_' + object_categories[np.nonzero(label[0])[1,0]] + '.jpg'
            #     save_dir = os.path.join(save_root_dir, basename)
            #     #
            #     plt.savefig(save_dir)
            #     print("saving!")
            # plt.close()
