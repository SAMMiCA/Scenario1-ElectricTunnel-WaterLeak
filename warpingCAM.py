import os
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
import skimage

######CAM
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2

import torch.nn.functional as F
# from torchvision.models.resnet import resnet101, resnet50
from PIL import Image
from network.resnet import resnet50

from ai_dataset import AI_Dataset
import pdb
import matplotlib.pyplot as plt


def returnCAM(feature_conv, weight_softmax, img, h_x,labels):
    # im_h, im_w = img.shape[1], img.shape[0]
    im_h, im_w = img.size()[2],img.size()[3]
    # print(im_h,im_w)
    probs, idx = h_x.sort(0, True)


    size_upsample = (im_w, im_h)  #####BE CAREFUL
    batch, nc, h, w = feature_conv.shape
    output_cam = []

    cams = torch.zeros((batch, 2, h,w))
    # fgs = torch.zeros((batch,1,im_h,im_w))

    for i in range(batch):


        cam_dict = {}

        for j in range(2):
            # j = j.item()
            cam = torch.mm(weight_softmax[j].clone().unsqueeze(0), (feature_conv[i].reshape((nc, h * w))))
            cam = cam.reshape(h, w)
            cam = cam - torch.min(cam)
            cam = cam / torch.max(cam)

            cams[i, j, :, :] = cam#*labels[j]
            cam_dict[j] =  cv2.resize(cam.cpu().detach().numpy(),size_upsample)

    cams = F.upsample(cams,(im_h,im_w),mode='bilinear',align_corners=False)


    # output_cam.append(cv2.resize(cam_img, size_upsample))
    return cams

def pixel2cam(depth, pixel_coord, fx, fy, cx, cy):
    x = (pixel_coord[0, :] - cx) / fx * pixel_coord[2, :]
    y = (pixel_coord[1, :] - cy) / fy * pixel_coord[2, :]
    z = pixel_coord[2, :]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    output = depth.reshape(-1, 1)*cam_coord
    return output

def cam2pixel(pix2cam, K, rot, trans):
    pcoords = np.matmul(rot, pix2cam.transpose())
    pcoords = pcoords + trans[:, None]
    pcoords = np.matmul(K, pcoords)
    X = pcoords[0, :]
    Y = pcoords[1, :]
    Z = pcoords[2, :]
    X_norm = 2*(X/Z)/(width - 1) - 1
    Y_norm = 2*(Y/Z)/(height - 1) - 1
    X_norm = X_norm.reshape(height, width)
    Y_norm = Y_norm.reshape(height, width)
    pixel_coords = np.stack((X_norm, Y_norm), axis=2)
    return pixel_coords

def set_id_grid(depth):
    global height
    global width
    height, width = depth.shape
    j_range = np.repeat(np.arange(0, height), width).reshape(height, width)
    i_range = np.repeat(np.arange(0, width), height).reshape(width, height).transpose()
    ones = np.ones((height, width))
    pixel_coords = np.stack((i_range, j_range, ones), axis=0)
    pixel_coords = pixel_coords.reshape(3, -1)
    return pixel_coords

if __name__=='__main__':
    base_dir = '/mnt/usb0/shyoon/ai28/dataset'
    option= 'crack2' #crack1/2/3, frost 1/2/3
    option2= 'light' #tunnel2

    crack_dir = os.path.join(base_dir, '%s'%(option))#crack
    crack_dir_left = os.path.join(crack_dir, '1')
    crack_dir_right = os.path.join(crack_dir, '0')
    normal_dir = os.path.join(base_dir, 'normal_%s'%option2)
    normal_dir_left = os.path.join(normal_dir, '1')
    normal_dir_right = os.path.join(normal_dir, '0')
    depth_dir_left = os.path.join(normal_dir, '3')
    depth_dir_right = os.path.join(normal_dir, '2')
    img_list_normal_left = os.listdir(normal_dir_left)
    len_of_img = len(img_list_normal_left)
    # transformation matrix
    crack_txt_filename = os.path.join(crack_dir, 'abc.txt')
    f = open(crack_txt_filename, 'r')
    trans_mat = f.read().split('\n')
    # intrinsic parameter
    fx = 512
    fy = 512
    cx = 512
    cy = 288
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    ##CAM
    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    inv_normalize = transforms.Normalize(
        mean=[-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
        std=[1 / std_img[0], 1 / std_img[1], 1 / std_img[2]]
    )
    resnet = resnet50(pretrained=True).cuda()

    state_dict = torch.load("./model/model_resnet_er_v3.pth")
    resnet.load_state_dict(state_dict)
    resnet.eval()

    # To tensor
    toTensor = transforms.ToTensor()

    #Left abnormal
    #Right normal
    cnt = 0

    for i in [51,52,53,54]: #range(len_of_img):
        img_name_format_left = str(i) + '_1.png'
        img_name_format_right = str(i) + '_0.png'
        # depth_name = str(i) +'_3.pfm'
        depth_name = '%d_2.pfm' % i
        cur_left_crack_filename = os.path.join(crack_dir_left, img_name_format_left)
        cur_left_depth_filename = os.path.join(depth_dir_left, depth_name)
        cur_right_normal_filename = os.path.join(normal_dir_right, img_name_format_right)
        cur_left_normal_filename = os.path.join(normal_dir_left, img_name_format_left)

        cur_left_crack = cv2.imread(cur_left_crack_filename)
        cur_right_normal = cv2.imread(cur_right_normal_filename)
        # cur_left_normal = cv2.imread(cur_left_normal_filename)

        # For Network / input crack left
        image = Image.open(cur_left_crack_filename).convert("RGB")
        image = np.float64(image) / 255.0
        image = image - np.array([0.485, 0.456, 0.406])
        image = image / np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).type(torch.FloatTensor)

        _,_,H,W = image.size()

        with torch.no_grad():
            output, features = resnet(image.cuda())
        pred = output[0].cpu().detach().numpy()
        pred_cls = pred.argmax() #pred_cls =0 : abnormal pred_cls =1 normal

        feature4 = features[3]

        h_x = F.softmax(output, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        params = list(resnet.parameters())
        weight_softmax = params[-2]

        CAMs = returnCAM(feature4, weight_softmax, image, h_x, pred)
        CAMs = CAMs.detach().cpu().numpy()
        CAMs_crack = CAMs[0][idx[0]]

        cur_left_depth = imageio.imread(cur_left_depth_filename)  # If Error: script: imageio_download_bin freeimage
        pixel_coords = set_id_grid(cur_left_depth)
        pix2cam = pixel2cam(cur_left_depth, pixel_coords, fx, fy, cx, cy)  # Camera coordinate (left crack)
        cur_trans = trans_mat[i].split(' ')
        cur_trans_mat = np.zeros((12, 1))
        for j in range(12):
            cur_trans_mat[j] = float(cur_trans[j])
        cur_trans_mat = cur_trans_mat.reshape(3, 4)
        cur_rot_mat = cur_trans_mat[:3, :3]
        cur_translation = cur_trans_mat[:3, 3]
        cur_translation = np.array([-1, 0, 0])
        pixel_coords = cam2pixel(pix2cam, K, cur_rot_mat,
                                 cur_translation)  # warping in cam coord, and camcoord2pixcoord (left crack to right crack (warped))
        pixel_coords_th = torch.from_numpy(pixel_coords)[None, ...].float()
        cur_right_normal_th = toTensor(cur_right_normal)[None, ...].float()
        cur_left_restored = F.grid_sample(cur_right_normal_th, pixel_coords_th, padding_mode='zeros', align_corners=True)

        left_restored = cur_left_restored[0].permute(1, 2, 0).detach().cpu().numpy()
        left_ori = cur_left_crack
        cl = CAMs_crack < 0.4
        ncl = CAMs_crack >= 0.4
        cl_stack = np.stack((cl, cl, cl), 2)
        ncl_stack = np.stack((ncl, ncl, ncl), 2)

        # pdb.set_trace()
        if pred_cls==0:
            left_final = (255 * left_restored * (1 - cl_stack) + left_ori * cl_stack) / 255
            error = np.abs(cv2.cvtColor(np.uint8(255 * left_final), cv2.COLOR_BGR2GRAY) - cv2.cvtColor(cur_left_normal,cv2.COLOR_BGR2GRAY))
        else:
            left_final = left_ori
            error = np.zeros((H,W))

        fig = plt.figure(figsize=(16, 9), dpi=120)
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)

        ax1.title.set_text('%s image(left)'%option)
        ax2.title.set_text('%s image(left) with CAM'%option)
        ax3.title.set_text('warped normal image(right2left)')
        ax4.title.set_text('reconstructed image(left)')
        ax5.title.set_text('normal image(left)')
        ax6.title.set_text('error map')

        ax1.imshow(left_ori)
        ax2.imshow(left_ori)
        if pred_cls == 0:
            ax2.imshow(np.uint8(255 * CAMs[0][idx[0]]), cmap='jet', alpha=0.3)
        ax3.imshow(left_restored)
        ax4.imshow(left_final)
        ax5.imshow(cur_left_normal)
        ax6.imshow(error)

        # plt.imshow(left_final)

        # plt.show()
        plt.savefig(os.path.join('./output','%s_warp'%option,'%04d.png'%cnt))
        plt.close()
        print("saving figure %04d"%cnt)
        cnt+=1

    # plt.imshow(CAMs_crack)
    # plt.show()

    # for i in [1,100,150]:#range(len_of_img):
    #     img_name_format_left = str(i) + '_1.png'
    #     img_name_format_right = str(i) + '_0.png'
    #     depth_name = str(i) +'_3.pfm'
    #     cur_left_crack_filename = os.path.join(crack_dir_left, img_name_format_left)
    #     cur_left_depth_filename = os.path.join(depth_dir_left, depth_name)
    #     cur_right_normal_filename = os.path.join(normal_dir_right, img_name_format_right)
    #     cur_left_crack = cv2.imread(cur_left_crack_filename)
    #     cur_right_normal = cv2.imread(cur_right_normal_filename)
    #     cur_left_depth = imageio.imread(cur_left_depth_filename) #If Error: script: imageio_download_bin freeimage
    #     pixel_coords = set_id_grid(cur_left_depth)
    #     pix2cam = pixel2cam(cur_left_depth, pixel_coords, fx, fy, cx, cy) #Camera coordinate (left crack)
    #     cur_trans = trans_mat[i].split(' ')
    #     cur_trans_mat = np.zeros((12, 1))
    #     for j in range(12):
    #         cur_trans_mat[j] = float(cur_trans[j])
    #     cur_trans_mat = cur_trans_mat.reshape(3, 4)
    #     cur_rot_mat = cur_trans_mat[:3, :3]
    #     cur_translation = cur_trans_mat[:3, 3]
    #     cur_translation = np.array([-1, 0, 0])
    #     pixel_coords = cam2pixel(pix2cam, K, cur_rot_mat, cur_translation) #warping in cam coord, and camcoord2pixcoord (left crack to right crack (warped))
    #     pixel_coords_th = torch.from_numpy(pixel_coords)[None, ...].float()
    #     cur_right_normal_th = toTensor(cur_right_normal)[None, ...].float()
    #     cur_left_restored = F.grid_sample(cur_right_normal_th, pixel_coords_th, padding_mode='zeros', align_corners=True)
    #
    #     left_restored = cur_left_restored[0].permute(1, 2, 0).detach().cpu().numpy()
    #     left_ori = cur_left_crack
    #     cl = CAMs_crack <0.4
    #     ncl = CAMs_crack >=0.4
    #     cl_stack = np.stack((cl,cl,cl),2)
    #     ncl_stack = np.stack((ncl, ncl, ncl), 2)
    #
    #     # pdb.set_trace()
    #
    #     left_final = (255*left_restored*(1-cl_stack) + left_ori*cl_stack)/255
    #
    #     fig = plt.figure(figsize=(13, 7), dpi=100)
    #     ax1 = fig.add_subplot(2, 2, 1)
    #     ax2 = fig.add_subplot(2, 2, 2)
    #     ax3 = fig.add_subplot(2, 2, 3)
    #     ax4 = fig.add_subplot(2, 2, 4)
    #
    #     ax1.title.set_text('cracked image(left)')
    #     ax2.title.set_text('cracked image(left) with CAM')
    #     ax3.title.set_text('warped normal image(right2left)')
    #     ax4.title.set_text('reconstructed image(left)')
    #
    #     ax1.imshow(left_ori)
    #     ax2.imshow(left_ori)
    #     ax2.imshow(np.uint8(255 * CAMs[0][idx[0]]), cmap = 'jet', alpha = 0.3)
    #     ax3.imshow(left_restored)
    #     ax4.imshow(left_final)
    #
    #     # plt.imshow(left_final)
    #     plt.show()