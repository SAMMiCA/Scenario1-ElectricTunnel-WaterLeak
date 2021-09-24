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

import pdb
import matplotlib.pyplot as plt

###Fundamental Matrix Estimation###
from scipy.spatial.transform import Rotation as rot
from RT_code.models.models import DeepVONet, DeepFMatResNet18


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

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


if __name__=='__main__':
    base_dir = '/mnt/usb0/shyoon/ai28/dataset/pose_abnormal'
    option1= 'original' #abnormal1/2/3, frost 1/2/3
    option2= 'fusion' #tunnel2
   
    normal_dir = os.path.join(base_dir, '%s'%option1)
    # normal_dir_left = os.path.join(normal_dir, '1')
    # normal_dir_right = os.path.join(normal_dir, '0')

    abnormal_dir = os.path.join(base_dir, '%s' % (option2))  # abnormal
    abnormal_dir_left = os.path.join(abnormal_dir, '1')
    normal_dir_right = os.path.join(abnormal_dir, '0')
    
    depth_dir_left = os.path.join(abnormal_dir, '3') #'3'??
    depth_dir_right = os.path.join(abnormal_dir, '2') #'2'??
    img_list_normal_left = os.listdir(abnormal_dir_left)
    len_of_img = len(img_list_normal_left)
    
    # transformation matrix
    abnormal_txt_filename = os.path.join(abnormal_dir, 'abc.txt')
    f = open(abnormal_txt_filename, 'r')
    trans_mat = f.read().split('\n')
    # intrinsic parameter
    fx = 512
    fy = 512
    cx = 512
    cy = 288
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    K_th = torch.from_numpy(K).float()[None, ...]
    inv_K = K_th.inverse()
    # To tensor
    toTensor = transforms.ToTensor()
    ## hyper-parameter ##
    height_ = 576
    width_ = 1024
    backproj = BackprojectDepth(batch_size=1, height=height_, width=width_)
    reproject = Project3D(batch_size=1, height=height_, width=width_)
    
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

    posenet = DeepFMatResNet18(6, 'ETR').cuda()
    resultModelFile = torch.load("/mnt/usb0/shyoon/ai28/RT_code/analysis/experiment_12_lr0.0001_batch8/logs/batchLosses/20210427-174501/result_12")
    posenet.load_state_dict(resultModelFile)
    posenet.eval()

    # To tensor
    toTensor = transforms.ToTensor()

    #Left abnormal
    #Right normal
    cnt = 0

    for i in range(len_of_img):
        img_name_format_left = str(i) + '_1.png'
        img_name_format_right = str(i) + '_0.png'

        depth_name_right = str(i) + '_2.pfm'
        depth_name_left = str(i) + '_3.pfm'

        cur_left_abnormal_filename = os.path.join(abnormal_dir_left, img_name_format_left)
        # cur_left_normal_filename = os.path.join(normal_dir_left, img_name_format_left)
        cur_right_normal_filename = os.path.join(normal_dir_right, img_name_format_right)

        cur_left_depth_filename = os.path.join(depth_dir_left, depth_name_left)
        cur_right_depth_filename = os.path.join(depth_dir_right, depth_name_right)

        cur_left_abnormal = cv2.imread(cur_left_abnormal_filename)
        cur_right_normal = cv2.imread(cur_right_normal_filename)
        # cur_left_normal = cv2.imread(cur_left_normal_filename)

        cur_left_abnormal_gray = cv2.imread(cur_left_abnormal_filename,flags=cv2.IMREAD_GRAYSCALE)
        cur_right_normal_gray = cv2.imread(cur_right_normal_filename, flags=cv2.IMREAD_GRAYSCALE)

        cur_left_depth = imageio.imread(cur_left_depth_filename)
        cur_right_depth = imageio.imread(cur_right_depth_filename)
        cur_right_depth_th = toTensor(cur_right_depth)
        cur_left_depth_th = toTensor(cur_left_depth)

        # For pose estimation network
        size = 512
        left_Img = (cur_left_abnormal_gray - 127.5) / 127.5
        right_Img = (cur_right_normal_gray - 127.5) / 127.5
        left_Img = np.expand_dims(left_Img, axis=0)
        right_Img = np.expand_dims(right_Img, axis=0)
        hl, wl = left_Img.shape[1], left_Img.shape[2]
        hr, wr = right_Img.shape[1], right_Img.shape[2]

        left_Img = left_Img[:, int(hl / 2) - int(size / 2): int(hl / 2) + int(size / 2),
                   int(wl / 2) - int(size / 2): int(wl / 2) + int(size / 2)]
        right_Img = right_Img[:, int(hr / 2) - int(size / 2): int(hr / 2) + int(size / 2),
                    int(wr / 2) - int(size / 2): int(wr / 2) + int(size / 2)]

        left_Img= torch.from_numpy(left_Img).unsqueeze(0).type(torch.FloatTensor).cuda()
        right_Img = torch.from_numpy(right_Img).unsqueeze(0).type(torch.FloatTensor).cuda()

        with torch.no_grad():
            input = torch.cat((left_Img,right_Img),1)
            RT = posenet(input)

        # For Network / input abnormal left
        image = Image.open(cur_left_abnormal_filename).convert("RGB")
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
        CAMs_abnormal = CAMs[0][idx[0]]


        temp= RT[0].detach().cpu().numpy()[:3]
        r1 = rot.from_euler('xyz', temp*np.pi/180)
        mat = torch.from_numpy(r1.as_matrix())
        angle = r1.as_rotvec()
        angle = angle*180/np.pi
        print('angle: ',angle)


        cur_trans = trans_mat[i].split(' ')
        cur_trans_mat = torch.zeros((12, 1))
        for j in range(12):
            cur_trans_mat[j] = float(cur_trans[j])
        cur_trans_mat = cur_trans_mat.reshape(3, 4)
        copy_mat = cur_trans_mat

        cur_trans_mat = cur_trans_mat.float()[None, ...]
        cur_trans_mat[0, :, 3] = torch.Tensor([-1, 0, 0])
        #
        # cur_trans_mat = torch.zeros((3, 4))
        # cur_trans_mat[:3,:3]=mat
        #
        # cur_trans_mat = cur_trans_mat.float()[None, ...]
        # cur_trans_mat[0, :, 3] = torch.Tensor([-1, 0, 0])
        # print("mat of sequence %d\n"%cnt,mat)

        pose_error = np.mean(np.abs(temp))

        if pose_error > 2.8:
            rotation = True
        else:
            rotation = False
        print("pose error: ",pose_error)

        cam_pts = backproj(cur_right_depth_th, inv_K) #right depth
        pixel_coords = reproject(cam_pts, K_th, cur_trans_mat)

        cur_right_normal_th = toTensor(cur_right_normal)[None, ...].float()
        cur_left_restored = F.grid_sample(cur_right_normal_th, pixel_coords, padding_mode='zeros', align_corners=True)
        # cur_left_restored = (255 * cur_left_restored.squeeze().cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)

        left_restored = cur_left_restored[0].permute(1, 2, 0).detach().cpu().numpy()
        left_ori = cur_left_abnormal
        cl = CAMs_abnormal < 0.3
        ncl = CAMs_abnormal >= 0.3
        cl_stack = np.stack((cl, cl, cl), 2)
        ncl_stack = np.stack((ncl, ncl, ncl), 2)

        # pdb.set_trace()
        if pred_cls==0:
            left_final = (255 * left_restored * (1 - cl_stack) + left_ori * cl_stack) / 255
            # error = np.abs(cv2.cvtColor(np.uint8(255 * left_final), cv2.COLOR_BGR2GRAY) - cv2.cvtColor(cur_left_normal,cv2.COLOR_BGR2GRAY))
        else:
            left_final = left_ori
            # error = np.zeros((H,W))

        red_box = np.zeros((height_,width_,3))
        red_box[:,:,0] = 255*np.ones((height_,width_))

        fig = plt.figure(figsize=(16, 9), dpi=120)
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)

        ax1.title.set_text('%s image(left)'%option1)
        ax2.title.set_text('%s image(left) with CAM'%option1)
        ax3.title.set_text('normal image(right)')
        ax4.title.set_text('warped normal image(right2left)')
        ax5.title.set_text('reconstructed image(left)')
        ax6.title.set_text('normal image(left)')
        # ax6.title.set_text('error map')

        ax1.imshow(left_ori)
        ax2.imshow(left_ori)
        if pred_cls == 0:
            ax2.imshow(np.uint8(255 * CAMs[0][idx[0]]), cmap='jet', alpha=0.3)
        ax3.imshow(cur_right_normal)
        # if rotation:
        #     ax4.imshow(np.uint8(red_box))
        # else:
        ax4.imshow(left_restored)
        ax5.imshow(left_final)
        ax6.imshow(left_final)
        # ax6.imshow(error)


        # plt.show()
        plt.savefig(os.path.join('./output','pose_abnormal','%04d.png'%cnt))
        plt.close()
        print(os.path.join('./output','pose_abnormal','%04d.png'%cnt))
        print("saving figure %04d"%cnt)
        cnt+=1

