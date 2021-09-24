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
from RT_code.models.models import DeepVONet,DeepFMatResNet18


if __name__=='__main__':
    base_dir = '/mnt/usb0/shyoon/ai28/dataset'
    option= 'pose_demo' #pose_demo

    base_dir = os.path.join(base_dir, '%s'%(option))#crack

    normal_dir_left = os.path.join(base_dir, '1')
    normal_dir_right = os.path.join(base_dir, '0')
    img_list_normal_left = os.listdir(normal_dir_left)
    len_of_img = len(img_list_normal_left)

    # transformation matrix
    pose_txt_filename = os.path.join(base_dir, 'abc.txt')
    f = open(pose_txt_filename, 'r')
    trans_mat_gt = f.read().split('\n')
    # intrinsic parameter

    # To tensor
    toTensor = transforms.ToTensor()
    ## hyper-parameter ##


    posenet = DeepFMatResNet18(6,'ETR').cuda() #DeepVONet().cuda()
    resultModelFile = torch.load("/mnt/usb0/shyoon/ai28/RT_code/analysis/experiment_12_lr0.0001_batch8/logs/batchLosses/20210427-174501/temp")

    posenet.load_state_dict(resultModelFile)
    posenet.eval()

    #Left abnormal
    #Right normal
    cnt = 0
    right_cnt = 0
    wrong_cnt = 0
    right_error = 0
    wrong_error = 0

    for i in range(len_of_img):
        img_name_format_left = str(i) + '_1.png'
        img_name_format_right = str(i) + '_0.png'

        cur_left_filename = os.path.join(normal_dir_left, img_name_format_left)
        cur_right_filename = os.path.join(normal_dir_right, img_name_format_right)

        cur_left_normal = cv2.imread(cur_left_filename)
        cur_right_normal = cv2.imread(cur_right_filename)

        cur_left_normal_gray = cv2.imread(cur_left_filename,flags=cv2.IMREAD_GRAYSCALE)
        cur_right_normal_gray = cv2.imread(cur_right_filename, flags=cv2.IMREAD_GRAYSCALE)

        # For pose estimation network
        size = 512
        left_Img = (cur_left_normal_gray - 127.5) / 127.5
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
            # RT = posenet(left_Img,right_Img)
            # print(RT)
            temp= RT[0].detach().cpu().numpy()[:3]
            # print(temp)
            r1 = rot.from_euler('xyz', temp*np.pi/180)
            mat = torch.from_numpy(r1.as_matrix())
            angle = r1.as_rotvec()
            angle = angle*180/np.pi
            print('angle: ',angle)



            cur_trans = RT[0].detach().cpu().numpy()
            cur_trans_mat = torch.zeros((3, 4))
            cur_trans_mat[:3,:3]=mat


            cur_trans_mat = cur_trans_mat.float()[None, ...]
            cur_trans_mat[0, :, 3] = torch.Tensor([-1, 0, 0])
            # print("mat of sequence %d\n"%i,mat)

            # pdb.set_trace()
            cur_trans_mat_gt = trans_mat_gt[i].split(' ')
            trans_mat = np.zeros((12, 1))
            for j in range(12):
                trans_mat[j] = float(cur_trans_mat_gt[j])

            pose_gt = trans_mat.reshape(3,4)[:3,:3]

            pose_error = np.mean(np.abs(temp))

            # pose_error = torch.norm(cur_trans_mat[0, :3, :3] - np.eye(3))

            # pdb.set_trace()
            # pose_error = torch.norm(cur_trans_mat[0,:3,:3]-pose_gt)
            if pose_gt[0,0]==1.0:
                rotate_gt = False
            else:
                rotate_gt = True

            if pose_error >1.7:
                rotate = True
            else:
                rotate = False

            print("pose error: %d"%i,pose_error)
            print("Estimate:%s, GT:%s"%(rotate,rotate_gt))


            if rotate_gt==rotate:
                right_cnt+=1
                right_error += pose_error
            else:
                wrong_cnt+=1
                wrong_error += pose_error

            #
            fig = plt.figure(figsize=(9, 6), dpi=80)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            #
            ax1.title.set_text('left image')
            ax2.title.set_text('right image')

            ax1.imshow(cur_left_normal)
            ax2.imshow(cur_right_normal)
            if rotate:
                fig.text(0.2,0.8,'Rotation (Pred): %s '%rotate,fontsize=15,color="red")

            else:
                fig.text(0.2,0.8,'Rotation (Pred): %s '%rotate,fontsize=15, color="blue")

            fig.text(0.6,0.8,'Rotation GT: %s ' % rotate_gt, fontsize=15)

            fig.tight_layout()
            #
            # plt.show()
            plt.savefig(os.path.join('./output','%s'%option,'%04d.png'%cnt))
            # plt.close()
            cnt+=1

    print("right",right_cnt)
    print("right mean", right_error/right_cnt)
    print("wrong",wrong_cnt)
    print("wrong mean", wrong_error/wrong_cnt)
    print("ACC:", right_cnt/(right_cnt+wrong_cnt))