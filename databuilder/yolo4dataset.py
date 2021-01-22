

import os
import sys
sys.path.append("..")
sys.path.append("../utils")
import utils.data_augment as dataAug
import utils.tools as tools

import torch
from torch.utils.data import Dataset
import config.yolov4_config as cfg
import numpy as np
import random
import cv2
from skimage.io import imread, imsave
import torch.nn.functional as F
from utils.tools import xyzwhd2xyzxyz
from PIL import Image, ImageFont, ImageDraw
def gray2rgb(image):
            w, h = image.shape
            image += np.abs(np.min(image))
            image_max = np.abs(np.max(image))
            if image_max > 0:
                image /= image_max
            ret = np.empty((w, h, 3), dtype=np.uint8)
            ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
            return ret
class YOLO4_3DDataset(Dataset):

    def __init__(self, ImageDataset, classes, img_size=(640, 160, 640)):
        self.img_size = img_size  # For Multi-training

        self.__image_dataset = ImageDataset
        self.classes = classes
        self.num_classes = len(classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))

    def __len__(self):
        return len(self.__image_dataset)


    def __getitem__(self, item):
        assert item <= len(self), 'index range error'
        #do_aug = (self.anno_file_type == 'train')
        do_aug=False
        if do_aug:
            img_org, bboxes_org, img_name = self.__image_dataset[item]
            img_org, bboxes_org = self.__data_aug(img_org, bboxes_org) #bboxes_org.shape=(N, 5) xyxyClass
            #img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

            item_mix = random.randint(0, len(self)-1)
            img_mix, bboxes_mix, _ = self.__image_dataset[item_mix]
            img_mix, bboxes_mix = self.__data_aug(img_mix, bboxes_mix)
            #img_mix = img_mix.transpose(2, 0, 1)

            img, bboxes = dataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
            del img_mix, bboxes_mix
        else:
            img_org, bboxes_org, img_name = self.__image_dataset[item]
            #img_org = img_org.transpose(2, 0, 1)
            img = img_org
            bboxes = bboxes_org



        del img_org, bboxes_org
        img_size = self.img_size

        #img = torch.from_numpy(img).float()
        if len(img.size())==5:
            #img is a batch of data, doesn't need resize
            pass
        else:
            if len(img.size())==4 or len(img.size())==5:
                org_img_shape = img.size()[:3]
                if (img_size==org_img_shape):
                    pass
                else:
                    img = img.permute((3, 0, 1, 2)).unsqueeze(0)
                    img = F.interpolate(img, size=img_size, mode='trilinear')
                    img = img[0]
                    img = img.permute((1, 2, 3, 0))
            else:
                org_img_shape = img.size()[:2]
                if (img_size==org_img_shape):
                    pass
                else:
                    img = img.permute((2, 0, 1)).unsqueeze(0)
                    img = F.interpolate(img, size=img_size, mode='bilinear')
                    img = img[0]
                    img = img.permute((1, 2, 0))
            if len(img.size())==4:
                resized_boxes = bboxes[:, :6] + 0.0
                for i in range(3): #3D
                    resized_boxes[:, i::3] = resized_boxes[:, i::3] * img_size[i] / org_img_shape[i]
                bboxes[:, :6] = resized_boxes
            else:
                resized_boxes = bboxes[:, :4] + 0.0
                for i in range(2): #2D
                    resized_boxes[:, i::2] = resized_boxes[:, i::2] * img_size[i] / org_img_shape[i]
                bboxes[:, :4] = resized_boxes

        if 0:
            boxes = bboxes
            scale = [1, 1, 1]
            ori_data = img
            for i in range(int(boxes[0][0][2]), int(boxes[0][0][5]), 3):
                #TY Image
                img = Image.fromarray(((ori_data[0].squeeze().numpy() * 255.0).astype('uint8'))[:,:,i], 'L')
                #img = Image.fromarray(TY_ori_data[i,:,:], 'L')
                img = img.convert(mode='RGB')
                draw = ImageDraw.Draw(img)
                scale = [1, 1, 1]
                for bx in boxes[0]:
                    z_bot, z_top, y_bot, y_top, x_bot, x_top =bx[0]*scale[0], bx[3]*scale[0], bx[1]*scale[1], bx[4]*scale[1], bx[2]*scale[2], bx[5]*scale[2]
                    if int(x_bot) <= i <= int(x_top):
                        draw.rectangle(
                            [(y_bot, z_bot),(y_top, z_top)],
                            outline ="red", width=2)
                img.save('debug/TY_' + str(i)+'.png')

        list_label_sbbox, list_label_mbbox, list_label_lbbox, list_sbboxes, list_mbboxes, list_lbboxes = [],[],[],[],[],[]
        for i in range(len(bboxes)):
            valid_bboxes = np.array([box for box in bboxes[i] if (not (box[0]==0 and box[3]==0))])
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__creat_label(valid_bboxes, img_size)
            label_sbbox = torch.from_numpy(label_sbbox).float()
            label_mbbox = torch.from_numpy(label_mbbox).float()
            label_lbbox = torch.from_numpy(label_lbbox).float()

            sbboxes = torch.from_numpy(sbboxes).float()
            mbboxes = torch.from_numpy(mbboxes).float()
            lbboxes = torch.from_numpy(lbboxes).float()

            list_label_sbbox.append(label_sbbox)
            list_label_mbbox.append(label_mbbox)
            list_label_lbbox.append(label_lbbox)
            list_sbboxes.append(sbboxes)
            list_mbboxes.append(mbboxes)
            list_lbboxes.append(lbboxes)
        list_label_sbbox, list_label_mbbox, list_label_lbbox, list_sbboxes, list_mbboxes, list_lbboxes = \
            torch.stack(list_label_sbbox), torch.stack(list_label_mbbox), torch.stack(list_label_lbbox), \
            torch.stack(list_sbboxes), torch.stack(list_mbboxes), torch.stack(list_lbboxes)
        #to B,C,X,Y,Z
        img = img.permute((0, 4, 1, 2, 3))
        return img, list_label_sbbox, list_label_mbbox, list_label_lbbox, list_sbboxes, list_mbboxes, list_lbboxes, img_name

    def __creat_label(self, bboxes, img_size):
        """
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "yxhw"; and scale bbox'
           yxhw by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.

        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.

        """

        anchors = np.array(cfg.MODEL["ANCHORS3D"])
        if 0:
            anchors = [
                [[29., 19., 14.],[19., 29., 14.],[29., 14., 19.]],
                [[24., 30., 32.], [46., 31., 18.], [57., 43., 44.]], #[46., 28., 30.],
                [[77., 62., 54.],[62., 77., 54.],[77., 54., 62.]],
            ]
            anchors = np.array(anchors)
            anchors[0] = anchors[0] / 8.
            anchors[1] = anchors[1] / 16.
            anchors[2] = anchors[2] / 32.
        if 0:
            anchors = [
                [[77., 62., 54.], [62., 77., 54.], [62., 54., 77.]],
                [[77., 62., 54.], [62., 77., 54.], [62., 54., 77.]],
                [[77., 62., 54.], [62., 77., 54.], [62., 54., 77.]],
                #[[24., 30., 32.], [46., 28., 30.], [46., 31., 18.]],
                #[[29., 19., 14.], [19., 29., 14.], [14., 19., 29.]],
            ]
            anchors = np.array(anchors)
            anchors[0] = anchors[0] / 8.
            anchors[1] = anchors[1] / 16.
            anchors[2] = anchors[2] / 32.

        strides = np.array(cfg.MODEL["STRIDES"])
        #train_output_size = img_size / strides
        anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]
        d_xyz = 6
        d_cls = 2
        label = [np.zeros((int(img_size[0] / strides[i]), int(img_size[1] / strides[i]), int(img_size[2] / strides[i]), anchors_per_scale, d_xyz+d_cls+self.num_classes))
                                                                      for i in range(3)]
        for i in range(3):
            label[i][..., 7] = 1.0

        bboxes_yxhw = [np.zeros((15, 6)) for _ in range(3)]   # Darknet the max_num is 30
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:6]
            bbox_class_ind = int(bbox[6])
            if len(bbox) >= 8:
                bbox_mix = bbox[7]
            else:
                bbox_mix = None

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)

            # convert "zyxzyx" to "zyxdhw"
            bbox_yxhw = np.concatenate([(bbox_coor[3:] + bbox_coor[:3]) * 0.5,
                                        bbox_coor[3:] - bbox_coor[:3]], axis=-1)

            bbox_yxhw_scaled = 1.0 * bbox_yxhw[np.newaxis, :] / strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_yxhw = np.zeros((anchors_per_scale, 6))
                anchors_yxhw[:, 0:3] = np.floor(bbox_yxhw_scaled[i, 0:3]).astype(np.int32) + 0.5  # 0.5 for compensation
                anchors_yxhw[:, 3:6] = anchors[i]

                iou_scale = tools.iou_xywh_numpy(bbox_yxhw_scaled[i][np.newaxis, :], anchors_yxhw)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    zind, yind, xind = np.floor(bbox_yxhw_scaled[i, 0:3]).astype(np.int32)

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    label[i][zind, yind, xind, iou_mask, 0:6] = bbox_yxhw
                    label[i][zind, yind, xind, iou_mask, 6:7] = 1.0
                    label[i][zind, yind, xind, iou_mask, 7:8] = bbox_mix
                    label[i][zind, yind, xind, iou_mask, 8:] = one_hot_smooth

                    #bbox_ind = int(bbox_count[i] % 15)  # BUG : 150为一个先验值,内存消耗大
                    bbox_ind = int(bbox_count[i])
                    bboxes_yxhw[i][bbox_ind, :6] = bbox_yxhw
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                zind, yind, xind = np.floor(bbox_yxhw_scaled[best_detect, 0:3]).astype(np.int32)

                label[best_detect][zind, yind, xind, best_anchor, 0:6] = bbox_yxhw
                label[best_detect][zind, yind, xind, best_anchor, 6:7] = 1.0
                label[best_detect][zind, yind, xind, best_anchor, 7:8] = bbox_mix
                label[best_detect][zind, yind, xind, best_anchor, 8:] = one_hot_smooth

                #bbox_ind = int(bbox_count[best_detect] % 15)
                bbox_ind = int(bbox_count[best_detect])
                bboxes_yxhw[best_detect][bbox_ind, :6] = bbox_yxhw
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_yxhw

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes