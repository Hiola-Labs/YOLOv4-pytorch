import os
import numpy as np
import torch
from torch.utils import data
import logging
from PIL import Image, ImageFont, ImageDraw
import config.yolov4_config as cfg
#from /home/lab402/bak/eason_thesis/program_update_v1/model.py
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''

    '''Preprocess true boxes to training input format
    # Eaosn: 3Dmodel
    Parameters
    ----------
    true_boxes: array, shape=(m, T, 7)
        Absolute x_min, y_min,z_min, x_max, y_max,z_max, class_id relative to input_shape.
    input_shape: array-like, dhw, multiples of 32
    anchors: array, shape=(N, 3), whd
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xyzwhd are reletive value

    '''
    assert (true_boxes[..., 6]<num_classes).all(), 'class id must be less than num_classes'

    num_layers = len(anchors)//3 # default setting
    use_two_scale=False
    if use_two_scale:
        num_layers = 2
    else:
        num_layers = 3
    #anchors = [[29., 19., 14.], [24., 30., 32.], [46., 28., 30.], [46., 31., 18.], [57., 43., 44.], [77., 62., 54.]]
    anchor_mask = [[5], [1, 2, 3, 4], [0]
                   ] if not use_two_scale else [[1, 2, 3, 4, 5], [0]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    boxes_xyz = (true_boxes[..., 0:3] + true_boxes[..., 3:6]) // 2

    boxes_whd = true_boxes[..., 3:6] - true_boxes[..., 0:3]
    true_boxes[..., 0:3] = boxes_xyz/input_shape[::-1]
    true_boxes[..., 3:6] = boxes_whd/input_shape[::-1]

    m = true_boxes.shape[0]

    if not use_two_scale:
        grid_shapes = [input_shape//{0: 16, 1: 8, 2: 4}[layer]
                       for layer in range(num_layers)]
    else:
        grid_shapes = [input_shape//{0:8, 1:2}[layer] for layer in range(num_layers)]

    # y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
    #     dtype='float32') for l in range(num_layers)]
    # y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], grid_shapes[l][2], len(anchor_mask[l]), 5+num_classes),
    #                    dtype='float32') for l in range(num_layers)]

    y_true = [np.zeros((m, grid_shapes[layer][0], grid_shapes[layer][1], grid_shapes[layer][2], len(anchor_mask[layer]), 7+num_classes),
                       dtype='float32') for layer in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # valid_mask = boxes_whd[..., 0] > 0
    # valid_mask = (boxes_whd[..., 0]>0 and boxes_whd[..., 1]>0 and boxes_whd[..., 2]>0)
    #0312 as is
    # valid_mask = (boxes_whd[..., 0] > 0) * (boxes_whd[..., 1] > 0) * (boxes_whd[..., 2]>0)*(boxes_xyz[...,0]>0)*(boxes_xyz[...,1]>0)*(boxes_xyz[...,2]>0)
    #0312 to be
    valid_mask = (boxes_whd[..., 0] > 0) * (boxes_whd[..., 1] > 0) * (boxes_whd[..., 2]>0)*(boxes_xyz[...,0]>0)*(boxes_xyz[...,1]>0)*(boxes_xyz[...,2]>0)

    for b in range(m):
        # Discard zero rows.
        whd = boxes_whd[b, valid_mask[b]]
        if len(whd)==0: continue

        whd = np.expand_dims(whd, -2)

        # actually no diff origin whd shape is (1,3) after change both to (1,1,3)

        box_maxes = whd / 2.
        box_mins = -box_maxes
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_whd = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_vol = intersect_whd[..., 0] * intersect_whd[..., 1] * intersect_whd[..., 2]
        box_vol = whd[..., 0] * whd[..., 1] * whd[..., 2]
        anchor_vol = anchors[..., 0] * anchors[..., 1] * anchors[..., 2]
        iou = intersect_vol / (box_vol + anchor_vol - intersect_vol)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            for layer in range(num_layers):
                if n in anchor_mask[layer]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[layer][2]).astype('int32')
                    j = np.floor(
                        true_boxes[b, t, 1]*grid_shapes[layer][1]).astype('int32')
                    k = np.floor(
                        true_boxes[b, t, 2]*grid_shapes[layer][0]).astype('int32')

                    mask = anchor_mask[layer].index(n)
                    c = true_boxes[b, t, 6].astype('int32')
                    y_true[layer][b, k,j, i, mask, 0:6] = true_boxes[b,t, 0:6]
                    y_true[layer][b, k, j, i, mask, 6] = 1
                    y_true[layer][b, k,j, i, mask, 7+c] = 1
    return y_true

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
def get_random_data(img_vol,annotation_line, input_shape, num_classes, random=True, max_boxes=30, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True, crop_scale=1.0):
    '''random preprocessing for real-time data augmentation'''

    line = annotation_line.split(',', 4)
    # img_vol = np.load(line[0])
    # img_vol = np.load(line[0].rsplit('.', 1)[0]+'_equalized.npy')
    boxes = line[-1].split(' ')

    box = np.array([np.array(list(map(int, box.split(','))))
                    for box in boxes])

    img_z, img_y, img_x = img_vol.shape

    model_z, model_y, model_x = input_shape

    target_box = box[0]

    x_1, y_1, z_1, x_2, y_2, z_2, _ = target_box
    tumor_size_x, tumor_size_y, tumor_size_z = x_2-x_1, y_2-y_1, z_2-z_1
    # rescale_ratio = 1.0
    # rescale_ratio=rand(0.8,1.2)
    rescale_ratio=1.0
    offset_x,offset_y,offset_z=0,0,0

    need_print=False

    min_size_x = np.clip((tumor_size_x/rescale_ratio)//3,
                         min(15/rescale_ratio,tumor_size_x/rescale_ratio), (model_x-10)/rescale_ratio)
    x_low = int(max(0, x_1+(min_size_x/rescale_ratio)-(model_x/rescale_ratio)))
    #x_high = int(min(img_x-2, x_2-(min_size_x/rescale_ratio) + (model_x/rescale_ratio))-(model_x/rescale_ratio))
    x_high = int(x_2-(min_size_x/rescale_ratio))
    x_high = min(x_high, img_x-2-(model_x/rescale_ratio))
    x_high = max(x_high, x_low)
    new_x1 = int(np.random.randint(min(x_low,x_high), max(x_low,x_high)+1))
    new_x2 = new_x1+np.floor(model_x/rescale_ratio)

    min_size_y = np.clip((tumor_size_y/rescale_ratio)//3,
                         min(15/rescale_ratio,tumor_size_y/rescale_ratio), (model_y-10)/rescale_ratio)
    y_low = int(max(0, y_1+(min_size_y/rescale_ratio)-(model_y/rescale_ratio)))
    y_high = int(y_2-(min_size_y/rescale_ratio))
    y_high = min(y_high, img_y-2-(model_y/rescale_ratio))
    y_high = max(y_high, y_low)

    new_y1 = int(np.random.randint(min(y_low,y_high), max(y_low,y_high)+1))
    new_y2 = new_y1+np.floor(model_y/rescale_ratio)

    min_size_z = np.clip((tumor_size_z/rescale_ratio)//3,
                         min(15/rescale_ratio,tumor_size_z/rescale_ratio), (model_z-10)/rescale_ratio)
    z_low = int(max(0, z_1+(min_size_z/rescale_ratio)-(model_z/rescale_ratio)))
    #z_high = int(min(img_z-2, z_2-(min_size_z/rescale_ratio) +
    #                 (model_z/rescale_ratio))-(model_z/rescale_ratio))
    z_high = int(z_2-(min_size_z/rescale_ratio))
    z_high = min(z_high, img_z-2-(model_z/rescale_ratio))
    z_high = max(z_high, z_low)
    new_z1 = int(np.random.randint(min(z_low,z_high), max(z_low,z_high)+1))
    new_z2 = new_z1+np.floor(model_z/rescale_ratio)
    assert new_x1>=0 and new_y1>=0 and new_z1>=0
    assert new_x2<=img_x and new_y2<=img_y and new_z2<=img_z
    # if tumor_size_x<=model_x/rescale_ratio:
    #     # x_low = int(max(0, x_1+(tumor_size_x*(1-((1/2)*tumor_size_x *
    #     #                                          rescale_ratio/model_x)))-(model_x/rescale_ratio)))
    #     # x_high = int(min(img_x, x_2-(tumor_size_x*(1-((1/2)*tumor_size_x *
    #     #                                               rescale_ratio/model_x)))+(model_x/rescale_ratio))-(model_x/rescale_ratio))
    #     x_low = int(max(0, x_1+(15/rescale_ratio)-(model_x/rescale_ratio)))
    #     x_high = int(min(img_x, x_2-(15/rescale_ratio) +
    #                      (model_x/rescale_ratio))-(model_x/rescale_ratio))
    #     # print('x_low, x_high', [x_low, x_high])
    #     new_x1 = int(np.random.randint(x_low, x_high+1))
    #     new_x2 = new_x1+np.floor(model_x/rescale_ratio)
    # else:
    #     need_print=True
    #     # index_x = np.random.randint(
    #     #     np.ceil(tumor_size_x*rescale_ratio/model_x))
    #     index_x = np.random.randint(
    #         np.floor(tumor_size_x*rescale_ratio/model_x)+1)
    #     x_low=int(max(0,x_1-model_x/(5*rescale_ratio)))
    #     x_high = int(max(0, x_1-model_x/(6*rescale_ratio)))+1
    #     start_x = int(np.random.randint(x_low, x_high))
    #     new_x1=start_x+index_x*(model_x/rescale_ratio)
    #     new_x2 = new_x1+np.floor(model_x/rescale_ratio)
    #     if new_x2>img_x:
    #         new_x2=img_x
    #         new_x1 = new_x2-np.floor(model_x/rescale_ratio)
    #     if new_x2 > x_2+model_x/(5*rescale_ratio):
    #         new_x2 = x_2+model_x/(5*rescale_ratio)
    #         new_x1 = new_x2-np.floor(model_x/rescale_ratio)

    # if tumor_size_y <= model_y/rescale_ratio:
    #     y_low = int(
    #         max(0, y_1+(15/rescale_ratio)-(model_y/rescale_ratio)))
    #     y_high = int(min(img_y, y_2-(15/rescale_ratio) +
    #                      (model_y/rescale_ratio))-(model_y/rescale_ratio))
    #     # y_low = int(
    #     #     max(0, y_1+(tumor_size_y*(1-((1/2)*tumor_size_y*rescale_ratio/model_y)))-(model_y/rescale_ratio)))
    #     # y_high = int(min(img_y, y_2-(tumor_size_y*(1-((1/2)*tumor_size_y *
    #     #                                               rescale_ratio/model_y)))+(model_y/rescale_ratio))-(model_y/rescale_ratio))
    #     new_y1 = int(np.random.randint(y_low, y_high+1))
    #     new_y2 = new_y1+np.floor(model_y/rescale_ratio)
    # else:
    #     need_print = True
    #     indey_y = np.random.randint(
    #         np.floor(tumor_size_y*rescale_ratio/model_y)+1)
    #     y_low = int(max(0, y_1-model_y/(5*rescale_ratio)))
    #     y_high = int(max(0, y_1-model_y/(6*rescale_ratio)))+1
    #     start_y = int(np.random.randint(y_low, y_high))
    #     new_y1 = start_y+indey_y*(model_y/rescale_ratio)
    #     new_y2 = new_y1+np.floor(model_y/rescale_ratio)
    #     if new_y2 > img_y:
    #         new_y2 = img_y
    #         new_y1 = new_y2-np.floor(model_y/rescale_ratio)
    #     if new_y2 > y_2+model_y/(5*rescale_ratio):
    #         new_y2 = y_2+model_y/(5*rescale_ratio)
    #         new_y1 = new_y2-np.floor(model_y/rescale_ratio)



    # if tumor_size_z <= model_z/rescale_ratio:

    #     z_low = int(
    #         max(0, z_1+(15/rescale_ratio)-(model_z/rescale_ratio)))
    #     z_high = int(min(img_z, z_2-(15/rescale_ratio) +
    #                      (model_z/rescale_ratio))-(model_z/rescale_ratio))
    #     # z_low = int(
    #     #     max(0, z_1+(tumor_size_z*(1-((1/2)*tumor_size_z*rescale_ratio/model_z)))-(model_z/rescale_ratio)))
    #     # z_high = int(min(img_z, z_2-(tumor_size_z*(1-((1/2)*tumor_size_z *
    #     #                                               rescale_ratio/model_z)))+(model_z/rescale_ratio))-(model_z/rescale_ratio))

    #     new_z1 = int(np.random.randint(z_low, z_high+1))
    #     new_z2 = new_z1+np.floor(model_z/rescale_ratio)
    # else:
    #     need_print = True
    #     indez_z = np.random.randint(
    #         np.floor(tumor_size_z*rescale_ratio/model_z)+1)
    #     z_low = int(max(0, z_1-model_z/(5*rescale_ratio)))
    #     z_high = int(max(0, z_1-model_z/(6*rescale_ratio)))+1
    #     start_z = int(np.random.randint(z_low, z_high))
    #     new_z1 = start_z+indez_z*(model_z/rescale_ratio)
    #     new_z2 = new_z1+np.floor((model_z/rescale_ratio))
    #     if new_z2 > img_z:
    #         new_z2 = img_z
    #         new_z1 = new_z2-np.floor((model_z/rescale_ratio))
    #     if new_z2 > z_2+model_z/(5*rescale_ratio):
    #         new_z2 = z_2+model_z/(5*rescale_ratio)
    #         new_z1 = new_z2-np.floor((model_z/rescale_ratio))

    # 11/01 change not must include the whole true box

    new_z1, new_z2, new_y1, new_y2, new_x1, new_x2 = int(new_z1), int(
        new_z2), int(new_y1), int(new_y2), int(new_x1), int(new_x2)

    #offset_x = int(np.floor(((model_x-int((new_x2-new_x1)*rescale_ratio))//2)))
    #offset_y = int(np.floor(((model_y-int((new_y2-new_y1)*rescale_ratio))//2)))
    #offset_z= int(np.floor(((model_z-int((new_z2-new_z1)*rescale_ratio))//2)))

    img_vol = img_vol[new_z1:new_z2, new_y1:new_y2, new_x1:new_x2].copy()
    assert img_vol.shape == (model_z, model_y, model_x)
    box[:, [0, 3]] = (box[:, [0, 3]]-new_x1)*rescale_ratio#+offset_x
    box[:, [1, 4]] = (box[:, [1, 4]]-new_y1)*rescale_ratio#+offset_y
    box[:, [2, 5]] = (box[:, [2, 5]]-new_z1)*rescale_ratio#+offset_z
    need_to_delete = []
    for i in range(len(box)):
        box[i, 0] = max(0, box[i, 0])
        box[i, 1] = max(0, box[i, 1])
        box[i, 2] = max(0, box[i, 2])
        box[i, 3] = min(model_x, box[i, 3])
        box[i, 4] = min(model_y, box[i, 4])
        box[i, 5] = min(model_z, box[i, 5])
        if box[i, 0] >= box[i, 3] or box[i, 1] >= box[i, 4] or box[i, 2] >= box[i, 5]:
            need_to_delete.append(i)
    box = np.delete(box, need_to_delete, axis=0)

    box_data = np.zeros((max_boxes, 7))

    if not random:
        img_vol=img_vol.astype(float)
        if 0:
            # flip image or not#Eason add flip
            flip = rand() < .5
            if flip:
                img_vol = img_vol[:, :, ::-1]
                # FLIP_LEFT_RIGHT

            reverse = rand() > .5
            if reverse:
                img_vol = img_vol[::-1, :, :]
                # REVERSE_SLICE
            # flip image or not#Eason add flip
            rotate=rand() > .5
            if rotate:
                img_vol = img_vol[:, ::-1, :]

            contrast=rand()>.5
            if 0 and contrast:
                scale=rand(0.5,2.0)
                img_vol=np.clip(img_vol*scale,0.0,255.0)


            # correct boxes
            if len(box) > 0:
                if flip:
                    box[:, [0, 3]] = model_x - box[:, [3, 0]]  # Eason add flip

                if reverse:
                    box[:, [2, 5]] = model_z - \
                        box[:, [5, 2]]  # Eason add reverse

                if rotate:
                    box[:, [1, 4]] = model_y - \
                        box[:, [4, 1]]  # Eason add reverse

        # with open('0610_anchors.txt','a+') as f:
        #     for sub_box in box:
        #         f.write(str(sub_box)+'\n')
        image_data = np.stack(img_vol/255)
        box_data[:len(box)] = box
        return image_data, box_data

    ####### not use
    # resize image Eason:change the scale from 0.25,2 to .75,1.25
    new_ar = w/h * rand(1-jitter, 1+jitter)/rand(1-jitter, 1+jitter)
    scale = rand(.75, 1.25)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    img_vol = img_vol.resize((nw, nh), Image.BICUBIC)

    # Eason: close it

    # place image
    dx = (w-nw)//2
    # dx = int(rand(0, w-nw))
    dy = (h-nh)//2
    # dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w, h), (0, 0, 0))
    new_image.paste(img_vol, (dx, dy))
    img_vol = new_image

    # flip image or not
    flip = rand() < .5
    if flip:
        img_vol = img_vol.transpose(Image.FLIP_LEFT_RIGHT)
    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1/rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(img_vol)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
        box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
        if flip:
            box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def data_generator(annotation_lines, batch_size, sub_batch_size, input_shape, anchors, num_classes, train_BG, GT_num, model_path=''):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    data_iter = 0
    GT_num = min(GT_num, sub_batch_size)
    BG_num = int(sub_batch_size-GT_num)
    global now_cnt

    while True:
        data_iter += 1
        image_data = []
        box_data = []
        for b in range(int(batch_size/sub_batch_size)):
            if i == 0:
                np.random.shuffle(annotation_lines)

            line = annotation_lines[i]
            img_vol = np.load(line.split(',', 4)[0])
            fail_file = line.split(',', 4)[0].rsplit('.', 1)[0]+'_fail_list\\'+model_path+'\\fail.txt'

            if not os.path.exists(fail_file) or len(open(fail_file).readlines()) == 0:
                target_BG_num = 0
            else:
                target_BG_num = min(BG_num, len(open(fail_file).readlines()))
                print('train BG: ', target_BG_num)
            target_GT_num = sub_batch_size-target_BG_num

            for _ in range(target_GT_num):
                image, box = get_random_data(
                    img_vol, line, input_shape, num_classes, random=False)
                image_data.append(image)
                box_data.append(box)

            for _ in range(target_BG_num):
                image, box = get_fail_background(
                    img_vol, line, input_shape, num_classes, fail_file, random=False)
                image_data.append(image)
                box_data.append(box)

            i = (i+1) % n

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        #y_true = preprocess_true_boxes(
        #    box_data, input_shape, anchors, num_classes)

        image_data = np.expand_dims(image_data, -1)
        assert not np.any(np.isnan(image_data))
        #eason return zyx image and y_true as anchor ground_truth map in zyx order
        #yield [image_data, *y_true], np.zeros(batch_size)
        yield [image_data, box_data], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, sub_batch_size, input_shape, anchors, num_classes, train_BG, GT_num, model_path=''):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, sub_batch_size, input_shape, anchors, num_classes, train_BG, GT_num, model_path)


class AbusNpyFormat(data.Dataset):
    def __init__(self, testing_mode, root, enable_CV=False, crx_fold_num=0, crx_partition='train', augmentation=False, include_fp=False, batch_size=0):
        fold_list_root = '/home/lab402/User/eason_thesis/program_update_v1/5_fold_list/'
        self.crx_partition = crx_partition
        # EASON code
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        fold_num = crx_fold_num
        train_path = fold_list_root + 'five_fold_train_' + \
            str(fold_num)+'_separate.txt'
        val_path = fold_list_root + 'five_fold_val_'+str(fold_num)+'_separate.txt'
        test_path = fold_list_root + 'five_fold_test_'+str(fold_num)+'.txt'

        input_shape = cfg.TRAIN["TRAIN_IMG_SIZE"]#(640, 160, 640) #(160,160,160)#(96, 96, 96)
        train_set = open(train_path).readlines()
        #train_set = [_.replace('/home/lab402/User/eason_thesis/ABUS_data/', '') for _ in train_set]
        #train_set = [root + '/converted_640_160_640/' + _.replace('/', '_') for _ in train_set]

        val_set = open(val_path).readlines()
        self.val_set=val_set
        #val_set = [_.replace('/home/lab402/User/eason_thesis/ABUS_data/', '') for _ in val_set]
        #val_set = [root + '/converted_640_160_640/' + _.replace('/', '_') for _ in val_set]

        batch_size = batch_size #18
        sub_batch_size = batch_size #18
        class_names = ['tumor'] #class_names = get_classes(classes_path)
        num_classes = 1 #num_classes = len(class_names)
        #anchors = get_anchors(anchors_path)
        anchors = [[29., 19., 14.], [24., 30., 32.], [46., 28., 30.], [46., 31., 18.], [57., 43., 44.], [77., 62., 54.]]
        #train_set = train_set[:100]
        self.eason_training_data = data_generator_wrapper(
                                    train_set, batch_size, sub_batch_size, input_shape, anchors, num_classes, train_BG=False, GT_num=sub_batch_size)

        self.eason_validation_data = data_generator_wrapper(
                                    val_set, batch_size, sub_batch_size, input_shape, anchors, num_classes, train_BG=False, GT_num=sub_batch_size)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #d = next(self.eason_training_data)
        self.set_size = 250 #len(train_set)//batch_size
        self.gt = []
        self.root = root.rstrip('/') + '/'
        self.aug = augmentation

        TY = 0
        if TY or (crx_partition!='train') :#for 640
            input_shape = (640, 160, 640)
            file_part = 'val' if crx_partition=='valid' else crx_partition
            if testing_mode==1 and file_part=='val':
                file_part = 'test'
            fold_list_file = fold_list_root + 'five_fold_{}_{}.txt'.format(file_part, crx_fold_num)

            with open(fold_list_file, 'r') as f:
                self.gt = f.read().splitlines()
            self.gt = [_.replace('/home/lab402/User/eason_thesis/ABUS_data/', '') for _ in self.gt]
            self.set_size = len(self.gt)
            self.aug = augmentation
        if 0:
            if include_fp:
                print('FP training mode...')
                with open(self.root + 'annotations/fp_{}.txt'.format(crx_fold_num), 'r') as f:
                    lines = f.read().splitlines()
            else:
                print('Normal mode....')
                with open(self.root + 'annotations/rand_all.txt', 'r') as f:
                    lines = f.read().splitlines()

            folds = []
            self.gt = []
            if enable_CV:
                assert crx_partition in ['train', 'valid'] , 'crx_partition must be train or valid with type str when cross validation enabled '
                for fi in range(5):
                    if fi == 4:
                        folds.append(lines[int(fi*0.2*len(lines)):])
                    else:
                        folds.append(lines[int(fi*0.2*len(lines)):int((fi+1)*0.2*len(lines))])
                #folds=[f0, f1, f2, f3, f4]
                cut_set = folds.pop(crx_fold_num)
                if crx_partition == 'train':
                    for li in folds:
                        self.gt += li
                elif crx_partition == 'valid':
                    self.gt = cut_set
            else:
                self.gt = lines

            self.set_size = len(self.gt)
            self.aug = augmentation
        self.img_size = (640,160,640) #(640,160,640)
        print('Dataset info: Cross-validation {}, partition: {}, fold number {}, data augmentation {}'\
            .format(enable_CV, crx_partition, crx_fold_num, self.aug))

    def get_data_lines(self):
        return self.val_set

    def __getitem__(self, index):
        aug_mode, index = self._get_aug_index(index)
        Load_TY = 0 #for 640
        if (self.crx_partition!='train'):
            Load_TY=1
        if Load_TY:
            # 0: original, 1: flip Z, 2: flip X, 3: flip ZX
            aug_mode, index = self._get_aug_index(index)
            line = self.gt[index]
            line = line.split(',', 4)

            # Only for L, W, H regression
            size = (640,160,640)
            gt_scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))
            scale = (4,4,4)

            # numpy array data (x,y,z) is not in the same order as gt label, which is (z,y,x)
            # line[0] = 'JuaShenData/JuaShenData/Converted/2-020/02287/LAP/1.3.6.1.4.1.47779.1.004.npy'
            # line[0] = 'JuaShenData/JuaShenData/Converted/2-020/02287/LAP/1.3.6.1.4.1.47779.1.004.npy'
            npy_name = self.root + 'converted_{}_{}_{}/'.format(self.img_size[0], self.img_size[1], self.img_size[2]) + line[0].replace('/', '_')
            ori_data = np.load(npy_name)
            TY_ori_data = ori_data
            ori_data = torch.from_numpy(ori_data)
            ori_data = torch.transpose(ori_data, 0, 2).contiguous()
            ori_data = ori_data.view(1,self.img_size[0],self.img_size[1],self.img_size[2]).to(torch.float32)


            true_boxes = line[-1].split(' ')
            true_boxes = list(map(lambda box: box.split(','), true_boxes))
            true_boxes = [list(map(int, box)) for box in true_boxes]

            TY_Image = ori_data# claim transposed to zyx
            TY_Box = true_boxes# claim zyx
            true_boxes = [[box[0] * gt_scale[0], box[1] * gt_scale[1], box[2] * gt_scale[2],
                           box[3] * gt_scale[0], box[4] * gt_scale[1], box[5] * gt_scale[2], ] for box in true_boxes]
            data, boxes = self._flipTensor(ori_data, true_boxes, aug_mode = aug_mode)
            scale = gt_scale
            ori_data = ori_data.unsqueeze(0).permute((0, 2, 3, 4, 1))
            ori_data = ori_data/255.0
            boxes = [boxes]
            if 0:
                for i in range(int(boxes[0][0]['x_bot']), int(boxes[0][0]['x_top']), 1):
                    #TY Image
                    img = Image.fromarray(((ori_data[0].squeeze().numpy() * 255.0).astype('uint8'))[:,:,i], 'L')
                    #img = Image.fromarray(TY_ori_data[i,:,:], 'L')
                    img = img.convert(mode='RGB')
                    draw = ImageDraw.Draw(img)
                    for bx in boxes[0]:
                        z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot'], bx['z_top'], bx['y_bot'], bx['y_top'], bx['x_bot'], bx['x_top']
                        if int(x_bot) <= i <= int(x_top):
                            #z_bot,y_bot = int(z_bot), int(y_bot)
                            #z_top,y_top = int(z_top), int(y_top)

                            draw.rectangle(
                                [(y_bot, z_bot),(y_top, z_top)],
                                outline ="red", width=2)
                    img.save('debug/TY_' + str(i)+'.png')


        #eason reads
        #overwrite_line = '/home/lab402/User/eason_thesis/ABUS_data/JuaShenData/JuaShenData/Converted/2-020/02287/LAP/1.3.6.1.4.1.47779.1.004.npy,668,160,665,218,7,260,336,73,321,0'

        # for box in boxes:
        #     if box['z_bot'] <= 0 or box['x_bot'] <= 0:
        #         print("A box is out of bound...")
        Load_EASON = 1 #for 96
        if (self.crx_partition!='train'):
            Load_EASON=0
        if Load_EASON:
            d = next(self.eason_training_data)
            image_data_pack, box_data_pack = d[0]
            box_data_pack = [[box for box in each_box_data if (box[3]*box[4]*box[5])>0] for each_box_data in box_data_pack]
            gt_scale = [1., 1., 1.]
            new_data = []
            new_boxes = []
            for ori_data, true_boxes in zip(image_data_pack, box_data_pack):
                ori_data = torch.from_numpy(ori_data).squeeze()
                ori_data = torch.transpose(ori_data, 0, 2).contiguous()
                ori_data = ori_data.unsqueeze(dim=0).to(torch.float32)
                data, boxes = self._flipTensor(ori_data, true_boxes, aug_mode=aug_mode)
                new_data.append(data)
                new_boxes.append(boxes)
            scale = gt_scale
            ori_data = torch.stack(new_data).permute((0, 2, 3, 4, 1))
            #ori_data = ori_data/255.0
            boxes = new_boxes

            #ori_data = torch.from_numpy(image_data)
            #ori_data = torch.transpose(ori_data, 1, 3).contiguous()
            #ori_data = ori_data.to(torch.float32)
            #.view(-1,96,96,96,1)
            if 0:
                for i in range(int(boxes[0][0]['x_bot']), int(boxes[0][0]['x_top']), 1):
                    #TY Image
                    img = Image.fromarray(((ori_data[0].squeeze().numpy() * 255.0).astype('uint8'))[:,:,i], 'L')
                    #img = Image.fromarray(TY_ori_data[i,:,:], 'L')
                    img = img.convert(mode='RGB')
                    draw = ImageDraw.Draw(img)
                    for bx in boxes[0]:
                        z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot'], bx['z_top'], bx['y_bot'], bx['y_top'], bx['x_bot'], bx['x_top']
                        if int(x_bot) <= i <= int(x_top):
                            #z_bot,y_bot = int(z_bot), int(y_bot)
                            #z_top,y_top = int(z_top), int(y_top)

                            draw.rectangle(
                                [(y_bot, z_bot),(y_top, z_top)],
                                outline ="red", width=2)
                    img.save('debug/ES_' + str(i)+'.png')

            if 0:
                for i in range(int(boxes[0][0]['x_bot']), int(boxes[0][0]['x_top']), 3):
                    #TY Image
                    img = Image.fromarray(((ori_data[0].squeeze().numpy() * 255.0).astype('uint8'))[:,:,i], 'L')
                    #img = Image.fromarray(TY_ori_data[i,:,:], 'L')
                    img = img.convert(mode='RGB')
                    draw = ImageDraw.Draw(img)
                    scale = [1, 1, 1]
                    for bx in boxes[0]:
                        z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot']*scale[0], bx['z_top']*scale[0], bx['y_bot']*scale[1], bx['y_top']*scale[1], bx['x_bot']*scale[2], bx['x_top']*scale[2]
                        if int(x_bot) <= i <= int(x_top):
                            draw.rectangle(
                                [(y_bot, z_bot),(y_top, z_top)],
                                outline ="red", width=2)
                    img.save('debug/EASON_' + str(i)+'.png')
                print("print visualize done ", boxes[0][0]['x_bot'], " ~ ", boxes[0][0]['x_top'])

        return ori_data, boxes

    def draw_img(self, volume, line):


        line = line.split(',', 4)
        boxes = line[-1].split(' ')
        true_box = np.array([np.array(list(map(int, box.split(','))))
                            for box in boxes])
        true_box[:, [0, 2]] = true_box[:, [2, 0]]
        true_box[:, [3, 5]] = true_box[:, [5, 3]]

        file_name = line[0]
        transverse_vol = np.load(file_name)
        transverse_vol.setflags(write=1)
        color_image_vol=[]
        for now_img in transverse_vol:
            now_img = Image.fromarray(now_img)
            now_img=now_img.convert('RGB')
            now_img = np.array(now_img)
            color_image_vol.append(now_img)
        transverse_vol = np.array(color_image_vol)

        coronal_vol = []
        for idy in range((transverse_vol.shape[1])):
            img = transverse_vol[:, idy, :]
            coronal_vol.append(img)
        coronal_vol=np.array(coronal_vol)

        sagittal_vol = []
        for idx in range((transverse_vol.shape[2])):
            img = transverse_vol[:, :, idx]
            sagittal_vol.append(img)
        sagittal_vol=np.array(sagittal_vol)
        sagittal_vol=np.flipud(sagittal_vol)

        thickness = 1
        for box_idx in range(len(true_box)):
            Z_start, Y_start, X_start, Z_end, Y_end, X_end, Class_no = true_box[box_idx]
            # Transverse view
            for idx_Z in range(Z_start, Z_end):
                now_img = transverse_vol[idx_Z]
                now_img = Image.fromarray(now_img)
                now_img=now_img.convert('RGB')
                draw = ImageDraw.Draw(now_img)

                for i in range(thickness):
                    draw.rectangle(
                        [X_start + i, Y_start + i,  X_end - i, Y_end - i],
                        outline=(255,0,0))
                del draw
                transverse_vol[idx_Z] = np.array(now_img)

            # Coronal view
            for idx_Y in range(Y_start, Y_end):
                now_img = coronal_vol[idx_Y]
                now_img = Image.fromarray(now_img)
                now_img = now_img.convert('RGB')
                draw = ImageDraw.Draw(now_img)

                for i in range(thickness):
                    draw.rectangle(
                        [X_start + i, Z_start + i,  X_end - i, Z_end - i],
                        outline=(255, 0, 0))
                del draw
                coronal_vol[idx_Y] = np.array(now_img)

            # Saggital view
            for idx_X in range(X_start, X_end):
                now_img = sagittal_vol[idx_X]
                now_img = Image.fromarray(now_img)
                now_img = now_img.convert('RGB')
                draw = ImageDraw.Draw(now_img)

                for i in range(thickness):
                    draw.rectangle(
                        [Y_start + i,Z_start + i, Y_end - i, Z_end - i],
                        outline=(255, 0, 0))
                del draw
                sagittal_vol[idx_X] = np.array(now_img)
            directory = 'debug/'
            if not os.path.exists(directory + 'transverse_view/') : os.mkdir(directory + 'transverse_view/')
            if not os.path.exists(directory + 'coronal_view/') : os.mkdir(directory + 'coronal_view/')
            if not os.path.exists(directory + 'sagittal_view/') : os.mkdir(directory + 'sagittal_view/')
            for idx,img in enumerate(transverse_vol):
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img.save(directory + 'transverse_view/' +
                            line[0].split('/')[-1].rsplit('.', 1)[0]+'_'+str(idx)+'.png')

            for idx,img in enumerate(coronal_vol):
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img.save(directory + 'coronal_view/' +
                            line[0].split('/')[-1].rsplit('.', 1)[0]+'_'+str(idx)+'.png')

            for idx,img in enumerate(sagittal_vol):
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img.save(directory + 'sagittal_view/' +
                            line[0].split('/')[-1].rsplit('.', 1)[0]+'_'+str(idx)+'.png')

    def __len__(self):
        if self.aug:
            return 4*self.set_size
        else:
            return self.set_size


    def _flipTensor(self, data, true_boxes, aug_mode=0):
        img_size_z, img_size_y, img_size_x = data[0].shape
        boxes = [{
            'z_bot': box[0],
            'z_top': box[3],
            'z_range': box[3] - box[0] + 1,
            'z_center': (box[0] + box[3]) / 2,
            'y_bot': box[1],
            'y_top': box[4],
            'y_range': box[4] - box[1] + 1,
            'y_center': (box[1] + box[4]) / 2,
            'x_bot': box[2],
            'x_top': box[5],
            'x_range': box[5] - box[2] + 1,
            'x_center': (box[2] + box[5]) / 2,
        } for box in true_boxes]

        if aug_mode == 1:
            data = torch.flip(data, [1])
            boxes = [{
                'z_bot': max(0, img_size_z - box['z_top']),
                'z_top': img_size_z - box['z_bot'],
                'z_range': box['z_range'],
                'z_center': img_size_z - box['z_center'],
                'y_bot': box['y_bot'],
                'y_top': box['y_top'],
                'y_range': box['y_range'],
                'y_center': box['y_center'],
                'x_bot': box['x_bot'],
                'x_top': box['x_top'],
                'x_range': box['x_range'],
                'x_center': box['x_center'],
            } for box in boxes]
        elif aug_mode == 2:
            data = torch.flip(data, [3])
            boxes = [{
                'z_bot': box['z_bot'],
                'z_top': box['z_top'],
                'z_range': box['z_range'],
                'z_center': box['z_center'],
                'y_bot': box['y_bot'],
                'y_top': box['y_top'],
                'y_range': box['y_range'],
                'y_center': box['y_center'],
                'x_bot': max(0, img_size_x - box['x_top']),
                'x_top': img_size_x - box['x_bot'],
                'x_range': box['x_range'],
                'x_center': img_size_x - box['x_center'],
            } for box in boxes]

        elif aug_mode == 3:
            data = torch.flip(data, [1,3])
            boxes = [{
                'z_bot': max(0, img_size_z - box['z_top']),
                'z_top': img_size_z - box['z_bot'],
                'z_range': box['z_range'],
                'z_center': img_size_z - box['z_center'],
                'y_bot': box['y_bot'],
                'y_top': box['y_top'],
                'y_range': box['y_range'],
                'y_center': box['y_center'],
                'x_bot': max(0, img_size_x - box['x_top']),
                'x_top': img_size_x - box['x_bot'],
                'x_range': box['x_range'],
                'x_center': img_size_x - box['x_center'],
            } for box in boxes]

        return data, boxes
    def getID(self, index):
        if index >= len(self.gt):
            return 'unknow'
        aug_mode, index = self._get_aug_index(index)
        return self.getFilePath(index).replace('/', '_') + '_' + str(aug_mode)

    def getName(self, index):
        return self.getFilePath(index).replace('/', '_')

    def getFilePath(self, index):
        aug_mode, index = self._get_aug_index(index)
        line = self.gt[index]
        line = line.split(',', 4)
        return line[0]

    def _get_aug_index(self, index):
        # 0: original, 1: flip Z, 2: flip X, 3: flip ZX
        aug_mode = index // self.set_size
        index = index % self.set_size
        return aug_mode, index
