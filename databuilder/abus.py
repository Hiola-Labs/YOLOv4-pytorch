from .abus_data_eason import AbusNpyFormat
#from .abus_data import AbusNpyFormat
from .image3d import ImageDetect3DDataset
from .bounding_box import BoxList
import torch
import numpy as np
from torch.utils.data import Dataset
#usage:
#for COCO
#   imageDataset = torchvision.datasets.coco.CocoDetection(root, ann_file)
#   final_dataset = ImageDetect3DDataset(imageDataset, transforms)
#for ABUS
#   imageDataset = ABUSDetectionDataset(
#       root, transform=None, target_transform=None, transforms=None, \
#       crx_valid=False, crx_fold_num=0, crx_partition='train', \
#       augmentation=False, include_fp=False)
#   final_dataset = ImageDetect3DDataset(imageDataset, transforms)

class ABUSDetectionDataset(Dataset):
    """`NTU402 ABUS Detection Dataset`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    # read from AbusNpyFormat(root, annFile)
    def __init__(self, testing_mode, root, transform=None, target_transform=None, transforms=None, \
            crx_valid=False, crx_fold_num=0, crx_partition='train', \
            augmentation=False, include_fp=False, batch_size=0):

        self.abusNpy = AbusNpyFormat(testing_mode, root, \
            crx_valid, crx_fold_num, crx_partition, \
            augmentation, include_fp, batch_size)

    def __len__(self):
        return len(self.abusNpy)

    def getCatIds(self):
        return [0, 1] #background=0  tumor=1

    def __getitem__(self, index):

        imgs, targets = self.abusNpy[index]
        #return_img = []
        return_anno = []
        return_id = []
        max_len_anno = 0
        for img, target in zip(imgs, targets):
            max_len_anno = max(len(target), max_len_anno)
        for img, target in zip(imgs, targets):
            anno = []
            for item in target:
                x1 = item['x_bot']
                y1 = item['y_bot']
                z1 = item['z_bot']
                x2 = item['x_top']
                y2 = item['y_top']
                z2 = item['z_top']
                category_id=1
                bbox_mix = 1.
                anno.append([z1, y1, x1, z2, y2, x2, category_id, bbox_mix])
            #data and notation in ZYXC shape
            #return_img.append((img.permute(1,2,3,0).float() / 255.0).numpy())
            #return_img.append(img.numpy())
            if len(anno) < max_len_anno:
                for i in range((max_len_anno - len(anno))):
                    anno.append([0, 0, 0, 0, 0, 0, 0, 0])
            return_anno.append(np.array(anno, dtype=np.float32))
            return_id.append(self.abusNpy.getID(index))
        #np.array(return_img)

        return imgs, np.stack(return_anno), return_id

    def get_img_info(self, index):
        return {'height':self.abusNpy.img_size[0], \
            'width':self.abusNpy.img_size[2], \
            'depth':self.abusNpy.img_size[1]}
