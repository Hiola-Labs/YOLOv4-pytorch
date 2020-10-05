import os
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from tqdm import tqdm

from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import config.yolov4_config as cfg

from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator
from eval.evaluator import Evaluator

import utils.gpu as gpu

class lightenYOLOv4(pl.LightningModule):
    def __init__(self, weight_path, resume, accumulate=None):
        # precision=16 for fp16

        super().__init__()
        self.model = Build_Model(weight_path=weight_path, resume=resume)
        self.criterion = YoloV4Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.evaluator = Evaluator(self.model, showatt=False)

    # how you want your model to do inference/predictions
    def forward(self, img):
        p, p_d = self.model(img)
        return p, p_d
    """
    def training_epoch_end(self,outputs):
        #  the function is called after every epoch is completed
        # calculating average loss
        avg_loss = 0
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # creating log dictionary
        result = pl.TrainResult()
        result.log('val/loss_epoch', avg_loss)
        return result
    """


    # the train loop INDEPENDENT of forward.
    def training_step(self, batch, batch_idx):
        img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, _ = batch

        p, p_d = self(img)
        loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)


        result = pl.TrainResult(minimize=loss)
        #result = pl.TrainResult(loss)
        #result.log('train_loss_ciou', loss_ciou)
        #result.log('train_loss_conf', loss_conf)
        #result.log('train_loss_cls', loss_cls)
        result.log('train_loss', loss, on_step=True, on_epoch=True)
        return result
        '''
        #https://www.learnopencv.com/tensorboard-with-pytorch-lightning/
        logs={"train_loss": loss,
            "train_loss_ciou":loss_ciou,
            "train_loss_conf":loss_conf,
            "train_loss_cls":loss_cls,}
        batch_dictionary={
            #REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            #optional for batch logging purposes
            "log": logs,
        }
        return batch_dictionary

        '''



    def validation_epoch_end(self, outputs):
        print(outputs)
        APs = self.evaluator.calc_APs()
        mAP = 0
        for i in APs:
            mAP += APs[i]
        mAP = mAP / self.model.getNC()
        result = pl.EvalResult()
        result.log('val/mAP_epoch', torch.Tensor([mAP]).cuda())
        #trainer.logger_connector.logged_metrics
        return result

    def validation_step(self, batch, batch_idx):
        img_batch, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_name = batch

        for idx, img in tqdm(zip(img_name, img_batch)):
            # CHW -> HWC
            img = img.cpu().numpy().transpose(1, 2, 0)
            bboxes_prd = self.evaluator.get_bbox(img, multi_test=False, flip_test=False)
            self.evaluator.store_bbox(idx, bboxes_prd)
        
        #img_batch = img_batch.cpu().numpy().transpose(0, 3, 2, 1)
        #img_batch = torch.tensor(img_batch)
        p, p_d = self(img_batch)
        loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)
        
        self.log('val_loss_ciou', loss_ciou)
        self.log('val_loss_conf', loss_conf)
        self.log('val_loss_cls', loss_cls)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = batch

        p, p_d = self(img)
        loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)
        #, loss_ciou, loss_conf, loss_cls
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"],
                        momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        return optimizer
