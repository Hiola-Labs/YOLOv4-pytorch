import logging
import utils.gpu as gpu
from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from eval.froc import calculate_FROC
from utils.tools import *
from torch.utils.tensorboard import SummaryWriter
import config.yolov4_config as cfg
from utils import cosine_lr_scheduler
from utils.log import Logger

from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator

from databuilder.abus import ABUSDetectionDataset
from databuilder.yolo4dataset import YOLO4_3DDataset
from tqdm import tqdm


class Trainer(object):
    def __init__(self, testing_mode, weight_path, checkpoint_save_dir, resume, gpu_id, accumulate, fp_16, writer, logger, crx_fold_num):
        self.data_root = 'datasets/abus'
        init_seeds(0)
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.accumulate = accumulate
        self.fp_16 = fp_16
        self.writer = writer
        self.logger = logger
        self.weight_path = weight_path
        self.checkpoint_save_dir = checkpoint_save_dir
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if self.multi_scale_train:print('Using multi scales training')
        else:print('train img size is {}'.format(cfg.TRAIN["TRAIN_IMG_SIZE"]))
        self.logger.info('augmentation=True, crx_fold_num= {}'.format(crx_fold_num))
        self.testing_mode = testing_mode
        self.crx_fold_num = crx_fold_num
        train_dataset = ABUSDetectionDataset(testing_mode, augmentation=True, crx_fold_num= crx_fold_num, crx_partition= 'train', crx_valid=True, include_fp=False, root=self.data_root,
            batch_size=cfg.TRAIN["BATCH_SIZE"])
        self.train_dataset = YOLO4_3DDataset(train_dataset, classes=[0, 1], img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        #self.train_dataset = data.Build_Dataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])

        self.epochs = cfg.TRAIN["YOLO_EPOCHS"] if cfg.MODEL_TYPE["TYPE"] == 'YOLOv4' else cfg.TRAIN["Mobilenet_YOLO_EPOCHS"]
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=1,
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True, pin_memory=True
                                           )


        test_dataset = ABUSDetectionDataset(testing_mode, augmentation=False, crx_fold_num= crx_fold_num, crx_partition= 'valid', crx_valid=True, include_fp=False, root=self.data_root,
            batch_size=cfg.VAL["BATCH_SIZE"])

        self.test_dataset = YOLO4_3DDataset(test_dataset, classes=[0, 1], img_size=cfg.VAL["TEST_IMG_SIZE"])
        self.test_dataloader = DataLoader(self.test_dataset,
                                            batch_size=1,
                                            num_workers=cfg.VAL["NUMBER_WORKERS"],
                                            shuffle=False, pin_memory=True
                                            )
        #sum([p.flatten().size(0) for p in self.model.parameters()])
        self.model = Build_Model(weight_path=weight_path, resume=resume, dims=3).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])

        self.criterion = YoloV4Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"], dims=3)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.TRAIN["LR_INIT"],
                                                          lr_min=cfg.TRAIN["LR_END"],
                                                          warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))
        if weight_path:
            if resume: self.__load_resume_weights(weight_path, load_as_pretrained=False)
            if not resume: self.__load_resume_weights(weight_path, load_as_pretrained=True)
        self.logger.info(self.model)
    def __load_resume_weights(self, weight_path, load_as_pretrained):
        last_weight = os.path.join(weight_path)
        chkpt = torch.load(last_weight, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])
        self.logger.info('__load_resume_weights, last_weight= {}, load_as_pretrained:{}'.format(last_weight, load_as_pretrained))
        if not load_as_pretrained:
            if chkpt['epoch'] is not None:
                self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
        del chkpt

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP

        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}

        torch.save(chkpt, os.path.join(self.checkpoint_save_dir, 'backup_epoch%g.pt'%epoch))
        torch.save(chkpt, os.path.join(self.checkpoint_save_dir, 'lastest_epoch.pt'))
        if epoch==0 or (self.best_mAP == mAP and mAP>0):
            torch.save(chkpt, os.path.join(self.checkpoint_save_dir, "best.pt"))
        #torch.save(chkpt, os.path.join(self.checkpoint_save_dir, "last.pt"))
        del chkpt



    def train(self):
        writer = self.writer
        logger = self.logger
        logger.info("Training start,img size is: {},batchsize is: {:d},work number is {:d}".format(cfg.TRAIN["TRAIN_IMG_SIZE"],cfg.TRAIN["BATCH_SIZE"],cfg.TRAIN["NUMBER_WORKERS"]))
        logger.info("Train datasets number is : {}".format(len(self.train_dataset)))

        if self.fp_16: self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)
        logger.info("        =======  start  training   ======     ")
        #area_small, area_big, plt = self.evaluate()
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            self.model.train()

            mloss = torch.zeros(4)
            logger.info("===Epoch:[{}/{}]===".format(epoch, self.epochs))
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_names)  in tqdm(enumerate(self.train_dataloader)):
                imgs = imgs[0]
                label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = \
                    label_sbbox[0], label_mbbox[0], label_lbbox[0], sbboxes[0], mbboxes[0], lbboxes[0]
                img_names = [_[0] for _ in img_names]
                self.scheduler.step(len(self.train_dataloader)*epoch + i)
                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.model(imgs)
                loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)

                if self.fp_16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # Accumulate gradient for x batches before optimizing
                if i % self.accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_ciou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)
                # len(self.train_dataloader) / (cfg.TRAIN["BATCH_SIZE"]) * epoch + iter
                # Print batch results
                if i % 10 == 0:

                    logger.info("  === Epoch:[{:3}/{}],step:[{:3}/{}],img_size:[{}],total_loss:{:.4f}|loss_ciou:{:.4f}|loss_conf:{:.4f}|loss_cls:{:.4f}|lr:{:.4f}".format(
                        epoch, self.epochs, i, len(self.train_dataloader) - 1, self.train_dataset.img_size,mloss[3], mloss[0], mloss[1],mloss[2],self.optimizer.param_groups[0]['lr']
                    ))
                    if writer:
                        writer.add_scalar('loss_ciou', mloss[0],
                                        len(self.train_dataloader) * epoch + i)
                        writer.add_scalar('loss_conf', mloss[1],
                                        len(self.train_dataloader) * epoch + i)
                        writer.add_scalar('loss_cls', mloss[2],
                                        len(self.train_dataloader) * epoch + i)
                        writer.add_scalar('train_loss', mloss[3],
                                        len(self.train_dataloader) * epoch + i)
                        writer.add_scalar('train_lr', self.optimizer.param_groups[0]["lr"],
                                        len(self.train_dataloader) * epoch + i)
                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1) % 10 == 0:
                    self.train_dataset.img_size = random.choice(range(10, 20)) * 32

            if epoch % 1==0: #tag:Val
                if cfg.TRAIN["DATA_TYPE"] == 'VOC' or cfg.TRAIN["DATA_TYPE"] == 'ABUS':
                    area_small, area_big, plt = self.evaluate()
                    logger.info("===== Validate =====".format(epoch, self.epochs))
                    if writer:
                        writer.add_scalar('AUC_10mm', area_small, epoch)
                        writer.add_scalar('AUC_15mm', area_big, epoch)
                    self.__save_model_weights(epoch, area_big)
                    logger.info('save weights done')
                    logger.info("  ===test AUC:{:.3f}".format(area_big))

            end = time.time()
            logger.info("  ===cost time:{:.4f}s".format(end - start))
        logger.info("=====Training Finished.   best_test_mAP:{:.3f}%====".format(self.best_mAP))

    def evaluate(self):
        logger = self.logger
        logger.info("Evaluate start,img size is: {},batchsize is: {:d},work number is {:d}".format(cfg.VAL["TEST_IMG_SIZE"], cfg.VAL["BATCH_SIZE"], cfg.VAL["NUMBER_WORKERS"]))
        logger.info("Test datasets number is : {}".format(len(self.test_dataloader)))

        if self.fp_16: self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)
        logger.info("        =======  start  evaluate   ======     ")
        start = time.time()
        self.model.eval()
        mloss = torch.zeros(4)
        pred_result_path=os.path.join(self.checkpoint_save_dir, 'evaluate')
        self.evaluator = Evaluator(self.model, showatt=False, pred_result_path=pred_result_path, box_top_k=256)
        self.evaluator.clear_predict_file()
        TOP_K = 50
        with torch.no_grad():
            start_time=time.time()
            if 1:
                fold_list_root = '/home/lab402/User/eason_thesis/program_update_v1/5_fold_list/'
                # EASON code
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                fold_num = self.crx_fold_num
                val_path = fold_list_root + 'five_fold_val_'+str(fold_num)+'_separate.txt'
                val_path = fold_list_root + 'five_fold_train_'+str(fold_num)+'_separate.txt'
                test_path = fold_list_root + 'five_fold_test_'+str(fold_num)+'.txt'
                val_set = open(val_path).readlines()
                val_set = val_set[:100]
                for line in val_set:
                    line = line.split(',', 4)

                    #true_box = np.array([np.array(list(map(int, box.split(','))))
                    #                    for box in boxes])
                    img_vol = np.load(line[0])
                    img_vol = torch.from_numpy(img_vol)
                    img_vol = torch.transpose(img_vol, 0, 2).contiguous() # from xyz to zyx

                    img = img_vol.unsqueeze(dim=0).cuda().float() / 255.0
                    img_name = line[0].replace('/home/lab402/User/eason_thesis/ABUS_data/', '').replace('/','_')
                    img = img.to(self.device)


                    #for img, img_name in zip(imgs, img_names):
                    bboxes_prd = self.evaluator.get_bbox(img, multi_test=False, flip_test=False)
                    #if len(bboxes_prd) > 0:
                    #    bboxes_prd[:, :6] = (bboxes_prd[:, :6] / img.size(1)) * cfg.VAL['TEST_IMG_BBOX_ORIGINAL_SIZE'][0]
                    self.evaluator.store_bbox(img_name, bboxes_prd)

            if 0:
                for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_names)  in tqdm(enumerate(self.test_dataloader)):
                    imgs = imgs[0]
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = \
                        label_sbbox[0], label_mbbox[0], label_lbbox[0], sbboxes[0], mbboxes[0], lbboxes[0]
                    img_names = [_[0] for _ in img_names]
                    imgs = imgs.to(self.device)
                    for img, img_name in zip(imgs, img_names):
                        bboxes_prd = self.evaluator.get_bbox(img, multi_test=False, flip_test=False)
                        if len(bboxes_prd) > 0:
                            bboxes_prd[:, :6] = (bboxes_prd[:, :6] / img.size(1)) * cfg.VAL['TEST_IMG_BBOX_ORIGINAL_SIZE'][0]
                        self.evaluator.store_bbox(img_name, bboxes_prd)

            print("Average time cost: {:.2f} sec.".format((time.time() - start_time)/len(self.test_dataloader)))
            npy_dir = pred_result_path
            #npy_format = npy_dir + '/{}_0.npy'
            npy_format = npy_dir + '/{}'
            area_small, area_big, plt = calculate_FROC(self.data_root, npy_dir, npy_format, size_threshold=20, th_step=0.01)
            plt.savefig(os.path.join(self.checkpoint_save_dir, 'froc_test.png'))

        end = time.time()
        logger.info("  ===cost time:{:.4f}s".format(end - start))
        return area_small, area_big, plt

    def evaluate_and_logTB(self):
        writer = self.writer
        logger = self.logger