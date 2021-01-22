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
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import config.yolov4_config as cfg
from utils import cosine_lr_scheduler
from utils.log import Logger

from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator

from databuilder.abus import ABUSDetectionDataset
from databuilder.yolo4dataset import YOLO4_3DDataset
from tqdm import tqdm
from trainer import Trainer


if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default=None, help='weight file path')#weight/darknet53_448.weights
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=-1, help='whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    parser.add_argument('--accumulate', type=int, default=1, help='batches to accumulate before optimizing')
    parser.add_argument('--fp_16', type=bool, default=False, help='whither to use fp16 precision')
    parser.add_argument('--exp_name', type=str, default='debug', help='log experiment name')
    parser.add_argument('--crx_valid', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--stem_channels', type=int, default=32)
    parser.add_argument('--csp_first_channels', type=int, default=64)

    opt = parser.parse_known_args()[0]
    writer = SummaryWriter(log_dir=opt.log_path + '/' + opt.exp_name)
    logger = Logger(log_file_name=opt.log_path + '/' + opt.exp_name + '/log.txt', log_level=logging.DEBUG, logger_name='YOLOv4').get_log()
    checkpoint_save_dir = 'checkpoint/' + opt.exp_name
    if not os.path.exists(checkpoint_save_dir):
        os.mkdir(checkpoint_save_dir)

    weight_path = opt.weight_path
    resume = opt.resume

    #resume = True
    #weight_path = 'checkpoint/YOLO_ABUS_d30_Stem16_lr25/backup_epoch140.pt'
    trainer = Trainer(testing_mode=0,
            weight_path=weight_path,
            checkpoint_save_dir=checkpoint_save_dir,
            resume=resume,
            gpu_id=opt.gpu_id,
            accumulate=opt.accumulate,
            fp_16=opt.fp_16,
            writer=writer,
            logger=logger,
            crx_fold_num=opt.crx_valid,
            train_batch_size=opt.batch_size,
            epochs=opt.epochs
            )

    trainer.train()