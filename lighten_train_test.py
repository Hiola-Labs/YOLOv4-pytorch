import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import utils.datasets as data

from lighten_model import lightenYOLOv4
import config.yolov4_config as cfg
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint



from custom_model_checkpoint import CustomModelCheckpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/mobilenetv3.pth', help='weight file path')#weight/darknet53_448.weights
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=-1, help='whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--fp_16', type=bool, default=False, help='whither to use fp16 precision')
    parser.add_argument('--exp_name', type=str, default='debug', help='log experiment name')
    opt = parser.parse_args()



    train_dataset = data.Build_Dataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
    test_dataset = data.Build_Dataset(anno_file_type="test", img_size=cfg.VAL["TEST_IMG_SIZE"])

    train_dataloader = DataLoader(train_dataset,
                                        batch_size=cfg.TRAIN["BATCH_SIZE"],
                                        num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                        shuffle=True, pin_memory=True
                                        )
    test_dataloader = DataLoader(test_dataset,
                                        batch_size=cfg.TRAIN["BATCH_SIZE"],
                                        num_workers=cfg.VAL["NUMBER_WORKERS"],
                                        shuffle=False, pin_memory=True
                                        )

    if 0:
        train_dataloader = DataLoader(train_dataset,
                                            batch_size=cfg.TRAIN["BATCH_SIZE"],
                                            num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                            shuffle=False, pin_memory=True,
                                            sampler=[0, 1, 2, 3]
                                            )
        test_dataloader = DataLoader(test_dataset,
                                            batch_size=1, #cfg.TRAIN["BATCH_SIZE"],
                                            num_workers=cfg.VAL["NUMBER_WORKERS"],
                                            shuffle=False, pin_memory=True,
                                            sampler=[0, 1, 2, 3]
                                            )


    model = lightenYOLOv4(
        weight_path=opt.weight_path,
        resume=opt.resume
    )

    checkpoint_callback = CustomModelCheckpoint(monitor='val/mAP_epoch',
        filepath='checkpoint/' + opt.exp_name + '-{epoch:02d}',
        verbose=True,
        prefix=opt.exp_name,
        mode = 'max',
        save_last=True)
    #
    tb_logger = pl_loggers.TensorBoardLogger('log/', name=opt.exp_name)
    trainer = pl.Trainer(gpus=-1, logger=tb_logger, checkpoint_callback=checkpoint_callback)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)