# coding=utf-8
# project
DATA_PATH = "/home/lab402/p08922003/YOLOv4-pytorch/dataset_link"
PROJECT_PATH = "/home/lab402/p08922003/YOLOv4-pytorch/data"
DETECTION_PATH = "/home/lab402/p08922003/YOLOv4-pytorch"

MODEL_TYPE = {"TYPE": 'YOLOv4'}  #YOLO type:YOLOv4, Mobilenet-YOLOv4 or Mobilenetv3-YOLOv4
MODEL_INPUT_CHANNEL = 1
CONV_TYPE = {"TYPE": 'DO_CONV'}  #conv type:DO_CONV or GENERAL

ATTENTION = {"TYPE": 'NONE'}  #attention type:SEnet、CBAM or NONE
CONV_LAYER_FACTOR = [4, 4] #4, 8 for 4_8_128  4, 4 for 4_4_64
# train
TRAIN = {
         "DATA_TYPE": 'ABUS',  #DATA_TYPE: VOC ,COCO or Customer
         "TRAIN_IMG_SIZE": (320, 160, 320), #(128, 128, 128), #(96, 96, 96), #(640, 160, 640),#(256, 64, 256),
         "AUGMENT": True,
         #for 640
         "BATCH_SIZE": 1,
         #for 96
         #"BATCH_SIZE": 4,
         "MULTI_SCALE_TRAIN": False,
         "IOU_THRESHOLD_LOSS": 0.5,
         #for 640
         "YOLO_EPOCHS": 15,
         #for 96
         #"YOLO_EPOCHS": 100,
         "Mobilenet_YOLO_EPOCHS": 120,
         "NUMBER_WORKERS": 2,
         "MOMENTUM": 0.9,
         "WEIGHT_DECAY": 0.0005,
         "LR_INIT": 5e-5, #1e-4,
         "LR_END": 4.95e-5, #1e-6,
         #for 640
         "WARMUP_EPOCHS": 4 #40  # or None
         #for 96
         #"WARMUP_EPOCHS": 10 #40  # or None
         }


# val
VAL = {
        #"TEST_IMG_SIZE": 416,
        "TEST_IMG_SIZE": (640, 160, 640), #(640, 160, 640),#(256, 64, 256), #
        "BATCH_SIZE": 1,
        "NUMBER_WORKERS": 2,
        "CONF_THRESH": 0.01, #0.005,
        "NMS_THRESH": 0.45,
        "MULTI_SCALE_VAL": True,
        "FLIP_VAL": True,
        "Visual": True,
        "TEST_IMG_BBOX_ORIGINAL_SIZE": (640, 160, 640)
        }

Customer_DATA = {"NUM": 1, #your dataset number
                 "CLASSES":['aeroplane'],# your dataset class
        }
ABUS_DATA = {"NUM": 2, #your dataset number
                 "CLASSES":['background', 'tumor'],# your dataset class
        }

VOC_DATA = {"NUM": 20, "CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor'],
        }

COCO_DATA = {"NUM":80,"CLASSES":['person',
'bicycle',
'car',
'motorcycle',
'airplane',
'bus',
'train',
'truck',
'boat',
'traffic light',
'fire hydrant',
'stop sign',
'parking meter',
'bench',
'bird',
'cat',
'dog',
'horse',
'sheep',
'cow',
'elephant',
'bear',
'zebra',
'giraffe',
'backpack',
'umbrella',
'handbag',
'tie',
'suitcase',
'frisbee',
'skis',
'snowboard',
'sports ball',
'kite',
'baseball bat',
'baseball glove',
'skateboard',
'surfboard',
'tennis racket',
'bottle',
'wine glass',
'cup',
'fork',
'knife',
'spoon',
'bowl',
'banana',
'apple',
'sandwich',
'orange',
'broccoli',
'carrot',
'hot dog',
'pizza',
'donut',
'cake',
'chair',
'couch',
'potted plant',
'bed',
'dining table',
'toilet',
'tv',
'laptop',
'mouse',
'remote',
'keyboard',
'cell phone',
'microwave',
'oven',
'toaster',
'sink',
'refrigerator',
'book',
'clock',
'vase',
'scissors',
'teddy bear',
'hair drier',
'toothbrush',]}


# model
MODEL = {"ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj(12,16),(19,36),(40,28)
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj(36,75),(76,55),(72,146)
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]],  # Anchors for big obj(142,110),(192,243),(459,401)
         "ANCHORS3D":[[[ 1.4375  ,  1.4375  ,  1.4375  ],
                        [ 2.875   ,  2.875   ,  2.875   ],
                        [ 3.5     ,  3.5     ,  3.5     ]],
                        [[ 2.84375 ,  2.84375 ,  2.84375 ],
                        [ 3.34375 ,  3.34375 ,  3.34375 ],
                        [ 5.5625  ,  5.5625  ,  5.5625  ]],

                        [[ 3.21875 ,  3.21875 ,  3.21875 ],
                        [ 5.53125 ,  5.53125 ,  5.53125 ],
                        [10.921875, 10.921875, 10.921875]]], #shape (STRIDES, Anchors_PER_SCALE, 3 element for 3D ZYX Anchor length)
         "STRIDES":[8, 16, 32],
         "ANCHORS_PER_SCLAE":3
         }