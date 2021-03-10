import os
import torch as t
import torch.nn as nn
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from utils.average import AverageVal
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

logger = AverageVal()
faster_rcnn = FasterRCNNVGG16()

trainer = FasterRCNNTrainer(faster_rcnn, logger).cuda()
trainer.load("checkpoints/fasterrcnn_03090654_0.6984418117245029")

for i in range(1, 192):
    img_path = "movie/input/{}.jpg".format(i)
    img = read_image(img_path)
    img = t.from_numpy(img)[None]
    opt.caffe_pretrain = False  # this model was trained from caffe-pretrained model
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
    output_path = "movie/output/{}.jpg".format(i)
    vis_bbox(
        at.tonumpy(img[0]),
        at.tonumpy(_bboxes[0]),
        at.tonumpy(_labels[0]).reshape(-1),
        at.tonumpy(_scores[0]).reshape(-1),
        figsize=(15, 8.5),
    )
    plt.axis("off")
    plt.savefig(output_path)
