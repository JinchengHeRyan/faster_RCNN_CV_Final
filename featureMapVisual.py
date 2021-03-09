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
import matplotlib.pyplot as plt

img = read_image("misc/demo.jpg")
img = t.from_numpy(img)[None]

logger = AverageVal()
faster_rcnn = FasterRCNNVGG16()

trainer = FasterRCNNTrainer(faster_rcnn, logger).cuda()

trainer.load("model/chainer_best_model_converted_to_pytorch_0.7053.pth")
opt.caffe_pretrain = True  # this model was trained from caffe-pretrained model

print(img.shape)

h = trainer.faster_rcnn.getFeatureMap(img)
print(h)
