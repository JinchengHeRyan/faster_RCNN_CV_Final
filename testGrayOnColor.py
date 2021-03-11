from utils.config import opt
from train import eval
from data.dataset import Dataset, TestDataset
from torch.utils import data as data_
from utils.average import AverageVal
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

opt.voc_data_dir = "/mingback/students/jincheng/data/VOC2007/VOCdevkit/VOC2007"
dataset = Dataset(opt)
print("load data")
dataloader = data_.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,  # pin_memory=True,
    num_workers=opt.num_workers,
)
testset = TestDataset(opt)
test_dataloader = data_.DataLoader(
    testset,
    batch_size=1,
    num_workers=opt.test_num_workers,
    shuffle=False,
    pin_memory=True,
)

faster_rcnn = FasterRCNNVGG16()
print("model construct completed")

logger = AverageVal()
trainer = FasterRCNNTrainer(faster_rcnn, logger).cuda()
trainer.load("checkpoints/trainedOnGray/fasterrcnn_03110128_0.6618494866421214")
eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
print(eval_result["map"])
