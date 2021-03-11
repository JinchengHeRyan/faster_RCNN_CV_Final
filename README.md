# Final Project of computer vision course about the topic of faster-rcnn 

## 1. Introduction

We have changed the source code to run without using visdom to visual the result, at least it could not run on our computers. Instead, we add logger to record all the loss, accuracy information we need for plotting. For the code we submit, we do not include the weights of the model because each of them are nearly 500M which are too big for uploading. 

The paper we read is [this paper](https://arxiv.org/abs/1506.01497)

## 2. Performance

All the records of the performace are in the directory `logs/`, there are three sub-folders under this directory, they are `logs/faster_rcnn`, `logs/faster_rcnn_train_onGray`, `logs/faster_rcnn_vggPretrained`. Those directories store the logs of the model trained on original images without vgg pretrained, and the model trained on gray scale images without vgg pretrained, and the model trained on original images with vgg pretrained model respectively. We obtain the mAP result similar to the origin paper, and the result is as the following. 

| Implementation | mAP|
| :---------: | :------: |
| origin paper | 0.699|
| Our model trained on original data without vgg Pretrained | 0.6984|
| Our model trained on original data with vgg Pretrained | 0.6962 |
| Our model trained on gray data without vgg Pretrained | 0.6681 |

We also see the speed, they are all basically 20 minutes for one epoch trained on one NVIDIA 1080TI. 

## 3. Run the code

The file `requirements.txt` is in the project folder, at least it can work on our group's computers, this file is not as same as the original author's file, we found out after installing his requirements some code still cannot work, so we change the source code and have our own version of requirements. 

```shell
CUDA_VISIBLE_DEVICES=0 python train.py train --plot-every=100
```

This line of bash shell can let you easily run the code of training the model. 

In the notebook `demo.ipynb`, it includes the code of showing how to predict and show the image. However it needs to load the already existed weight of the model, but they are too big to upload. This notebook can also show the feature map. 
