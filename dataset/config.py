# -*- coding: utf-8 -*-
"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just choose in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,
    ResNet18,ResNet34,ResNet50,ResNet101,ResNet152,mnist_net,TSL16
"""

import sys
class DefaultConfig():

    train_data_path = 'dataset/train'
    test_data_path = 'dataset/test'

    # if model_name == 'InceptionV3':
    #     normal_size = 75#minSize
    # elif model_name == 'Xception':
    #     normal_size = 71#minSize
    # else:
    #     normal_size = 224
    classNumber = 5
    normal_size = 224
    channles = 3 # or 3 or 1
    lr = 0.0001     #可以调成0.01或0.1试试

    lr_reduce_patience = 10  # 需要降低学习率的训练步长
    early_stop_patience = 100  # 提前终止训练的步长

    data_augmentation = False
    monitor = 'val_loss'
    cut = False
    rat = 0.1 #if cut,img[slice(h*self.rat,h-h*self.rat),slice(w*self.rat,w-w*self.rat)]
    feature_extract = True      #此处参数为True表示只更新重新形成的图层参数，为False表示微调整个模型

config = DefaultConfig()
