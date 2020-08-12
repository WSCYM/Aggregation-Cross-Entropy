# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from models.seq_module import ACE
from torch.autograd import Variable
from models.solver import seq_solver
from utils.basic import timeSince
from torch.utils.data import DataLoader
from utils.data_loader import ImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../log/snapshot/model-{:0>2d}.pkl')
parser.add_argument('--total_epoch', type=int, default=50, help='total epoch number')
parser.add_argument('--train_path', type=str, default='../data/train.txt')
parser.add_argument('--test_path', type=str, default='../data/test.txt')
parser.add_argument('--train_batch_size', type=int, default=50, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=50, help='testing batch size')
parser.add_argument('--last_epoch', type=int, default=0, help='last epoch')
parser.add_argument('--class_num', type=int, default=26, help='class number')
parser.add_argument('--dict', type=str, default='_abcdefghijklmnopqrstuvwxyz')
opt = parser.parse_args()
print(opt)

import torchvision.models as models

class ResnetEncoderDecoder(nn.Module):
    def __init__(self, loss_layer):
        super(ResnetEncoderDecoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        #二维卷积,输入通道数1,输出通道数64
        self.conv  = nn.Conv2d(1,   64,   kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        #卷积层之后会添加BatchNorm2d进行数据归一化处理,使数据Relu之前不会因为数据过大导致网络性能不稳定
        self.bn = nn.BatchNorm2d(64)
        self.cnn = nn.Sequential(*list(resnet.children())[4:-2])
        self.out = nn.Linear(512, opt.class_num+1)
        self.loss_layer = loss_layer(opt.dict)

    def forward(self, input, labels):
        # bs*1*100*100
        input = F.relu(self.bn(self.conv(input)), True)
        # bs*64*100*100
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        # 50* 64 * 50 * 50
        input = self.cnn(input)
        # 50 * 512 * 7 * 7
        input = input.permute(0,2,3,1)
        #50 * 7 * 7 * 512
        input = F.softmax(self.out(input),dim=-1)
        # 50 * 7 * 7 * 27
        labels = labels.cuda()

        return  self.loss_layer(input,labels)


if __name__ == "__main__":

    model = ResnetEncoderDecoder(ACE).cuda()
    print(model)

    optimizer = optim.Adadelta(model.parameters())

    if opt.last_epoch != 0:
        check_point = torch.load(opt.model_path.format(opt.last_epoch))
        model.load_state_dict(check_point['state_dict'])
        optimizer.load_state_dict(check_point['optimizer'])
        # 动态调整learning rate,epoch为opt.total_epoch时，学习率变成learning rate*gamma
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [opt.total_epoch], gamma = 0.1, last_epoch = opt.last_epoch)    
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [opt.total_epoch], gamma = 0.1)    


    train_set = ImageDataset(file_name = opt.train_path, length = 5000, class_num = opt.class_num)
    lmdb_train = DataLoader(train_set, batch_size=opt.train_batch_size, shuffle=True, num_workers=0) 

    test_set = ImageDataset(file_name = opt.test_path, length = 1000, class_num = opt.class_num)
    lmdb_test = DataLoader(test_set, batch_size=opt.test_batch_size, shuffle=False, num_workers=0) 


    the_solver = seq_solver(model = model,
                        lmdb = [lmdb_train, lmdb_test],
                        optimizer = optimizer, 
                        scheduler = scheduler,
                        total_epoch = opt.total_epoch,
                        model_path = opt.model_path,
                        last_epoch = opt.last_epoch)

    the_solver.forward()

