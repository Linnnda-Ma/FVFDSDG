# -*- coding:utf-8 -*-
from deeplearning.models.unet import UnetBlock, SaveFeatures
from deeplearning.models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
from torch import nn
import torch
import torch.nn.functional as F
from deeplearning.models.HFC_filter import HFCFilter
from deeplearning.models.gaussian_mixup import *

class Projector(nn.Module):
    def __init__(self, output_size=1024):
        super(Projector, self).__init__()
        self.conv = nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(131072, output_size)

    def forward(self, x_in):
        x = self.conv(x_in)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x


class UNetFVFDSDG(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        first_layer = layers[0]
        other_layers = layers[1:]
        base_layers = nn.Sequential(*other_layers)
        self.width_list = tuple(range(5, 50, 5))
        self.sigma_list = tuple(range(2, 22, 2))
        self.first_layer = first_layer
        self.rn = base_layers

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [1, 3, 4, 5]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1)  # 用1x1卷积替代上采样

        # 使用嵌套循环生成所有组合
        self.hfc_list = nn.ModuleList()

        # 外层循环遍历 width_list
        for width in self.width_list:
            # 内层循环遍历 sigma_list
            for sigma in self.sigma_list:
                # 对于每个组合，创建 HFCFilter 并添加到 hfc_list 中
                self.hfc_list.append(HFCFilter(width, sigma))

        self.mixup_filter = GaussianMixUp(sub_low_ratio=1, sub_mask=True,is_clamp=True)

        # Add a convolution layer to reduce channels from 270 to 64
        self.reduce_channels = nn.Conv2d(96, 64, kernel_size=1)  # Add this line

    def forward_first_layer(self, x, tau=0.1):
        x = torch.cat(self.mixup_filter(x),dim=1)  # [batch_size, num_filters, height, width]

        # Apply the channel reduction (to make it (8, 64, 224, 224))
        x = self.reduce_channels(x)

        channel_prompt_onehot = torch.softmax(self.channel_prompt / tau, dim=0)
        # print(f"channel_prompt_onehot.shape:{channel_prompt_onehot.shape}")
        # print(f"x.shape:{x.shape}")
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)

        return f_content, f_style

    def forward(self, x, tau=0.1):
        x = torch.cat(self.mixup_filter(x),dim=1)

        # Apply the channel reduction (to make it (8, 64, 224, 224))
        x = self.reduce_channels(x)

        channel_prompt_onehot = torch.softmax(self.channel_prompt / tau, dim=0)
        # print(f"channel_prompt_onehot.shape:{channel_prompt_onehot.shape}")
        # print(f"x.shape:{x.shape}")
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)

        x = F.relu(self.rn(f_content))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        output = self.up5(x)

        return output

    def close(self):
        for sf in self.sfs: sf.remove()
