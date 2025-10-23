# -*- coding: UTF-8 -*-
"""
@Function:
@File: HFC_filter.py
@Date: 2021/7/26 15:02
@Author: Hever
"""
import os

from torch import nn
from torch.nn import functional as F
import torch
import cv2



# 高频部分通常代表图像中的细节、边缘和纹理，而低频部分代表的是图像中的平滑区域或均匀的背景。
class HFCFilter(nn.Module):

    def __init__(self, filter_width=23, nsig=20, ratio=4, sub_low_ratio=1, sub_mask=False, is_clamp=True):
        super(HFCFilter, self).__init__()
        self.gaussian_filter = Gaussian_kernel(filter_width, nsig=nsig)  # 用于生成高斯核
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)  # 用于对图像进行池化（降采样）
        # 将图像的分辨率降低一半，同时保留图像中的一些关键信息。

        self.max = 1.0  # 设置高频滤波的最大值
        self.min = -1.0  # 设置高频滤波的最小值
        self.ratio = ratio  # 用于调整高频部分的强度
        self.sub_low_ratio = sub_low_ratio  # 用于调整低频部分的强度
        self.sub_mask = sub_mask  # 是否应用掩膜
        self.is_clamp = is_clamp  # 是否限制结果范围

    def median_padding(self, x):
    #对输入张量 x 应用中值填充。
        mask = torch.ones_like(x)
        m_list = []
        batch_size = x.shape[0]
        for i in range(x.shape[1]):
            m_list.append(x[:, i].view([batch_size, -1]).median(dim=1).values.view(batch_size, -1) + 0.2)
        median_tensor = torch.cat(m_list, dim=1)
        median_tensor = median_tensor.unsqueeze(2).unsqueeze(2)
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x


    def forward(self, x):
        x = self.median_padding(x)  # 对输入图像进行中值填充
        mask = torch.ones_like(x)
        gaussian_output = self.gaussian_filter(x)  # 使用高斯滤波器进行处理
        gaussian_output = torch.nn.functional.interpolate(gaussian_output, size=(256,256), mode='bilinear', align_corners=False)

        res = self.ratio * (x - self.sub_low_ratio * gaussian_output)  # 提取高频部分
        if self.is_clamp:
            res = torch.clamp(res, self.min, self.max)  # 限制输出的范围
        if self.sub_mask:
            res = (res + 1) * mask - 1  # 应用掩膜
        return res  # 返回处理后的图像


#生成一个二维高斯核（Gaussian kernel），用于滤波操作，并返回该核。
def get_kernel(kernel_len=16, nsig=10):  # nsig 标准差 ，kernlen=16核尺寸
    GaussianKernel = cv2.getGaussianKernel(kernel_len, nsig) \
                     * cv2.getGaussianKernel(kernel_len, nsig).T
    return GaussianKernel

#该类用于创建一个高斯卷积核，并应用于输入的图像数据。
#高斯卷积核是一种常用于图像处理的滤波器，用于进行模糊、去噪等操作。
class Gaussian_kernel(nn.Module):
    def __init__(self,
                 # device,
                 kernel_len, nsig=20):
        #kernel_len:卷积核的大小（长度） nsig:高斯分布的标准差，控制高斯核的模糊程度。标准差越大，滤波器的作用越广，模糊程度越强。
        super(Gaussian_kernel, self).__init__()
        #初始化 nn.Module 的一些基本属性和功能（如模型参数管理、GPU支持等）。
        self.kernel_len = kernel_len
        kernel = get_kernel(kernel_len=kernel_len, nsig=nsig)  # 获得高斯卷积核
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展两个维度
        # self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        #高斯核是一个固定的滤波器。并将它赋值给 self.weight。

        self.padding = torch.nn.ReplicationPad2d(int(self.kernel_len / 2))
        #将输入图像边缘的值进行复制来填充图像。

    def forward(self, x):  # x1是用来计算attention的，x2是用来计算的Cs
        x = self.padding(x)

        # 对三个channel分别做卷积
        res = []
        for i in range(x.shape[1]):
            res.append(F.conv2d(x[:, i:i + 1], self.weight))
        x_output = torch.cat(res, dim=1)
        # 将所有单通道卷积的输出合并为一个多通道的输出。
        # 假设有 C 个通道，最终拼接后的结果会是一个形状为 [batch_size, C, height, width] 的张量。
        return x_output


if __name__ == '__main__':
    from torchvision import transforms
    from PIL import Image, ImageDraw
    import random
    from tqdm import tqdm

    img = Image.open('/home/lihaojin/data/retinal_vessel/drive/train/0/image.png').convert('RGB')
    label = Image.open('/home/lihaojin/data/retinal_vessel/drive/train/0/label.png').convert('L')
    mask = Image.new("L", (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(((0, 0), (511, 511)), fill=255)
    img = transforms.ToTensor()(img).unsqueeze(0)
    mask = transforms.ToTensor()(mask).unsqueeze(0)
    label = transforms.ToTensor()(label).unsqueeze(0)
    #将图像数据从 PIL 图像格式或 NumPy 数组格式转换为 PyTorch 的 Tensor 格式。
    #Tensor 是一个多维的数据结构，类似于矩阵或多维数组。
    #unsqueeze(0) 用来创建批次维度，将 Tensor 从 [C, H, W] 扩展到 [1, C, H, W]，其中 1 表示批次大小为 1。

    # 以下是生成HFC效果大图（高频滤波器）
    # # mid = (27, 9)
    # mid = (25, 11)  # 中心位置的坐标（用于控制滤波器的范围）
    # step = (4, 1)  # 步长，控制滤波器范围扩展的步长
    # amp = (6, 10)  # 振幅，用于调整滤波器的强度

    # width_list = list(range(mid[0] - amp[0] * step[0], mid[0] + amp[0] * step[0] + 1, step[0]))
    # sigma_list = list(range(mid[1] - amp[1] * step[1], mid[1] + amp[1] * step[1] + 1, step[1]))
    #根据 mid、amp 和 step 生成一系列滤波器的宽度和标准差值。

    # print(width_list)
    # print(sigma_list)
    # hfc_list_list = [[HFCFilter(width, sigma, sub_low_ratio=1, sub_mask=True, is_clamp=True) for sigma in sigma_list] for width in width_list]
    #创建高频滤波器列表，包含了不同宽度 (width) 和标准差 (sigma) 的高频滤波器。

    # result_list = []
    #
    # for hfc_list in hfc_list_list:
    #     temp_list = []
    #     for hfc_filter in hfc_list:
    #         filtered = ((hfc_filter(img, mask) + 1) / 2).squeeze().numpy().transpose((1, 2, 0))[:, :, ::-1] * 255
    #         temp_list.append(filtered)
    #     result_list.append(cv2.hconcat(temp_list))
    # final_result = cv2.vconcat(result_list)
    #
    # cv2.imwrite('/home/lihaojin/final.png', final_result)

    # 以下是生成cutmix效果图
    #通过不同的高频滤波器对图像进行处理，生成不同高频特征图，然后将这些特征图进行随机切割和混合，最终生成混合图像。
    target_dir = '/home/lihaojin/visual'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    cv2.imwrite(os.path.join(target_dir, 'image.png'), (img.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'mask.png'), mask.squeeze().numpy() * 255)
    cv2.imwrite(os.path.join(target_dir, 'label.png'), label.squeeze().numpy() * 255)

    #应用多个高频滤波器。
    h1 = HFCFilter(3, 3, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h2 = HFCFilter(5, 6, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h3 = HFCFilter(11, 9, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h4 = HFCFilter(21, 20, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h5 = HFCFilter(41, 30, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h6 = HFCFilter(81, 40, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h7 = HFCFilter(101, 80, sub_low_ratio=1, sub_mask=True, is_clamp=True)

    #应用高频滤波器，生成高频图像。
    high1 = (h1(img, mask) + 1) / 2
    high2 = (h2(img, mask) + 1) / 2
    high3 = (h3(img, mask) + 1) / 2
    high4 = (h4(img, mask) + 1) / 2
    high5 = (h5(img, mask) + 1) / 2
    high6 = (h6(img, mask) + 1) / 2
    high7 = (h7(img, mask) + 1) / 2

    #保存高频图像。
    cv2.imwrite(os.path.join(target_dir, 'high1.png'), (high1.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high2.png'), (high2.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high3.png'), (high3.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high4.png'), (high4.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high5.png'), (high5.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high6.png'), (high6.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high7.png'), (high6.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])

    len_list = [50, 100, 150, 200, 250, 300, 350, 400]
    #len_list 定义了随机选择的切割区域的大小（宽度和高度）。

    high_list = [high1, high2, high3, high4, high5, high6, high7]
    # high_list 是一个包含了 7 张高频图像的列表，将用于后续的切割和混合操作。

    random.seed(0)

    for i in tqdm(range(7)):
        for j in range(7):
            for k in range(5):
                width = random.choice(len_list)
                height = random.choice(len_list)
                #确定了切割区域的左上角坐标。
                left = random.randint(0, 512 - width)
                up = random.randint(0, 512 - height)
                # print(left, up, width, height)
                result = high_list[i].clone()

                result[:, :, left:left + width, up:up + height] = high_list[j][:, :, left:left + width, up:up + height]

                cv2.imwrite(os.path.join(target_dir, 'mix_' + str(i + 1) + '_' + str(j + 1) + '_' + str(k) + '.png'),
                            (result.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
