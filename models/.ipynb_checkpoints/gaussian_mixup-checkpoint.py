import torch
import torch.nn as nn
import torch.nn.functional as F
from deeplearning.models.HFC_filter import HFCFilter


class GaussianMixUp(nn.Module):
    def __init__(self, width_list=tuple(range(5, 50, 5)), sigma_list=tuple(range(2, 22, 2)),
                 mixup_size=50, sub_low_ratio=1, sub_mask=True, is_clamp=True):
        super(GaussianMixUp, self).__init__()

        # 定义可学习的参数 channel_prompt
        self.channel_prompt = nn.Parameter(torch.randn(90))

        # 创建 HFC 滤波器
        self.hfc_list = nn.ModuleList()
        for width in width_list:
            for sigma in sigma_list:
                self.hfc_list.append(
                    HFCFilter(width, sigma, sub_low_ratio=sub_low_ratio, sub_mask=sub_mask, is_clamp=is_clamp))

        # 定义 lambda_params 为可训练的参数, 初始化为标准正态分布，形状是 (num_images, num_images)
        self.lambda_param = nn.Parameter(torch.randn(1) * 0.1)

    def forward(self, image):
        *_, w, h = image.shape

        lambda_value = torch.sigmoid(self.lambda_param)

        # 对图像应用 HFC 滤波器并获取结果
        filtered_images = []
        for hfc_filter in self.hfc_list:
            result = hfc_filter(image)
            filtered_images.append(result)

        # 获取前 70 个最大值的索引及其对应的图像
        top_values, top_indices = torch.topk(self.channel_prompt.view(-1), 64)
        top_images = [filtered_images[i] for i in top_indices]  # 根据 top_indices 获取图像

        # 按顺序两两混合图像
        mixed_images = []
        for i in range(0, len(top_images), 2):  # 步长为 2，确保两两混合
            if i + 1 < len(top_images):  # 确保不会越界
                img_1 = top_images[i]
                img_2 = top_images[i + 1]

                # 进行 MixUp 操作，lambda_value 广播到每对图像
                mixed_image = lambda_value * img_1 + (1 - lambda_value) * img_2
                mixed_images.append(mixed_image)

        return mixed_images
