import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        """
        初始化 ResNet50 模型
        Args:
            num_classes (int): 分类类别数量
            pretrained (bool): 是否使用预训练权重
        """
        super(CustomResNet50, self).__init__()
        
        # 加载预训练的 ResNet50 模型
        self.model = models.resnet18(pretrained=True)
        
        # 获取特征提取层的输出特征数
        num_ftrs = self.model.fc.in_features
        
        # 修改最后的全连接层以匹配目标类别数
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入图像张量
        Returns:
            输出预测结果
        """
        return self.model(x)

# 使用示例
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型实例（假设有10个类别）
    model = CustomResNet50(num_classes=10)
    model = model.to(device)

    # 打印模型结构
    print(model)

    # 测试模型前向传播
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"输出张量形状: {output.shape}")

    # 计算模型参数总量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params:,}")

