import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations (train.csv).
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        # 获取唯一的标签类别
        self.classes = sorted(self.data_frame.iloc[:, 1].unique())
        self.num_classes = len(self.classes)
        # 创建标签到索引的映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.data_frame.iloc[idx, 1]
        
        # 将标签转换为索引
        label_idx = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label_idx

# 示例用法
if __name__ == "__main__":
    # 假设 train.csv 和图片都在当前目录下
    csv_file = "./leaves/train.csv"
    root_dir = "./leaves"  # 图片所在目录
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片大小
        transforms.ToTensor(),         # 转换为张量
    ])

    full_dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 计算训练集和测试集的大小
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    # 使用random_split划分数据集
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 打印数据集信息
    print(f"总数据集大小: {total_size}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别数量: {full_dataset.num_classes}")
    
    # 测试读取训练数据
    for images, labels in train_loader:
        print("训练集批次形状:", images.shape)
        print("训练集标签形状:", labels.shape)
        print("训练集标签示例:")
        print(labels[:5])  # 显示前5个标签
        break
        
    # 测试读取测试数据
    for images, labels in test_loader:
        print("测试集批次形状:", images.shape)
        print("测试集标签形状:", labels.shape)
        print("测试集标签示例:")
        print(labels[:5])  # 显示前5个标签
        break