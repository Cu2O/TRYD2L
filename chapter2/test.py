import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from model import CustomResNet50
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载测试数据
train_csv_path = "./leaves/train.csv"  # 替换为你的 train.csv 文件路径
train_data = pd.read_csv(train_csv_path)
test_csv_path = "./leaves/test.csv"  # 替换为你的 test.csv 文件路径
test_data = pd.read_csv(test_csv_path)

# 加载模型
num_classes = 176  # 替换为你的类别数量
model = CustomResNet50(num_classes=num_classes)
checkpoint_path = "best_model.pth"  # 替换为你的权重文件路径
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# 类别映射
class_to_idx = {cls: idx for idx, cls in enumerate(sorted(train_data['label'].unique()))}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 预测函数
def predict(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)  # 获取预测的类别索引
    return predicted_idx.item()

# 遍历测试数据并进行预测
predictions = []
for _, row in test_data.iterrows():
    img_path = row['image']  # 替换为 test.csv 中图片路径的列名
    image_path= os.path.join("./leaves", img_path)
    predicted_idx = predict(image_path, model, transform, device)
    predicted_class = idx_to_class[predicted_idx]
    predictions.append(predicted_class)

# 输出预测结果
test_data['label'] = predictions
print(test_data[['image', 'label']])

# 保存预测结果到 CSV 文件
output_csv_path = "./leaves/test_predictions.csv"  # 替换为你的输出路径
test_data.to_csv(output_csv_path, index=False)
print(f"预测结果已保存到 {output_csv_path}")