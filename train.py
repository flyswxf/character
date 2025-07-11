import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import os

# 自定义数据集类
class FourCornerDataset(Dataset):
    def __init__(self, csv_file, img_dir, font_path, transform=None):
        self.data = pd.read_csv(csv_file)  # 假设CSV包含"character"和"four_corner"列
        self.img_dir = img_dir
        self.transform = transform
        self.font_path = font_path
        self.font_size = 64
        self.image_size = (64, 64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        char = self.data.iloc[idx]['character']
        four_corner = str(self.data.iloc[idx]['four_corner']).zfill(5)  # 确保5位
        img_path = os.path.join(self.img_dir, f"{char}.png")

        # 如果图像不存在，动态生成
        if not os.path.exists(img_path):
            self._generate_char_image(char, img_path)

        # 加载图像
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        if self.transform:
            image = self.transform(image)

        # 将四角号码转换为标签（每位数字为0-9）
        labels = [int(d) for d in four_corner]
        labels = torch.tensor(labels, dtype=torch.long)

        return image, labels

    def _generate_char_image(self, char, save_path):
        # 生成汉字图像
        image = Image.new('L', self.image_size, color=255)  # 白色背景
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(self.font_path, self.font_size)
        text_size = draw.textbbox((0, 0), char, font=font)
        text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]
        position = ((self.image_size[0] - text_width) // 2, (self.image_size[1] - text_height) // 2)
        draw.text(position, char, fill=0, font=font)  # 黑色文字
        image.save(save_path)

# CNN模型定义
class FourCornerCNN(nn.Module):
    def __init__(self):
        super(FourCornerCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 5 * 10)  # 5位数字，每位10个类别
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, 5, 10)  #  reshape为(批次, 5位, 10个类别)
        return x

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
])

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # (batch_size, 5, 10)
            loss = 0
            for i in range(5):  # 对每位数字计算损失
                loss += criterion(outputs[:, i, :], labels[:, i])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 预测函数
def predict_four_corner(model, char, font_path, device, transform):
    model.eval()
    # 生成单字图像
    image = Image.new('L', (64, 64), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 64)
    text_size = draw.textbbox((0, 0), char, font=font)
    text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]
    position = ((64 - text_width) // 2, (64 - text_height) // 2)
    draw.text(position, char, fill=0, font=font)
    
    # 预处理图像
    image = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(image)  # (1, 5, 10)
        _, predicted = torch.max(output, dim=2)  # 每位取最大概率
        four_corner = ''.join([str(p.item()) for p in predicted[0]])
    return four_corner

# 主程序
def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_file = "four_corner_data.csv"  # 假设数据文件
    img_dir = "char_images"
    font_path = "SimSun.ttf"  # 宋体字体文件路径
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001

    # 创建数据集和数据加载器
    dataset = FourCornerDataset(csv_file, img_dir, font_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = FourCornerCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # 保存模型
    torch.save(model.state_dict(), "four_corner_model.pth")

    # 示例：预测《汉语大字典》中一个字的四角号码
    test_char = "汉"
    predicted_code = predict_four_corner(model, test_char, font_path, device, transform)
    print(f"汉字 '{test_char}' 的预测四角号码: {predicted_code}")

if __name__ == "__main__":
    main()