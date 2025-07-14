import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import wandb
import config

# 自定义数据集类
class FourCornerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        four_corner = str(self.data.iloc[idx]['four_corner']).zfill(5)  # 确保5位
        img_path = os.path.join(self.img_dir, f"{idx}.png")

        # 加载图像
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        if self.transform:
            image = self.transform(image)

        # 将四角号码转换为标签（每位数字为0-9）
        labels = [int(d) for d in four_corner]
        labels = torch.tensor(labels, dtype=torch.long)

        return image, labels



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
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss})

# 预测函数
def predict_four_corner(model, char, csv_file, img_dir, device, transform):
    model.eval()
    
    # 从CSV中找到汉字对应的索引
    data = pd.read_csv(csv_file)
    try:
        idx = data[data['character'] == char].index[0]
    except IndexError:
        return f"汉字 '{char}' 在数据中未找到。"

    # 加载对应的图像
    img_path = os.path.join(img_dir, f"{idx}.png")
    if not os.path.exists(img_path):
        return f"图像文件 '{img_path}' 不存在。"

    image = Image.open(img_path).convert('L')
    
    # 预处理图像
    image = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(image)  # (1, 5, 10)
        _, predicted = torch.max(output, dim=2)  # 每位取最大概率
        four_corner = ''.join([str(p.item()) for p in predicted[0]])
    return four_corner

# 测试函数
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = [0] * 5
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = 0
            for i in range(5):
                loss += criterion(outputs[:, i, :], labels[:, i])
                _, predicted = torch.max(outputs[:, i, :], 1)
                correct[i] += (predicted == labels[:, i]).sum().item()
            test_loss += loss.item()
            total += labels.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracies = [c / total for c in correct]
    avg_accuracy = sum(accuracies) / 5

    print(f'Test Loss: {avg_loss:.4f}')
    for i in range(5):
        print(f'Accuracy of digit {i+1}: {accuracies[i]:.4f}')
    print(f'Average Accuracy: {avg_accuracy:.4f}')

    wandb.log({
        "test_loss": avg_loss,
        "average_accuracy": avg_accuracy,
        "accuracy_digit_1": accuracies[0],
        "accuracy_digit_2": accuracies[1],
        "accuracy_digit_3": accuracies[2],
        "accuracy_digit_4": accuracies[3],
        "accuracy_digit_5": accuracies[4],
    })

# 主程序
def main():
    # 初始化wandb
    wandb.init(project="four-corner-cnn", config=config.__dict__)

    # 从config模块加载配置
    device = config.DEVICE
    csv_file = config.CSV_FILE
    img_dir = config.IMAGE_DIR
    batch_size = config.BATCH_SIZE
    num_epochs = config.NUM_EPOCHS
    learning_rate = config.LEARNING_RATE
    model_save_path = config.MODEL_SAVE_PATH
    test_char = config.TEST_CHARACTER

    # 创建数据集
    dataset = FourCornerDataset(csv_file, img_dir, transform=transform)

    # 划分训练集和测试集
    test_split = 0.2
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = FourCornerCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 监视模型
    wandb.watch(model, log='all')

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # 测试模型
    test_model(model, test_loader, criterion, device)

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至 {model_save_path}")

    # 示例：预测一个字的四角号码
    predicted_code = predict_four_corner(model, test_char, csv_file, img_dir, device, transform)
    print(f"汉字 '{test_char}' 的预测四角号码: {predicted_code}")

    wandb.finish()

if __name__ == "__main__":
    main()