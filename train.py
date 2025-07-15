import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import pandas as pd
import os
import wandb
import config

# 自定义数据集类
class FourCornerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, corner_transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.corner_transform = corner_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        four_corner = str(self.data.iloc[idx]['four_corner']).zfill(5)
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        image = Image.open(img_path).convert('L')

        # 提取四个角
        w, h = image.size
        corner_size = int(w / 2)
        # Top-left, Top-right, Bottom-left, Bottom-right
        corners = [
            image.crop((0, 0, corner_size, corner_size)),
            image.crop((w - corner_size, 0, w, corner_size)),
            image.crop((0, h - corner_size, corner_size, h)),
            image.crop((w - corner_size, h - corner_size, w, h))
        ]

        # 应用变换
        x_full = self.transform(image) if self.transform else image
        corner_images = [self.corner_transform(c) if self.corner_transform else c for c in corners]

        labels = torch.tensor([int(d) for d in four_corner], dtype=torch.long)

        return (*corner_images, x_full), labels



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

# ResNet模型定义
class FourCornerResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(FourCornerResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        # 修改第一个卷积层以接受灰度图像
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改全连接层
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 5 * 10)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 5, 10) # reshape为(批次, 5位, 10个类别)
        return x

# 多分支模型定义
class CornerBranch(nn.Module):
    def __init__(self):
        super(CornerBranch, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # 添加Dropout
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class FullImageBranch(nn.Module):
    def __init__(self):
        super(FullImageBranch, self).__init__()
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
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # 添加Dropout
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class FourCornerHybridNN(nn.Module):
    def __init__(self):
        super(FourCornerHybridNN, self).__init__()
        self.branch_tl = CornerBranch() # Top-left
        self.branch_tr = CornerBranch() # Top-right
        self.branch_bl = CornerBranch() # Bottom-left
        self.branch_br = CornerBranch() # Bottom-right
        self.branch_full = FullImageBranch() # Full image for 5th digit

    def forward(self, x_tl, x_tr, x_bl, x_br, x_full):
        out_tl = self.branch_tl(x_tl)
        out_tr = self.branch_tr(x_tr)
        out_bl = self.branch_bl(x_bl)
        out_br = self.branch_br(x_br)
        out_full = self.branch_full(x_full)
        return torch.stack([out_tl, out_tr, out_bl, out_br, out_full], dim=1)

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

corner_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    # 早停法相关变量
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5 # 如果验证损失连续5个epoch没有改善，则停止训练

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for (x_tl, x_tr, x_bl, x_br, x_full), labels in train_loader:
            x_tl, x_tr, x_bl, x_br, x_full, labels = \
                x_tl.to(device), x_tr.to(device), x_bl.to(device), x_br.to(device), x_full.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(x_tl, x_tr, x_bl, x_br, x_full)  # (batch_size, 5, 10)
            loss = 0
            for i in range(5):  # 对每位数字计算损失
                loss += criterion(outputs[:, i, :], labels[:, i])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_tl, x_tr, x_bl, x_br, x_full), labels in val_loader:
                x_tl, x_tr, x_bl, x_br, x_full, labels = \
                    x_tl.to(device), x_tr.to(device), x_bl.to(device), x_br.to(device), x_full.to(device), labels.to(device)
                outputs = model(x_tl, x_tr, x_bl, x_br, x_full)
                loss = 0
                for i in range(5):
                    loss += criterion(outputs[:, i, :], labels[:, i])
                val_loss += loss.item()
        
        epoch_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss, "val_loss": epoch_val_loss, "learning_rate": scheduler.get_last_lr()[0]})

        # 早停法逻辑
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

# 预测函数
def predict_four_corner(model, char, csv_file, img_dir, device, transform, corner_transform):
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

    # 提取四个角
    w, h = image.size
    corner_size = int(w / 2)
    corners = [
        image.crop((0, 0, corner_size, corner_size)),
        image.crop((w - corner_size, 0, w, corner_size)),
        image.crop((0, h - corner_size, corner_size, h)),
        image.crop((w - corner_size, h - corner_size, w, h))
    ]

    # 应用变换
    x_full = transform(image).unsqueeze(0).to(device)
    corner_images = [corner_transform(c).unsqueeze(0).to(device) for c in corners]
    
    # 预测
    with torch.no_grad():
        output = model(*corner_images, x_full)  # (1, 5, 10)
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
        for (x_tl, x_tr, x_bl, x_br, x_full), labels in test_loader:
            x_tl, x_tr, x_bl, x_br, x_full, labels = \
                x_tl.to(device), x_tr.to(device), x_bl.to(device), x_br.to(device), x_full.to(device), labels.to(device)
            
            outputs = model(x_tl, x_tr, x_bl, x_br, x_full)
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
    # wandb.init(project="four-corner-cnn", config=config.__dict__)
    wandb.init(
        project="four-corner-cnn",
        config={
            "CSV_FILE": str(config.CSV_FILE),
            "IMAGE_DIR": str(config.IMAGE_DIR),
            "FONT_PATH": str(config.FONT_PATH),
            "IMAGE_SIZE": config.IMAGE_SIZE,
            "FONT_SIZE": config.FONT_SIZE,
            "DEVICE": str(config.DEVICE),  # torch.device 需要转换为字符串
            "BATCH_SIZE": config.BATCH_SIZE,
            "NUM_EPOCHS": config.NUM_EPOCHS,
            "LEARNING_RATE": config.LEARNING_RATE,
            "MODEL_SAVE_PATH": str(config.MODEL_SAVE_PATH),
            "TEST_CHARACTER": config.TEST_CHARACTER
        }
    )

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
    dataset = FourCornerDataset(csv_file, img_dir, transform=transform, corner_transform=corner_transform)

    # 划分训练集、验证集和测试集
    test_split = 0.2
    val_split = 0.1
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * (dataset_size - test_size))
    train_size = dataset_size - test_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    # model = FourCornerCNN().to(device)
    # model = FourCornerResNet(pretrained=True).to(device)
    model = FourCornerHybridNN().to(device)

    criterion = nn.CrossEntropyLoss()
    # 添加 L2 正则化 (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # 每10个epoch学习率乘以0.1

    # 监视模型
    wandb.watch(model, log='all')

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

    # 测试模型
    test_model(model, test_loader, criterion, device)

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至 {model_save_path}")

    # 示例：预测一个字的四角号码
    predicted_code = predict_four_corner(model, test_char, csv_file, img_dir, device, transform, corner_transform)
    print(f"汉字 '{test_char}' 的预测四角号码: {predicted_code}")

    wandb.finish()

if __name__ == "__main__":
    main()