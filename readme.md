# 汉字四角号码识别项目

## 项目简介

这是一个基于机器视觉的汉字四角号码识别项目。四角号码是一种汉字检字法，通过分析汉字的四个角落的笔画形状来为每个汉字分配一个5位数字编码。本项目使用卷积神经网络(CNN)、残差神经网络(ResNet)和多分支神经网络来自动识别汉字图像并预测其对应的四角号码。

## 项目结构

```
character/
├── config.py                    # 项目配置文件
├── train.py                     # 模型训练脚本
├── generate_images.py           # 汉字图像生成脚本
├── process_chars.py             # 字符数据处理脚本
├── scraper.py                   # 四角号码数据爬取脚本
├── debug_csv.py                 # CSV调试工具
├── four_corner_data.csv         # 汉字四角号码数据集
├── four_corner_data_more.csv    # 扩展汉字四角号码数据集
├── simsun.ttc                   # 宋体字体文件
├── requirement.txt              # 项目依赖
├── chinese/                     # 中文字符集目录
│   ├── 3500常用字.txt
│   ├── 7000常用字.txt
│   ├── Chinese16159.txt
│   └── Chinese7000.txt
└── char_images/                 # 生成的汉字图像目录
```

## 功能特性

### 1. 数据处理
- **字符提取**: 从文本文件中提取唯一汉字字符
- **四角号码爬取**: 自动从网站爬取汉字的四角号码数据
- **图像生成**: 将汉字渲染为64x64像素的灰度图像

### 2. 深度学习模型
项目实现了三种不同的神经网络架构：

- **FourCornerCNN**: 基础CNN模型，直接处理完整汉字图像
- **FourCornerResNet**: 基于ResNet18的预训练模型
- **FourCornerHybridNN**: 多分支混合模型，分别处理汉字的四个角落和完整图像

### 3. 训练特性
- 数据增强和预处理
- 早停法防止过拟合
- 学习率调度
- 支持GPU加速训练
- Weights & Biases集成用于实验跟踪

## 安装依赖

```bash
pip install -r requirement.txt
```

主要依赖包括：
- torch (PyTorch深度学习框架)
- torchvision (计算机视觉工具)
- Pillow (图像处理)
- pandas (数据处理)
- selenium (网页爬虫)

## 使用方法

### 1. 数据准备

首先处理字符数据：
```bash
python process_chars.py
```

爬取四角号码数据：
```bash
python scraper.py
```

生成汉字图像：
```bash
python generate_images.py
```

### 2. 模型训练

```bash
python train.py
```

训练过程中会自动：
- 加载数据集并进行训练/验证集划分
- 应用数据增强
- 训练选定的模型架构
- 保存最佳模型权重
- 输出训练日志和验证指标

## 配置说明

所有配置参数都在 <mcfile name="config.py" path="c:\Users\FeiFei\Desktop\code\character\config.py"></mcfile> 中定义：

```python
# 数据相关
CSV_FILE = 'four_corner_data.csv'        # 数据集文件
IMAGE_DIR = 'char_images'                # 图像目录
FONT_PATH = 'SimSun.ttc'                 # 字体文件

# 图像生成
IMAGE_SIZE = (64, 64)                    # 图像尺寸
FONT_SIZE = 64                           # 字体大小

# 训练参数
BATCH_SIZE = 32                          # 批次大小
NUM_EPOCHS = 20                          # 训练轮数
LEARNING_RATE = 0.001                    # 学习率
```

## 模型架构详解

### 多分支混合模型 (FourCornerHybridNN)
这是项目的核心创新，模型结构如下：
- **4个角落分支**: 分别处理汉字的左上、右上、左下、右下四个角落
- **完整图像分支**: 处理整个汉字图像用于预测第5位数字
- **输出**: 5位四角号码，每位都是0-9的数字分类

### 训练策略
- 使用交叉熵损失函数
- Adam优化器
- 学习率衰减调度
- Dropout正则化防止过拟合
- 早停法基于验证损失

## 数据集说明

- **字符来源**: 包含3500、7000、16159个常用汉字的文本文件
- **四角号码**: 通过网络爬虫从专业网站获取
- **图像格式**: 64x64像素灰度图像，白底黑字
- **数据增强**: 包括归一化等预处理步骤

## 性能优化

- 支持CUDA GPU加速
- 批量数据加载
- 内存优化的数据管道
- 多进程数据预处理

## 扩展功能

- 支持自定义字符集
- 可配置的模型架构
- 灵活的训练参数调整
- 实验跟踪和可视化

## 注意事项

1. 确保安装了Chrome浏览器和对应版本的ChromeDriver（用于数据爬取）
2. 字体文件 `simsun.ttc` 需要放在项目根目录
3. 训练前确保有足够的磁盘空间存储生成的图像
4. 建议使用GPU进行训练以获得更好的性能


        