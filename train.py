import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt

class FourCornerPredictor:
    def __init__(self, font_path="simsun.ttc", img_size=64):
        """
        初始化四角号码预测器
        
        参数:
        font_path: 字体文件路径
        img_size: 生成汉字图像的尺寸
        """
        self.font_path = font_path
        self.img_size = img_size
        self.model = None
        
    def generate_character_image(self, char):
        """
        生成单个汉字的灰度图像
        
        参数:
        char: 要生成图像的汉字字符
        
        返回:
        image: 生成的灰度图像数组
        """
        # 创建空白的灰度图像
        image = Image.new("L", (self.img_size, self.img_size), 255)
        draw = ImageDraw.Draw(image)
        
        try:
            # 尝试加载指定字体
            font = ImageFont.truetype(self.font_path, int(self.img_size * 0.7))
        except IOError:
            # 使用默认字体
            font = ImageFont.load_default()
        
        # 获取字符大小并居中绘制
        char_size = draw.textbbox((0, 0), char, font=font)[2:]
        position = ((self.img_size - char_size[0]) // 2, 
                    (self.img_size - char_size[1]) // 2)
        draw.text(position, char, 0, font=font)
        
        return np.array(image)

    def extract_corner_features(self, img):
        """
        从汉字图像中提取四角特征区域
        
        参数:
        img: 汉字灰度图像
        
        返回:
        corners: 四个角的图像区域数组
        """
        h, w = img.shape
        # 定义四个角的坐标区域 (左上、右上、左下、右下)
        region_height, region_width = h // 3, w // 3
        
        # 左上角区域
        top_left = img[:region_height, :region_width]
        # 右上角区域
        top_right = img[:region_height, w-region_width:w]
        # 左下角区域
        bottom_left = img[h-region_height:h, :region_width]
        # 右下角区域
        bottom_right = img[h-region_height:h, w-region_width:w]
        
        # 组合四角区域
        corners = np.array([top_left, top_right, bottom_left, bottom_right])
        return corners

    def build_feature_extractor(self):
        """
        构建特征提取器模型（CNN）
        
        返回:
        feature_model: 特征提取模型
        """
        input_layer = Input(shape=(self.img_size//3, self.img_size//3, 1))
        
        # 卷积层1
        x = Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        
        # 卷积层2
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        
        # 展平层
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        
        return Model(inputs=input_layer, outputs=x)
    
    def build_model(self):
        """
        构建完整的四角预测模型
        
        返回:
        model: 完整的预测模型
        """
        # 创建特征提取器
        feature_extractor = self.build_feature_extractor()
        
        # 为四个角创建四个独立的输入通道
        inputs = []
        for i in range(4):
            input_layer = Input(shape=(self.img_size//3, self.img_size//3, 1))
            inputs.append(input_layer)
        
        # 对四个角分别应用特征提取器
        features = [feature_extractor(inp) for inp in inputs]
        
        # 合并所有角的特征
        merged = Concatenate()(features)
        
        # 全连接层
        x = Dense(256, activation='relu')(merged)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # 四角号码的四个数字分别预测 (每个数字0-9)
        outputs = []
        for _ in range(4):
            outputs.append(Dense(10, activation='softmax')(x))
        
        # 创建完整模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_data(self, characters, corner_codes):
        """
        预处理训练数据
        
        参数:
        characters: 汉字列表
        corner_codes: 对应的四角号码列表
        
        返回:
        inputs: 预处理后的输入数据 (四个角的图像)
        outputs: 预处理后的输出数据 (四个角的编码)
        """
        # 存储四个角的图像数据
        inputs = [[] for _ in range(4)]
        outputs = [[] for _ in range(4)]
        
        # 处理每个汉字
        for char, code in zip(tqdm(characters, desc="Processing characters"), corner_codes):
            # 生成汉字图像
            img = self.generate_character_image(char)
            
            # 提取四角特征区域
            corners = self.extract_corner_features(img)
            
            # 归一化处理
            corners = corners / 255.0
            corners = corners[:, :, :, np.newaxis]  # 添加通道维度
            
            # 分离四个角的编码 (每个角一个数字)
            code_digits = [int(d) for d in str(code).zfill(4)]
            
            # 添加到对应的列表中
            for i in range(4):
                inputs[i].append(corners[i])
                outputs[i].append(code_digits[i])
        
        # 转换为NumPy数组并返回
        inputs = [np.array(corner) for corner in inputs]
        outputs = [to_categorical(np.array(corner), num_classes=10) for corner in outputs]
        
        return inputs, outputs

    def train(self, characters, corner_codes, epochs=50, batch_size=32, test_size=0.2):
        """
        训练模型
        
        参数:
        characters: 汉字列表
        corner_codes: 对应的四角号码列表
        epochs: 训练轮数
        batch_size: 批大小
        test_size: 测试集比例
        """
        # 预处理数据
        inputs, outputs = self.preprocess_data(characters, corner_codes)
        
        # 拆分训练集和测试集
        train_inputs, test_inputs = [], []
        train_outputs, test_outputs = [], []
        
        for i in range(4):
            # 分别拆分每个角的输入输出
            X_train, X_test, y_train, y_test = train_test_split(
                inputs[i], outputs[i], test_size=test_size, random_state=42
            )
            train_inputs.append(X_train)
            test_inputs.append(X_test)
            train_outputs.append(y_train)
            test_outputs.append(y_test)
        
        # 构建模型（如果没有已经构建）
        if self.model is None:
            self.build_model()
        
        # 训练模型
        history = self.model.fit(
            train_inputs, 
            train_outputs,
            validation_data=(test_inputs, test_outputs),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history

    def predict(self, char):
        """
        预测单个汉字的四角号码
        
        参数:
        char: 要预测的汉字
        
        返回:
        prediction: 预测的四角号码
        """
        # 生成图像
        img = self.generate_character_image(char)
        
        # 提取四角
        corners = self.extract_corner_features(img)
        corners = corners / 255.0
        corners = corners[:, :, :, np.newaxis]  # 添加通道维度
        
        # 预测四角编码
        predictions = self.model.predict([
            np.array([corners[0]]),
            np.array([corners[1]]),
            np.array([corners[2]]),
            np.array([corners[3]])
        ])
        
        # 组合预测结果
        corner_code = ""
        for pred in predictions:
            digit = np.argmax(pred[0])
            corner_code += str(digit)
        
        return corner_code

    def visualize_characters(self, chars, rows=2, cols=5):
        """
        可视化生成的汉字图像
        
        参数:
        chars: 要可视化的汉字列表
        rows: 显示行数
        cols: 显示列数
        """
        plt.figure(figsize=(15, rows*3))
        for i, char in enumerate(chars[:rows*cols]):
            img = self.generate_character_image(char)
            plt.subplot(rows, cols, i+1)
            plt.imshow(img, cmap='gray')
            plt.title(char)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# 示例使用
if __name__ == "__main__":
    # 实例化预测器
    predictor = FourCornerPredictor()
    
    # 准备数据 - 这里使用模拟数据
    # 实际应用中应从《四角号码新词典》加载数据
    training_chars = ["汉", "字", "编", "码", "预", "测", "机", "器", "学", "习"]
    training_codes = ["3710", "3040", "2312", "1160", "7122", "3760", "4793", "6603", "1241", "1241"]
    
    # 可视化生成的汉字图像
    print("汉字图像示例:")
    predictor.visualize_characters(training_chars[:5])
    
    # # 训练模型
    # print("\n开始训练模型...")
    # history = predictor.train(training_chars, training_codes, epochs=30, batch_size=16)
    
    # # 预测新字符
    # test_chars = ["测", "试", "汉", "语", "大", "字", "典"]
    # print("\n预测结果:")
    # for char in test_chars:
    #     prediction = predictor.predict(char)
    #     print(f"汉字 '{char}' 的预测四角号码: {prediction}")