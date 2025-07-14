import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import config

def generate_char_images(csv_file, font_path, output_dir, image_size=(64, 64), font_size=64):
    """
    从CSV文件中的字符列表生成图像。

    :param csv_file: 包含“character”列的CSV文件路径。
    :param font_path: 用于渲染字符的.ttf字体文件路径。
    :param output_dir: 保存生成图像的目录。
    :param image_size: 每个图像的尺寸（宽度，高度）。
    :param font_size: 渲染字符的字体大小。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取CSV文件
    try:
        data = pd.read_csv(csv_file)
        if 'character' not in data.columns:
            print(f"错误：CSV文件 {csv_file} 中未找到“character”列。")
            return
    except FileNotFoundError:
        print(f"错误：找不到CSV文件 {csv_file}。")
        return

    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"错误：找不到或无法加载字体文件 {font_path}。请确保该文件存在。")
        return

    print(f"开始从 {csv_file} 生成图像...")

    # 遍历每个字符并生成图像
    for index, row in data.iterrows():
        char = row['character']
        # 创建一个白色背景的灰度图像
        image = Image.new('L', image_size, color=255)
        draw = ImageDraw.Draw(image)

        # 计算文本大小和位置以使其居中
        try:
            # 使用 textbbox 来获取更准确的边界框
            text_bbox = draw.textbbox((0, 0), char, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            position = (
                (image_size[0] - text_width) // 2 - text_bbox[0],
                (image_size[1] - text_height) // 2 - text_bbox[1]
            )
        except AttributeError:
            # 兼容旧版Pillow
            text_width, text_height = draw.textsize(char, font=font)
            position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)


        # 在图像上绘制黑色文本
        draw.text(position, char, fill=0, font=font)

        # 保存图像
        image_path = os.path.join(output_dir, f"{index}.png")
        image.save(image_path)

    print(f"成功！所有图像已保存在 '{output_dir}' 文件夹中。")

if __name__ == "__main__":
    generate_char_images(
        csv_file=config.CSV_FILE,
        font_path=config.FONT_PATH,
        output_dir=config.IMAGE_DIR,
        image_size=config.IMAGE_SIZE,
        font_size=config.FONT_SIZE
    )