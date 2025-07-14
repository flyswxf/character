import pandas as pd

def debug_four_corner_csv(csv_path):
    """
    读取指定的CSV文件，并检查'four_corner'列中是否存在无法直接转换为整数的值。

    Args:
        csv_path (str): CSV文件的路径。
    """
    print(f"正在检查文件: {csv_path}")
    try:
        data = pd.read_csv(csv_path, dtype={'four_corner': str})
        print("CSV文件成功加载。开始检查'four_corner'列...")
        
        found_issues = False
        # 迭代每一行来检查
        for index, row in data.iterrows():
            four_corner_val = row['four_corner']
            try:
                # 尝试将值转换为整数
                int(float(four_corner_val))
            except (ValueError, TypeError):
                found_issues = True
                print(f"---\n发现问题数据!")
                print(f"  行号 (从0开始): {index}")
                print(f"  汉字: {row.get('character', 'N/A')}")
                print(f"  原始 'four_corner' 值: '{four_corner_val}'")
                print(f"  原始值的数据类型: {type(four_corner_val)}")

        if not found_issues:
            print("\n检查完成。在'four_corner'列中未发现明显的格式问题。")
        else:
            print("\n检查完成。已列出所有发现的问题数据。")

    except FileNotFoundError:
        print(f"错误: 文件未找到 at '{csv_path}'")
    except Exception as e:
        print(f"处理文件时发生未知错误: {e}")

if __name__ == "__main__":
    # 请将此路径替换为您的CSV文件的实际路径
    csv_file_path = 'four_corner_data.csv'
    debug_four_corner_csv(csv_file_path)