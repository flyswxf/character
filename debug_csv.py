import pandas as pd

def debug_four_corner_csv(csv_path, auto_remove=False, output_path=None):
    """
    读取指定的CSV文件，并检查'four_corner'列中是否存在无法直接转换为整数的值。
    可选择自动删除异常值。

    Args:
        csv_path (str): CSV文件的路径。
        auto_remove (bool): 是否自动删除异常值。
        output_path (str): 输出文件路径，如果为None则覆盖原文件。
    """
    print(f"正在检查文件: {csv_path}")
    try:
        data = pd.read_csv(csv_path, dtype={'four_corner': str})
        print("CSV文件成功加载。开始检查'four_corner'列...")
        
        found_issues = False
        problematic_indices = []
        
        # 迭代每一行来检查
        for index, row in data.iterrows():
            four_corner_val = row['four_corner']
            try:
                # 尝试将值转换为整数
                int(float(four_corner_val))
            except (ValueError, TypeError):
                found_issues = True
                problematic_indices.append(index)
                print(f"---\n发现问题数据!")
                print(f"  行号 (从0开始): {index}")
                print(f"  汉字: {row.get('character', 'N/A')}")
                print(f"  原始 'four_corner' 值: '{four_corner_val}'")
                print(f"  原始值的数据类型: {type(four_corner_val)}")

        if not found_issues:
            print("\n检查完成。在'four_corner'列中未发现明显的格式问题。")
        else:
            print(f"\n检查完成。共发现 {len(problematic_indices)} 行问题数据。")
            
            if auto_remove:
                print("\n正在自动删除异常值...")
                # 删除有问题的行
                cleaned_data = data.drop(problematic_indices)
                
                # 确定输出路径
                if output_path is None:
                    output_path = csv_path
                
                # 保存清理后的数据
                cleaned_data.to_csv(output_path, index=False)
                print(f"已删除 {len(problematic_indices)} 行异常数据")
                print(f"清理后的数据已保存到: {output_path}")
                print(f"原始数据行数: {len(data)}")
                print(f"清理后数据行数: {len(cleaned_data)}")
            else:
                print("\n提示: 如需自动删除异常值，请设置 auto_remove=True")

    except FileNotFoundError:
        print(f"错误: 文件未找到 at '{csv_path}'")
    except Exception as e:
        print(f"处理文件时发生未知错误: {e}")

if __name__ == "__main__":
    # 请将此路径替换为您的CSV文件的实际路径
    csv_file_path = 'four_corner_data_more.csv'
    
    # 设置为True来自动删除异常值
    auto_remove_anomalies = True
    
    # 可选：指定输出文件路径，如果为None则覆盖原文件
    # output_file_path = 'four_corner_data_more_cleaned.csv'
    output_file_path = None
    
    debug_four_corner_csv(csv_file_path, auto_remove=auto_remove_anomalies, output_path=output_file_path)