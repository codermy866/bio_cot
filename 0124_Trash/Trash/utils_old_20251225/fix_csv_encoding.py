#!/usr/bin/env python3
"""
修复CSV文件的中文编码问题
从Excel文件重新生成正确编码的CSV文件
"""

import pandas as pd
import os
from pathlib import Path

def fix_csv_encoding():
    """修复CSV文件的中文编码问题"""
    print("=== 修复CSV文件中文编码问题 ===")
    
    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()
    data_dir = project_root / "5centers_multi"
    
    # 备份原始CSV文件
    print("备份原始CSV文件...")
    train_csv_backup = data_dir / "train_labels_backup.csv"
    test_csv_backup = data_dir / "test_labels_backup.csv"
    
    if (data_dir / "train_labels.csv").exists():
        os.system(f"cp {data_dir}/train_labels.csv {train_csv_backup}")
        print(f"训练集CSV已备份到: {train_csv_backup}")
    
    if (data_dir / "test_labels.csv").exists():
        os.system(f"cp {data_dir}/test_labels.csv {test_csv_backup}")
        print(f"测试集CSV已备份到: {test_csv_backup}")
    
    # 处理训练集
    print("\n处理训练集...")
    train_xlsx = data_dir / "train_labels.xlsx"
    if train_xlsx.exists():
        try:
            # 读取Excel文件
            print("读取Excel文件...")
            df_train = pd.read_excel(train_xlsx)
            print(f"训练集数据形状: {df_train.shape}")
            print(f"训练集列名: {list(df_train.columns)}")
            
            # 显示前几行数据
            print("\n训练集前5行数据:")
            print(df_train.head())
            
            # 保存为UTF-8编码的CSV
            train_csv = data_dir / "train_labels.csv"
            df_train.to_csv(train_csv, index=False, encoding='utf-8')
            print(f"训练集CSV已保存到: {train_csv}")
            
            # 验证保存的文件
            print("\n验证保存的训练集CSV文件:")
            df_verify = pd.read_csv(train_csv, encoding='utf-8')
            print("前5行数据:")
            print(df_verify.head())
            
        except Exception as e:
            print(f"处理训练集时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"训练集Excel文件不存在: {train_xlsx}")
    
    # 处理测试集
    print("\n处理测试集...")
    test_xlsx = data_dir / "test_labels.xlsx"
    if test_xlsx.exists():
        try:
            # 读取Excel文件
            print("读取Excel文件...")
            df_test = pd.read_excel(test_xlsx)
            print(f"测试集数据形状: {df_test.shape}")
            print(f"测试集列名: {list(df_test.columns)}")
            
            # 显示前几行数据
            print("\n测试集前5行数据:")
            print(df_test.head())
            
            # 保存为UTF-8编码的CSV
            test_csv = data_dir / "test_labels.csv"
            df_test.to_csv(test_csv, index=False, encoding='utf-8')
            print(f"测试集CSV已保存到: {test_csv}")
            
            # 验证保存的文件
            print("\n验证保存的测试集CSV文件:")
            df_verify = pd.read_csv(test_csv, encoding='utf-8')
            print("前5行数据:")
            print(df_verify.head())
            
        except Exception as e:
            print(f"处理测试集时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"测试集Excel文件不存在: {test_xlsx}")
    
    print("\n=== 修复完成 ===")
    print("CSV文件已重新生成，中文字符应该能正确显示了")

if __name__ == "__main__":
    fix_csv_encoding() 