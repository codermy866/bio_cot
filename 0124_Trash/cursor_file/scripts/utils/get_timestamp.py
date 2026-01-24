#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件生成信息:
- 生成时间: 2025-12-24 19:30:00
- 生成需求: 用户要求使用nvidia-smi获取CUDA时间作为文件生成时间戳
- 生成原因: 需要准确的时间戳，CUDA时间与GPU工作环境一致，更准确
- 相关任务: 文件头注释规范和时间戳获取

文件功能: 获取当前时间戳的工具脚本，优先使用nvidia-smi获取CUDA时间
"""

import subprocess
import datetime
import sys

def get_timestamp():
    """
    获取当前时间戳
    优先使用nvidia-smi获取CUDA时间，如果不可用则使用系统时间
    
    Returns:
        str: 格式化的时间戳 'YYYY-MM-DD HH:MM:SS'
    """
    try:
        # 尝试使用nvidia-smi获取CUDA时间
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=timestamp', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi返回的时间格式通常是: YYYY/MM/DD HH:MM:SS.mmm
            timestamp_str = result.stdout.strip().split('\n')[0]
            # 转换为标准格式
            try:
                # 尝试解析nvidia-smi的时间格式
                if '/' in timestamp_str:
                    # 格式: YYYY/MM/DD HH:MM:SS.mmm 或 YYYY/MM/DD HH:MM:SS
                    # 去掉毫秒部分
                    timestamp_clean = timestamp_str.split('.')[0]
                    dt = datetime.datetime.strptime(timestamp_clean, '%Y/%m/%d %H:%M:%S')
                else:
                    # 其他格式，尝试解析
                    timestamp_clean = timestamp_str.split('.')[0].replace('T', ' ')
                    dt = datetime.datetime.fromisoformat(timestamp_clean)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, AttributeError):
                # 如果解析失败，使用系统时间
                pass
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # nvidia-smi不可用或出错，使用系统时间
        pass
    
    # 使用系统时间作为备选
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    timestamp = get_timestamp()
    print(timestamp)
    sys.exit(0)

