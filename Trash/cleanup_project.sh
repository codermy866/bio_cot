#!/bin/bash
# 项目清理脚本
# 用于清理不需要提交到GitHub的文件

echo "=== 开始清理项目文件 ==="

# 1. 清理Python缓存
echo "1. 清理Python缓存文件..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
echo "   ✓ Python缓存已清理"

# 2. 清理日志文件
echo "2. 清理日志文件..."
find . -name "*.log" -type f -delete 2>/dev/null
echo "   ✓ 日志文件已清理"

# 3. 清理临时文件
echo "3. 清理临时文件..."
find . -name "*.tmp" -delete 2>/dev/null
find . -name "*.bak" -delete 2>/dev/null
find . -name "*.swp" -delete 2>/dev/null
find . -name "*~" -delete 2>/dev/null
echo "   ✓ 临时文件已清理"

# 4. 统计清理结果
echo ""
echo "=== 清理完成 ==="
echo "剩余文件统计:"
echo "  Python文件: $(find . -name "*.py" -type f | wc -l)"
echo "  总文件数: $(find . -type f | wc -l)"
echo ""
echo "注意: 虚拟环境 my_retfound/ 已保留"
echo "      如需清理，请手动删除: rm -rf my_retfound/"

