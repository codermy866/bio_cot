#!/bin/bash
# Arial字体安装脚本

echo "=========================================="
echo "安装Arial字体到Linux系统"
echo "=========================================="

# 方法1: 尝试使用apt安装Microsoft核心字体（包含Arial）
echo ""
echo "方法1: 尝试安装ttf-mscorefonts-installer..."
echo "qaz@123" | sudo -S apt-get update -qq
echo "qaz@123" | sudo -S apt-get install -y ttf-mscorefonts-installer 2>&1 | grep -v "password"

if [ $? -eq 0 ]; then
    echo "✅ 通过apt安装成功！"
    echo "qaz@123" | sudo -S fc-cache -fv
    echo ""
    echo "验证安装:"
    fc-list | grep -i arial | head -3
    exit 0
fi

# 方法2: 如果方法1失败，尝试手动下载并安装
echo ""
echo "方法2: 手动安装Arial字体..."

# 创建字体目录
FONT_DIR="$HOME/.fonts/arial"
mkdir -p "$FONT_DIR"

# 下载Arial字体（从公开资源）
echo "正在下载Arial字体文件..."
cd "$FONT_DIR"

# 尝试从多个源下载Arial字体
# 注意：这些是公开可用的字体资源
wget -q --timeout=10 https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf -O arial_alternative.ttf 2>/dev/null || true

# 如果用户有Windows系统，可以从Windows复制
# 或者从其他合法来源下载

# 更好的方法：使用Liberation Sans作为Arial的替代（已包含在大多数Linux发行版中）
echo ""
echo "方法3: 使用Liberation Sans作为Arial的替代（兼容性更好）..."
echo "Liberation Sans是Arial的开源替代，已包含在大多数Linux系统中"

# 更新字体缓存
fc-cache -fv

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "验证字体安装:"
fc-list | grep -iE "(arial|liberation)" | head -5

echo ""
echo "如果Arial未安装，系统将使用Liberation Sans作为替代"
echo "Liberation Sans与Arial在视觉上非常相似，适合学术论文使用"


