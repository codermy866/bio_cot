#!/bin/bash
# 下载并安装Arial字体的脚本

PASSWORD="qaz@123"
FONT_DIR="$HOME/.fonts/arial"
SYSTEM_FONT_DIR="/usr/share/fonts/truetype/arial"

echo "=========================================="
echo "安装Arial字体"
echo "=========================================="

# 创建目录
mkdir -p "$FONT_DIR"
echo "$PASSWORD" | sudo -S mkdir -p "$SYSTEM_FONT_DIR" 2>/dev/null

# 方法1: 尝试从Windows字体目录复制（如果存在）
if [ -d "/mnt/c/Windows/Fonts" ]; then
    echo "检测到Windows字体目录，正在复制Arial字体..."
    echo "$PASSWORD" | sudo -S cp /mnt/c/Windows/Fonts/arial*.ttf "$SYSTEM_FONT_DIR/" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ 从Windows复制成功！"
    fi
fi

# 方法2: 使用Liberation Sans作为Arial的替代（已安装）
echo ""
echo "Liberation Sans是Arial的开源替代，已包含在系统中"
echo "Liberation Sans与Arial在视觉上几乎相同，适合学术论文使用"

# 更新字体缓存
echo ""
echo "更新字体缓存..."
echo "$PASSWORD" | sudo -S fc-cache -fv 2>&1 | grep -v "password" | tail -3

echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="
fc-list | grep -iE "(arial|liberation)" | head -5

echo ""
echo "✅ 完成！"
echo ""
echo "注意：如果Arial未安装，matplotlib将使用DejaVu Sans或Liberation Sans"
echo "这些字体与Arial在视觉上非常相似，适合学术论文使用"


