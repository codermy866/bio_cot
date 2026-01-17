#!/bin/bash
# 修复虚拟环境中的路径引用
# 虚拟环境复制后，需要更新激活脚本中的路径

VENV_DIR="/data2/hmy/VLM_Caus_Rm/my_retfound"
OLD_PATH="/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/my_retfound"
NEW_PATH="/data2/hmy/VLM_Caus_Rm/my_retfound"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 虚拟环境目录不存在: $VENV_DIR"
    exit 1
fi

echo "=========================================="
echo "🔧 修复虚拟环境路径引用"
echo "=========================================="
echo ""

# 需要更新的文件
FILES_TO_UPDATE=(
    "bin/activate"
    "bin/activate.csh"
    "bin/activate.fish"
    "pyvenv.cfg"
)

for file in "${FILES_TO_UPDATE[@]}"; do
    FILE_PATH="$VENV_DIR/$file"
    if [ -f "$FILE_PATH" ]; then
        echo "📝 更新: $file"
        # 使用sed更新路径
        sed -i "s|$OLD_PATH|$NEW_PATH|g" "$FILE_PATH"
        if [ $? -eq 0 ]; then
            echo "  ✅ 更新成功"
        else
            echo "  ⚠️  更新失败"
        fi
    else
        echo "  ⏭️  文件不存在: $file"
    fi
done

# 检查Python可执行文件中的路径
echo ""
echo "🔍 检查Python可执行文件..."
PYTHON_BIN="$VENV_DIR/bin/python"
if [ -f "$PYTHON_BIN" ]; then
    # Python可执行文件通常是符号链接或shebang，检查shebang
    if head -1 "$PYTHON_BIN" | grep -q "$OLD_PATH"; then
        echo "  ⚠️  Python可执行文件包含旧路径，需要重新创建"
        echo "  建议: 重新创建虚拟环境或使用虚拟环境修复工具"
    else
        echo "  ✅ Python可执行文件路径正常"
    fi
fi

echo ""
echo "=========================================="
echo "✅ 路径修复完成"
echo "=========================================="
echo ""
echo "⚠️  注意事项:"
echo "  1. 如果Python可执行文件有问题，可能需要重新创建虚拟环境"
echo "  2. 建议测试虚拟环境是否正常工作:"
echo "     source $VENV_DIR/bin/activate"
echo "     python --version"
echo "     which python"

