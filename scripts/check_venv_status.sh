#!/bin/bash
# 检查虚拟环境复制状态

TARGET_DIR="/data2/hmy/VLM_Caus_Rm"
LOG_FILE="$TARGET_DIR/venv_copy.log"
VENV_DIR="$TARGET_DIR/my_retfound"

echo "=========================================="
echo "📊 虚拟环境复制状态检查"
echo "=========================================="
echo ""

# 检查日志
if [ -f "$LOG_FILE" ]; then
    echo "📄 最新日志 (最后20行):"
    echo "----------------------------------------"
    tail -20 "$LOG_FILE"
    echo "----------------------------------------"
    echo ""
fi

# 检查目标目录
echo "📁 目标目录状态:"
if [ -d "$VENV_DIR" ]; then
    SIZE=$(du -sh "$VENV_DIR" 2>/dev/null | cut -f1)
    echo "  ✅ 虚拟环境目录已存在"
    echo "  📊 当前大小: $SIZE"
    
    # 检查关键文件
    if [ -f "$VENV_DIR/bin/activate" ]; then
        echo "  ✅ 激活脚本存在"
    else
        echo "  ⚠️  激活脚本不存在"
    fi
    
    if [ -d "$VENV_DIR/lib" ]; then
        PYTHON_COUNT=$(find "$VENV_DIR/lib" -name "python*" -type d 2>/dev/null | wc -l)
        echo "  ✅ Python库目录存在 ($PYTHON_COUNT 个Python版本)"
    fi
else
    echo "  ⏳ 虚拟环境目录尚未创建"
fi

echo ""

# 检查rsync进程
echo "🔄 运行中的复制进程:"
RSYNC_PIDS=$(pgrep -f "rsync.*my_retfound" || echo "")
if [ -n "$RSYNC_PIDS" ]; then
    ps aux | grep -E "rsync.*my_retfound" | grep -v grep
    echo ""
    echo "✅ 复制正在进行中..."
else
    echo "  ℹ️  没有运行中的复制进程"
    echo ""
    
    # 检查是否完成
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        SOURCE_SIZE=$(du -sb /data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/my_retfound 2>/dev/null | cut -f1)
        TARGET_SIZE=$(du -sb "$VENV_DIR" 2>/dev/null | cut -f1)
        
        if [ -n "$SOURCE_SIZE" ] && [ -n "$TARGET_SIZE" ]; then
            if [ "$SOURCE_SIZE" -eq "$TARGET_SIZE" ]; then
                echo "✅ 复制可能已完成，大小匹配"
            else
                echo "⚠️  大小不匹配，可能未完成"
                echo "   源: $SOURCE_SIZE bytes"
                echo "   目标: $TARGET_SIZE bytes"
            fi
        else
            echo "✅ 复制可能已完成，请手动验证"
        fi
    else
        echo "⚠️  复制可能未完成或未开始"
    fi
fi

echo ""
echo "=========================================="

