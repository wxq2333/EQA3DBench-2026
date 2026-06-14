#!/bin/bash
set -e

# ==========================================
# ⚙️ 1. 全局配置区
# ==========================================
# --- A. 唯一的 Excel 文件路径 (所有任务共用) ---
SHARED_EXCEL_FILE="/home/ubuntu/桌面/dataset_new/data-gemini-2.5-pro_quec.xlsx"

# --- B. API 配置信息 ---
API_NAME="gemini-2.5-pro"
GLOBAL_API_KEY="sk-sZwZhBKNWns1Xi2W48MU43rulcXnMRTc8cTCMwyDYcmAijjM"
GLOBAL_API_URL="https://api.wenwen-ai.com/v1/chat/completions"
GLOBAL_MODEL_NAME="gemini-2.5-pro"
# --- C. 需要运行的 5 个 Python 文件路径列表 ---
# 请在这里填入你那 5 个文件的具体路径
PY_FILES=(
#   "/home/ubuntu/桌面/control/env_xq/image_attack/bath_room/attack_bathroom/chat_image_muti_que_auto_pipeline.py"
  "/home/ubuntu/桌面/control/env_xq/image_attack/bed_room/attack_bed_room/chat_image_muti_que_auto_pipeline.py"
  "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/attack_diningroom/chat_image_muti_que_auto_pipeline.py"
  "/home/ubuntu/桌面/control/env_xq/image_attack/kitchen/attack_kitchen/chat_image_muti_que_auto_pipeline.py"
  "/home/ubuntu/桌面/control/env_xq/image_attack/living_room/attack_living_room/chat_image_muti_que_auto_pipeline.py"
  
  # 如果有更多，继续往下面加...
)

# ==========================================
# 🚀 2. 执行逻辑 (无需修改)
# ==========================================

echo "🚀 开始批量串行任务..."
echo "💾 共用 Excel: $(basename "$SHARED_EXCEL_FILE")"
echo "🔑 API Model: $GLOBAL_MODEL_NAME"
echo "📋 待处理脚本数: ${#PY_FILES[@]}"
echo ""

# 检查 Excel 是否存在 (只检查一次)
if [ ! -f "$SHARED_EXCEL_FILE" ]; then
    echo "❌ 错误：找不到共用的 Excel 文件 -> $SHARED_EXCEL_FILE"
    echo "请检查路径是否正确，或文件是否被移动。"
    exit 1
fi

for i in "${!PY_FILES[@]}"; do
    PY_FILE="${PY_FILES[$i]}"
    
    echo "------------------------------------------"
    echo "⏳ 任务 #$((i+1))/${#PY_FILES[@]}:"
    echo "   📄 脚本：$(basename "$PY_FILE")"
    echo "   📊 数据：$(basename "$SHARED_EXCEL_FILE")"
    echo "------------------------------------------"

    # 1. 检查 Python 文件是否存在
    if [ ! -f "$PY_FILE" ]; then
        echo "❌ 错误：找不到脚本 -> $PY_FILE"
        echo "   ⚠️ 跳过此任务，继续下一个..."
        continue
    fi

    # 2. 备份原脚本
    BACKUP_FILE="${PY_FILE}.bak.auto"
    cp "$PY_FILE" "$BACKUP_FILE"
    
    # 3. 动态注入配置
    
    # 替换 API_KEY
    sed -i "s|^API_KEY *= *\".*\"|API_KEY = \"${GLOBAL_API_KEY}\"|g" "$PY_FILE"
    
    # 替换 API_URL
    sed -i "s|^API_URL *= *\".*\"|API_URL = \"${GLOBAL_API_URL}\"|g" "$PY_FILE"
    
    # 替换 MODEL_NAME
    sed -i "s|^MODEL_NAME *= *\".*\"|MODEL_NAME = \"${GLOBAL_MODEL_NAME}\"|g" "$PY_FILE"
    
    # 替换 EXCEL_FILE (指向同一个共享文件)
    # 使用 @ 作为分隔符，防止路径中的 / 干扰
    sed -i "s|^EXCEL_FILE *= *\".*\"|EXCEL_FILE = \"${SHARED_EXCEL_FILE}\"|g" "$PY_FILE"
    
    echo "   ✅ 配置已更新 (Key, URL, Model, Excel)"

    # 4. 执行脚本
    echo "   ▶️  正在运行..."
    if python3 "$PY_FILE"; then
        echo "   🎉 成功：$(basename "$PY_FILE")"
        echo "   💡 结果已追加写入到：$SHARED_EXCEL_FILE"
    else
        echo "   ❌ 失败：$(basename "$PY_FILE")"
        echo "   ⚠️  正在恢复脚本原状..."
        cp "$BACKUP_FILE" "$PY_FILE"
        echo "   🔙 脚本已恢复。继续处理下一个任务..."
        # 如果希望遇到错误立即停止整个流程，取消下面这行的注释:
        # exit 1
    fi

    # (可选) 运行完后删除备份，保持整洁
    # rm -f "$BACKUP_FILE"
    
    echo ""
done

echo "=========================================="
echo "🏁 所有任务处理完毕！"
echo "📊 最终结果请查看：$SHARED_EXCEL_FILE"
echo "=========================================="