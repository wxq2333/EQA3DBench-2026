#!/bin/bash
set -e

# ========= 每次只改这里 =========
# MODEL_TAG="gemini"
# MODEL_TAG="deepseek"
# MODEL_TAG="minmax"
# MODEL_TAG="minimax-m2.7"
# MODEL_TAG="kimi_2.5"
# MODEL_TAG="gpt-4o"
# MODEL_TAG="grok-4-0709" 
# MODEL_TAG="gpt5"
# MODEL_TAG="deepseek-r1"
# MODEL_TAG="gpt-4o-mini"
MODEL_TAG="gemini-2.5-pro"

# API_NAME="gemini-2.5-flash" 
# API_NAME="deepseek-v3"
# API_NAME="MiniMaxM2.7"  #
# API_NAME="kimi-k2.5"
# API_NAME="gpt-4o"
API_NAME="gemini-2.5-pro"
API_URL="https://api.wenwen-ai.com/v1/chat/completions"
API_KEY="sk-sZwZhBKNWns1Xi2W48MU43rulcXnMRTc8cTCMwyDYcmAijjM"
API_MODEL="gemini-2.5-pro"

NEW_EXCEL_FILE="/home/ubuntu/桌面/dataset_new/data-${MODEL_TAG}_quec.xlsx"
# ===============================

PY_FILES=(
  "/home/ubuntu/桌面/control/env_xq/image_attack/bath_room/attack_bathroom/attack_pipeline_1.py"
  "/home/ubuntu/桌面/control/env_xq/image_attack/bed_room/attack_bed_room/attack_pipeline_1.py"
  "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/attack_diningroom/attack_pipeline_1.py"
  "/home/ubuntu/桌面/control/env_xq/image_attack/kitchen/attack_kitchen/attack_pipeline_1.py"
  "/home/ubuntu/桌面/control/env_xq/image_attack/living_room/attack_living_room/attack_pipeline_1.py"
)

for PY_FILE in "${PY_FILES[@]}"; do
    echo "=========================================="
    echo "Processing: $PY_FILE"
    echo "=========================================="

    if [ ! -f "$PY_FILE" ]; then
        echo "File not found: $PY_FILE"
        continue
    fi

    # 先备份
    cp "$PY_FILE" "${PY_FILE}.bak"

    # 1) 替换 EXCEL_FILE
    sed -i "s|^EXCEL_FILE *=.*|EXCEL_FILE = \"${NEW_EXCEL_FILE}\"|g" "$PY_FILE"

    # 2) 替换 API_CONFIGS 整个代码块
    python3 - <<PY
from pathlib import Path
import re

py_file = Path(r'''$PY_FILE''')
text = py_file.read_text(encoding='utf-8')

new_block = '''API_CONFIGS = [
    {
        "name": "${API_NAME}",
        "url": "${API_URL}",
        "key": "${API_KEY}",
        "model": "${API_MODEL}"
    }
]'''

pattern = r'API_CONFIGS\s*=\s*\[[\s\S]*?\]\n'
new_text, n = re.subn(pattern, new_block + '\n', text, count=1)

if n == 0:
    raise RuntimeError(f"Failed to replace API_CONFIGS in {py_file}")

py_file.write_text(new_text, encoding='utf-8')
print(f"Updated: {py_file}")
PY

    # 3) 执行该场景脚本
    python3 "$PY_FILE"

done