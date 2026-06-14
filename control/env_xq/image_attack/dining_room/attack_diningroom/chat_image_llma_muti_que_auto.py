import os
import base64
import pandas as pd
import requests
from tqdm import tqdm
import urllib3

# 禁用 HTTPS 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === 配置区 ===
# ⚠️ 安全提示：请确保此 Key 仅在受信任的环境中使用
API_KEY = "sk-sZwZhBKNWns1Xi2W48MU43rulcXnMRTc8cTCMwyDYcmAijjM"

# 第三方 API 地址 (已修正为包含 /chat/completions)
API_URL = "https://api.wenwen-ai.com/v1/chat/completions"

# 模型名称 (请确保该第三方平台确实支持此模型名，若报错请尝试 llama-3-70b-instruct 等)
MODEL_NAME = "claude-haiku-4-5-20251001-thinking"

# 图像路径列表
image_paths = [
   "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/generated_images/frame_000000.jpg",
   "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/generated_images/frame_000001.jpg",
   "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/generated_images/frame_000002.jpg"  # Add more images as needed
]

EXCEL_FILE = "/home/ubuntu/桌面/datasets/data-llma_que.xlsx"

# 初始配置
START_ROW_BASE = 77       # 起始问题行号
LOOP_COUNT = 5             # 循环次数
STEP_SIZE = 5              # 每次递增的行数

# === 工具函数 ===
def encode_image(path):
    """将图片编码为 Base64"""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 读取图片失败 {path}: {e}")
        return None

def get_answer_from_api(images, question):
    """
    使用 requests 调用接口获取回答
    """
    # 1. 编码所有图片
    image_b64_list = []
    for img_path in images:
        b64 = encode_image(img_path)
        if b64:
            image_b64_list.append(b64)
    
    if not image_b64_list:
        return "[Error: No valid images to send]"

    # 2. 构建消息体
    messages = [
        {
            "role": "system",
            "content": "You are a friendly home assistant robot in a living room. Your task is to understand the images presented to you and provide clear, helpful answers to any questions."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
                        As a home assistant robot, you are responsible for interpreting the images provided.
                        Your evaluation should focus on:
                        - Understanding the context and details of the images.
                        - Offering practical insights or answers based on what you see.
                        - Don't give me anything that doesn't appear in the images.
                        - The pictures I give you are from different locations in a room.
                        - I will provide you with images from different angles around a center to help you understand the environment.
                        Please respond to the following question using the information from the images:
                    """
                },
                {
                    "type": "text",
                    "text": question
                },
            ]
        },
    ]

    # 添加图片到消息体
    for image_b64 in image_b64_list:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }
        })

    # 3. 构建请求头和数据
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages
    }

    # 4. 发送请求
    try:
        response = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=120)
        response.raise_for_status()
        
        response_data = response.json()
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        else:
            return f"[API Error: Unexpected response format] {response_data}"
            
    except requests.exceptions.RequestException as e:
        return f"[Connection Error: {str(e)}]"
    except Exception as e:
        return f"[Exception: {str(e)}]"

# === 主程序 ===
def main():
    print(f"🚀 开始批量任务...")
    print(f"📂 Excel 文件：{EXCEL_FILE}")
    print(f"🔄 循环次数：{LOOP_COUNT}, 步长：{STEP_SIZE}")
    print(f"🤖 模型：{MODEL_NAME}")
    
    # 检查图片文件是否存在
    for path in image_paths:
        if not os.path.exists(path):
            print(f"⚠️  警告：图片文件不存在 - {path}")

    # 预先读取 Excel (为了效率，我们在内存中维护 df，最后一次性保存)
    try:
        df = pd.read_excel(EXCEL_FILE, header=None, dtype=str)
        print(f"✅ 成功加载 Excel，当前行数：{len(df)}")
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {EXCEL_FILE}")
        return

    total_success = 0
    total_fail = 0

    # === 核心循环：重复 5 次，每次行数 +5 ===
    #========================================================================这里修改后会有两次回答=========================================================
    for i in range(LOOP_COUNT):
        # 动态计算当前的问题行和答案行
        current_question_row = START_ROW_BASE + (i * STEP_SIZE)
        current_answer_row = current_question_row + 1
        
        print(f"\n{'='*40}")
        print(f"🔁 第 {i+1}/{LOOP_COUNT} 轮处理")
        print(f"   📍 问题行 (Excel): {current_question_row} -> 答案行：{current_answer_row}")
        print(f"{'='*40}")

        # 索引转换 (Excel 行号 -> 0-based 索引)
        q_row_idx = current_question_row - 1
        a_row_idx = current_answer_row - 1

        # 确保 DataFrame 有足够行数来写入答案
        # 如果当前答案行超出了现有行数，自动扩展 DataFrame
        while len(df) <= a_row_idx:
            # 添加新行，列数与现有最大列数一致
            new_row = [None] * max(1, df.shape[1])
            df.loc[len(df)] = new_row

        # 检查问题行是否有效
        if q_row_idx >= len(df):
            print(f"⚠️  跳过：问题行 {current_question_row} 超出当前 Excel 范围 (最大行号：{len(df)})")
            continue

        row_questions = df.iloc[q_row_idx]
        total_cols = len(row_questions)

        # 如果该行全为空，可以选择跳过或继续
        if row_questions.isna().all() and (row_questions == '').all():
            print(f"⚠️  跳过：第 {current_question_row} 行全为空")
            continue

        print(f"🔍 正在处理第 {current_question_row} 行，共 {total_cols} 列...")

        loop_success = 0
        loop_fail = 0

        for col_idx in tqdm(range(total_cols), desc=f"Processing Row {current_question_row}", leave=False):
            cell_value = row_questions.iloc[col_idx]
            
            # 跳过空单元格
            if pd.isna(cell_value) or str(cell_value).strip() == "":
                continue

            question = str(cell_value).strip()
            col_letter = chr(65 + (col_idx % 26)) 
            
            # 仅打印简短日志，避免刷屏
            # print(f"  📌 [{col_letter}] 问题：{question[:30]}...") 

            # 调用 API 获取回答
            answer = get_answer_from_api(image_paths, question)
            
            # 写入 DataFrame (直接修改内存中的 df)
            df.iat[a_row_idx, col_idx] = answer

            # 统计结果
            if answer.startswith("[Error]") or answer.startswith("[Connection") or answer.startswith("[Exception"):
                # print(f"    ❌ [{col_letter}] 失败：{answer[:50]}")
                loop_fail += 1
            else:
                # print(f"    ✅ [{col_letter}] 成功")
                loop_success += 1
        
        print(f"   📊 本轮结果 -> ✅ 成功：{loop_success}, ❌ 失败：{loop_fail}")
        total_success += loop_success
        total_fail += loop_fail

    # === 循环结束，一次性保存 ===
    try:
        df.to_excel(EXCEL_FILE, index=False, header=False)
        print(f"\n🎉 所有任务完成！")
        print(f"   📈 总成功：{total_success}")
        print(f"   📉 总失败：{total_fail}")
        print(f"   💾 结果已保存至文件：{EXCEL_FILE}")
        print(f"   📝 覆盖行范围：{START_ROW_BASE + 1} 到 {START_ROW_BASE + (LOOP_COUNT * STEP_SIZE)}")
    except Exception as e:
        print(f"❌ 保存文件时出错：{e}")

if __name__ == "__main__":
    main()