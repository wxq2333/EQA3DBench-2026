import os
import base64
import pandas as pd
import requests
from tqdm import tqdm
import urllib3

# 禁用 HTTPS 警告 (因为你之前的代码使用了 verify=False)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === 配置区 ===
# ⚠️ 安全提示：请确保此 Key 仅在受信任的环境中使用
API_KEY = "sk-YrS3MnJJWd661l1EsUmvR2RxAQFP4hScbtaZmWAC3ho0vxtv"

# 第三方 API 地址 (根据你的成功代码)
API_URL = "https://tb.api.mkeai.com/v1/chat/completions"

# 模型名称 (根据你的成功代码)
MODEL_NAME = "gemini-2.5-pro"

# 图像路径列表
image_paths = [
    "/home/ubuntu/桌面/control/env_xq/image_attack/bed_room/image_bedroom/frame_000000.jpg",
    "/home/ubuntu/桌面/control/env_xq/image_attack/bed_room/image_bedroom/frame_000001.jpg",
    "/home/ubuntu/桌面/control/env_xq/image_attack/bed_room/image_bedroom/frame_000002.jpg"
]

EXCEL_FILE = "/home/ubuntu/桌面/datasets/data-gemini_que.xlsx"
QUESTION_ROW = 42                     # Excel 中的问题行 (第 102 行)
ANSWER_ROW = QUESTION_ROW + 2          # 答案写入下一行 (第 103 行)

# === 工具函数 ===
def encode_image(path):
    """将图片编码为 Base64"""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 读取图片失败 {path}: {e}")
        return None

def get_answer_from_gemini(images, question):
    """
    使用 requests 调用第三方 mkeai 接口获取回答
    """
    # 1. 编码所有图片
    image_b64_list = []
    for img_path in images:
        b64 = encode_image(img_path)
        if b64:
            image_b64_list.append(b64)
    
    if not image_b64_list:
        return "[Error: No valid images to send]"

    # 2. 构建消息体 (完全复刻你成功的代码逻辑)
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
        # 注意：这里沿用你之前的 verify=False 以兼容某些自签名证书的代理
        response = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=120)
        response.raise_for_status()  # 如果状态码不是 200，抛出异常
        
        response_data = response.json()
        
        # 提取回答
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
    print(f"🚀 开始任务...")
    print(f"📂 Excel 文件：{EXCEL_FILE}")
    print(f"🖼️  图片数量：{len(image_paths)}")
    print(f"🤖 模型：{MODEL_NAME}")
    print(f"🔗 接口：{API_URL}")
    
    # 检查图片文件是否存在
    for path in image_paths:
        if not os.path.exists(path):
            print(f"⚠️  警告：图片文件不存在 - {path}")

    try:
        # 读取 Excel (无表头)
        df = pd.read_excel(EXCEL_FILE, header=None, dtype=str)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {EXCEL_FILE}")
        return

    # 索引转换 (Excel 行号 -> 0-based 索引)
    q_row_idx = QUESTION_ROW - 1
    a_row_idx = ANSWER_ROW - 1

    # 确保 DataFrame 有足够行数来写入答案
    while len(df) <= a_row_idx:
        df.loc[len(df)] = [None] * max(1, df.shape[1])

    # 检查问题行是否有效
    if q_row_idx >= len(df):
        print(f"❌ 错误：指定的问题行 {QUESTION_ROW} 超出 Excel 范围 (最大行号: {len(df)})")
        return

    row_questions = df.iloc[q_row_idx]
    total_cols = len(row_questions)

    print(f"🔍 正在处理第 {QUESTION_ROW} 行，共 {total_cols} 列...")

    success_count = 0
    fail_count = 0

    for col_idx in tqdm(range(total_cols), desc="Processing Columns"):
        cell_value = row_questions.iloc[col_idx]
        
        # 跳过空单元格
        if pd.isna(cell_value) or str(cell_value).strip() == "":
            continue

        question = str(cell_value).strip()
        col_letter = chr(65 + (col_idx % 26)) # A, B, C...
        
        print(f"\n📌 [{col_letter}] 问题：{question[:50]}...")

        # 调用 API 获取回答
        answer = get_answer_from_gemini(image_paths, question)
        
        # 写入 DataFrame
        df.iat[a_row_idx, col_idx] = answer

        # 简单的结果反馈
        if answer.startswith("[Error]") or answer.startswith("[Connection") or answer.startswith("[Exception"):
            print(f"❌ 失败：{answer[:80]}")
            fail_count += 1
        else:
            print(f"✅ 成功：{answer[:60]}...")
            success_count += 1

    # 保存回原文件
    try:
        df.to_excel(EXCEL_FILE, index=False, header=False)
        print(f"\n🎉 任务完成！")
        print(f"   ✅ 成功：{success_count}")
        print(f"   ❌ 失败：{fail_count}")
        print(f"   📝 结果已保存至第 {ANSWER_ROW} 行")
    except Exception as e:
        print(f"❌ 保存文件时出错：{e}")

if __name__ == "__main__":
    main()