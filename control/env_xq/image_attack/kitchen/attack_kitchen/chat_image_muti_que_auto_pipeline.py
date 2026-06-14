import os
import base64
import pandas as pd
import requests
from tqdm import tqdm
import urllib3
import time

# 禁用 HTTPS 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === 配置区 ===
# ⚠️ 安全警告：请勿在生产环境中硬编码 Key，建议使用环境变量
API_KEY = "sk-sZwZhBKNWns1Xi2W48MU43rulcXnMRTc8cTCMwyDYcmAijjM"
API_URL = "https://api.wenwen-ai.com/v1/chat/completions"
MODEL_NAME = "gemini-2.5-pro"

# 图像路径列表
image_paths = [
    "/home/ubuntu/桌面/control/env_xq/image_attack/kitchen/images_attack/frame_000000.jpg",
    "/home/ubuntu/桌面/control/env_xq/image_attack/kitchen/images_attack/frame_000001.jpg",
    "/home/ubuntu/桌面/control/env_xq/image_attack/kitchen/images_attack/frame_000002.jpg"
]

EXCEL_FILE = "/home/ubuntu/桌面/dataset_new/data-gemini-2.5-pro_quec.xlsx"

# 【修改点 1】定义需要处理的问题行号列表
TARGET_QUESTION_ROWS = [2, 7, 12, 17, 22]

# === 工具函数 ===
def encode_image(path):
    """将图片编码为 Base64"""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 读取图片失败 {path}: {e}")
        return None

def get_answer_from_gemini(images, question, round_num=1):
    """
    使用 requests 调用第三方 mkeai 接口获取回答
    :param round_num: 1 表示第一轮，2 表示第二轮（用于微调提示词）
    """
    # 1. 编码所有图片
    image_b64_list = []
    for img_path in images:
        b64 = encode_image(img_path)
        if b64:
            image_b64_list.append(b64)
    
    if not image_b64_list:
        return "[Error: No valid images to send]"

    # 【修改点 2】根据轮次微调 System Prompt 或 User Prompt，避免两次回答完全一致
    round_instruction = ""
    if round_num == 1:
        round_instruction = "Please provide your primary analysis and answer."
    else:
        round_instruction = "Please provide a second, independent perspective or additional details that were not mentioned in your first response. Avoid repeating the exact same phrasing."

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
                    "text": f"""
                        As a home assistant robot, you are responsible for interpreting the images provided.
                        Your evaluation should focus on:
                        - Understanding the context and details of the images.
                        - Offering practical insights or answers based on what you see.
                        - Don't give me anything that doesn't appear in the images.
                        - The pictures I give you are from different locations in a room.
                        - I will provide you with images from different angles around a center to help you understand the environment.
                        
                        Current Task Round: {round_num}/2
                        Specific Instruction: {round_instruction}
                        
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
    print(f"🚀 开始多轮多行任务...")
    print(f"📂 Excel 文件：{EXCEL_FILE}")
    print(f"🎯 目标问题行：{TARGET_QUESTION_ROWS}")
    print(f"🖼️  图片数量：{len(image_paths)}")
    
    # 检查图片文件是否存在
    for path in image_paths:
        if not os.path.exists(path):
            print(f"⚠️  警告：图片文件不存在 - {path}")

    try:
        # 读取 Excel (无表头)
        df = pd.read_excel(EXCEL_FILE, header=None, dtype=str)
        print(f"📊 当前 Excel 行数：{len(df)}")
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {EXCEL_FILE}")
        return

    total_success = 0
    total_fail = 0

    # 【修改点 3】外层循环：遍历每一个指定的问题行
    for q_row in TARGET_QUESTION_ROWS:
        print(f"\n{'='*50}")
        print(f"🔍 正在处理第 {q_row} 行的问题...")
        print(f"{'='*50}")

        # 索引转换 (Excel 行号 -> 0-based 索引)
        q_row_idx = q_row - 1
        ans_row_1_idx = q_row      # 下一行 (第 Q+1 行)
        ans_row_2_idx = q_row + 1  # 下两行 (第 Q+2 行)

        # 确保 DataFrame 有足够行数来写入两轮答案
        # 需要保证至少有 ans_row_2_idx + 1 行 (因为索引是从0开始的)
        required_rows = ans_row_2_idx + 1
        while len(df) < required_rows:
            df.loc[len(df)] = [None] * max(1, df.shape[1])

        # 检查问题行是否有效
        if q_row_idx >= len(df):
            print(f"❌ 错误：指定的问题行 {q_row} 超出 Excel 范围")
            continue

        row_questions = df.iloc[q_row_idx]
        total_cols = len(row_questions)

        # 进度条描述包含当前行号
        pbar = tqdm(range(total_cols), desc=f"Processing Row {q_row}", leave=False)
        
        for col_idx in pbar:
            cell_value = row_questions.iloc[col_idx]
            
            # 跳过空单元格
            if pd.isna(cell_value) or str(cell_value).strip() == "":
                continue

            question = str(cell_value).strip()
            
            # --- 第一轮询问 ---
            answer_1 = get_answer_from_gemini(image_paths, question, round_num=1)
            df.iat[ans_row_1_idx, col_idx] = answer_1
            
            # 简单的结果反馈 (第一轮)
            if "Error" in answer_1 or "Exception" in answer_1:
                pbar.set_postfix({"Status": "Round1 Fail"})
                total_fail += 1
            else:
                total_success += 1

            # 可选：在两轮之间加一点延迟，防止触发频率限制 (Rate Limit)
            time.sleep(1) 

            # --- 第二轮询问 ---
            answer_2 = get_answer_from_gemini(image_paths, question, round_num=2)
            df.iat[ans_row_2_idx, col_idx] = answer_2

            # 简单的结果反馈 (第二轮)
            if "Error" in answer_2 or "Exception" in answer_2:
                pbar.set_postfix({"Status": "Round2 Fail"})
                total_fail += 1
            else:
                total_success += 1
                
            pbar.set_postfix({"Status": "OK"})

    # 保存回原文件
    try:
        # 建议：在实际运行前，可以手动备份文件，或者这里自动保存为副本
        backup_file = EXCEL_FILE.replace(".xlsx", "_backup.xlsx")
        if not os.path.exists(backup_file):
             # 仅在第一次运行时备份（简单逻辑），生产环境建议手动备份
             pass 
        
        df.to_excel(EXCEL_FILE, index=False, header=False)
        print(f"\n🎉 所有任务完成！")
        print(f"   ✅ 总成功次数：{total_success}")
        print(f"   ❌ 总失败次数：{total_fail}")
        print(f"   📝 结果已保存：")
        for q_row in TARGET_QUESTION_ROWS:
            print(f"      - 第 {q_row} 行问题的答案 -> 第 {q_row+1} 行 (第一轮) & 第 {q_row+2} 行 (第二轮)")
    except Exception as e:
        print(f"❌ 保存文件时出错：{e}")

if __name__ == "__main__":
    main()