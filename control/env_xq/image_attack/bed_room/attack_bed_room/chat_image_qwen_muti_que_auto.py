import base64
import pandas as pd
from tqdm import tqdm
import dashscope
from dashscope import MultiModalConversation

# === 配置区 ===
API_KEY = "sk-0a46c0017e094c3ea61c1f15dbdc5bbc"
dashscope.api_key = API_KEY

# 图像路径（逗号分隔）
image_paths = [
    "/home/ubuntu/桌面/control/env_xq/image_attack/bed_room/image_bedroom/frame_000000.jpg",
    "/home/ubuntu/桌面/control/env_xq/image_attack/bed_room/image_bedroom/frame_000001.jpg",
    "/home/ubuntu/桌面/control/env_xq/image_attack/bed_room/image_bedroom/frame_000002.jpg"
]

EXCEL_FILE = "/home/ubuntu/桌面/datasets/data-qwen_quec.xlsx"        # 你的 Excel 文件名
QUESTION_ROW = 47                     # Excel 中的问题行（第 I 行 = 第 9 行）
ANSWER_ROW = QUESTION_ROW + 2       # 答案写入下一行（第 J 行 = 第 10 行）

# === 工具函数 ===
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_answer_from_qwen(images, question):
    messages = [
        {"role": "system", "content": "You are a helpful home assistant robot."},
        {"role": "user", "content": []}
    ]
    instruction = (
        "Answer based ONLY on the provided images. "
        "The images show different views of a bedroom. "
        "Do not hallucinate. If unsure, say 'Not visible in the images.'"
    )
    messages[1]["content"].extend([
        {"text": instruction},
        {"text": question}
    ])
    for img in images:
        b64 = encode_image(img)
        messages[1]["content"].append({"image": f"data:image/jpeg;base64,{b64}"})
    
    try:
        resp = MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=messages,
            result_format='message'
        )
        if resp.status_code == 200:
            content = resp.output.choices[0].message.content
            if isinstance(content, list):
                return "".join(item.get("text", "") for item in content)
            return content
        else:
            return f"[API Error: {resp.code}]"
    except Exception as e:
        return f"[Exception: {str(e)}]"

# === 主程序 ===
def main():
    # 读取 Excel（无表头，保留原始布局）
    df = pd.read_excel(EXCEL_FILE, header=None, dtype=str)

    # 转为 0-based 索引
    q_row_idx = QUESTION_ROW - 1
    a_row_idx = ANSWER_ROW - 1

    # 确保 DataFrame 有足够行数
    while len(df) <= a_row_idx:
        df.loc[len(df)] = [None] * (df.shape[1] if df.shape[1] > 0 else 1)

    # 获取第 I 行（第9行）的所有列
    if q_row_idx >= len(df):
        print(f"❌ 指定的问题行 {QUESTION_ROW} 超出 Excel 范围！")
        return

    row_questions = df.iloc[q_row_idx]
    total_cols = len(row_questions)

    print(f"🔍 正在处理 Excel 第 {QUESTION_ROW} 行（共 {total_cols} 列）...")

    for col_idx in tqdm(range(total_cols), desc="Processing columns"):
        cell_value = row_questions.iloc[col_idx]
        if pd.isna(cell_value) or str(cell_value).strip() == "":
            continue  # 跳过空单元格

        question = str(cell_value).strip()
        print(f"\n📌 问题 (第{QUESTION_ROW}行, 列{chr(65+col_idx)}): {question}")

        answer = get_answer_from_qwen(image_paths, question)
        df.iat[a_row_idx, col_idx] = answer

        print(f"✅ 回答 (第{ANSWER_ROW}行, 列{chr(65+col_idx)}): {answer[:60]}...")

    # 保存回原文件
    df.to_excel(EXCEL_FILE, index=False, header=False)
    print(f"\n🎉 完成！答案已写入第 {ANSWER_ROW} 行，文件：{EXCEL_FILE}")

if __name__ == "__main__":
    main()