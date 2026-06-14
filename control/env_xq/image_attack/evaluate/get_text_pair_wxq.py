import openpyxl
import os

def generate_text_pairs(file_path):
    """
    从Excel文件生成文本对列表。假设第一行是"初始回答"，第二行是需要配对的文本。
    
    :param file_path: 输入的Excel文件路径
    :return: 文本对列表
    """
    # 打开Excel文件
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active  # 获取活动表（通常是第一个工作表）

    # 假设文本对的第一行是“初始回答”行
    initial_answers = sheet[161]  # 获取Excel的第5行（行号从0开始，因此是第5行）
    next_answers = sheet[162]  # 获取Excel的第6行（行号从0开始，因此是第6行）
    
    # 生成文本对
    text_pairs = []
    
    for i in range(len(initial_answers)):
        # 获取每一列的内容
        text_a = initial_answers[i].value
        text_b = next_answers[i].value
        
        # 删除换行符
        if text_a:
            text_a = text_a.replace("\n", " ").replace("\r", " ").strip()  # 移除换行符及多余的空格
        if text_b:
            text_b = text_b.replace("\n", " ").replace("\r", " ").strip()  # 同样处理文本B
        
        # 确保文本不是空的
        if text_a and text_b:
            text_pairs.append((text_a, text_b))
    
    return text_pairs

def save_text_pairs_to_file(text_pairs, output_file="text_pairs.txt"):
    """
    将生成的文本对保存到文本文件中。
    
    :param text_pairs: 文本对列表
    :param output_file: 输出的文本文件路径
    """
    # 确保文件路径的文件夹存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        # 写入文本对格式到文件
        f.write("text_pairs = [\n")
        for pair in text_pairs:
            f.write(f'    ("{pair[0]}", "{pair[1]}"),\n')
        f.write("]\n")

    print(f"文本对已保存到文件 {output_file}")

if __name__ == "__main__":
    # 输入 Excel 文件路径
    file_path = "/home/ubuntu/下载/副本数据集1.12.xlsx"
    # 生成文本对
    text_pairs = generate_text_pairs(file_path)
    # 保存文本对到指定路径
    save_text_pairs_to_file(text_pairs, output_file="/home/ubuntu/桌面/control/env_xq/image_attack/evaluate/text_pairs.txt")
