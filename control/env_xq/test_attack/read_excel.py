import pandas as pd
import json

def excel_to_qa_pairs(excel_path, question_rows, output_json_path):
    """
    从Excel中提取QA对（问题在指定行，答案在下一行对应列）
    
    :param excel_path: Excel文件路径
    :param question_rows: 问题所在的行号列表（如 [2, 4, 6]）
    :param output_json_path: 输出的JSON文件路径
    :return: 生成的QA对列表
    """
    df = pd.read_excel(excel_path, header=None)  # 无表头读取
    
    qa_pairs = []
    
    for row in question_rows:
        # 问题行索引 = row - 1（因为Excel行号从1开始，pandas从0开始）
        question_row = df.iloc[row - 1]
        answer_row = df.iloc[row+1] if row < len(df) else None  # 防止越界
        
        if answer_row is None:
            continue  # 跳过没有答案的行
        
        # 遍历当前行的每一列
        for col in range(len(question_row)):
            question = question_row[col]
            answer = answer_row[col] if not pd.isna(answer_row[col]) else "No answer found"
            
            if pd.isna(question):  # 跳过空问题
                continue
                
            qa_pair = {
                "question": str(question),
                "answer": f"In the images provided, {str(answer)}"
            }
            qa_pairs.append(qa_pair)
    
    # 保存JSON（直接显示中文）
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
    
    return qa_pairs


# 使用示例
excel_path = '/home/ubuntu/桌面/数据集6.21（带有初始回答和扰动回答）+-+副本.xlsx'
question_rows = [2,8,15]  # 指定问题行号（Excel行号，从1开始）
output_json_path = '/home/ubuntu/桌面/control/env_xq/test_attack/qa_pairs.json'

qa_pairs = excel_to_qa_pairs(excel_path, question_rows, output_json_path)
print(f"已生成 {len(qa_pairs)} 个QA对，保存至 {output_json_path}")