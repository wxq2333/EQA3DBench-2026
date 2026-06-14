import os
import re
import yaml
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

def clean_text_for_matching(text):
    """
    核心清洗函数：
    1. 移除 # 号
    2. 移除所有标点符号 (只保留字母、数字、空格)
    3. 转小写
    4. 压缩多余空格
    """
    if not isinstance(text, str):
        return ""
    
    # 1. 移除井号
    text = text.replace('#', '')
    
    # 2. 使用正则移除所有非字母数字和非空格的字符
    # \w 匹配字母数字和下划线，\s 匹配空格
    # 我们保留字母数字和空格，去掉其他所有标点
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. 转小写
    text = text.lower()
    
    # 4. 压缩空格 (多个变一个) 并去除首尾空格
    text = " ".join(text.split())
    
    return text

def load_qa_mapping(excel_path):
    """
    读取 Excel，构建 {清洗后的回答片段: 原始问题} 映射
    """
    print(f"正在读取并解析表格文件：{excel_path} ...")
    try:
        df = pd.read_excel(excel_path, sheet_name=0, header=None)
    except Exception as e:
        print(f"读取 Excel 失败：{e}")
        return {}

    qa_map = {}
    rows, cols = df.shape
    
    match_count = 0
    
    for col in range(cols):
        for row in range(rows - 1):
            cell_current = df.iloc[row, col]
            cell_next = df.iloc[row+1, col]
            
            # 识别问题行 (包含问号或典型疑问词)
            is_question = False
            if isinstance(cell_current, str):
                # 简单的疑问句判断
                if '?' in cell_current or re.match(r'^(What|Where|Is|Can|How|Describe|Are|Why|Who|Does|Do)', cell_current, re.IGNORECASE):
                    is_question = True
            
            # 如果当前是问题，下一行是非空长字符串，则视为回答
            if is_question and isinstance(cell_next, str) and len(cell_next) > 15:
                question_raw = str(cell_current).strip()
                answer_raw = str(cell_next).strip()
                
                # 关键步骤：清洗回答文本用于作为 Key
                answer_cleaned = clean_text_for_matching(answer_raw)
                
                if len(answer_cleaned) < 10:
                    continue
                
                # 策略：存储不同长度的切片，提高匹配成功率
                # 1. 前 50 个字符 (最独特)
                key_50 = answer_cleaned[:50]
                # 2. 前 100 个字符
                key_100 = answer_cleaned[:100]
                # 3. 完整清洗后的文本 (如果不太长)
                if len(answer_cleaned) <= 200:
                    qa_map[answer_cleaned] = question_raw
                
                qa_map[key_50] = question_raw
                qa_map[key_100] = question_raw
                
                # 4. 后 50 个字符 (防止前面被截断)
                if len(answer_cleaned) > 60:
                    qa_map[answer_cleaned[-50:]] = question_raw

                match_count += 1

    print(f"✅ 成功从表格中提取了 {match_count} 组问答对，生成了 {len(qa_map)} 个索引键。")
    return qa_map

def find_question_in_map(yaml_content, qa_map):
    """
    提取 YAML 第一行注释，清洗后去 map 中查找
    """
    lines = yaml_content.split('\n')
    first_line_comment = ""
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            # 提取 # 后面的内容
            text = stripped[1:].strip()
            if text:
                first_line_comment = text
                break
    
    if not first_line_comment:
        return None

    # 关键步骤：清洗 YAML 中的注释文本，使其格式与 Excel Key 完全一致
    yaml_cleaned = clean_text_for_matching(first_line_comment)
    
    if len(yaml_cleaned) < 10:
        return None

    # 1. 精确匹配 (检查各种长度的切片)
    # 检查完整内容 (如果短)
    if yaml_cleaned in qa_map:
        return qa_map[yaml_cleaned]
    
    # 检查前 50 字
    if yaml_cleaned[:50] in qa_map:
        return qa_map[yaml_cleaned[:50]]
        
    # 检查前 100 字
    if len(yaml_cleaned) >= 100 and yaml_cleaned[:100] in qa_map:
        return qa_map[yaml_cleaned[:100]]
        
    # 检查后 50 字
    if len(yaml_cleaned) >= 60 and yaml_cleaned[-50:] in qa_map:
        return qa_map[yaml_cleaned[-50:]]

    # 2. 模糊匹配 (作为备选)
    # 遍历 map 中的 key，看 yaml_cleaned 是否包含 key，或者 key 是否包含 yaml_cleaned
    for key, question in qa_map.items():
        if key in yaml_cleaned or yaml_cleaned in key:
            # 进一步验证相似度，防止误匹配短词
            ratio = SequenceMatcher(None, key, yaml_cleaned).ratio()
            if ratio > 0.6: 
                return question
                
    return None

def clean_filename_from_text(text):
    """将文本转换为安全的文件名"""
    safe_text = re.sub(r'[<>:"/\\|?*]', '', text)
    safe_text = " ".join(safe_text.split())
    safe_text = safe_text.replace(" ", "_")
    if len(safe_text) > 100:
        safe_text = safe_text[:100]
    return safe_text

def process_yaml_file(file_path, qa_map):
    print(f"\n处理文件: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_question = find_question_in_map(content, qa_map)
    
    if not original_question:
        # 调试信息：打印清洗后的 YAML 首行，方便排查
        lines = content.split('\n')
        raw_first = ""
        for l in lines:
            if l.strip().startswith('#'):
                raw_first = l.strip()
                break
        
        cleaned_debug = clean_text_for_matching(raw_first)
        print(f"⚠️  未在表格中找到匹配的回答。")
        print(f"   原始首行: {raw_first}")
        print(f"   清洗后特征: {cleaned_debug[:60]}...")
        
        user_input = input("是否手动输入原始问题? (直接输入问题或回车跳过): ").strip()
        if user_input:
            original_question = user_input
        else:
            print("跳过此文件。")
            return False

    print(f"✅ 匹配成功 -> 问题: {original_question}")
    
    safe_name_prefix = clean_filename_from_text(original_question)
    new_filename = f"{safe_name_prefix}.yaml"
    target_path = file_path.parent / new_filename
    
    if target_path.exists() and target_path != file_path:
        print(f"⚠️  目标文件 {new_filename} 已存在，跳过以避免覆盖。")
        return False

    header_comment = f"# Question: {original_question}\n"
    
    if not content.startswith(header_comment):
        new_content = header_comment + content
    else:
        new_content = content

    try:
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        if target_path != file_path:
            os.remove(file_path)
            print(f"🔄 重命名: {file_path.name} -> {new_filename}")
        else:
            print("💾 已更新内容。")
            
        return True
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def main():
    excel_file = "data-3.xlsx"
    if not os.path.exists(excel_file):
        excel_file = input("请输入 data-3.xlsx 的路径: ").strip().strip('"')
    
    if not os.path.exists(excel_file):
        print("找不到 Excel 文件。")
        return

    qa_map = load_qa_mapping(excel_file)
    if not qa_map:
        return

    base_dir = input("\n请输入 YAML 文件目录路径 (回车使用当前目录): ").strip()
    if not base_dir:
        base_dir = os.getcwd()
    
    path_obj = Path(base_dir)
    if not path_obj.exists():
        print("目录不存在。")
        return

    yaml_files = list(path_obj.glob("*.yaml"))
    if not yaml_files:
        print("未找到 .yaml 文件。")
        return

    print(f"开始处理 {len(yaml_files)} 个文件...\n")
    
    success = 0
    for f in yaml_files:
        if process_yaml_file(f, qa_map):
            success += 1
            
    print(f"\n🎉 完成。成功: {success}/{len(yaml_files)}")

if __name__ == "__main__":
    main()