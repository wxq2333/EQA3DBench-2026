#这个文件用于处理yaml文件的命名
import os
import re
import yaml
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

def load_qa_mapping(excel_path):
    """
    读取 Excel 文件，构建 {回答文本片段: 原始问题} 的映射字典。
    逻辑：遍历每一列，如果某行是问题，下一行是非空回答，则建立映射。
    为了容错，我们提取回答的前100个字符作为键。
    """
    print(f"正在读取表格文件：{excel_path} ...")
    try:
        # 读取所有 sheet，假设数据在第一个 sheet 或者所有 sheet 结构类似
        # header=None 因为表头比较复杂，我们按行处理
        df = pd.read_excel(excel_path, sheet_name=0, header=None)
    except Exception as e:
        print(f"读取 Excel 失败：{e}")
        return {}

    qa_map = {}
    rows, cols = df.shape
    
    # 遍历每一列 (因为不同场景的问题分布在不同的列)
    for col in range(cols):
        for row in range(rows - 1):
            cell_current = df.iloc[row, col]
            cell_next = df.iloc[row+1, col]
            
            # 简单的启发式规则：
            # 1. 当前单元格包含问号或者是典型的疑问句开头 (Where, What, Is, Can...) -> 视为问题
            # 2. 下一个单元格是非空的长文本 -> 视为回答
            is_question = False
            if isinstance(cell_current, str):
                if '?' in cell_current or re.match(r'^(What|Where|Is|Can|How|Describe|Are|Why|Who)', cell_current, re.IGNORECASE):
                    is_question = True
            
            if is_question and isinstance(cell_next, str) and len(cell_next) > 20:
                question = cell_current.strip()
                answer = cell_next.strip()
                
                # 清理回答中的换行和多余空格，以便匹配
                answer_clean = " ".join(answer.split())
                
                # 使用回答的前 80 个字符作为 Key (防止回答太长导致匹配困难，且保留独特性)
                # 同时也存储完整回答以防万一
                key_short = answer_clean[:80].lower()
                key_long = answer_clean[:200].lower()
                
                qa_map[key_short] = question
                qa_map[key_long] = question
                
                # 也可以尝试用回答的后半部分作为键，增加匹配率
                if len(answer_clean) > 100:
                    qa_map[answer_clean[-100:].lower()] = question

    print(f"成功从表格中提取了 {len(qa_map)} 组 问题-回答 映射关系。")
    return qa_map

def find_question_in_map(yaml_content, qa_map):
    """
    在 YAML 内容中查找与表格中回答匹配的片段，返回对应的原始问题。
    """
    # 获取 YAML 的第一行注释内容 (去掉 '# ')
    lines = yaml_content.split('\n')
    first_line_comment = ""
    
    for line in lines:
        if line.strip().startswith('#'):
            # 提取注释文本
            text = line.replace('#', '', 1).strip()
            if text:
                first_line_comment = text
                break
    
    if not first_line_comment:
        return None

    # 清理注释文本用于匹配
    comment_clean = " ".join(first_line_comment.split()).lower()
    
    # 策略 1: 精确匹配前缀
    for key, question in qa_map.items():
        if key in comment_clean or comment_clean.startswith(key):
            return question
            
    # 策略 2: 模糊匹配 (如果直接包含匹配不到)
    # 遍历 map 中的键，计算相似度
    best_match = None
    highest_ratio = 0.0
    
    for key in qa_map.keys():
        # 比较 key 和 comment_clean
        ratio = SequenceMatcher(None, key, comment_clean).ratio()
        # 或者检查 key 是否是 comment 的子串
        if key in comment_clean:
            ratio = 0.95 # 高置信度
            
        if ratio > highest_ratio and ratio > 0.6: # 阈值设为 0.6
            highest_ratio = ratio
            best_match = qa_map[key]
            
    return best_match

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
    
    # 1. 查找原始问题
    original_question = find_question_in_map(content, qa_map)
    
    if not original_question:
        print(f"⚠️  未在表格中找到匹配的回答。文件内容首行: {content.split(chr(10))[0][:50]}...")
        # 可选：让用户手动输入
        user_input = input("是否手动输入原始问题? (直接输入问题或回车跳过): ").strip()
        if user_input:
            original_question = user_input
        else:
            print("跳过此文件。")
            return False

    print(f"✅ 找到原始问题: {original_question}")
    
    # 2. 生成新文件名
    safe_name_prefix = clean_filename_from_text(original_question)
    new_filename = f"{safe_name_prefix}.yaml"
    
    target_path = file_path.parent / new_filename
    
    # 防止覆盖非目标文件
    if target_path.exists() and target_path != file_path:
        print(f"⚠️  目标文件 {new_filename} 已存在。")
        # 这里可以选择跳过或重命名，为了安全暂时跳过
        return False

    # 3. 构建新内容
    # 需求：把原始问题当作第一行注释加入到原有的yaml文件中
    # 原有文件第一行已经是注释 (# Based on...)，我们在它上面插入一行 # Question: ...
    
    header_comment = f"# Question: {original_question}\n"
    
    if content.startswith(header_comment):
        print("文件已包含该问题注释，仅检查重命名。")
        new_content = content
    else:
        new_content = header_comment + content

    # 4. 写入并重命名
    try:
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        if target_path != file_path:
            os.remove(file_path)
            print(f"🔄 重命名成功: {file_path.name} -> {new_filename}")
        else:
            print("💾 原地更新内容成功。")
            
        return True
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return False

def main():
    # 配置路径
    excel_file = "data-3.xlsx"  # 确保 excel 文件在当前目录，或填写绝对路径
    
    if not os.path.exists(excel_file):
        excel_file = input("请输入 data-3.xlsx 的完整路径: ").strip().strip('"')
    
    if not os.path.exists(excel_file):
        print(f"错误：找不到文件 {excel_file}")
        return

    # 加载映射
    qa_map = load_qa_mapping(excel_file)
    
    if not qa_map:
        print("未能从表格中提取有效数据，请检查表格格式。")
        return

    # 选择 YAML 目录
    base_dir = input("\n请输入包含 yaml 文件的目录路径 (直接回车使用当前目录): ").strip()
    if not base_dir:
        base_dir = os.getcwd()
    
    path_obj = Path(base_dir)
    if not path_obj.exists():
        print(f"错误：目录 {base_dir} 不存在。")
        return

    yaml_files = list(path_obj.glob("*.yaml"))
    if not yaml_files:
        print("未找到任何 .yaml 文件。")
        return

    print(f"\n找到 {len(yaml_files)} 个 YAML 文件。开始匹配和处理...\n")
    
    success_count = 0
    for yf in yaml_files:
        if process_yaml_file(yf, qa_map):
            success_count += 1
            
    print(f"\n🎉 处理完成。成功处理 {success_count}/{len(yaml_files)} 个文件。")

if __name__ == "__main__":
    main()