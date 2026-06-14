import os
import re
import time
import pandas as pd
from openai import OpenAI
from typing import Dict, Optional

# ==============================
# API Clients (Your Keys)
client_claude = OpenAI(
    api_key="sk-sZwZhBKNWns1Xi2W48MU43rulcXnMRTc8cTCMwyDYcmAijjM",
    base_url="https://api.wenwen-ai.com/v1",
)
client_gemini = OpenAI(
    api_key="sk-sZwZhBKNWns1Xi2W48MU43rulcXnMRTc8cTCMwyDYcmAijjM",
    base_url="https://api.wenwen-ai.com/v1",
)
client_gpt = OpenAI(
    api_key="sk-sZwZhBKNWns1Xi2W48MU43rulcXnMRTc8cTCMwyDYcmAijjM",
    base_url="https://api.wenwen-ai.com/v1",
)

MODEL_CLIENTS = {
    "claude-opus": (client_claude, "claude-haiku-4-5-20251001-thinking"),
    "Gemini-2.5": (client_gemini, "gemini-2.5-pro"),
    "GPT-4": (client_gpt, "gpt-4.1")
}

VERIFICATION_CHAIN = [
    ("Gemini-2.5", "claude-opus"),
    ("GPT-4", "Gemini-2.5"),
    ("claude-opus", "GPT-4")
]

MAX_RETRIES = 3
TIMEOUT = 60

# ======================
# Evaluation Prompts
# ======================
def get_initial_evaluation_prompt(question: str, response: str) -> str:
    return f"""You are an expert evaluator. Assess the following response to the given question on three dimensions:

Question: {question}
Response: {response}

Provide your evaluation in EXACTLY the following format:

Fluency: [1-5]
Accuracy: [1-5]
Relevance: [1-5]
Overall Score: [average, one decimal]
Justification: [brief explanation under 100 words]
"""

def get_attacked_evaluation_prompt(question: str, attacked_response: str, reference_response: str) -> str:
    return f"""You are an expert evaluator assessing a potentially adversarially modified response. 
Compare it to the original reference response on factual consistency.

Question: {question}
Reference Response (original): {reference_response}
Attacked Response: {attacked_response}

Evaluation Criteria:
- Fluency: Clarity and naturalness of language (1–5).
- Accuracy: Consistency with the reference response on key facts such as object existence, location, or critical attributes. ⚠️ If the attacked response contradicts the reference—e.g., claims an object "does not exist" when the reference states it exists, or gives a significantly different location—assign an Accuracy score of 3 or below.
- Relevance: Appropriateness and focus on the question (1–5).

Provide your evaluation in EXACTLY the following format:

Fluency: [1-5]
Accuracy: [1-5]
Relevance: [1-5]
Overall Score: [average, one decimal]
Justification: [brief explanation under 100 words]
"""

# ======================
# Core Evaluation Functions
# ======================
def call_model(model_name: str, messages: list) -> Optional[str]:
    if model_name not in MODEL_CLIENTS:
        return None
    client, actual_model = MODEL_CLIENTS[model_name]
    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=actual_model,
                messages=messages,
                timeout=TIMEOUT,
                max_tokens=500,
                temperature=0.0
            )
            content = completion.choices[0].message.content
            if content and len(content.strip()) > 20:
                return content
        except Exception as e:
            print(f"  ❌ {model_name} failed (attempt {attempt+1}): {str(e)[:100]}")
        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)
    return None

def parse_evaluation(text: str) -> Optional[Dict]:
    if not text:
        return None
    try:
        fluency = accuracy = relevance = None
        for line in text.split('\n'):
            if 'fluency' in line.lower():
                match = re.search(r'(\d)', line)
                if match:
                    fluency = int(match.group(1))
            elif 'accuracy' in line.lower():
                match = re.search(r'(\d)', line)
                if match:
                    accuracy = int(match.group(1))
            elif 'relevance' in line.lower():
                match = re.search(r'(\d)', line)
                if match:
                    relevance = int(match.group(1))
        
        if fluency is None or accuracy is None or relevance is None:
            return None

        overall = round((fluency + accuracy + relevance) / 3, 1)
        justification = "No justification."
        if 'justification' in text.lower():
            just_part = text.split('Justification:')[-1].strip()
            justification = " ".join(just_part.split())[:150]
        
        return {
            "fluency": fluency,
            "accuracy": accuracy,
            "relevance": relevance,
            "overall": overall,
            "justification": justification
        }
    except:
        return None

def evaluate_single_response(question: str, response: str, reference_response: Optional[str] = None) -> Dict:
    if not response or pd.isna(response) or len(str(response).strip()) < 5:
        return {"error": "Empty response"}
    
    if reference_response is not None:
        prompt = get_attacked_evaluation_prompt(question, response, reference_response)
    else:
        prompt = get_initial_evaluation_prompt(question, response)
    
    initial = {}
    for model in MODEL_CLIENTS:
        raw = call_model(model, [{"role": "user", "content": prompt}])
        parsed = parse_evaluation(raw) if raw else None
        if parsed:
            initial[model] = parsed
    
    if len(initial) < 2:
        if initial:
            evals = list(initial.values())
            avg = {
                "fluency": round(sum(e["fluency"] for e in evals) / len(evals), 1),
                "accuracy": round(sum(e["accuracy"] for e in evals) / len(evals), 1),
                "relevance": round(sum(e["relevance"] for e in evals) / len(evals), 1),
                "overall": round(sum(e["overall"] for e in evals) / len(evals), 1),
                "justification": " | ".join(e["justification"] for e in evals)[:150]
            }
            return avg
        else:
            return {"error": "All evaluators failed"}

    current = {k: v.copy() for k, v in initial.items()}
    for reviewer, target in VERIFICATION_CHAIN:
        if reviewer in MODEL_CLIENTS and target in current:
            if reference_response is not None:
                verify_prompt = f"""Review {target}'s eval of this attacked response:\n\nQ: {question}\nReference: {reference_response}\nAttacked: {response}\n\nEval: Fluency={current[target]['fluency']}, Accuracy={current[target]['accuracy']}, Relevance={current[target]['relevance']}, Just: {current[target]['justification']}\n\nProvide corrected scores in standard format."""
            else:
                verify_prompt = f"""Review {target}'s eval of this Q&A:\n\nQ: {question}\nA: {response}\n\nEval: Fluency={current[target]['fluency']}, Accuracy={current[target]['accuracy']}, Relevance={current[target]['relevance']}, Just: {current[target]['justification']}\n\nProvide corrected scores in standard format."""
            
            raw = call_model(reviewer, [{"role": "user", "content": verify_prompt}])
            parsed = parse_evaluation(raw)
            if parsed:
                current[target] = parsed

    evals = list(current.values())
    return {
        "fluency": round(sum(e["fluency"] for e in evals) / len(evals), 1),
        "accuracy": round(sum(e["accuracy"] for e in evals) / len(evals), 1),
        "relevance": round(sum(e["relevance"] for e in evals) / len(evals), 1),
        "overall": round(sum(e["overall"] for e in evals) / len(evals), 1),
        "justification": " | ".join(f"[{m}]{e['justification']}" for m, e in current.items())[:200]
    }

# ======================
# Main: 仅处理ERR部分（核心修改）
# ======================
def main(): 
    input_file = "/home/ubuntu/桌面/dataset_new/data-grok-4-0709_quec.xlsx"
    output_file = "/home/ubuntu/桌面/dataset_new/data-grok-4-0709_judge.xlsx"

    # 🔥 1. 读取已有的评分结果
    try:
        existing_df = pd.read_excel(output_file)
        print(f"✅ 加载已有评分文件：{len(existing_df)} 条记录")
        # 筛选出 包含ERR 的样本（需要重处理）
        err_mask = (
            existing_df['resp1_overall'].astype(str).str.contains('ERR') |
            existing_df['resp2_overall'].astype(str).str.contains('ERR') |
            existing_df['attacked_overall'].astype(str).str.contains('ERR')
        )
        err_samples = existing_df[err_mask]
        normal_samples = existing_df[~err_mask]
        print(f"🔍 需重处理样本：{len(err_samples)} 条 | 正常样本：{len(normal_samples)} 条")
    except FileNotFoundError:
        print("❌ 未找到已有评分文件，将从头处理")
        err_samples = pd.DataFrame()
        normal_samples = pd.DataFrame()

    # 读取原始数据
    df = pd.read_excel(input_file, header=None)
    new_results = []

    # 🔥 2. 仅处理 ERR 样本，跳过正常样本
    for col_idx in range(1, 11):
        col_letter = chr(ord('B') + col_idx - 1)
        print(f"\nProcessing column {col_letter}...")
        
        row = 1
        sample_id = 1
        
        while row < len(df):
            question_cell = df.iloc[row, col_idx]
            if pd.isna(question_cell) or str(question_cell).strip() == "":
                row += 1
                continue
            
            # 匹配当前样本：列 + 编号
            if not err_samples.empty:
                match = err_samples[
                    (err_samples['column'] == col_letter) & 
                    (err_samples['sample_id'] == sample_id)
                ]
                if len(match) == 0:
                    # 🚀 非ERR样本，直接跳过，不调用API
                    sample_id += 1
                    row += 5
                    continue

            # ✅ 仅重处理 ERR 样本
            question = str(question_cell).strip()
            print(f"  重处理 Sample {sample_id}: Q = {question[:50]}...")
            
            resp1 = df.iloc[row + 1, col_idx] if (row + 1) < len(df) else ""
            resp2 = df.iloc[row + 2, col_idx] if (row + 2) < len(df) else ""
            resp_attacked = df.iloc[row + 3, col_idx] if (row + 3) < len(df) else ""

            eval1 = evaluate_single_response(question, resp1) if resp1 and not pd.isna(resp1) else {"error": "No response"}
            eval2 = evaluate_single_response(question, resp2) if resp2 and not pd.isna(resp2) else {"error": "No response"}
            eval_attacked = evaluate_single_response(question, resp_attacked, reference_response=resp1) if resp_attacked and not pd.isna(resp_attacked) else {"error": "No response"}
            
            new_results.append({
                "column": col_letter,
                "sample_id": sample_id,
                "question": question,
                "resp1_overall": eval1.get("overall", "ERR"),
                "resp1_fluency": eval1.get("fluency", "ERR"),
                "resp1_accuracy": eval1.get("accuracy", "ERR"),
                "resp1_relevance": eval1.get("relevance", "ERR"),
                "resp1_just": eval1.get("justification", "")[:100],
                
                "resp2_overall": eval2.get("overall", "ERR"),
                "resp2_fluency": eval2.get("fluency", "ERR"),
                "resp2_accuracy": eval2.get("accuracy", "ERR"),
                "resp2_relevance": eval2.get("relevance", "ERR"),
                "resp2_just": eval2.get("justification", "")[:100],
                
                "attacked_overall": eval_attacked.get("overall", "ERR"),
                "attacked_fluency": eval_attacked.get("fluency", "ERR"),
                "attacked_accuracy": eval_attacked.get("accuracy", "ERR"),
                "attacked_relevance": eval_attacked.get("relevance", "ERR"),
                "attacked_just": eval_attacked.get("justification", "")[:100],
            })
            
            sample_id += 1
            row += 5

    # 🔥 3. 合并：正常样本 + 重处理后的ERR样本
    if not existing_df.empty:
        final_df = pd.concat([normal_samples, pd.DataFrame(new_results)], ignore_index=True)
        # 按列和样本ID排序，恢复原始顺序
        final_df = final_df.sort_values(['column', 'sample_id']).reset_index(drop=True)
    else:
        final_df = pd.DataFrame(new_results)

    # 保存最终结果
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        final_df.to_excel(writer, sheet_name='Evaluation_Results', index=False)
    
    print(f"\n✅ 处理完成！结果已保存到 '{output_file}'")
    print(f"📊 最终：正常{len(normal_samples)}条 + 重处理{len(new_results)}条")

if __name__ == "__main__":
    main()
