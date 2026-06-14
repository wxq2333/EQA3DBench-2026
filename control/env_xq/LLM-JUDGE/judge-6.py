import os
import re
import time
import pandas as pd
from openai import OpenAI
from typing import Dict, Optional

# ==============================
# API Clients (Your Keys)
# ==============================
client_qwen = OpenAI(
    api_key="sk-0a46c0017e094c3ea61c1f15dbdc5bbc",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
client_gemini = OpenAI(
    api_key="sk-IAg99sNg9pTZ4wY5aGjL9QFx6HTMa6gCAovFhsdhOLnrBQbr",
    base_url="https://tb.api.mkeai.com/v1",
)
client_gpt = OpenAI(
    api_key="sk-sZwZhBKNWns1Xi2W48MU43rulcXnMRTc8cTCMwyDYcmAijjM",
    base_url="https://api.wenwen-ai.com/v1",
)

MODEL_CLIENTS = {
    "Qwen-Max": (client_qwen, "qwen-max"),
    "Gemini-2.5": (client_gemini, "gemini-2.5-pro"),
    "GPT-4": (client_gpt, "gpt-4.1")
}

VERIFICATION_CHAIN = [
    ("Gemini-2.5", "Qwen-Max"),
    ("GPT-4", "Gemini-2.5"),
    ("Qwen-Max", "GPT-4")
]

MAX_RETRIES = 2
TIMEOUT = 30

# ======================
# Core Evaluation Functions (Simplified for Batch)
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
    if not text: return None
    try:
        fluency = accuracy = relevance = None
        for line in text.split('\n'):
            if 'fluency' in line.lower():
                match = re.search(r'(\d)', line)
                if match: fluency = int(match.group(1))
            elif 'accuracy' in line.lower():
                match = re.search(r'(\d)', line)
                if match: accuracy = int(match.group(1))
            elif 'relevance' in line.lower():
                match = re.search(r'(\d)', line)
                if match: relevance = int(match.group(1))
        
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

def evaluate_single_response(question: str, response: str) -> Dict:
    """Run full cross-validation chain for one response"""
    if not response or pd.isna(response) or len(str(response).strip()) < 5:
        return {"error": "Empty response"}
    
    # Step 1: Initial evaluations
    initial = {}
    for model in MODEL_CLIENTS:
        prompt = get_initial_evaluation_prompt(question, response)
        raw = call_model(model, [{"role": "user", "content": prompt}])
        parsed = parse_evaluation(raw) if raw else None
        if parsed:
            initial[model] = parsed
    
    if len(initial) < 2:
        # Fallback: use available evals
        if initial:
            evals = list(initial.values())
            avg = {
                "fluency": round(sum(e["fluency"] for e in evals)/len(evals), 1),
                "accuracy": round(sum(e["accuracy"] for e in evals)/len(evals), 1),
                "relevance": round(sum(e["relevance"] for e in evals)/len(evals), 1),
                "overall": round(sum(e["overall"] for e in evals)/len(evals), 1),
                "justification": " | ".join(e["justification"] for e in evals)[:150]
            }
            return avg
        else:
            return {"error": "All evaluators failed"}

    # Step 2: Cross-validation chain
    current = {k: v.copy() for k, v in initial.items()}
    for reviewer, target in VERIFICATION_CHAIN:
        if reviewer in MODEL_CLIENTS and target in current:
            prompt = f"""Review {target}'s eval of this Q&A:\n\nQ: {question}\nA: {response}\n\nEval: Fluency={current[target]['fluency']}, Accuracy={current[target]['accuracy']}, Relevance={current[target]['relevance']}, Just: {current[target]['justification']}\n\nProvide corrected scores in standard format."""
            raw = call_model(reviewer, [{"role": "user", "content": prompt}])
            parsed = parse_evaluation(raw)
            if parsed:
                current[target] = parsed

    # Step 3: Consensus
    evals = list(current.values())
    return {
        "fluency": round(sum(e["fluency"] for e in evals)/len(evals), 1),
        "accuracy": round(sum(e["accuracy"] for e in evals)/len(evals), 1),
        "relevance": round(sum(e["relevance"] for e in evals)/len(evals), 1),
        "overall": round(sum(e["overall"] for e in evals)/len(evals), 1),
        "justification": " | ".join(f"[{m}]{e['justification']}" for m, e in current.items())[:200]
    }

# ======================
# Main: Process Excel
# ======================

def main():
    input_file = "/home/ubuntu/桌面/datasets/data-2 (1).xlsx"
    output_file = "/home/ubuntu/桌面/datasets/data_out.xlsx" 

    
    df = pd.read_excel(input_file, header=None)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

    # Prepare result structure
    results = []

    # Columns B to K → indices 1 to 10
    for col_idx in range(1, 11):
        col_letter = chr(ord('B') + col_idx - 1)
        print(f"\nProcessing column {col_letter}...")
        
        row = 1  # 0-based index; row 1 = Excel row 2
        sample_id = 1
        
        while row < len(df):
            question_cell = df.iloc[row, col_idx]
            if pd.isna(question_cell) or str(question_cell).strip() == "":
                row += 1
                continue
                
            question = str(question_cell).strip()
            print(f"  Sample {sample_id}: Q = {question[:50]}...")
            
            # Get responses from row+2, row+3, row+4 (Excel rows 4,5,6)
            resp1 = df.iloc[row+1, col_idx] if (row + 1) < len(df) else ""
            resp2 = df.iloc[row+2, col_idx] if (row + 2) < len(df) else ""
            resp_attacked = df.iloc[row+3, col_idx] if (row + 3) < len(df) else ""


            # Print what was actually read

            print(f"    → Resp1 (Excel row {row+2+1}): '{resp1}'")
            print(f"    → Resp2 (Excel row {row+3+1}): '{resp2}'")
            print(f"    → Attacked (Excel row {row+4+1}): '{resp_attacked}'")

            # Evaluate each
            eval1 = evaluate_single_response(question, resp1) if resp1 and not pd.isna(resp1) else {"error": "No response"}
            eval2 = evaluate_single_response(question, resp2) if resp2 and not pd.isna(resp2) else {"error": "No response"}
            eval_attacked = evaluate_single_response(question, resp_attacked) if resp_attacked and not pd.isna(resp_attacked) else {"error": "No response"}
            
            # Store result
            results.append({
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
            row += 5  # Skip to next sample block (6 rows per sample)

    # Save results
    result_df = pd.DataFrame(results)
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='Evaluation_Results', index=False)
    
    print(f"\n✅ Done! Results saved to '{output_file}'")

if __name__ == "__main__":
    main()