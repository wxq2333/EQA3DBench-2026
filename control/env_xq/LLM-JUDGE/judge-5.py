import pandas as pd
import re
import time
from typing import Dict, Optional
from openai import OpenAI

# ==============================
# API Clients (Keep your keys)
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
    api_key="sk-oviGSMINj4x6sn0PN55cLwqlPuvjafsCEacYUblju6en8OPG",
    base_url="https://api.n1n.ai/v1",
)

MODEL_CLIENTS = {
    "Qwen-Max": (client_qwen, "qwen-max"),
    "Gemini-2.5": (client_gemini, "gemini-2.5-pro"),
    "GPT-4": (client_gpt, "gpt-4")
}

VERIFICATION_CHAIN = [
    ("Gemini-2.5", "Qwen-Max"),
    ("GPT-4", "Gemini-2.5"),
    ("Qwen-Max", "GPT-4")
]

CATEGORIES = ["物品定位", "空间推理", "属性识别", "状态识别", "功能推理"]

# ==============================
# Prompts (same as yours)
# ==============================
def get_initial_evaluation_prompt() -> str:
    return """
You are an expert AI response evaluator. Your task is to assess the quality of a given response to a specific question based on three criteria:

1. **Fluency**: Is the response clear, grammatically correct, coherent, and natural-sounding?
2. **Accuracy**: Is the response factually correct, precise, and free from hallucinations or errors?
3. **Relevance**: Does the response directly address the question without unnecessary digressions?

Assign an integer score from 1 to 5 for each criterion:
- 1 = Very Poor
- 2 = Poor
- 3 = Fair
- 4 = Good
- 5 = Excellent

Then compute an **Overall Score** as the average of the three scores, rounded to one decimal place.

Finally, provide a concise justification (2–3 sentences) explaining your evaluation.

Output your response in the following exact format—do not add any extra text:

Fluency: [score]  
Accuracy: [score]  
Relevance: [score]  
Overall Score: [X.X]  
Justification: [your justification]
""".strip()

def get_cross_validation_prompt(question: str, response: str, target_model: str, target_eval: dict) -> str:
    return f"""
You are a senior AI evaluation auditor. Your job is to critically review another expert model's assessment of a response and determine whether it is fair, accurate, and well-reasoned.

**Original Question:**  
{question}

**Response Being Evaluated:**  
{response}

**Evaluation to Review (from {target_model}):**  
Fluency: {target_eval['fluency']}  
Accuracy: {target_eval['accuracy']}  
Relevance: {target_eval['relevance']}  
Overall Score: {target_eval['overall']}  
Justification: {target_eval['justification']}

Your task:
1. Analyze whether the above evaluation is justified by the response.
2. Identify any potential over-scoring, under-scoring, factual misunderstandings, or logical flaws.
3. Provide your own corrected evaluation of the response—not of the evaluator, but of the original response itself—based on your independent judgment.

Even if you largely agree with the original scores, you must still output a full evaluation in the required format.

Output your response in the following exact format—no additional commentary:

Fluency: [score]  
Accuracy: [score]  
Relevance: [score]  
Overall Score: [X.X]  
Justification: [2–3 sentences explaining your reasoning and any adjustments]
""".strip()

# ==============================
# Utilities
# ==============================
def parse_evaluation(text: str) -> Optional[Dict]:
    if not text or not isinstance(text, str):
        print("⚠️ Empty or invalid model output.")
        return None

    # Debug: print raw output
    print("📄 Raw model output:")
    print(repr(text[:500]))  # Show first 500 chars with escapes

    try:
        # Case-insensitive, flexible spacing, allow prefixes/suffixes
        fluency_match = re.search(r"fluency\s*[:：]\s*(\d)", text, re.IGNORECASE)
        accuracy_match = re.search(r"accuracy\s*[:：]\s*(\d)", text, re.IGNORECASE)
        relevance_match = re.search(r"relevance\s*[:：]\s*(\d)", text, re.IGNORECASE)
        overall_match = re.search(r"overall\s*score\s*[:：]\s*([\d.]+)", text, re.IGNORECASE)

        if not all([fluency_match, accuracy_match, relevance_match, overall_match]):
            missing = []
            if not fluency_match: missing.append("Fluency")
            if not accuracy_match: missing.append("Accuracy")
            if not relevance_match: missing.append("Relevance")
            if not overall_match: missing.append("Overall Score")
            print(f"⚠️ Missing fields in evaluation: {missing}")
            return None

        fluency = int(fluency_match.group(1))
        accuracy = int(accuracy_match.group(1))
        relevance = int(relevance_match.group(1))
        overall = float(overall_match.group(1))

        # Extract justification (from "Justification:" to end, case-insensitive)
        just_match = re.search(r"justification\s*[:：]?\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        justification = just_match.group(1).strip() if just_match else ""

        # Clean up justification (remove trailing newlines, extra spaces)
        justification = justification.split("\n")[0][:200]  # Take first line, max 200 chars

        return {
            "fluency": fluency,
            "accuracy": accuracy,
            "relevance": relevance,
            "overall": overall,
            "justification": justification,
            "raw": text
        }
    except Exception as e:
        print(f"⚠️ Parsing error: {e}")
        return None
 
def call_model(client, model_name: str, messages: list, timeout: int = 30) -> Optional[str]:
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            timeout=timeout,
            max_tokens=600
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return None

# ==============================
# Evaluation Pipeline (Same as Yours)
# ==============================
def get_initial_evaluations(question: str, response: str) -> Dict[str, dict]:
    results = {}
    for name, (client, model) in MODEL_CLIENTS.items():
        messages = [
            {"role": "system", "content": get_initial_evaluation_prompt()},
            {"role": "user", "content": f"Question: {question}\n\nResponse: {response}"}
        ]
        output = call_model(client, model, messages)
        if output:
            parsed = parse_evaluation(output)
            if parsed:
                results[name] = parsed
    return results

def run_cross_validation(question: str, response: str, initial_evals: Dict[str, dict]) -> Dict[str, dict]:
    corrected = {}
    for reviewer, target in VERIFICATION_CHAIN:
        if target not in initial_evals:
            continue
        reviewer_client, reviewer_model = MODEL_CLIENTS[reviewer]
        target_eval = initial_evals[target]
        prompt = get_cross_validation_prompt(question, response, target, target_eval)
        messages = [{"role": "user", "content": prompt}]
        output = call_model(reviewer_client, reviewer_model, messages, timeout=40)
        if output:
            corrected_eval = parse_evaluation(output)
            if corrected_eval:
                corrected[target] = corrected_eval
    return corrected

def compute_consensus(corrected_evals: Dict[str, dict]) -> Optional[Dict]:
    if not corrected_evals:
        return None
    n = len(corrected_evals)
    f = sum(e["fluency"] for e in corrected_evals.values()) / n
    a = sum(e["accuracy"] for e in corrected_evals.values()) / n
    r = sum(e["relevance"] for e in corrected_evals.values()) / n
    overall = round((f + a + r) / 3, 1)
    justifications = [f"[{model}] {e['justification']}" for model, e in corrected_evals.items()]
    combined_just = " | ".join(just[:120] for just in justifications)
    return {
        "fluency": round(f, 1),
        "accuracy": round(a, 1),
        "relevance": round(r, 1),
        "overall": overall,
        "justification": combined_just
    }

# ==============================
# Main Function (Adapted to Your Table Structure)
# ==============================
def main():
    input_file = "/home/ubuntu/桌面/datasets/data1.xlsx"
    output_file = "/home/ubuntu/桌面/datasets/data_out.xlsx"

    df = pd.read_excel(input_file, header=None)  # No header assumed
    print(f"📊 Loaded {len(df)} rows (headerless).")

    # Add column names for clarity (optional, not used in logic)
    # We assume data starts from row 0

    # Initialize result columns (we'll store in new DataFrame or extend)
    result_cols = []
    for cat in CATEGORIES:
        for metric in ["attack_fluency", "attack_accuracy", "attack_relevance", "attack_overall", "attack_justification"]:
            result_cols.append(f"{cat}_{metric}")

    # Create a results dictionary keyed by (group_index, category_index)
    # But easier: process every 3 rows per category within each 15-row block
    total_rows = len(df)
    group_size = 15  # 5 categories × 3 rows (question, initial, attacked)

    # Prepare output structure
    output_data = []

    for start in range(0, total_rows, group_size):
        group = df.iloc[start:start+group_size]
        if len(group) < 15:
            print(f"⚠️ Incomplete group at {start}, skipping.")
            continue

        group_results = {}
        valid = True

        for i, cat in enumerate(CATEGORIES):
            q_row = group.iloc[i * 3 + 0]      # question
            attacked_row = group.iloc[i * 3 + 2]  # 攻击后回答

            # Assume first column contains the content
            question = str(q_row.iloc[0]).strip() if pd.notna(q_row.iloc[0]) else ""
            attacked_resp = str(attacked_row.iloc[0]).strip() if pd.notna(attacked_row.iloc[0]) else ""

            if not question or attacked_resp in ["", "nan"]:
                print(f"  ⚠️ Missing data for {cat} in group {start//15}")
                valid = False
                break

            # Check if already scored (we’ll store scores in a separate dict for now)
            # For simplicity, always re-score (or you can save to file later)

            print(f"\n[{start//15}] Evaluating {cat}...")
            print(f"Q: {question[:50]}...")

            # Run full consensus pipeline
            initial = get_initial_evaluations(question, attacked_resp)
            if len(initial) < 2:
                print(f"  ⚠️ Initial eval failed for {cat}")
                valid = False
                break

            corrected = run_cross_validation(question, attacked_resp, initial)
            if not corrected:
                print(f"  ⚠️ Cross-validation failed for {cat}")
                valid = False
                break

            consensus = compute_consensus(corrected)
            if not consensus:
                print(f"  ⚠️ Consensus failed for {cat}")
                valid = False
                break

            group_results[cat] = consensus
            time.sleep(1)  # Avoid rate limits

        if valid:
            # Flatten results into a row
            flat_row = {}
            for cat in CATEGORIES:
                res = group_results[cat]
                flat_row[f"{cat}_attack_fluency"] = res["fluency"]
                flat_row[f"{cat}_attack_accuracy"] = res["accuracy"]
                flat_row[f"{cat}_attack_relevance"] = res["relevance"]
                flat_row[f"{cat}_attack_overall"] = res["overall"]
                flat_row[f"{cat}_attack_justification"] = res["justification"]
            output_data.append(flat_row)
        else:
            # Append empty row to keep alignment
            empty_row = {col: None for col in result_cols}
            output_data.append(empty_row)

        # Save progress periodically
        if (start // group_size + 1) % 2 == 0:
            result_df = pd.DataFrame(output_data)
            with pd.ExcelWriter(output_file) as writer:
                df.to_excel(writer, sheet_name="Raw Data", index=False, header=False)
                result_df.to_excel(writer, sheet_name="Scores", index=False)
            print(f"💾 Progress saved to {output_file}")

    # Final save
    result_df = pd.DataFrame(output_data)
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name="Raw Data", index=False, header=False)
        result_df.to_excel(writer, sheet_name="Scores", index=False)
    print(f"\n🎉 All done! Results in '{output_file}' (Sheet: 'Scores')")

if __name__ == "__main__":
    main()