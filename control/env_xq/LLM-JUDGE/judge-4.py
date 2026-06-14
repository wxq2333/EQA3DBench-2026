#实现多个模型循环打分制度，三方达成共识后给出平均得分
"""
Circular Cross-Validation Evaluator (English Prompts)

Validation Chain:
  Gemini-2.5 → reviews Qwen-Max's evaluation
  GPT-4      → reviews Gemini-2.5's evaluation
  Qwen-Max   → reviews GPT-4's evaluation

Final consensus: average of the three corrected evaluations.
"""

import re
from typing import Dict, Optional
from openai import OpenAI

# ==============================
# API Clients (Replace with your own keys if needed)
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

# Model registry
MODEL_CLIENTS = {
    "Qwen-Max": (client_qwen, "qwen-max"),
    "Gemini-2.5": (client_gemini, "gemini-2.5-pro"),
    "GPT-4": (client_gpt, "gpt-4")
}

# Circular validation chain: (reviewer, target_to_review)
VERIFICATION_CHAIN = [
    ("Gemini-2.5", "Qwen-Max"),
    ("GPT-4", "Gemini-2.5"),
    ("Qwen-Max", "GPT-4")
]

# ==============================
# Prompt Templates (English)
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

def get_cross_validation_prompt(
    question: str,
    response: str,
    target_model: str,
    target_eval: dict
) -> str:
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
# Utility Functions
# ==============================
def parse_evaluation(text: str) -> Optional[Dict]:
    """Parse model output into structured scores."""
    try:
        fluency = int(re.search(r"Fluency:\s*(\d+)", text).group(1))
        accuracy = int(re.search(r"Accuracy:\s*(\d+)", text).group(1))
        relevance = int(re.search(r"Relevance:\s*(\d+)", text).group(1))
        overall = float(re.search(r"Overall Score:\s*([\d.]+)", text).group(1))
        justification_match = re.search(r"Justification:\s*(.+)", text, re.DOTALL)
        justification = justification_match.group(1).strip() if justification_match else ""
        return {
            "fluency": fluency,
            "accuracy": accuracy,
            "relevance": relevance,
            "overall": overall,
            "justification": justification,
            "raw": text
        }
    except Exception as e:
        print(f"⚠️ Parsing failed: {e}")
        return None

def call_model(client, model_name: str, messages: list, timeout: int = 30) -> Optional[str]:
    """Call model safely."""
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            timeout=timeout,
            max_tokens=350
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return None

# ==============================
# Phase 1: Initial Evaluations
# ==============================
def get_initial_evaluations(question: str, response: str) -> Dict[str, dict]:
    print("🔄 Phase 1: Collecting initial evaluations...")
    results = {}
    for name, (client, model) in MODEL_CLIENTS.items():
        print(f"  → {name} is evaluating...")
        messages = [
            {"role": "system", "content": get_initial_evaluation_prompt()},
            {"role": "user", "content": f"Question: {question}\n\nResponse: {response}"}
        ]
        output = call_model(client, model, messages)
        if output:
            print(output)
            parsed = parse_evaluation(output)
            if parsed:
                results[name] = parsed
    return results

# ==============================
# Phase 2: Circular Cross-Validation
# ==============================
def run_cross_validation(
    question: str,
    response: str,
    initial_evals: Dict[str, dict]
) -> Dict[str, dict]:
    """
    Each reviewer critiques a target's evaluation and outputs their own corrected scores.
    Returns a dict: {target_model: corrected_evaluation_by_reviewer}
    """
    print("\n🔄 Phase 2: Circular cross-validation...")
    corrected = {}

    for reviewer, target in VERIFICATION_CHAIN:
        if target not in initial_evals:
            print(f"  ⚠️ Skipping {reviewer} → {target} (target missing)")
            continue

        reviewer_client, reviewer_model = MODEL_CLIENTS[reviewer]
        target_eval = initial_evals[target]

        prompt = get_cross_validation_prompt(
            question=question,
            response=response,
            target_model=target,
            target_eval=target_eval
        )

        print(f"\n  🔍 {reviewer} is reviewing {target}'s evaluation...")
        messages = [{"role": "user", "content": prompt}]
        output = call_model(reviewer_client, reviewer_model, messages, timeout=40)

        if output:
            print(output)
            corrected_eval = parse_evaluation(output)
            if corrected_eval:
                # Store under target name: this is the reviewer's view of what the score *should be*
                corrected[target] = corrected_eval
            else:
                print(f"    ❌ Failed to parse {reviewer}'s review")
        else:
            print(f"    ❌ {reviewer} call failed")

    return corrected

# ==============================
# Phase 3: Compute Final Consensus
# ==============================
def compute_consensus(corrected_evals: Dict[str, dict]) -> Optional[Dict]:
    if not corrected_evals:
        return None

    n = len(corrected_evals)
    f = sum(e["fluency"] for e in corrected_evals.values()) / n
    a = sum(e["accuracy"] for e in corrected_evals.values()) / n
    r = sum(e["relevance"] for e in corrected_evals.values()) / n
    overall = round((f + a + r) / 3, 1)

    # Combine justifications (truncate for readability)
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
# Main Execution
# ==============================
def main():
    # 🧪 Test case — modify as needed
    question = "What is the capital of France?"
    response = "The capital of France is Paris. It is known for the Eiffel Tower and its rich cultural history."

    print("=" * 70)
    print("🔁 Circular Cross-Validation Evaluator (English Prompts)")
    print(f"Question: {question}")
    print(f"Response: {response}")
    print("=" * 70)

    # Phase 1
    initial = get_initial_evaluations(question, response)
    if len(initial) < 2:
        print("❌ Not enough initial evaluations. Exiting.")
        return

    # Phase 2
    corrected = run_cross_validation(question, response, initial)

    # Phase 3
    consensus = compute_consensus(corrected)

    # Output final result
    print("\n" + "=" * 70)
    print("✅ FINAL CONSENSUS SCORE (After Circular Cross-Validation)")
    print("=" * 70)
    if consensus:
        print(f"Fluency:    {consensus['fluency']}")
        print(f"Accuracy:   {consensus['accuracy']}")
        print(f"Relevance:  {consensus['relevance']}")
        print(f"Overall:    {consensus['overall']}")
        print(f"\nJustification: {consensus['justification']}")
    else:
        print("⚠️ Could not generate consensus.")

if __name__ == "__main__":
    main()