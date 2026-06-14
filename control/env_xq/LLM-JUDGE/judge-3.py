#这个部分三个模型可以独立地给出评分，针对流畅度，准确度，相关性给出评分，以及最终得分
import os
from openai import OpenAI

# ==============================
# Step 1: 定义评分提示词（不变）
# ==============================
def get_evaluator_prompt():
    return """
You are an expert evaluator tasked with assessing the quality of a response to a given question. Please evaluate the provided response based on the following three criteria:

1. **Fluency**: How clear, coherent, grammatically correct, and natural-sounding is the response?
2. **Accuracy**: How factually correct, precise, and free from errors or hallucinations is the response?
3. **Relevance**: How well does the response address the specific question asked, without unnecessary digressions?

For each criterion, assign an integer score from 1 to 5, where:
- 1 = Very Poor
- 2 = Poor
- 3 = Fair
- 4 = Good
- 5 = Excellent

After scoring each dimension, provide a **final overall score** (also on a 1–5 scale), calculated as the average of the three scores rounded to one decimal place.

Finally, write a concise justification (2–3 sentences) explaining your evaluation.

Format your response exactly as follows:

Fluency: [score]  
Accuracy: [score]  
Relevance: [score]  
Overall Score: [X.X]  
Justification: [your explanation]
"""

# ==============================
# Step 2: 创建各模型的 Client
# ==============================
# Qwen-Max (DashScope)
client_qwen = OpenAI(
    api_key="sk-0a46c0017e094c3ea61c1f15dbdc5bbc",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Gemini-2.5 (via third-party proxy)
client_gemini = OpenAI(
    api_key="sk-IAg99sNg9pTZ4wY5aGjL9QFx6HTMa6gCAovFhsdhOLnrBQbr",
    base_url="https://tb.api.mkeai.com/v1",
)

# GPT-4 (via n1n.ai proxy)
client_gpt = OpenAI(
    api_key="sk-oviGSMINj4x6sn0PN55cLwqlPuvjafsCEacYUblju6en8OPG",
    base_url="https://api.n1n.ai/v1",
)

# ==============================
# Step 3: 封装评分函数
# ==============================
def evaluate_with_model(client, model_name, question, response, model_label):
    print(f"\n{'='*60}")
    print(f"Evaluating with {model_label} ({model_name})")
    print(f"{'='*60}")
    
    evaluator_prompt = get_evaluator_prompt()
    messages = [
        {"role": "system", "content": evaluator_prompt},
        {"role": "user", "content": f"Question: {question}\n\nResponse: {response}"}
    ]
    
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            timeout=30  # 可选：防止卡死
        )
        
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    print(delta.content, end="", flush=True)
        print()  # 换行
        
    except Exception as e:
        print(f"❌ Error during evaluation with {model_label}: {e}")

# ==============================
# Step 4: 主程序 - 输入问题与回答
# ==============================
if __name__ == "__main__":
    # 你可以替换成任意问题和模型生成的回答
    question = "What is the capital of France?"
    response_to_evaluate = "The capital of France is Paris. It is known for the Eiffel Tower and its rich cultural history."

    # 调用三个模型进行评分
    evaluate_with_model(client_qwen, "qwen-max", question, response_to_evaluate, "Qwen-Max")
    evaluate_with_model(client_gemini, "gemini-2.5-pro", question, response_to_evaluate, "Gemini-2.5 (Proxy)")
    evaluate_with_model(client_gpt, "gpt-4", question, response_to_evaluate, "GPT-4 (Proxy)")