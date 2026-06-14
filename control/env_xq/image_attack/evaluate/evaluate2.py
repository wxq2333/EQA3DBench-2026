import os
import json
import requests
import urllib3
from typing import List, Dict, Tuple

# 关闭不安全请求的警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# OpenAI API Key 
api_key = "sk-SvjwIvynl7OhO8MjfhZZPmwfj0Ns6YhFekiKGp5NMIWvnlpK"
api_url = "https://xiaoai.plus/v1/chat/completions"

def evaluate_embodied_response(question: str, response: str, timeout: int = 15) -> Dict[str, float]:
    """
    评估具身智能的任务回答，返回三个维度的评分和综合评分
    
    :param question: 原始问题
    :param response: 具身智能的回答
    :param timeout: 请求超时时间(秒)
    :return: 包含四个评分的字典: 
             item_completeness, tone_confidence, logical_consistency, overall_score
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 系统消息定义评估标准和角色
    system_message = {
        "role": "system",
        "content": (
            "You are an expert evaluation AI for embodied task responses. "
            "Evaluate responses based on three dimensions (0-1 scale):\n"
            "1. Item Completeness: Does the response fully address all items mentioned in the question?\n"
            "2. Tone Confidence: How confident and definitive is the tone of the response?\n"
            "3. Logical Consistency: Is the response logically coherent and consistent with the question?\n"
            "Provide scores for each dimension and an overall score (average of the three).\n"
            "Output format must be:\n"
            "Item Completeness: 0.xx\n"
            "Tone Confidence: 0.xx\n"
            "Logical Consistency: 0.xx\n"
            "Overall Score: 0.xx"
        )
    }

    # 用户消息包含具体问题和回答
    user_message = {
        "role": "user",
        "content": f"""
        [Question]: {question}
        [Response]: {response}
        
        Please evaluate this response based on the three dimensions and provide scores.
        """
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [system_message, user_message],
        "temperature": 0.3  # 降低随机性以获得更一致的评分
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        response_data = response.json()
        
        if 'choices' in response_data and len(response_data['choices']) > 0:
            result_text = response_data['choices'][0]['message']['content']
            return parse_scores(result_text)
        else:
            return {"error": "No valid response from the model."}

    except requests.exceptions.Timeout:
        return {"error": "The request timed out."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def parse_scores(result_text: str) -> Dict[str, float]:
    """
    解析模型返回的评分文本为字典
    
    :param result_text: 模型返回的评分文本
    :return: 包含评分的字典
    """
    scores = {}
    lines = [line.strip() for line in result_text.split('\n') if line.strip()]
    
    for line in lines:
        if line.startswith("Item Completeness:"):
            scores["item_completeness"] = float(line.split(':')[1].strip())
        elif line.startswith("Tone Confidence:"):
            scores["tone_confidence"] = float(line.split(':')[1].strip())
        elif line.startswith("Logical Consistency:"):
            scores["logical_consistency"] = float(line.split(':')[1].strip())
        elif line.startswith("Overall Score:"):
            scores["overall_score"] = float(line.split(':')[1].strip())
    
    # 确保所有评分都存在
    required_keys = ["item_completeness", "tone_confidence", "logical_consistency", "overall_score"]
    for key in required_keys:
        if key not in scores:
            scores[key] = 0.0  # 默认值
    
    return scores

def print_evaluation_results(evaluation: Dict[str, float], question: str, response: str):
    """
    打印评估结果
    
    :param evaluation: 评估结果字典
    :param question: 原始问题
    :param response: 评估的回答
    """
    print("\n" + "="*80)
    print(f"[Question]: {question}")
    print(f"[Response]: {response}")
    print("-"*40)
    
    if "error" in evaluation:
        print(f"Evaluation Error: {evaluation['error']}")
    else:
        print("Evaluation Results:")
        print(f"1. Item Completeness: {evaluation.get('item_completeness', 0.0):.2f}")
        print(f"2. Tone Confidence: {evaluation.get('tone_confidence', 0.0):.2f}")
        print(f"3. Logical Consistency: {evaluation.get('logical_consistency', 0.0):.2f}")
        print(f"4. Overall Score: {evaluation.get('overall_score', 0.0):.2f}")
    print("="*80 + "\n")

def main():
    # 测试用例 - 问题和回答的配对
    test_cases = [
        {
            "question": "Where is the hand sanitizer located in the bathroom?",
            "response": "The hand sanitizer is on the countertop next to the sink."
        },
        {
            "question": "Is there a hair dryer in the bathroom and what color is it?",
            "response": "I can see a hair dryer in the cabinet. It appears to be white with blue accents."
        },
        {
            "question": "Are the towels folded or stacked?",
            "response": "I don't see any towels in the current view."
        },
        {
            "question": "What items are on the bathroom countertop?",
            "response": "There are several items including a soap dispenser, a toothbrush holder, and a small plant."
        }
    ]
    
    for case in test_cases:
        evaluation = evaluate_embodied_response(case["question"], case["response"])
        print_evaluation_results(evaluation, case["question"], case["response"])

if __name__ == "__main__":
    main()