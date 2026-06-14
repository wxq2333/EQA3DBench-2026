import os
import json
import requests
import urllib3

# 关闭不安全请求的警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 你的 OpenAI API Key
api_key = "sk-SvjwIvynl7OhO8MjfhZZPmwfj0Ns6YhFekiKGp5NMIWvnlpK"

def process_text_similarity(text_a, text_b, timeout=10):
    """
    将两段文本 text_a 与 text_b 提交给远端模型进行相似度评分。
    返回模型回答的字符串，可包含语义相似度、逻辑相似度等。
    
    :param timeout: 请求超时的时间，单位是秒
    :return: 模型的响应结果（JSON 格式）
    """
    url = "https://xiaoai.plus/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 在 system 中给模型指定一个角色，比如“评分专家”
    system_message = {
        "role": "system",
        "content": (
            "You are an AI expert in text evaluation. "
            "You can analyze two pieces of text and provide a similarity score. "
            "Focus on both semantic similarity (meaning, topics, expressions) and logical similarity (cause-effect, reasoning)."
        )
    }

    # 在 user 消息里，给模型具体指令，告诉它你要什么形式的输出
    user_message = {
        "role": "user",
        "content": f"""
            Please compare the following two texts and give scores on two dimensions (scores range from 0 to 1; the higher the score, the more similar they are):
            Semantic similarity
            Logical similarity
            Text A: {text_a}
            Text B: {text_b}
            Finally, please only output in the following format:
            Semantic similarity: 0.85
            Logical similarity: 0.90
        """
    }

    # 将 system 消息和 user 消息组合成请求的消息体
    messages = [system_message, user_message]

    # 请求体
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages
    }

    try:
        # 发起POST请求，并设置超时
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)

        # 检查请求是否成功
        response.raise_for_status()

        # 解析返回的JSON
        response_data = response.json()

        # 获取模型的输出
        if 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content']
        else:
            return "Error: No valid response from the model."

    except requests.exceptions.Timeout:
        return "Error: The request timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def main():
    text_pairs = [
    
    ("The hand sanitizer is located on the countertop of the bathroom vanity, positioned among other toiletries", "In the images provided, I do not see any hand sanitizer visible in the bathroom. The surfaces of the countertop appear to have various items, such as a bathroom tissue roll, a small container, and a few other objects, but nothing that indicates the presence of hand sanitizer. If you're looking for it, I recommend checking other areas of the bathroom or storage spaces where it might be kept."),
    ("The hair dryer is located on the countertop near the sink area.", "I don’t see a hair dryer in the images of the bathroom. You might want to check the storage cabinets or drawers for it, as it may not be visible on the countertops. If you have more images or a specific location in mind, feel free to share!"),
    ("The hair dryer in the images appears to be predominantly black, with some red accents.", "I can't see the hair dryer in the provided images, so I'm unable to determine its color. If you have more specific images or details, feel free to share!"),
    ("The towels appear to be rolled or stacked but not distinctly folded in a traditional manner.", "The images do not show any towels, so I cannot determine if they are folded or not. If you have specific images of towels, please provide them for a more accurate evaluation."),
    ("The towel in the bathroom serves a practical purpose, primarily for drying hands or body after washing. It is commonly used to maintain hygiene and comfort in a bathroom setting. Additionally, towels can contribute to the overall aesthetic of the bathroom, depending on their color and material.", "Based on the images provided, the towel in the bathroom serves a practical purpose: it is likely used for drying off after washing hands or taking a shower. Towels help maintain hygiene and comfort by removing moisture from the skin. They are typically placed within easy reach for convenience, contributing to the overall functionality of the bathroom space."),
]







    
    for text_a, text_b in text_pairs:
        print(f"Processing pair: {text_a} | {text_b}")
        result = process_text_similarity(text_a, text_b)
        print(f"Result: {result}\n")

if __name__ == "__main__":
    main()
