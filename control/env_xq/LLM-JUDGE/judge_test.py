from openai import OpenAI

# 1. 配置 Gemini 客户端 (使用你代码中的配置)
client_gemini = OpenAI(
    api_key="sk-SSKJb4nDqFShEyNg88e4E1icSLRVsDGYudBSFsBeuAyfHDad", # 你的 Key
    base_url="https://api.zestai.pro/v1" # 你的 Base URL
)

def test_gemini_connection():
    print("🚀 正在测试 Gemini 连接...")
    print(f"   URL: https://api.zestai.pro/v1")
    print("-" * 50)
    
    try:
        # 2. 发送一个最简单的请求
        # 注意：虽然我们用的是 OpenAI SDK，但后端模型名需填对
        response = client_gemini.chat.completions.create(
            model="gemini-2.5-pro", # 确保模型名正确
            messages=[
                {"role": "user", "content": "你好，请用中文回复：测试连接成功！"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        # 3. 解析结果
        content = response.choices[0].message.content
        print("✅ 连接成功！收到回复：")
        print(content)
        
    except Exception as e:
        print("❌ 连接失败！错误信息：")
        print(str(e))

if __name__ == "__main__":
    test_gemini_connection()