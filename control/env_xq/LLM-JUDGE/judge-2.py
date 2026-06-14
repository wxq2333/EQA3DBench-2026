#利用LLM作为评判回答是否合理的评判器，1.0版本，多个llm共同作用来进行评判  
#分为三个阶段：1.创建；2.审查；3.修订
#参考论文：Towards Reasoning in Large Language Models via Multi-Agent Peer Review Collaboration



#step1.创建第一阶段：创建。 同行评审过程始于每个代理提交自己的解决方案。 具体来说，当面对一个问题，三个代理分别生成问题的回答
#选用三种不同的大模型作为评判代理，能够发挥不同模型的长处千问，GPT4o，gemini
import os
from openai import OpenAI

# #qwen-max
# client = OpenAI(
#     api_key="sk-0a46c0017e094c3ea61c1f15dbdc5bbc",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# completion = client.chat.completions.create(
#     model="qwen3-max",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "你是谁？"},
#     ],
#     stream=True
# )
# for chunk in completion:
#     print(chunk.choices[0].delta.content, end="", flush=True)

# #gemini-2.5 
# client = OpenAI(
#     api_key="sk-IAg99sNg9pTZ4wY5aGjL9QFx6HTMa6gCAovFhsdhOLnrBQbr",
#     base_url="https://tb.api.mkeai.com/v1",
# )
# completion = client.chat.completions.create(
#     model="gemini-2.5-pro",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "你是谁？"},
#     ],
#     stream=True
# )
# for chunk in completion:
#     print(chunk.choices[0].delta.content, end="", flush=True)


#gpt-4
client = OpenAI(
    api_key="sk-oviGSMINj4x6sn0PN55cLwqlPuvjafsCEacYUblju6en8OPG",
    base_url="https://api.n1n.ai/v1",
)

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    stream=True
) 

for chunk in stream:
    if chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            print(delta.content, end="", flush=True)
print()