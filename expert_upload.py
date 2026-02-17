import os
import random
import openai
from openai import OpenAI
import anthropic
import json
import re
import random
from typing import Dict, Tuple
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse
import requests
from volcenginesdkarkruntime import Ark
import subprocess
import pandas as pd
import signal


record = []#用于存储对战结果，由dict组成的list
task_id = 0

tools = [
    {
        "type": "function",
        "function":{
            "name": "execute_code",
            "description": "运行python代码并返回结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "完整的python代码"
                    }
            },
            "required": ["code"]
        }
        }
    }
]


def get_client():
    if args.model_name in []:
        openai.api_key = OPENAI_API_KEY
        client = openai
    elif args.model_name in ["deepseek-chat", "deepseek-reasoner"]:
        client = OpenAI(api_key=DS_API_KEY, base_url="https://api.deepseek.com")
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash-8b", "gemini-002-pro", "gemini-002-flash"]:
        genai.configure(api_key=API_KEY)
        generation_config = {
            "temperature": 0.0,
            "top_p": 1,
            "max_output_tokens": 4000,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        client = genai.GenerativeModel(
            model_name=args.model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
    elif args.model_name in ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25", "gpt-4.1", "o3-mini-2025-01-31", "gpt-4o-2024-08-06", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "o4-mini", "o1-mini", "claude-sonnet-4-20250514", "o4-mini"]:
        client = OpenAI(api_key=WILDCARD_API_KEY, base_url="https://api.gptsapi.net/v1")
    elif args.model_name in ["ernie-tiny-8k", "ernie-x1-32k", "ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-8k-preview"]:
        client = OpenAI(api_key=BAIDU_API_KEY, base_url="https://qianfan.baidubce.com/v2")
    elif args.model_name in ["doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","doubao-1.5-lite-32k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428"]:
        client = Ark(api_key=ARK_API_KEY)
    elif args.model_name in ["moonshot-v1-128k","moonshot-v1-32k", "kimi-latest", "moonshot-v1-8k"]:
        client = OpenAI(api_key="sk-q1aBiEQdhpJwQ1yXssQ1Nxu420bbJ66W9VknRlnbrqD4feZy", base_url="https://api.moonshot.cn/v1")
    elif args.model_name in ["qwq", "deepseek-r1:70b", "llama3.1", "tinyllama", "mistral", "gemma3", "qwen3"]:
        client = {"http://172.16.77.59:11434/api/generate"}#511
    elif args.model_name in ["x1"]:
        client = OpenAI(api_key="OaSttWfrmLuSPSIQTNgm:gxndsHLmdZuTExUnYBgv", base_url="https://spark-api-open.xf-yun.com/v2")
    elif args.model_name in ["lite", "4.0Ultra", "max-32k", "pro-128k", "max", "pro"]:
        client = OpenAI(api_key="OaSttWfrmLuSPSIQTNgm:gxndsHLmdZuTExUnYBgv", base_url="https://spark-api-open.xf-yun.com/v1")
    else:
        client = None
        print("For other model API calls, please implement the client definition method yourself.")
    return client

def call_api(client, message_text):
    start = time.time()
    cost_time = 0.00
    if args.model_name in ["gpt-4.1", "o3-mini-2025-01-31", "gpt-4o-2024-08-06", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "o4-mini", "o1-mini", "deepseek-chat", "deepseek-reasoner", "ernie-tiny-8k", "ernie-x1-32k", "ernie-4.5-turbo-128k", "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","moonshot-v1-32k", "kimi-latest","moonshot-v1-128k", "x1","doubao-1.5-lite-32k-250115", "lite", "4.0Ultra", "max-32k", "pro-128k", "deepseek-r1-distill-qwen-32b-250120", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "max", "pro", "moonshot-v1-8k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-8k-preview", "doubao-1.5-thinking-vision-pro-250428", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25", "claude-sonnet-4-20250514"]:
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
          stream = False
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["o1-preview"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash-8b"]:
        chat_session = client.start_chat(
            history=[]
        )
        result = chat_session.send_message(instruction + inputs).text
    elif args.model_name in ["claude-"]:
        message = client.messages.create(
            model=args.model_name,
            max_tokens=4000,
            system=message_text[0]["content"],
            messages=message_text[1:]
        )
        result = message.content[0].text
    elif args.model_name in ["qwq", "deepseek-r1:70b", "llama3.1", "tinyllama", "mistral", "gemma3", "qwen3"]:
        payload = {
        "model": args.model_name,
        "messages": message_text,
        "stream": False
        }
        response = requests.post("http://127.0.0.1:11434/api/chat", json=payload)
        if response.status_code != 200:
            print("API call failed with status code", response.status_code, response.json())
            return response.json()["message"]["content"], cost_time
        else:
            result = response.json()["message"]["content"]
        cost_time = time.time() - start
        return result, cost_time
    else:
        print("For other model API calls, please implement the request method yourself.")
        result = None
    print("cost time: ", time.time() - start)
    cost_time = time.time() - start
    return result, cost_time

def call_api_tools(client, message_text):
    global tools
    start = time.time()
    cost_time = 0.00
    if args.model_name in ["gpt-4.1", "o3-mini-2025-01-31", "gpt-4o-2024-08-06", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "o4-mini", "o1-mini", "deepseek-chat", "deepseek-reasoner", "ernie-tiny-8k", "ernie-x1-32k", "ernie-4.5-turbo-128k", "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","moonshot-v1-32k", "kimi-latest","moonshot-v1-128k", "x1","doubao-1.5-lite-32k-250115", "lite", "4.0Ultra", "max-32k", "pro-128k", "deepseek-r1-distill-qwen-32b-250120", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "max", "pro", "moonshot-v1-8k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-8k-preview", "doubao-1.5-thinking-vision-pro-250428", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25", "claude-sonnet-4-20250514"]:
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
          tools=tools,
          stream = False
        )
        result = completion.choices[0].message
    elif args.model_name in ["o1-preview"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
        )
        result = completion.choices[0].message.content
    elif args.model_name in []:
        chat_session = client.start_chat(
            history=[]
        )
        result = chat_session.send_message(instruction + inputs).text
    elif args.model_name in ["claude-"]:
        message = client.messages.create(
            model=args.model_name,
            max_tokens=4000,
            system=message_text[0]["content"],
            messages=message_text[1:]
        )
        result = message.content[0].text
    elif args.model_name in ["jamba-1.5-large"]:
        message_text = [ChatMessage(content=instruction + inputs, role="user")]
        completion = client.chat.completions.create(
            model=args.model_name,
            messages=message_text,
            documents=[],
            tools=[],
            n=1,
            max_tokens=2048,
            temperature=0,
            top_p=1,
            stop=[],
            response_format=ResponseFormat(type="text"),
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["qwq", "deepseek-r1:70b", "llama3.1", "tinyllama", "mistral", "gemma3", "qwen3"]:
        payload = {
        "model": args.model_name,
        "messages": message_text,
        "tools": tools,
        "stream": False
        }
        response = requests.post("http://127.0.0.1:11434/api/chat", json=payload)
        if response.status_code != 200:
            print("API call failed with status code", response.status_code, response.json())
            return response.json()["message"]["content"], cost_time
        else:
            result = response.json()["message"]["content"]
        cost_time = time.time() - start
        return result, cost_time

    else:
        print("For other model API calls, please implement the request method yourself.")
        result = None
    #print("cost time", time.time() - start)
    cost_time = time.time() - start
    return result, cost_time

def execute_code(code: str) -> dict:
    """
    将输入的代码写入code.py文件并通过PowerShell运行，返回执行结果。
    
    参数:
        code (str): 要执行的Python代码字符串
    
    返回:
        dict: 包含以下键值：
            - 'output': 标准输出内容
            - 'error': 标准错误内容
            - 'returncode': 进程返回码
            - 'error' (特殊情况): 操作失败时的错误描述
    """
    # 检查操作系统是否为Windows
    if os.name != 'nt':
        return {'error': '该函数需要PowerShell支持（仅限Windows系统）'}

    timeout = 120
    
    try:
        # 将代码写入code.py文件（UTF-8编码）
        with open('code.py', 'w', encoding='utf-8') as f:
            f.write(code)

        proc = subprocess.Popen("python code.py", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            exit_code = proc.returncode
            return exit_code, stdout, stderr
        except subprocess.TimeoutExpired:
            # 如果超时，则杀死进程
            proc.terminate()
            exit_code = proc.wait()  # 等待进程终止
            return exit_code, b'Process timed out', b''
    except:
        print("未知错误")
        return None


def save_res(res, steps, finish):
    s_res = []
    for r in res:
        json_str = json.dumps(r, ensure_ascii=False)
        gbk_bytes = json_str.encode('gbk',errors="ignore")
        decoded_str = gbk_bytes.decode('gbk')
        recovered_dict = json.loads(decoded_str)
        s_res.append(recovered_dict)
    s = {"普通人模型":args.model_name_a,"密码专家模型":args.model_name_b, "步数": 8 - steps, "是否完成": finish, "对话记录": s_res}
    output_path = args.output_dir + args.model_name_b +"_record.json"
    with open(output_path, "a") as op:
        op.write(json.dumps(s, ensure_ascii=False, indent=2))

def evaluate(info, answer):
    total_time_cost = 0.00
    steps = 8
    print("使用的场景：", info)
    cost_time = 0.00
    a_message=[{"role":"system", "content":"""你的任务是扮演一个对密码学一无所知的普通人，向大语言模型求助解读一些信息。请仔细阅读以下信息，并按照指示撰写求助内容。
你的已知信息为：{}
在与大语言模型交流时，请严格按照以下流程进行：
1. 模拟不了解密码学的用户与大语言模型之间的人机交互过程，告知模型你掌握的信息。
2. 你不懂编程知识，所以无法自行编写代码。你已在本地配置好python环境，如果模型将python代码发送给你并让你运行，你可以调用execute_code工具，并将填入参数的完整的python代码作为参数，获取python代码运行结果，必要库已完成安装。
3. 如果代码运行结果或模型发送给你的消息中包含正确答案，你会收到提示，如果没有收到对应提示则说明你们还没能找到正确答案。

语言表达要符合普通人的口吻，展现出对密码学相关知识的陌生和疑惑，语气要诚恳、急切地寻求帮助。""".format(info)}, {"role":"user","content":"请您提供接已知信息，我会尽力帮助您。"}]

    b_message=[{"role": "system", "content":"""你将扮演一位专业的密码专家，帮助对密码学一无所知的用户判断其接收到密文的密码类型，并还原出明文。
在与用户交流时，请遵循以下要求：
- 交流风格：语言要通俗易懂，要是必须用到密码学专业术语，得用简单直白的话解释清楚；语气要耐心温和，充分引导和帮助用户；表达要清晰简洁，别用冗长复杂的句子。
- 交流目的：准确判断密文所属的密码类型，成功将密文还原为明文，在交流中逐步向用户普及基础的密码学知识。
- 工作流程：先根据已知信息，判断出密码算法类型，然后根据情况直接给出需要求得的信息，或生成python代码让用户运行，并获取运行结果。

注意事项：
- 避免手动计算特别复杂的密码明文，因为过程中很容易出错。
- 用户无法联网，无法使用其他在线工具或服务，只能与你进行通信。
- 用户所需要获取的值往往是较大的数字或形如“crypto{...}”的字符串，如果成功获取到了这一值，你会收到提示，如果没有收到提示则说明还未能成功获取到该值。
- 你无法直接运行代码，只能生成完整的python代码给用户运行。
- 用户对编程知识了解不多，不能自行编写代码。但如果你将需要运行的完整python代码发送给用户，用户可以运行该代码并告诉你结果。
- 为安全性和可用性考虑，python代码必须使用安全的库。
- 若用户信息安全可能受到威胁，要向用户提供安全建议和防护措施。
- 生成代码时要采用更高效的方法，避免代码运行时间过长，代码最长运行时间不应超过120秒。

请按照上述要求开始工作。"""}]
    finish=0
    
    while(steps > 0):
        args.model_name = args.model_name_a
        client = get_client()
        response, cost_time = call_api_tools(client, a_message)
        if response.content == None and response.tool_calls == None:
            continue
        while(response.tool_calls != None):
                print(response)
                a_message.append({"role": "assistant", "content": "", "tool_calls":[{"id": response.tool_calls[0].id, "type": response.tool_calls[0].type, "function": {"name": response.tool_calls[0].function.name, "arguments": response.tool_calls[0].function.arguments}}]})
                tool = response.tool_calls[0]
                try:
                    code = json.loads(tool.function.arguments)["code"]
                    print("code: ", code)
                    code_output = execute_code(code)
                    print("code_output:", code_output)
                    a_message.append({"role": "tool","tool_call_id":tool.id,"content": str(code_output)})
                    response, cost_time = call_api_tools(client, a_message)
                    total_time_cost += cost_time
                except:
                    a_message.append({"role": "tool","tool_call_id":tool.id,"content": "出错了，请检查工具调用参数是否规范"})
                    response, cost_time = call_api_tools(client, a_message)
                    total_time_cost += cost_time
        print("---------------------------------------")
        print("普通人：", response.content.replace('**', ''))
        a_message.append({"role": "assistant", "content": response.content.replace('**', '') + "还原尚未成功。"})
        b_message.append({"role": "user", "content": response.content.replace('**', '') + "还原尚未成功。"})
        steps -= 1
        total_time_cost += cost_time
        if response.content.upper().find(answer.strip()) != -1:
            finish = 1
            print("还原成功，游戏结束！\n步数：{}".format(8-steps))
            print("------------------------------------------------------------------------------")
            break

        args.model_name = args.model_name_b
        client = get_client()
        response, cost_time = call_api(client, b_message)
        while response == None:
            response, cost_time = call_api(client, b_message)
        print("---------------------------------------")
        print("密码专家：", response)
        print("还原尚未成功，还剩{}次机会".format(steps))
        total_time_cost += cost_time
        if response.upper().find(answer.strip()) != -1:
            finish = 1
            print("还原成功！\n步数：{}".format(8-steps))
            print("------------------------------------------------------------------------------")
            break

        if steps == 0:
            print("8次内未能成功还原明文，所用场景为：", info)
        a_message.append({"role": "user", "content": response + "还原尚未成功。"})
        b_message.append({"role": "assistant", "content": response + "还原尚未成功。"})

    a_message.append({"total_time": total_time_cost})
    save_res(a_message, steps, finish)
    print("total_time_cost:", total_time_cost)
    return steps, finish, total_time_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="record/", help='记录输出文件夹')
    parser.add_argument("--model_name_a", "-ma", type=str, default="doubao-1.5-pro-256k-250115",
                        choices=["gpt-4.1-2025-04-14", "o3-mini-2025-01-31", "gpt-4o-2024-11-20", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14",
                                 "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25",
                                 "ernie-tiny-8k", "ernie-x1-32k", "ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-8k-preview",
                                 "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","doubao-1.5-lite-32k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428",
                                 "deepseek-chat", "deepseek-reasoner",
                                 "x1", "lite", "4.0Ultra", "max-32k", "pro-128k", "max", "pro",
                                 "qwq","deepseek-r1:70b",
                                 "moonshot-v1-32k", "kimi-latest","moonshot-v1-128k", "moonshot-v1-8k"], help='提问方1')#普通人
    parser.add_argument("--model_name_b", "-mb", type=str, default="deepseek-chat",
                        choices=["gpt-4.1-2025-04-14", "o3-mini-2025-01-31", "gpt-4o-2024-11-20", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14",
                                 "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25",
                                 "ernie-tiny-8k", "ernie-x1-32k", "ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-8k-preview",
                                 "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","doubao-1.5-lite-32k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428",
                                 "deepseek-chat", "deepseek-reasoner",
                                 "x1", "lite", "4.0Ultra", "max-32k", "pro-128k", "max", "pro",
                                 "qwq","deepseek-r1:70b",
                                 "moonshot-v1-32k", "kimi-latest","moonshot-v1-128k", "moonshot-v1-8k", "all"], help='被提问方')#密码专家
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all", help='[已知信息]')
    parser.add_argument("--model_name", "-m", type=str, default="deepseek-reasoner", help='缓存区')
    args = parser.parse_args()
    model_finished = ["gpt-4.1-2025-04-14", "o3-mini-2025-01-31", "gpt-4o-2024-11-20", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14",
                                 "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25",
                                 "ernie-x1-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "ernie-4.5-turbo-32k",
                                 "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428",
                                 "deepseek-chat", "deepseek-reasoner",
                                 "x1", "lite", "4.0Ultra", "pro-128k", "pro",
                                 "moonshot-v1-32k", "kimi-latest", "moonshot-v1-128k",]
    models_list = ["tinyllama", "qwq", "mistral", "gemma3", "qwen3", "llama3.1"]
    if args.model_name_b == "all":
        for model in models_list:
            args.model_name_b=model
            if args.assigned_subjects == "all":
                dtype_dict={"id": int, "question": str, "answer": str}
                df = pd.read_csv('expert_data.csv', encoding='GB18030', dtype=dtype_dict)
                for index in range(0, 14):
                    info = df.loc[index, 'question'] if isinstance(df.loc[index, 'question'], str) else ''
                    answer = df.loc[index, 'answer'].upper() if isinstance(df.loc[index, 'answer'], str) else ''
                    evaluate(info, answer)
            else:
                print("?")
    elif args.assigned_subjects == "all":
        dtype_dict={"id": int, "question": str, "answer": str}
        df = pd.read_csv('expert_data.csv', encoding='GB18030', dtype=dtype_dict)
        for index in range(0, 14):
            info = df.loc[index, 'question'] if isinstance(df.loc[index, 'question'], str) else ''
            answer = df.loc[index, 'answer'].upper() if isinstance(df.loc[index, 'answer'], str) else ''
            evaluate(info, answer)
    else:
        print("?")
