
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

def pylint(res):
    code = ''
    with open('code.py', 'w', encoding='utf-8') as f:
        f.write(code)

def compute_elo(ra, rb, result_a, k=32):
    """
    计算两位玩家比赛后的新ELO评分
    
    参数:
    ra (float): 玩家A的当前评分
    rb (float): 玩家B的当前评分
    result_a (float): 玩家A的比赛结果 (1=胜利，0=失败，0.5=平局)
    k (int): 调整系数，决定评分变化幅度 (默认32)
    
    返回:
    tuple: (玩家A的新评分, 玩家B的新评分)
    """
    # 计算玩家A和B的预期胜率
    k = 32 * 3000/(ra + rb)
    expected_a = 1 / (1 + 10 ** ((rb - ra) / 400))
    expected_b = 1 - expected_a  # 预期胜率互为补数
    
    # 根据结果计算新评分
    new_ra = ra + k * (result_a - expected_a)
    new_rb = rb + k * ((1 - result_a) - expected_b)
    
    return round(new_ra, 2), round(new_rb, 2)

def select_match(players: Dict[str, float]) -> Tuple[str, str]:
    """
    从选手池中随机选择两名评分差≤400的不同选手
    保证最多进行1000次尝试防止无限循环
    
    参数:
        players (Dict[str, float]): 选手字典 {选手ID: 当前评分}
    
    返回:
        Tuple[str, str]: 选中选手的ID元组
    
    抛出:
        ValueError: 当无法找到符合条件的选手对时
    """
    player_list = list(players.items())
    
    # 安全机制：最多尝试1000次
    max_attempts = 1000
    for _ in range(max_attempts):
        # 随机选择第一个选手
        a_id, a_rating = random.choice(player_list)
        
        # 生成符合条件的候选对手列表
        candidates = [
            (pid, pr) for pid, pr in player_list
            if pid != a_id and abs(pr - a_rating) <= 400
        ]
        
        if candidates:
            b_id, _ = random.choice(candidates)
            return (a_id, b_id)
    
    raise ValueError("未找到符合条件的选手对，请检查评分数据")

def takeSecond(elem):
    return elem[1]

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
    f_result = []
    global task_id
    
    try:
        # 将代码写入code.py文件（UTF-8编码）
        with open('./g_code/code.py', 'w', encoding='utf-8') as f:
            f.write(code)

        corr = 0
        dtype_dict={"测试用例1参数a": str, "测试用例1参数b": str, "测试用例2参数a": str, "测试用例2参数b": str, "测试用例3参数a": str, "测试用例3参数b": str, "测试用例4参数a": str, "测试用例4参数b": str, "测试用例5参数a": str, "测试用例5参数b": str}
        df = pd.read_csv('gene_tasks.csv', encoding='GB18030', dtype=dtype_dict)
        for index in range(1, 6):
            plaintext = df.loc[task_id, '测试用例{}明文'.format(index)] if isinstance(df.loc[task_id, '测试用例{}明文'.format(index)], str) else ''
            a = df.loc[task_id, '测试用例{}参数a'.format(index)] if isinstance(df.loc[task_id, '测试用例{}参数a'.format(index)], str) else ''
            b = df.loc[task_id, '测试用例{}参数b'.format(index)] if isinstance(df.loc[task_id, '测试用例{}参数b'.format(index)], str) else ''
            answer = df.loc[task_id, '测试用例{}密文'.format(index)].upper() if isinstance(df.loc[task_id, '测试用例{}密文'.format(index)], str) else ''
            print("p:", plaintext)
            print("a:", a)
            print("b:", b)
            print("answer:", answer)
            if a == "" and b == "":
                # 通过PowerShell执行Python脚本
                s_result = subprocess.run(
                    ['powershell', '-Command', 'python ./g_code/code.py -p "{}"'.format(plaintext)],
                    capture_output=True,   # 捕获标准输出和错误
                    text=True,             # 自动解码为字符串
                    check=False,            # 不自动抛出非零退出码异常
                    timeout=5
                )
                print("输出：", s_result.stdout)
                if s_result.stdout.upper().find(answer) == -1:
                    s_result = subprocess.run(
                        ['powershell', '-Command', 'python ./g_code/code.py "{}"'.format(plaintext)],
                        capture_output=True,   # 捕获标准输出和错误
                        text=True,             # 自动解码为字符串
                        check=False,            # 不自动抛出非零退出码异常
                        timeout=5
                    )
                    print("输出：", s_result.stdout)
            elif a != "" and b == "":
                # 通过PowerShell执行Python脚本
                s_result = subprocess.run(
                    ['powershell', '-Command', 'python ./g_code/code.py -p "{}" -a {}'.format(plaintext, a)],
                    capture_output=True,   # 捕获标准输出和错误
                    text=True,             # 自动解码为字符串
                    check=False,            # 不自动抛出非零退出码异常
                    timeout=5
                )
                print("输出：", s_result.stdout)
                if s_result.stdout.upper().find(answer) == -1:
                    s_result = subprocess.run(
                        ['powershell', '-Command', 'python ./g_code/code.py "{}" "{}"'.format(plaintext, a)],
                        capture_output=True,   # 捕获标准输出和错误
                        text=True,             # 自动解码为字符串
                        check=False,            # 不自动抛出非零退出码异常
                        timeout=5
                    )
                    print("输出：", s_result.stdout)
            elif a != "" and b != "":
                # 通过PowerShell执行Python脚本
                s_result = subprocess.run(
                    ['powershell', '-Command', 'python ./g_code/code.py -p "{}" -a {} -b {}'.format(plaintext, a, b)],
                    capture_output=True,   # 捕获标准输出和错误
                    text=True,             # 自动解码为字符串
                    check=False,            # 不自动抛出非零退出码异常
                    timeout=5
                )
                print("输出：", s_result.stdout)
                if s_result.stdout.upper().find(answer) == -1:
                    s_result = subprocess.run(
                        ['powershell', '-Command', 'python ./g_code/code.py "{}" "{}" "{}"'.format(plaintext, a, b)],
                        capture_output=True,   # 捕获标准输出和错误
                        text=True,             # 自动解码为字符串
                        check=False,            # 不自动抛出非零退出码异常
                        timeout=5
                    )
                    print("输出：", s_result.stdout)

            f_result.append({
                'output': s_result.stdout,
                'error': s_result.stderr,
                'returncode': s_result.returncode
            })
            print("输出：", s_result.stdout)
            if s_result.stdout is not None:
                if s_result.stdout.upper().find(answer) != -1:
                    corr += 1
        print("corr: ", corr)
        # 格式化返回结果
        return f_result, corr
    
    except OSError as e:
        # 文件操作异常处理
        return {'output': s_result.stdout,
                   'error': f'文件操作失败: {e}'}, corr
    except FileNotFoundError:
        # PowerShell或Python未找到的情况
        return {'output': s_result.stdout,
'error': '未找到PowerShell或Python解释器'}, corr
    except Exception as e:
        # 其他未知异常处理
        return {'output': s_result.stdout,
'error': f'执行过程中发生未知错误: {e}'}, corr

def extract_code(text):
    pattern = r"```python([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        pattern = r"<code>([\s\S]*?)</code>"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            print("code extract failed\n" + text)
            return text

def generate_code(model, task_description: str) -> str:
    """
    调用指定的LLM生成完成特定密码任务的Python代码。
    """
    args.model_name=model
    client=get_client()
    prompt = [{"role":"user", "content":f"按照以下任务要求生成一段Python代码：{task_description}\n将代码用“```python”和“```”包裹，并确保符合以下条件：1. 正确实现预期功能；2. 适当进行代码注释，让用户更容易理解代码；3. 代码要易于修改，避免对代码进行部分修改时的扩散风险；4. 避免使用有安全风险或漏洞的工作模式、函数或库；5. 对非标准输入做处理，防止意外情况导致的程序运行中止；6. 代码中不要包含input()等要求用户进行输入的函数。"}]
    response, time = call_api(client, prompt)
    return response, time

def score_code(model, code: str) -> float:
    """
    调用评分LLM对代码的指定维度进行评分。
    """
    output_path=args.output_dir +"/response/" + model+"_score_response.json"
    args.model_name=model
    client=get_client()
    prompt = [{"role":"user","content":"""你的任务是从正确性、可读性、可维护性、安全性和鲁棒性5个维度对一段目的是{}Python代码进行打分，各项满分均为100分。以下是要评估的Python代码：
<python_code>
{}
</python_code>
各维度说明如下：
- 正确性：代码是否能正确实现预期功能。
- 可读性：代码注释密度是否足够高，能够让用户容易理解代码。
- 可维护性：代码是否易于修改，对代码进行部分修改时是否有扩散影响的风险。
- 安全性：代码是否使用了有安全风险或漏洞的工作模式、函数、库。
- 鲁棒性：代码是否对异常情况做了处理，面对非标准输出是否能正常运行并进行错误处理。

对于每个维度，请先在<思考>标签中详细分析你的评分理由，然后在<评分>标签中给出0 - 100的具体分数。例如：
<思考>
[在此详细说明你对该维度的分析]
</思考>
<评分>X</评分>

在完成所有5个维度的评分后，请按照以下格式输出总分：
<总分>
正确性：[正确性得分]
可读性：[可读性得分]
可维护性：[可维护性得分]
安全性：[安全性得分]
鲁棒性：[鲁棒性得分]
</总分>

请现在开始你的评估。""".format(args.assigned_task, code)}]
    response, time = call_api(client, prompt)
    print(response)
    scores = extract_scores(response)
    save(response, model,output_path)
    return scores

def save(res, model, output_path):
    s = {"打分模型":model,"响应":res}
    s["响应"] = s["响应"].encode(encoding="gbk",errors="ignore").decode(encoding="gbk")
    with open(output_path, "a") as op:
        op.write(json.dumps(s, ensure_ascii=False, indent=2))



def extract_scores(output: str) -> Dict[str, float]:
    """
    从评分LLM的输出中提取各项得分。
    """
    scores = {'正确性': -1, '可读性': -1, '可维护性': -1, '安全性': -1, '鲁棒性': -1}
    dimensions = ['正确性', '可读性', '可维护性', '安全性', '鲁棒性']
    lines = output.split('\n')
    
    # 预编译通用正则模板
    pattern_template = r'{dim}[^0-9]*(\d+)'

    for dim in dimensions:
        # 为每个维度创建专属正则
        current_pattern = pattern_template.format(dim=dim)
    
        for line in lines:
            line = line.strip()
            match = re.search(current_pattern, line)
            if match:
                scores[dim] = int(match.group(1))
    return scores


def main():
    score_llms = ["deepseek-chat", "ernie-4.5-turbo-vl-32k-preview", "doubao-1.5-pro-256k-250115", "ernie-x1-turbo-32k", "kimi-latest"]  # 替换为实际的评分LLM标识
    for i in range(0, 5):
        print("{}第{}次生成{{{}}}的Python代码...".format(args.model_name_a, i+1, args.assigned_task))
        response, time = generate_code(args.model_name_a, args.assigned_task)
        print("生成的代码：")
        print(response)
        generated_code = extract_code(response)
        if generated_code.find("input(") != -1:
            print("代码包含input()函数，重新生成...")
            continue
        exe_result, corr = execute_code(generated_code)
        print(exe_result)
        if corr == 0:
            if i < 7:
                print("代码测试全部不通过，重新生成...")
                continue
            else:
                print("已生成代码达5次，继续评分")
        else:
            break

    print("""
    步骤2：对生成的代码进行打分...""")
    
    all_scores = []
    for lm in score_llms:
        print(f"{lm}正在打分...")
        # 假设每个LLM对应不同的客户端或配置，这里简化为同一客户端
        scores = score_code(lm, response)  # 示例，仅调用一次维度
        print(scores)
        # 实际应用中应循环所有维度并收集评分
        # 这里为了示例，手动创建一个完整的评分字典
        # 您需要根据实际情况调用评分函数获取各维度评分
        single_scores = {
            "打分模型":lm,
            "正确性": scores['正确性'],
            "可读性": scores['可读性'],  
            "可维护性": scores['可维护性'],  
            "安全性": scores['安全性'],
            "鲁棒性": scores['鲁棒性']
        }
        all_scores.append(single_scores)
    
    print("步骤3：提取并存储评分结果到JSON文件...")
    results = {"model":args.model_name_a, "task": args.assigned_task, "scores": all_scores}
    ss = results["scores"]
    avg_score = {"被打分模型":args.model_name_a, "正确性": sum(item["正确性"] for item in ss)/(len(ss)),"可读性":sum(item["可读性"] for item in ss)/(len(ss)),"可维护性":sum(item["可维护性"] for item in ss)/(len(ss)),"安全性":sum(item["安全性"] for item in ss)/(len(ss)), "鲁棒性":sum(item["鲁棒性"] for item in ss)/(len(ss)), "corr": corr}
    results["scores"].append(avg_score)
    result_file = args.output_dir + args.model_name_a + "_scores_results.json"
    with open(result_file, 'a', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"评分结果已保存到 {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="code_record/", help='记录输出文件夹')
    parser.add_argument("--model_name_a", "-ma", type=str, default=random.choice(["doubao-1.5-pro-256k-250115", "ernie-4.5-turbo-128k", "deepseek-chat"]),
                        choices=["gpt-4.1-2025-04-14", "o3-mini-2025-01-31", "gpt-4o-2024-11-20", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14",
                                 "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25",
                                 "ernie-tiny-8k", "ernie-x1-32k", "ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-8k-preview",
                                 "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","doubao-1.5-lite-32k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428",
                                 "deepseek-chat", "deepseek-reasoner",
                                 "x1", "lite", "4.0Ultra", "max-32k", "max", "pro",
                                 "moonshot-v1-32k", "kimi-latest","moonshot-v1-128k", "moonshot-v1-8k",
                                 "all"], help='代码生成模型')
    parser.add_argument("--assigned_task", "-a", type=str, default="all")
    args = parser.parse_args()
    model_finished = ["gpt-4.1-2025-04-14", "o3-mini-2025-01-31", "gpt-4o-2024-11-20", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14",
                                 "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25",
                                 "ernie-x1-turbo-32k", "ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k","ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "ernie-4.5-turbo-32k",
                                 "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428",
                                 "deepseek-chat", "deepseek-reasoner",
                                 "x1", "lite", "4.0Ultra", "pro-128k", "pro",
                                 "moonshot-v1-32k", "kimi-latest", "moonshot-v1-128k"
                                 "qwq"]
    if args.model_name_a == "all":
        for model in ["tinyllama", "mistral", "gemma3", "qwen3", "llama3.1"]:
            args.model_name_a=model
            if args.assigned_task == "all":
                df = pd.read_csv('gene_tasks.csv', encoding='GB18030')
                for i in range(0, 5):
                    task_id = i
                    args.assigned_task = df.loc[i, "密码生成任务要求"] if isinstance(df.loc[i, '密码生成任务要求'], str) else ''
                    main()
                args.assigned_task = "all"#重置
            else:
                main()
    elif args.assigned_task == "all":
        df = pd.read_csv('gene_tasks.csv', encoding='GB18030')
        for i in range(0, 5):
            task__id = i
            args.assigned_task = df.loc[i, "密码生成任务要求"]
            main()
    else:
        main()
