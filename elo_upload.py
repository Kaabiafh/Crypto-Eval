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

record = []#用于存储对战结果，由dict组成的list

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
    elif args.model_name in ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25", "gpt-4.1", "o3-mini-2025-01-31", "gpt-4o-2024-08-06", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "o4-mini", "o1-mini-2024-09-12", "claude-sonnet-4-20250514", "o4-mini"]:
        client = OpenAI(api_key=WILDCARD_API_KEY, base_url="https://api.gptsapi.net/v1")
    elif args.model_name in ["ernie-tiny-8k", "ernie-x1-32k", "ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-8k-preview"]:
        client = OpenAI(api_key=BAIDU_API_KEY, base_url="https://qianfan.baidubce.com/v2")
    elif args.model_name in ["doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","doubao-1.5-lite-32k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428"]:
        client = Ark(api_key=ARK_API_KEY)
    elif args.model_name in ["moonshot-v1-128k","moonshot-v1-32k", "kimi-latest", "moonshot-v1-8k"]:
        client = OpenAI(api_key="sk-q1aBiEQdhpJwQ1yXssQ1Nxu420bbJ66W9VknRlnbrqD4feZy", base_url="https://api.moonshot.cn/v1")
    elif args.model_name in ["qwq", "deepseek-r1:70b"]:
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
    if args.model_name in ["gpt-4.1", "o3-mini-2025-01-31", "gpt-4o-2024-08-06", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "o4-mini", "o1-mini-2024-09-12", "deepseek-chat", "deepseek-reasoner", "ernie-tiny-8k", "ernie-x1-32k", "ernie-4.5-turbo-128k", "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","moonshot-v1-32k", "kimi-latest","moonshot-v1-128k", "x1","doubao-1.5-lite-32k-250115", "lite", "4.0Ultra", "max-32k", "pro-128k", "deepseek-r1-distill-qwen-32b-250120", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "max", "pro", "moonshot-v1-8k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-8k-preview", "doubao-1.5-thinking-vision-pro-250428", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25", "claude-sonnet-4-20250514"]:
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
    elif args.model_name in ["qwq", "deepseek-r1:70b"]:
        payload = {
            "model": args.model_name,
            "prompt": message_text,
            "stream": False
        }
        response = requests.post("http://172.16.77.59:11434/api/generate", json=payload, timeout=300)
        if response.status_code != 200:
            print("API call failed with status code", response.status_code, response.json())
            return response.json()["response"]["message"], cost_time
        else:
            result = response.json()["response"]["message"]
        return result, cost_time
    else:
        print("For other model API calls, please implement the request method yourself.")
        result = None
    #print("cost time", time.time() - start)
    cost_time = time.time() - start
    return result, cost_time

def load_cipher():
    dataset = load_dataset("Kaif-d-d-d/cipher")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def format_example(question, options, language, cot_content=""):
    if cot_content == "":
        if language == "English":
            cot_content = "Let's think step by step."
        elif language == "中文":
            cot_content = "让我们一步步进行思考。"
    if cot_content.startswith("A:"):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_answer(text):
    match = re.search(r'猜测：\s*(\S+)', text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)

def extract_again(text):
    # 增强的正则表达式，支持全角/半角冒号、括号及多种空白字符
    pattern = r"猜测[\s\u3000]*[:：]?[\s\u3000]*\(?(\S+)\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return "无"

def single_request(client, message_text):
    exist = True
    cost_time = 0.00

    try:
        response, cost_time = call_api(client, message_text)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return "无", None, cost_time
    pred = extract_answer(response)
    return pred, response, cost_time


def evaluate(subjects, model):
    total_time_cost = 0.00
    steps = 40
    print("提问者：" + model + "\n被提问者：" + args.model_name_b)
    print("使用的密码算法：", subjects)
    cost_time = 0.00
    a_message=[{"role":"system", "content":"""# 角色
你是一个密码算法猜测专家，用户使用了某种密码算法对自己的信息进行加密，通过与用户的多轮对话，你要在尽量少的提问次数内，用判断性问题获取该算法相关信息，猜测出用户使用的密码算法。
已知用户使用的密码算法在以下列表中：["凯撒密码","维吉尼亚密码","仿射密码","栅栏密码","Playfair密码","自动密钥密码","ROT13密码","摩斯电码","一次性密码本","Enigma密码机", "AES","DES","3DES","Blowfish", "Serpent","RC4","ChaCha20","Salsa20", "IDEA", "CAST-128","Camellia", "ARIA","SEED","GOST28147-89","MARS","RC6","RSA", "ElGamal", "NTRUEncrypt","Kyber", "McEliece", "Rabin","ECDSA", "SHA-256", "SHA-3","BLAKE2","Whirlpool", "RIPEMD-160", "MD5","Argon2","SM2", "SM3", "SM4", "ZUC"]
# 任务要求
## 提问流程
1. 每次提问前，仔细查看对话记录和列表中尚未猜测到的密码算法的特点并进行总结，根据总结结果，制定接下来的提问和猜测策略，每次提问的目的应该是排除掉尽可能多的选项而非确认是否是特定的某一选项，例如，在前期提问是否是现代密码/古典密码/对称密码算法等密码算法类型的效率会比较高。
2. 根据制定的提问和猜测策略，提出一个可以明确用“是”或“不是”回答的判断性问题，获取关于用户使用密码算法的更多信息，注意：用户的回答只可能在“是”或“不是”中二选一。
3. 在输出的最后，以“猜测：X”的格式附上当前认为最有可能的密码算法，其中“X”为具体的密码算法名称，无论目前是否有足够的线索指向某一算法，都应把最可能的算法作为猜测，如果不进行猜测会浪费一次机会。
4. 等待用户回答。

## 提问策略
- 问题应具有针对性，能够有效排除一些不可能的算法。
- 每次提问前，仔细查看对话记录并进行总结，根据已有信息决定接下来的提问策略。
- 问题应该能明确以“是”或“不是”来回答，避免歧义或混淆的情况。

## 输出格式
每次回复需严格按照以下格式输出：
- 问题：[具体问题]
- 猜测：[算法名称]

## 交流风格
- 语言简洁明了，避免使用过于复杂的专业术语，确保用户能够理解问题。

# 注意事项
- 每次回答只能提出一个问题，提交一个对密码算法的猜测，然后必须等待用户输入才能进行下一次提问和猜测，如果一次提出一个以上的问题，只有第一个问题会得到回答。
- 用户是诚实的，用户的回答一定真实。
"""}, {"role":"user", "content":"现在，开始提问吧"}]

    b_message=[{"role": "system", "content":"""# 角色
你参与了一个密码算法猜测游戏，在这个游戏中，你使用了{}密码算法对你的信息进行加密，对方会向你提问，试图猜出你使用的密码算法。

# 任务要求
- **回答规则**：对于对方的提问，回答只能在“是”、“不是”中二选一，除此之外，不要输出任何内容。如果问题不能用以上选项回答，你可以拒绝回答并要求用户换一个能用以上选项回答的问题。每次只回答一个问题，如果对方一次提出多个问题，只对第一个问题进行回答。除提问外，对方每次还会进行猜测，但猜测正确与否与你无关，不需要作出响应。
- **互动要求**：配合对方完成游戏，不要故意作出错误回答，保持游戏的公平性和趣味性。

# 游戏流程
1. 等待对方提问。
2. 根据对方的提问，基于你使用的密码算法给出相应回答。
3. 持续进行上述步骤，直到对方猜出算法。 """.format(subjects)}]
    output_summary_path = args.output_dir + args.model_name + "_summary.json"
    finish=0
    
    while(steps > 0):
        args.model_name = model
        client = get_client()
        guess, response, cost_time = single_request(client, a_message)
        if response == None:
            continue
        print("---------------------------------------")
        print("提问者：", response)
        a_message.append({"role": "assistant", "content": response})
        b_message.append({"role": "user", "content": response})
        steps -= 1
        total_time_cost += cost_time
        if guess.find(subjects) != -1 and guess.find(subjects) < 10:
            finish = 1
            print("猜测正确，游戏结束！\n猜测次数：{}".format(40-steps))
            print("------------------------------------------------------------------------------")
            break

        args.model_name = args.model_name_b
        client = get_client()
        guess, response, cost_time = single_request(client, b_message)
        while response == None:
            guess, response, cost_time = single_request(client, b_message)
        print("---------------------------------------")
        print("被提问者：", response)
        print("猜测错误，还剩{}次机会".format(steps))
        total_time_cost += cost_time
        if steps == 0:
            print("次数耗尽，未能猜出算法……所用算法为：", subjects)
        a_message.append({"role": "user", "content": response})
        b_message.append({"role": "assistant", "content": response})

    a_message.append({"total_time": total_time_cost})
    save_res(a_message, steps, finish, model)
    print("total_time_cost:", total_time_cost)
    return steps, finish, total_time_cost



def save_res(res, steps, finish, model):
    global record
    res[0] = {"提问者" + str(len(record)): model, "被提问者": args.model_name_b, "提问次数": 40-steps, "是否完成": finish}
    for each in res[1:-1]:#从第二个开始
        each["role"] = each["role"].replace("user", "被提问者")
        each["role"] = each["role"].replace("assistant", "提问者")
        each["content"] = each["content"].encode(encoding="gbk",errors="ignore").decode(encoding="gbk")
    record.append(res)

def save_match_record(output_record_path):
    global record
    with open(output_record_path, "r") as ip:
        while ip == None:
            ip = open(output_record_path, "r")
        c_record = json.load(ip)
    c_record.append(record)
    with open(output_record_path, "w") as op:
        op.write(json.dumps(c_record, ensure_ascii=False, indent=2))

def save_match_summary(output_summary_path):
    res = [{"提问者": args.model_name_a, "被提问者": args.model_name_b, "提问次数": 40-steps, "结果": finish}]
    with open(output_summary_path, "a") as op:
        op.write(json.dumps(summary))


def update_rank(win):
    with open("elo/elo.json", "r") as elo_ip:
        current_rank = json.load(elo_ip)
        rank_a = current_rank[args.model_name_a]
        rank_c = current_rank[args.model_name_c]

    n_rank_a, n_rank_c = compute_elo(rank_a, rank_c, win)
    current_rank[args.model_name_a] = n_rank_a
    current_rank[args.model_name_c] = n_rank_c
    n_rank = sorted(current_rank.items(), key=takeSecond, reverse=True)#对新elo分降序排列
    with open("elo/elo.json", "w") as elo_op:
        elo_op.write(json.dumps(dict(n_rank), ensure_ascii=False, indent=2))
    if win == 1:
        print(args.model_name_a + "胜利！")
        print(args.model_name_c + "失败。")
    elif win == 0:
        print(args.model_name_c + "胜利！")
        print(args.model_name_a + "失败。")
    else:
        print(args.model_name_a + "与" + args.model_name_c + "平局！")
    print(args.model_name_a + ":\t" + str(rank_a) + "->" + str(n_rank_a))
    print(args.model_name_c + ":\t" + str(rank_c) + "->" + str(n_rank_c))
    print("------------------------------------------------------------------------------")
    record.append({args.model_name_a: str(rank_a) + "->" + str(n_rank_a), args.model_name_c: str(rank_c) + "->" + str(n_rank_c), "win": win})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="elo/match_record/", help='对战记录输出文件夹')
    parser.add_argument("--model_name_a", "-ma", type=str, default="deepseek-chat",
                        choices=["gpt-4.1", "o3-mini-2025-01-31", "gpt-4o-2024-08-06", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "o4-mini", "o1-mini-2024-09-12",
                                 "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25", "claude-sonnet-4-20250514",
                                 "ernie-tiny-8k", "ernie-x1-turbo-32k", "ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-turbo-32k",
                                 "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","doubao-1.5-lite-32k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428",
                                 "deepseek-chat", "deepseek-reasoner",
                                 "x1", "lite", "4.0Ultra", "max-32k", "pro-128k", "max", "pro",
                                 "qwq","deepseek-r1:70b",
                                 "moonshot-v1-32k", "kimi-latest","moonshot-v1-128k", "moonshot-v1-8k"], help='提问方1')#提问方1
    parser.add_argument("--model_name_b", "-mb", type=str, default=random.choice(["doubao-1.5-pro-256k-250115", "ernie-4.5-turbo-128k", "deepseek-chat"]),
                        choices=["gpt-4.1", "o3-mini-2025-01-31", "gpt-4o-2024-08-06", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "o4-mini", "o1-mini-2024-09-12",
                                 "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25", "claude-sonnet-4-20250514",
                                 "ernie-tiny-8k", "ernie-x1-turbo-32k", "ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-turbo-32k",
                                 "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","doubao-1.5-lite-32k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428",
                                 "deepseek-chat", "deepseek-reasoner",
                                 "x1", "lite", "4.0Ultra", "max-32k", "pro-128k", "max", "pro",
                                 "qwq","deepseek-r1:70b",
                                 "moonshot-v1-32k", "kimi-latest","moonshot-v1-128k", "moonshot-v1-8k"], help='被提问方')#被提问方
    parser.add_argument("--model_name_c", "-mc", type=str, default="ernie-x1-32k",
                        choices=["gpt-4.1", "o3-mini-2025-01-31", "gpt-4o-2024-08-06", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "o4-mini", "o1-mini-2024-09-12",
                                 "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25", "claude-sonnet-4-20250514",
                                 "ernie-tiny-8k", "ernie-x1-turbo-32k", "ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-turbo-32k",
                                 "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","doubao-1.5-lite-32k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428",
                                 "deepseek-chat", "deepseek-reasoner",
                                 "x1", "lite", "4.0Ultra", "max-32k", "pro-128k", "max", "pro",
                                 "qwq","deepseek-r1:70b",
                                 "moonshot-v1-32k", "kimi-latest","moonshot-v1-128k", "moonshot-v1-8k"], help='提问方2')#提问方2
    parser.add_argument("--assigned_subjects", "-a", type=str, default="RSA", choices=["凯撒密码","维吉尼亚密码","仿射密码","栅栏密码","Playfair密码","自动密钥密码","ROT13密码","摩斯电码","一次性密码本","Enigma密码机", "AES","DES","3DES","Blowfish", "Serpent","RC4","ChaCha20","Salsa20","IDEA","CAST-128","Camellia", "ARIA","SEED","GOST28147-89","MARS","RC6","RSA", "ElGamal", "NTRUEncrypt","Kyber","McEliece","Rabin","ECDSA", "SHA-256", "SHA-3","BLAKE2","Whirlpool", "RIPEMD-160", "MD5","Argon2","SM2","SM3","SM4","祖冲之算法", "rand"], help='被提问方使用的算法名称')
    parser.add_argument("--model_name", "-m", type=str, default="deepseek-reasoner", help='没用')
    parser.add_argument("--random", "-r", type=int, default=None, help='随机匹配对战，输入值为次数')
    args = parser.parse_args()
    
    if args.assigned_subjects == "rand":
        assigned_subjects = random.choice(["凯撒密码","维吉尼亚密码","仿射密码","栅栏密码","Playfair密码","自动密钥密码","ROT13密码","摩斯电码","一次性密码本","Enigma密码机", "AES","DES","3DES","Blowfish", "Serpent","RC4","ChaCha20", "Salsa20","IDEA","Camellia","SEED","GOST28147-89","MARS","RSA", "ElGamal","Kyber", "McEliece","Rabin", "Ed25519","ECDSA", "SHA-256", "BLAKE2","RIPEMD-160","MD5","SHA-512","SM2","SM3","SM4","祖冲之算法"])
    else:
        assigned_subjects = args.assigned_subjects
    os.makedirs(args.output_dir, exist_ok=True)
    output_record_path = args.output_dir + "match_record.json"
    output_summary_path = args.output_dir + "match_summary.json"

    if args.random:
        for i in range(0, args.random):
            with open("elo/elo.json", "r") as elo:
                args.model_name_a, args.model_name_c = select_match(json.load(elo))
            print(args.model_name_a + "  vs  " + args.model_name_c)
            print("---------------------------------------")
            with open(output_record_path, "r") as ip:
                if ip:
                    match_id = json.load(ip)[-1][0]["match_id"] + 1
                else:
                    match_id = 1
            if args.assigned_subjects == "rand":
                assigned_subjects = random.choice(["凯撒密码","维吉尼亚密码","仿射密码","栅栏密码","Playfair密码","自动密钥密码","ROT13密码","摩斯电码","一次性密码本","Enigma密码机", "AES","DES","3DES","Blowfish", "Serpent","RC4","ChaCha20", "Salsa20","IDEA","Camellia","SEED","GOST28147-89","MARS","RSA", "ElGamal","Kyber", "McEliece","Rabin", "Ed25519", "SHA-256", "BLAKE2","RIPEMD-160","MD5", "SHA-512", "SM2","SM3","SM4","祖冲之算法", "DSA", "NTRU"])
            record = [{"match_id": match_id, "提问者1": args.model_name_a, "提问者2": args.model_name_c, "被提问者": args.model_name_b, "密码算法": assigned_subjects}]
            steps_a, finish_a, time_a = evaluate(assigned_subjects, args.model_name_a)
            steps_c, finish_c, time_c = evaluate(assigned_subjects, args.model_name_c)

            if steps_a+finish_a > steps_c+finish_c:
                win = 1
            elif steps_a+finish_a == steps_c+finish_c:
                win = 0.5
            else:
                win = 0
            update_rank(win)
            save_match_record(output_record_path)
    else:
        steps_a, finish_a, time_a = evaluate(assigned_subjects,args.model_name_a)
        steps_c, finish_c, time_c = evaluate(assigned_subjects, args.model_name_c)

        if steps_a+finish_a > steps_c+finish_c:
            win = 1
        elif steps_a+finish_a == steps_c+finish_c:
            win = 0.5
        else:
            win = 0
    
        update_rank(win)