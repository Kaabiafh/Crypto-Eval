import os
import openai
from openai import OpenAI
#import anthropic
#import google.generativeai as genai
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse
import requests
from volcenginesdkarkruntime import Ark


random.seed(123456)

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
    elif args.model_name in ["qwq", "llama3.1", "mistral", "gemma3", "qwen3", "tinyllama"]:
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
    elif args.model_name in ["qwq", "llama3.1", "mistral", "gemma3", "qwen3", "tinyllama"]:
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
            cost_time = time.time()-start
        return result, cost_time
    else:
        print("For other model API calls, please implement the request method yourself.")
        result = None
    #print("cost time", time.time() - start)
    cost_time = time.time() - start
    return result, cost_time


def load_cipher():
    dataset = load_dataset("Kaif-d-d-d/code")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res = {}
    for each in test_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def format_example(code, answer):
    cot_content = "一步步进行思考。"
    example = "<Code>\n{}\n</code>".format(code)
    if answer == "":
        example += "Answer: "
    else:
        example += "Answer: " + answer + "\n\n"
    return example


def extract_answer(text):
    pattern = r"Answer is (.+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        pattern = r"答案(?:选择|是)?\s*[:：]\s*(.+)"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            print("1st answer extract failed\n" + text)
            return extract_again(text)


def extract_again(text):
    match = re.search(r"[aA]nswer[\:：]\s(.+)", text)
    if match:
        return match.group(1)
    else:
        match = re.search(r"答案[\:：](.+)", text)
        if match:
            return match.group(1)
        else:
            return text

def single_request(client, single_question, cot_examples_dict, exist_result):
    exist = True
    cost_time = 0.00
    q_id = single_question["code_id"]
    for each in exist_result:
        if q_id == each["code_id"]:
            pred = extract_answer(each["model_outputs"])
            return pred, each["model_outputs"], exist, cost_time
    exist = False
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["code"]
    answer_0 = single_question["answer_0"]
    answer_1 = single_question["answer_1"]
    answer_2 = single_question["answer_2"]
    prompt = "下面是一个关于{}的代码识别题示例。一步步进行思考。" \
             " 在响应的最后，添加一个\"答案是(X)\"，输出用户问题的答案。\n\n" \
        .format(category)
    for each in cot_examples:
        prompt += format_example(each["code"], each["answer_0"])
    input_text = format_example(question, "")
    if args.model_name == "o1-mini":
        message = [{"role":"user", "content":prompt + input_text}]
    else:
        message = [{"role":"system", "content": prompt},{"role":"user", "content": input_text}]
    try:
        response, cost_time = call_api(client, message)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None, exist, cost_time
    pred = extract_answer(response)
    return pred, response, exist, cost_time


def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as ip:
                    res = json.load(ip)
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                        if not each["pred"]:
                            x = random.randint(0, len(each["options"]) - 1)
                            if x == each["answer"]:
                                category_record[category]["corr"] += 1
                            else:
                                category_record[category]["wrong"] += 1
                        elif each["pred"].find(each["answer_0"]) != -1 or each["pred"].find(each["answer_1"]) != -1 or each["pred"].find(each["answer_2"]) != -1:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    merged = False
    for i, single in enumerate(res):
        if single["code_id"] == curr["code_id"]:
            res[i] = curr
            merged = True
    if not merged:
        res.append(curr)
    return res


def evaluate(subjects):
    client = get_client()
    total_time_cost = 0.00
    exist_time = 0
    test_df, dev_df = load_cipher()
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir + args.model_name + "/", subject + "_result.json")
        output_summary_path = os.path.join(args.output_dir + args.model_name + "/", subject + "_summary.json")
        with open(output_summary_path, 'a') as file:
            pass
        if os.path.getsize(output_summary_path) != 0:
            with open(output_summary_path, "r") as f:
                if f is not None:
                    line = f.readlines()[-1]
                    exist_time = float(line[0:line.rfind("s/it")])
        total_time_cost += exist_time
        res, category_record = update_result(output_res_path)

        for each in tqdm(test_data):
            cost_time = 0.00
            label_0 = each["answer_0"]
            label_1 = each["answer_1"]
            label_2 = each["answer_2"]
            category = subject
            pred, response, exist, cost_time = single_request(client, each, dev_df, res)
            total_time_cost += cost_time
            if response is not None:
                res, category_record = update_result(output_res_path)
                if category not in category_record:
                    category_record[category] = {"corr": 0.0, "wrong": 0.0}
                each["pred"] = pred.encode(encoding="gbk",errors="ignore").decode(encoding="gbk")
                each["model_outputs"] = response.encode(encoding="gbk",errors="ignore").decode(encoding="gbk")#接受中文响应时写入要ensure_ascii=False（见save_res）使用gbk字符集，但希腊字母、上标、下标等在gbk字符集中无法表示，所以此处去除非gbk字符集的字符。
                print("id: ", each["code_id"])
                print(each["model_outputs"])
                merge_result(res, each)
                labels = [label_0, label_1, label_2]
                print("label:{}\tpred:{}\tmodel:{}".format(labels,pred,args.model_name))
                if pred.find(label_0) != -1:
                    print("matched")
                    category_record[category]["corr"] += 1
                elif pred.find(label_1) != -1:
                    print("matched")
                    category_record[category]["corr"] += 1
                elif pred.find(label_2) != -1:
                    print("matched")
                    category_record[category]["corr"] += 1
                else:
                    category_record[category]["wrong"] += 1
                save_res(res, output_res_path)
                save_summary(category_record, output_summary_path)
                res, category_record = update_result(output_res_path)
            else:
                continue
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)
        print(category_record)
        with open(output_summary_path, "a") as op:
            op.write("\n")
            op.write(str(total_time_cost) + "s/it")



def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["code_id"] not in exist_q_id:
            exist_q_id.append(each["code_id"])
            temp.append(each)
        else:
            continue
    res = temp
    with open(output_res_path, "w") as op:
        op.write(json.dumps(res, ensure_ascii=False, indent=2))


def save_summary(category_record, output_summary_path):
    total_corr = 0.0
    total_wrong = 0.0
    print(category_record)
    for k, v in category_record.items():
        if k == "total":
            continue
        category_record[k]["acc"] = v["corr"] / (v["corr"] + v["wrong"])
#        total_corr += v["corr"]
#        total_wrong += v["wrong"]
#    total_acc = total_corr / (total_corr + total_wrong)
#    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": total_acc}
    with open(output_summary_path, "w") as op:
        op.write(json.dumps(category_record))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="code_anal/results/")
    parser.add_argument("--model_name", "-m", type=str, default="deepseek-chat",
                        choices=["gpt-4.1", "o3-mini-2025-01-31", "gpt-4o-2024-08-06", "o1", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "o4-mini", "o1-mini",
                                 "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25", "claude-sonnet-4-20250514",
                                 "ernie-tiny-8k", "ernie-x1-turbo-32k", "ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k", "ernie-speed-8k", "ernie-lite-8k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "deepseek-r1-distill-qianfan-70b", "ernie-4.5-turbo-32k",
                                 "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","doubao-1.5-lite-32k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428",
                                 "qwq", "deepseek-r1:70b", "llama3.1"
                                 "deepseek-chat", "deepseek-reasoner",
                                 "lite", "4.0Ultra", "max-32k", "pro-128k", "max", "pro",
                                 "moonshot-v1-32k", "kimi-latest","moonshot-v1-128k", "moonshot-v1-8k", "all"])
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")	
    args = parser.parse_args()

    models_finished = ["qwq", "deepseek-r1:70b", "llama3.1", "deepseek-chat", "gpt-4.1", "o3-mini-2025-01-31", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "o4-mini",
                                 "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview-03-25", "claude-sonnet-4-20250514",
                                  "ernie-tiny-8k", "ernie-speed-8k", "ernie-lite-8k", "moonshot-v1-8k", 
                                  "ernie-x1-turbo-32k", "ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k",
                                 "ernie-4.5-turbo-128k", "ernie-speed-128k", "ernie-speed-pro-128k", "ernie-lite-pro-128k", "ernie-4.5-turbo-vl-32k-preview", "ernie-4.5-turbo-vl-32k", "ernie-x1-turbo-32k", "ernie-4.5-turbo-32k",
                                 "doubao-1.5-thinking-pro-250415", "doubao-1.5-pro-32k-250115", "doubao-1.5-pro-256k-250115","doubao-1.5-lite-32k-250115", "deepseek-r1-distill-qwen-32b-250120", "doubao-1.5-thinking-vision-pro-250428",
                                 "deepseek-reasoner",
                                 "lite", "4.0Ultra", "pro-128k", "max", "pro",
                                 "moonshot-v1-32k", "kimi-latest","moonshot-v1-128k"]

    models_list = ["qwq", "llama3.1", "mistral", "gemma3", "qwen3", "tinyllama"]
    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.model_name == "all":
        for model in models_list:
            args.model_name = model
            os.makedirs(args.output_dir + args.model_name, exist_ok=True)
            evaluate(assigned_subjects)
    elif args.model_name == "openai":
        for model in openai_list:
            args.model_name = model
            os.makedirs(args.output_dir + args.model_name, exist_ok=True)
            evaluate(assigned_subjects)
    elif args.model_name =="fast":
        for model in fast_list:
            args.model_name = model
            os.makedirs(args.output_dir + args.model_name, exist_ok=True)
            evaluate(assigned_subjects)
    else:
        os.makedirs(args.output_dir + args.model_name, exist_ok=True)
        evaluate(assigned_subjects)