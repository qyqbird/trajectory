import requests
import time
from concurrent.futures import ProcessPoolExecutor
from common_utils import OpenAI_client
from ccks.offline_batch import data_parser
import json


def qwen2_7B_online_api(question):
    # python -m vllm.entrypoints.openai.api_server --model Qwen2-7B-Instruct-AWQ --max-model-len 8096
    # 想要批处理，怎么搞呢,无法批处理。尝试异步调用: openai 似乎不行呀
    # concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.
    chat_response = OpenAI_client.chat.completions.create(
        model="Qwen2-7B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "你是一名知识图谱，信息抽取专家，负责解答用户的抽取任务"},
            {"role": "user", "content": question},
        ],
        temperature=0,
        frequency_penalty=2,
        max_tokens=1024,
        stop="True",
        response_format={"type": "json_object"},    # 很慢了
    )
    return chat_response.choices[0].message.content


def request_api(question):
    # python -m vllm.entrypoints.openai.api_server --model Qwen2-7B-Instruct-AWQ --max-model-len 8096
    # http://localhost:8000/v1/completions   直接完成completion
    url = "http://localhost:8000/v1/chat/completions"
    data_infos = {'prompt': question, 
                  'model': "Qwen2-7B-Instruct-AWQ",
                  "temperature":0,
                  "max_tokens":1024,
                #   "stop":"True",
                #   "response_format":{"type": "json_object"}
                  }
    try:
        print(f"\n{question}\n")
        reponse = requests.post(url=url, json=data_infos)
        result = eval(reponse.text)

    except Exception as e:
        print(e)
        result = "ERROR"
    
    print(result)
    return result

def async_by_openapi_test():
    data = data_parser()
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=1) as pool:
        global_idx = 0
        # qwen2_7B_online_api
        for result in pool.map(request_api, data):  
            print(result)

if __name__ == '__main__':
    async_by_openapi_test()

