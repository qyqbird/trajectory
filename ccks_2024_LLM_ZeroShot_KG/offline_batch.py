import json
import time
from vllm import LLM, SamplingParams
from common_utils import OpenAI_client, data_parser
from prompt_utils import create_repair_json_prompt, create_uie_prompt_construct
import pprint
import re


def format_align(info_fields, completions):
    result = []

    second_recall_list = {}
    for idx, completion in enumerate(completions):
        try:
            first_json = json.loads(completion)
            #后验去重
            for entity_type, second_dict in first_json.items():
                key_set = set()
                value_set = set()
                for entity, third_dict in second_dict.items():
                    for attri, value in third_dict.items():
                        if attri in key_set:
                            continue
                        else:
                            key_set.add(attri)
                            if type(value) == list:
                                value = list(set(value))
            pprint.pprint(first_json)
        except Exception as e:
            # print(f"Prompt: {prompt!r}, \nGenerated text: {generated_text!r}")
            print(e)
            print(f"ERROR:{completion}")
            completion = "{}"
            second_recall_list[idx] = completion
        result.append(completion)

    llm_repair_jsonformat(second_recall_list)
    return result

def delete_unrelated(completions):
    result = []
    for completion in completions:
        # print(completion)
        #1. JSON + schema:xxx
        if 'schema:' in completion:
            idx = completion.index('schema:')
            completion = completion[:idx].strip()
        if 'Question:' in completion:
            completion = completion.replace('Question:', '')
        #2. ```JSON``` 正则匹配
        if type(completion) == list:
            completion = completion[0]
        res = re.findall(r'.*```json(.*)```.*', completion, re.DOTALL)
        if len(res) > 0:
            completion = res[0].strip()
        completion = completion.strip().replace("\n", "")

        result.append(completion)
    return result

def post_process(data, completion):
    completion = delete_unrelated(completion)
    completion = format_align(data, completion)

    fo = open('data/qwen2-7B-instruct-final.json', 'w')
    for input,res in zip(data, completion):
        input[-1]["output"] = res
        fo.write(json.dumps(input[-1], ensure_ascii=False) + "\n")
    fo.close()


def llm_repair_jsonformat(questions):
    prompts = [create_uie_prompt_construct(info) for info in questions]
    # 的确可以batch size 操作
    sampling_params = SamplingParams(temperature=0.1, 
                                     top_p=0.95, 
                                     max_tokens = 512, 
                                     presence_penalty=-0.3,
                                     frequency_penalty=0.1,
                                     repetition_penalty=0.9,
                                     stop=["}}}"],
                                     include_stop_str_in_output=True
                                     )
    # 这几个参数还是很难调和的比较好
    llm = LLM(model="/root/Qwen2-7B-Instruct-AWQ", enforce_eager=True,
              trust_remote_code=True,revision="v1.1.8",)
    # print(questions)
    outputs = llm.generate(prompts, sampling_params)
    
    result = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        result.append(generated_text)
    return result

def qwen2_7B_offline_api(questions):
    prompts = [create_uie_prompt_construct(info) for info in questions]
    # 的确可以batch size 操作
    sampling_params = SamplingParams(temperature=0.1, 
                                     top_p=0.95, 
                                     max_tokens = 512, 
                                     presence_penalty=-0.3,
                                     frequency_penalty=0.1,
                                     repetition_penalty=0.9,
                                     stop=["}}}"],
                                     include_stop_str_in_output=True
                                     )
    # 这几个参数还是很难调和的比较好
    llm = LLM(model="/root/Qwen2-7B-Instruct-AWQ", enforce_eager=True,
              trust_remote_code=True,revision="v1.1.8",)
    # print(questions)
    outputs = llm.generate(prompts, sampling_params)
    
    result = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        result.append(generated_text)
    return result

def vllm_UniversalNER_7B(question):
    # python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/UniNER-7B-all --max-model-len 2048
    chat_response = OpenAI_client.chat.completions.create(
        model="/root/autodl-tmp/UniNER-7B-all",
        messages=[
            {"role": "assistant", "content": "You are a helpful assistant."},
            # {"role": "system", "content": "你是一名知识图谱，信息抽取专家，负责解答用户的抽取任务"},
            {"role": "user", "content": "hello world"},
        ],
        max_tokens=1024
        # response_format={"type": "json_object"},    # 配置了一些参数后，throughout明显变慢 7.8 token/s
    )
    content = chat_response.choices[0].message.content
    print(type(content))
    print(content)
    return content


def task_pipe():
    data = data_parser()

    start_time = time.time()
    completion = qwen2_7B_offline_api(data)
    total_time = time.time() - start_time
    mean_time = total_time * 1000 / len(data)
    print(f"total time:{total_time}s\nmean request:{mean_time}ms")

    completion = post_process(data, completion)

if __name__ == '__main__':
    task_pipe()
