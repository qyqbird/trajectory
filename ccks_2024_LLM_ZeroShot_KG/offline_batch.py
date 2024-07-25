import json
import time
from vllm import LLM, SamplingParams
from common_utils import OpenAI_client, data_parser, art_works_clean, get_schema_keyset
from prompt_utils import create_repair_json_prompt, create_uie_prompt_construct
import pprint
import re
from collections import defaultdict

llm = LLM(model="/root/Qwen2-7B-Instruct-AWQ", enforce_eager=True,
              trust_remote_code=True,revision="v1.1.8",max_seq_len_to_capture=8192)

def schema_output_format_align(info_fields, completion):
    #1. 保证JSON格式 
    try:
        first_json = json.loads(completion)
    except json.decoder.JSONDecodeError:
        #TODO 计划生成2个，然后取第二个 18/1000; 或者2次召回
        # print(f"JSON ERROR:\n{completion}")
        first_json = {}

    # 2. 保证JSON key在schema中  3. 保证value在input中 4. key, value去重
    input = info_fields[3]
    schema_keyset = get_schema_keyset(info_fields[2])
    Key_value_set = set()
    result = defaultdict(defaultdict)

    for entity_type, second_dict in first_json.items():
        if type(entity_type)!=str or type(second_dict) != dict:
            continue
        if entity_type not in schema_keyset:
            continue
        
        for entity, third_dict in second_dict.items():
            if type(entity)!= str or type(third_dict) != dict:
                continue
            if entity not in input:
                continue
            inner_dict = {}
            for attri, value in third_dict.items():
                if attri not in schema_keyset:
                    continue
                if type(value) == list:
                    value = [art_works_clean(item) for item in value]
                    value = list(filter(lambda x: x in input, value))
                    value = list(set(value))
                    inner_dict[attri] = value
                elif type(value) == str:
                    if value not in input:
                        third_dict[attri] = "无"
                    value = art_works_clean(value)
                    inner_dict[attri] = value
            result[entity_type][entity] = inner_dict

    # if first_json != result:
    #     pprint.pprint(first_json)
    #     pprint.pprint(result)

    result = json.dumps(result, ensure_ascii=False)
    return result

def delete_unrelated(completion):
    # print(completion)
    #1. JSON + schema:xxx
    old = completion
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
    # completion = completion.replace("《","").replace("》","")
    # if old != completion:
    #     print(f"{old}\n{completion}")
    return completion

def post_process(data, completions):
    # print(completion)
    completions = [delete_unrelated(completion) for completion in completions]
    completions = [schema_output_format_align(data[idx], completion) for idx, completion in enumerate(completions)]

    fo = open('data/qwen2-7B-instruct-final.json', 'w')
    for input, res in zip(data, completions):
        input[-1]["output"] = res
        fo.write(json.dumps(input[-1], ensure_ascii=False) + "\n")
    fo.close()


def llm_repair_jsonformat(questions):
    prompts = [create_repair_json_prompt(info[0]) for info in questions]
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
    # print(questions)
    outputs = llm.generate(prompts, sampling_params)
    
    result = {}
    idx = -1
    for output in outputs:
        generated_text = output.outputs[0].text
        idx += 1
        result[idx] = generated_text
    return result

def qwen2_7B_offline_api(questions):
    prompts = [create_uie_prompt_construct(info) for info in questions]
    # 的确可以batch size 操作
    sampling_params = SamplingParams(temperature=0.1, 
                                     top_p=0.95, 
                                     max_tokens = 512, 
                                     presence_penalty=0.0,
                                     frequency_penalty=0.1,
                                     repetition_penalty=0.9,
                                     stop=["}}}"],
                                     include_stop_str_in_output=True
                                     )
    # 这几个参数还是很难调和的比较好
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
