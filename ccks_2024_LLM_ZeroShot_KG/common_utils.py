from openai import OpenAI
import json
import re
from vllm import LLM
import yaml
import pprint
from collections import defaultdict

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

OpenAI_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_model(model_path, enable_lora=False):
    llm = LLM(model=model_path, 
              enforce_eager=True, enable_lora=enable_lora,
              trust_remote_code=True,revision="v1.1.8",
              max_seq_len_to_capture=8192)
    return llm

def data_parser(filename):
    data_container = []
    for line in open(filename):
        input_dict = json.loads(line.strip())
        ID = input_dict['id']
        source = "default"
        if 'source' in input_dict:
            source = input_dict['source']
        instruction_dict = json.loads(input_dict['instruction'])
        instruction = instruction_dict['instruction']
        schema = instruction_dict['schema'][0]
        input = instruction_dict['input']

        details = {}
        entity_type = schema['entity_type']
        attributes = schema['attributes']
        
        format_info = [ID, instruction, schema, input, source, input_dict]
        data_container.append(format_info)
    return data_container

def get_schema_keyset(schema):
    attribute_keys = []
    
    entity_type = schema['entity_type']
    attributes = schema['attributes']
    attribute_keys.append(entity_type)
    attribute_keys.extend(attributes.keys())
    
    return attribute_keys

def art_works_clean(completion):
    #1. 《作品》  2. 话剧《作品》
    res = re.findall(r'.*《(.*)》.*', completion, re.DOTALL)
    if res:
        completion = res[0]
    return completion


def get_config(model_name, config_path):
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    params = config[model_name]
    return params

# get_config("qwen2-7b-awq", 'config/model.yaml')

def schema_output_format_align(info_fields, completion):
    try:
        first_json = json.loads(completion)
        print(first_json)
    except json.decoder.JSONDecodeError:
        print(f"JSON ERROR:\n{completion}")
        first_json = {}

    # 2. 保证JSON key在schema中  3. 保证value在input中 4. key, value去重
    input = info_fields[3]
    schema_keyset = get_schema_keyset(info_fields[2])
    Key_value_set = set()
    result = defaultdict(dict)

    log_detail = []
    for entity_type, second_dict in first_json.items():
        if type(entity_type)!=str or type(second_dict) != dict:
            log_detail.append(f"first_json {type(entity_type)}:{type(second_dict)}")
            continue
        if entity_type not in schema_keyset:
            log_detail.append(f"ERROR ENTITY_TYPE:{entity_type}")
            continue
        
        for entity, third_dict in second_dict.items():
            if type(entity)!= str or type(third_dict) != dict:
                log_detail.append(f"second_dict {type(entity)}:{type(third_dict)}")
                continue
            if entity in ["未提及",''] or entity not in input:
                log_detail.append(f"ERROR ENTITY:{entity}")
                continue

            entity = art_works_clean(entity)
            inner_dict = {}
            for attri, value in third_dict.items():
                if attri not in schema_keyset:
                    log_detail.append(f"ERROR attri:{attri}")
                    continue
                if type(value) == list:
                    value = [art_works_clean(item) for item in value]
                    value = list(filter(lambda x: x in input, value))
                    value = list(set(value))
                    inner_dict[attri] = value
                elif type(value) == str:
                    if value in ["未提及", '']:
                        value = "无"
                    if value == "无":
                        continue
                    
                    # ['星座', '国籍', "生肖", "民族"] 大部分推理也是错的
                    if value not in input:
                        if attri in ['出生日期', '性别'] and len(value) > 1 and value[:-1] in input:
                            log_detail.append(f"缩进规则:{attri}: {value} --> {value[:-1]}")
                            value = value[:-1]
                        else:
                            log_detail.append(f"无法推理：{attri}: {value} --> 无")
                            value = "无"
                            
                    value = art_works_clean(value)
                    inner_dict[attri] = value
                    third_dict[attri] = value

            result[entity_type][entity] = inner_dict
    if len(log_detail) > 0:
        pprint.pprint(f"log_detail:{input}\n{log_detail}")
        pprint.pprint(dict(result))

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
    # if old != completion:
    #     print(f"{old}\n{completion}")
    return completion


def offline_post_process():
    data = data_parser('data/ccks2024复赛.json')
    middle_result = data_parser('data/qwen2-72B-middle.json')
    completions = [middle[-1]['output'] for middle in middle_result]

    completions = [delete_unrelated(completion) for completion in completions]
    completions = [schema_output_format_align(data[idx], completion) for idx, completion in enumerate(completions)]

    with open('data/qwen2-72B-final.json', 'w') as fo:
        for input, res in zip(data, completions):
            input[-1]["output"] = res
            fo.write(json.dumps(input[-1], ensure_ascii=False) + "\n")


if __name__ ==  '__main__':
    offline_post_process()