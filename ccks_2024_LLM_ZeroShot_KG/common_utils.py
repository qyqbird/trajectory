from openai import OpenAI
import json
import re
from vllm import LLM
import yaml

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

def data_parser(filename=None):
    data_container = []
    for line in open('data/ccks2024复赛.json'):
    # for line in open('data/test'):
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