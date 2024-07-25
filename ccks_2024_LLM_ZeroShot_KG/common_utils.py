from openai import OpenAI
import json
import re

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

OpenAI_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

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