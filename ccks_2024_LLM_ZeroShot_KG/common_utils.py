from openai import OpenAI
import json

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
        instruction_dict = json.loads(input_dict['instruction'])
        instruction = instruction_dict['instruction']
        schema = instruction_dict['schema'][0]
        input = instruction_dict['input']

        details = {}
        entity_type = schema['entity_type']
        attributes = schema['attributes']
        
        format_info = [ID, instruction, schema, input, input_dict]
        # prompt = prompt_construct(format_info)
        # format_info.append(prompt)
        data_container.append(format_info)
    return data_container