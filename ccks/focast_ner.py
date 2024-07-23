import json
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def data_parser(filename=None):
    data_container = []
    # for line in open('ccks2024复赛.json'):
    for line in open('test'):
        input_dict = json.loads(line.strip())
        ID = input_dict['id']
        instruction_dict = json.loads(input_dict['instruction'])
        instruction = instruction_dict['instruction']
        schema = instruction_dict['schema'][0]
        input = instruction_dict['input']

        details = {}
        entity_type = schema['entity_type']
        attributes = schema['attributes']
        
        format_info = [ID, instruction, schema, input]
        prompt = prompt_construct(format_info)
        format_info.append(prompt)
        data_container.append(format_info)
    return data_container


WARNING_INFO = "输出必须满足 1.JSON 2. 无关信息不输出"
def prompt_construct(info_fields):
    schema = json.dumps(info_fields[2], ensure_ascii=False)
    # print(schema)
    prompt = info_fields[1] + "\nschema:" + schema + "\nInput:" + info_fields[3] + "\n" + WARNING_INFO
    return prompt

def qwen2_7B_api(question):
    # python -m vllm.entrypoints.openai.api_server --model Qwen2-7B-Instruct-AWQ --max-model-len 8096
    chat_response = client.chat.completions.create(
        model="Qwen2-7B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "你是一名知识图谱，信息抽取专家，负责解答用户的抽取任务"},
            {"role": "user", "content": question},
        ],
        temperature=0,
        frequency_penalty=2,
        max_tokens=1024,
        # response_format={"type": "json_object"},    # 配置了一些参数后，throughout明显变慢 7.8 token/s
    )
    content = chat_response.choices[0].message.content.replace("    ", "")
    return content

def vllm_UniversalNER_7B(question):
    # python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/UniNER-7B-all --max-model-len 2048
    chat_response = client.chat.completions.create(
        model="/root/autodl-tmp/UniNER-7B-all",
        messages=[
            {"role": "assistant", "content": "You are a helpful assistant."},
            # {"role": "system", "content": "你是一名知识图谱，信息抽取专家，负责解答用户的抽取任务"},
            {"role": "user", "content": "hello world"},
        ],
        max_tokens=1024
        # response_format={"type": "json_object"},    # 配置了一些参数后，throughout明显变慢 7.8 token/s
    )
    content = chat_response.choices[0].message.content.replace("    ", "")
    return content

def hf_UniversalNER_7B(question):
    pass

def instruct_all():
    data = data_parser()
    fo = open('qwen2-7B-final', 'w')
    for info in data:
        response = qwen2_7B_api(info[-1])
        fo.write(response + "\n")
    fo.close()

if __name__ == '__main__':
    instruct_all()
