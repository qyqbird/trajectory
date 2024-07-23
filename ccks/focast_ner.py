import json
from openai import OpenAI
import time
from vllm import LLM, SamplingParams


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
    
    # 数据分组，batch 推理
    return data_container


WARNING_INFO = "输出 1.纯简洁的JSON对象，非markdown格式"
def prompt_construct(info_fields):
    schema = json.dumps(info_fields[2], ensure_ascii=False)
    # print(schema)
    prompt = info_fields[1] + "\nschema:" + schema + "\nInput:" + info_fields[3] + "\n" + WARNING_INFO
    return prompt

def qwen2_7B_online_api(question):
    # python -m vllm.entrypoints.openai.api_server --model Qwen2-7B-Instruct-AWQ --max-model-len 8096
    # 想要批处理，怎么搞呢,无法批处理
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
        stop="True",
        response_format={"type": "json_object"},    # 很慢了
    )
    return chat_response.choices[0].message.content

def qwen2_7B_offline_api(questions):
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens = 512, repetition_penalty=2)
    llm = LLM(model="/root/Qwen2-7B-Instruct-AWQ", enforce_eager=True,trust_remote_code=True,revision="v1.1.8",)
    outputs = llm.generate(questions, sampling_params)

    result = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        result.append(generated_text)
    
    return result

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
    content = chat_response.choices[0].message.content
    print(type(content))
    print(content)
    return content

def hf_UniversalNER_7B(question):
    pass

def instruct_all():
    data = data_parser()
    fo = open('qwen2-7B-final', 'w')
    start_time = time.time()
    # for info in data:
    response = qwen2_7B_offline_api(data)
    fo.write(response + "\n")
    
    consume = (time.time() - start_time) * 1000 / len(data)
    fo.close()
    print(f"mean request:{consume}")

if __name__ == '__main__':
    instruct_all()
