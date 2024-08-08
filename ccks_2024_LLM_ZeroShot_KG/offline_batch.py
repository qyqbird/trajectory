import json
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from common_utils import OpenAI_client, data_parser, schema_output_format_align, delete_unrelated
from prompt_utils import create_uie_prompt_construct


def post_process(data, completions):
    with open('data/qwen2-72B-middle.json', 'w') as middle:
        for input, res in zip(data, completions):
            input[-1]["output"] = res
            print(res)
            middle.write(json.dumps(input[-1], ensure_ascii=False) + "\n")

    completions = [delete_unrelated(completion) for completion in completions]
    completions = [schema_output_format_align(data[idx], completion) for idx, completion in enumerate(completions)]

    with open('data/qwen2-72B-final.json', 'w') as fo:
        for input, res in zip(data, completions):
            input[-1]["output"] = res
            fo.write(json.dumps(input[-1], ensure_ascii=False) + "\n")

def qwen2_7B_offline_api(questions):
    prompts = [create_uie_prompt_construct(info) for info in questions]
    # 的确可以batch size 操作
    sampling_params = SamplingParams(temperature=0, 
                                     top_p=0.95, 
                                     max_tokens = 512, 
                                     presence_penalty=0.1,
                                     frequency_penalty=0.1,
                                     repetition_penalty=0.9,
                                     stop=["}}}"],
                                     include_stop_str_in_output=True
                                     )
    llm = LLM(model="/root/Qwen2-7B-Instruct-AWQ", enforce_eager=True,
              trust_remote_code=True,revision="v1.1.8",max_seq_len_to_capture=8192)
    # 没有生效：max_seq_len_to_capture
    outputs = llm.generate(prompts, sampling_params)
    
    result = []
    for output in outputs:
        #output.finished
        completion_1 = output.outputs[0].text.strip()
        result.append(completion_1)
    return result


def qwen2_7B_offline_lora_api(questions, ner_lora_path):
    prompts = [create_uie_prompt_construct(info) for info in questions]
    # 的确可以batch size 操作
    sampling_params = SamplingParams(temperature=0, 
                                     top_p=0.95, 
                                     max_tokens = 512, 
                                     presence_penalty=0.1,
                                     frequency_penalty=0.1,
                                     repetition_penalty=0.9,
                                     stop=["}}}"],
                                     include_stop_str_in_output=True
                                     )
    llm = LLM(model="/root/Qwen2-7B-Instruct-AWQ", enforce_eager=True,enable_lora=True,
              trust_remote_code=True,revision="v1.1.8",max_seq_len_to_capture=8192,max_lora_rank=32)
    # 没有生效：max_seq_len_to_capture, 同理max_lora_rank 没啥用
    outputs = llm.generate(prompts, sampling_params,lora_request=LoRARequest("ner_adapter", 1, ner_lora_path))
    
    result = []
    for output in outputs:
        #output.finished
        completion_1 = output.outputs[0].text.strip()
        result.append(completion_1)
    return result

def qwen2_72B_offline_api(questions):
    prompts = [create_uie_prompt_construct(info) for info in questions]
    # python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/Qwen2-72B-Instruct-AWQ --max-model-len 1024  成功启动
    #L20 48G显卡， OOM。原因是：max_seq_len=32768,  设置的参数失效。解决方案：直接去config.json 修改
    sampling_params = SamplingParams(temperature=0, 
                                     top_p=0.95, 
                                     max_tokens = 512, 
                                     presence_penalty=0.0,
                                     frequency_penalty=0.0,
                                     repetition_penalty=1.0,
                                     stop=["}}}"],
                                     include_stop_str_in_output=True
                                     )
    # 在Qwen源头进行修改max_pos
    llm = LLM(model="/root/autodl-tmp/Qwen2-72B-Instruct-AWQ",enforce_eager=True,
              max_seq_len_to_capture=4096,gpu_memory_utilization=0.95)
    # max_seq_len_to_capture=4096 不影响最初模型加载
    # gpu_memory_utilization 也不影响
    outputs = llm.generate(prompts, sampling_params)
    
    result = []
    for output in outputs:
        #output.finished
        completion_1 = output.outputs[0].text.strip()
        result.append(completion_1)
    return result

def UniversalNER_7B_offline_api(questions):
    #27G
    prompts = [create_uie_prompt_construct(info) for info in questions]
    sampling_params = SamplingParams(temperature=0, 
                                     top_p=0.95, 
                                     max_tokens = 512, 
                                     presence_penalty=0.1,
                                     frequency_penalty=0.1,
                                     repetition_penalty=0.9,
                                     stop_token_ids=["}}}"],
                                     )
    # include_stop_str_in_output=True 没生效，结果没包含 }}}
    llm = LLM(model="/root/autodl-tmp/UniNER-7B-all", enforce_eager=True,
              trust_remote_code=True,revision="v1.1.8",max_seq_len_to_capture=8192)
    outputs = llm.generate(prompts, sampling_params)
    
    result = []
    for output in outputs:
        #output.finished
        completion_1 = output.outputs[0].text.strip()
        result.append(completion_1)
    return result


def task_pipe():
    data = data_parser('data/ccks2024复赛.json')
    start_time = time.time()
    completion = qwen2_72B_offline_api(data)
    # model_name = "qwen2-7B-lora-awq"
    # lora_path = "/root/workspace/Qwen2/examples/sft/output_qwen"
    # completion = qwen2_7B_offline_lora_api(data, lora_path)
    total_time = time.time() - start_time
    mean_time = total_time * 1000 / len(data)
    print(f"total time:{total_time}s\nmean request:{mean_time}ms")
    completion = post_process(data, completion)

if __name__ == '__main__':
    task_pipe()
