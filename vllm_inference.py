from openai import OpenAI
from HF_inference import questions
import time
import torch
from utils import running_time
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

@running_time
def bf16_test():
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-7B
# 吞吐量 52 tokens/s
    for question in questions:
        chat_response = client.chat.completions.create(
            model="/root/Qwen2-7B",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]
        )
        print("Chat response:", chat_response)
        print('\n-------------\n')

# bf16_test()

@running_time
def awq_test():
    # 2.x s
    # python -m vllm.entrypoints.openai.api_server --model /root/Qwen2-7B-Instruct-AWQ
    for question in questions:
        chat_response = client.chat.completions.create(
            model="/root/Qwen2-7B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]
        )
        print("Chat response:", chat_response)
        print(type(chat_response))
    print(f"{torch.cuda.max_memory_allocated(0)//1e6}MB")


def vllm_code_test():
    # 平均1.xs
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens = 512, repetition_penalty=2)
    llm = LLM(model="/root/Qwen2-7B", revision="v1.1.8", trust_remote_code=True)
    start_time = time.time()
    outputs = llm.generate(questions, 
                    sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}\nGenerated text: {generated_text}")
    consume = 1000*(time.time() - start_time)/len(questions)
    print(f"{torch.cuda.max_memory_allocated(0)//1e6}MB")
    print(f"mean time:{consume}")


def vllm_UniversalNER_7B_online(question):
    # python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/UniNER-7B-all --max-model-len 2048
    chat_response = OpenAI_client.chat.completions.create(
        model="/root/autodl-tmp/UniNER-7B-all",
        messages=[
            {"role": "assistant", "content": "You are a helpful assistant."},
            # {"role": "system", "content": "你是一名知识图谱，信息抽取专家，负责解答用户的抽取任务"},
            {"role": "user", "content": "hello world"},
        ],
        max_tokens=1024,
        response_format={"type": "json_object"},    # 配置了一些参数后，throughout明显变慢 7.8 token/s
    )
    content = chat_response.choices[0].message.content
    print(type(content))
    print(content)
    return content

if __name__ == '__main__':
    awq_test()