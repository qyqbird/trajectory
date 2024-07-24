import json
import time
from vllm import LLM, SamplingParams
from common_utils import OpenAI_client, data_parser
import pprint
import re

pre_instruction = """你是知识图谱、实体抽取、知识结构化专家。根据输入实体类型(entity type)的schema描述，从文本中抽取出相应的实体实例和其属性信息，并返回json对象。
json格式:
{entity type:
    {entity:
        {attribute key:attribute value（attribute value为空不输出;数值转为string类型;True/False转换为是或否;注意key,value都要去重）}
    }
}
1.只输出JSON对象，不要多余的信息,JSON中所有字段都要去重
2.不存在属性值就不输出，属性存在多值以列表返回
3.
"""

EXAMPLE = """\nExample1:\nschema: [{\"entity_type\": \"运动员\", \"attributes\": {\"运动项目\": \"体育运动中的特定活动或比赛。\", \"主要奖项\": \"因卓越表现而获得的荣誉性奖项。\",  \"出生地\": \"人物出生的具体地点。\", \"死亡日期\": \"去世的时间。\"}}]
input: 叶乔波，女，1964年6月3日出生于吉林省长春市  。中国女子速滑运动员。北京冬奥组委运动员委员会委员  。 10岁进>入长春市业余体校速滑班，12岁入选八一速滑队。1991年首次夺得500米短道速滑世界冠军；1992年获第十六届冬奥会两枚银牌  ，为中国冬季项目实现冬奥会上奖牌零的突破。同年在挪威举行的世界短距离速滑锦标赛上，>获女子1000米速滑冠军，并夺得女子全能世界冠军，成为中国和亚洲第一个短距离速滑全能世界冠军；至1993年春季赛事结束，她共获得14个世界冠军，其中包括全部女子500米速滑金牌，创造了世界冰坛的“大满贯”战绩；1994年带伤夺得第17届冬奥会女子1000米速滑铜牌  ；冬奥会后因伤退役。1994年结束运动员生涯。 2000年清华大学经管学院毕业，少将（2006年晋升）。  2021年12月，由叶乔波等55位世界冠军共同唱响《我们北京见》MV发布。
Anwser: "{\"运动员\": {\"叶乔波\": {\"运动项目\": \"速滑\", \"主要奖项\": [\"500米短道速滑世界冠军\", \"第十六届冬奥会两枚银牌\", \"女子1000米速滑冠军\", \"女子全能世界冠军\", \"全部女子50 >米速滑金牌\", \"第17届冬奥会女子1000米速滑铜牌\"],  \"出生地\": \"吉林省长春市\"}}}\n\n
"""

def prompt_construct(info_fields):
    schema = json.dumps(info_fields[2], ensure_ascii=False)
    prompt = f"{pre_instruction}\n{EXAMPLE}\nQuestion:\nschema:{schema}\nInput:{info_fields[3]}\nAnwser(只输出JSON):"
    return prompt

def format_align(info_fields, completion):
    # 值校验
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

    return completion

def delete_unrelated(completion):
    #1. JSON + schema:xxx
    if 'schema:' in completion:
        idx = completion.index('schema:')
        completion = completion[:idx].strip()
    if 'Question:' in completion:
        completion = completion.replace('Question:', '')
    #2. ```JSON``` 正则匹配
    res = re.findall(r'.*```json(.*)```*', completion, re.DOTALL)
    if len(res) > 0:
        completion = res[0].strip()
    completion = completion.strip().replace("\n", "")
    return completion

def post_process(data, completion):
    print(completion)
    completion = delete_unrelated(completion)
    completion = format_align(data, completion)

    fo = open('data/qwen2-7B-instruct-final.json', 'w')
    for input,res in zip(data, completion):
        input[-1]["output"] = res
        fo.write(json.dumps(input[-1], ensure_ascii=False) + "\n")
    fo.close()

def qwen2_7B_offline_api(questions):
    prompts = [prompt_construct(info) for info in questions]
    # 的确可以batch size 操作
    sampling_params = SamplingParams(temperature=0.1, 
                                     top_p=0.95, 
                                     max_tokens = 512, 
                                     presence_penalty=-2,
                                    #  repetition_penalty=2,
                                     )
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
    total_time = time.time()
    consume = (total_time - start_time) * 1000 / len(data)
    print(f"total time:{total_time}s\nmean request:{consume}ms")

    completion = post_process(data, completion)

if __name__ == '__main__':
    task_pipe()
