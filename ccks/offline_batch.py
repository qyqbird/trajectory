import json
import time
from vllm import LLM, SamplingParams
from common_utils import OpenAI_client



def data_parser(filename=None):
    data_container = []
    for line in open('ccks2024复赛.json'):
    # for line in open('test'):
        input_dict = json.loads(line.strip())
        ID = input_dict['id']
        instruction_dict = json.loads(input_dict['instruction'])
        instruction = instruction_dict['instruction']
        schema = instruction_dict['schema'][0]
        input = instruction_dict['input']

        details = {}
        entity_type = schema['entity_type']
        attributes = schema['attributes']
        
        format_info = [input_dict, ID, instruction, schema, input]
        prompt = prompt_construct(format_info)
        format_info.append(prompt)
        data_container.append(format_info)
    
    # 数据分组，batch 推理
    return data_container

pre_instruction = """你是知识图谱、实体抽取、知识结构化专家。根据输入实体类型(entity type)的schema描述，从文本中抽取出相应的实体实例和其属性信息（不存在的属性不输出, 属性存在多值就返回列表），并返回json对象(额外信息不要输出)。
json格式:
{entity_type:
    {entity:
        {attribute_key:attribute_value（attribute_value为空不输出;数值类型统一转为string类型;True/False转换为是或否;注意key,value都要去重）
        }
    }
}
只输出JSON对象，不需要任何其他信息,JSON注意字段去重
"""

EXAMPLE = """schema: [{\"entity_type\": \"运动员\", \"attributes\": {\"运动项目\": \"体育运动中的特定活动或比赛。\", \"主要奖项\": \"因卓越表现而获得的荣誉性奖项。\", \"毕业院校\": \"人物所毕业的学校。\", \"出生地\": \"人物出生的具体地点。\"}}], 
\"input\": \"叶乔波，女，1964年6月3日出生于吉林省长春市  。中国女子速滑运动员。北京冬奥组委运动员委员会委员  。 10岁进>入长春市业余体校速滑班，12岁入选八一速滑队。1991年首次夺得500米短道速滑世界冠军；1992年获第十六届冬奥会两枚银牌  ，为中国冬季项目实现冬奥会上奖牌零的突破。同年在挪威举行的世界短距离速滑锦标赛上，>获女子1000米速滑冠军，并夺得女子全能世界冠军，成为中国和亚洲第一个短距离速滑全能世界冠军；至1993年春季赛事结束，她共获得14个世界冠军，其中包括全部女子500米速滑金牌，创造了世界冰坛的“大满贯”战绩；1994年带伤夺得第17届冬奥会女子1000米速滑铜牌  ；冬奥会后因伤退役。1994年结束运动员生涯。 2000年清华大学经管学院毕业，少将（2006年晋升）。  2021年12月，由叶乔波等55位世界冠军共同唱响《我们北京见》MV发布。\"}",
"Anwser": "{\"运动员\": {\"叶乔波\": {\"运动项目\": \"速滑\", \"主要奖项\": [\"500米短道速滑世界冠军\", \"第十六届冬奥会两枚银牌\", \"女子1000米速滑冠军\", \"女子全能世界冠军\", \"全部女子50 >米速滑金牌\", \"第17届冬奥会女子1000米速滑铜牌\"], \"毕业院校\": \"清华大学经管学院\", \"出生地\": \"吉林省长春市\"}}}"
"""

def prompt_construct(info_fields):
    schema = json.dumps(info_fields[3], ensure_ascii=False)
    prompt = f"{pre_instruction}\nExample:{EXAMPLE}\n用户请求:\nschema:{schema}\nInput:{info_fields[4]}\nAnswer(对JSONkey，value都要去重):"
    return prompt

def post_process(completion):
    print(completion)
    res = completion.replace("\n", "")[1:]
    res = res.replace('```', '')
    return res

def qwen2_7B_offline_api(questions):
    prompts = [question[-1] for question in questions]
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
        print(f"Prompt: {prompt!r}, \nGenerated text: {generated_text!r}")

        post_res = post_process(generated_text)
        result.append(post_res)
    
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


def instruct_all():
    data = data_parser()
    fo = open('qwen2-7B-instruct-final.json', 'w')
    start_time = time.time()
    # for info in data:
    response = qwen2_7B_offline_api(data)
    for input,res in zip(data, response):
        input[0]["output"] = res
        fo.write(json.dumps(input[0], ensure_ascii=False) + "\n")
    
    consume = (time.time() - start_time) * 1000 / len(data)
    fo.close()
    print(f"mean request:{consume}")

if __name__ == '__main__':
    instruct_all()
