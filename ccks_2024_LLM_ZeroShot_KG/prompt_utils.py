import json

def load_examples(filename="config/prompt_examples.json"):
    with open(filename) as fo:
        prompt_examples = fo.read()
    parsed_json = json.loads(prompt_examples)
    return parsed_json
PROMPT_EXAMPLES = load_examples()
# print(PROMPT_EXAMPLES['default']['input'])
# schema = PROMPT_EXAMPLES['default']['schema']
# output = PROMPT_EXAMPLES['default']['output']
# print(json.loads(output))


pre_instruction = """你是知识图谱、实体抽取、知识结构化专家。根据输入实体类型(entity type)的schema描述，从文本中抽取出相应的实体实例和其属性信息，并返回json对象。
json格式:
{entity type:
    {entity:
        {attribute key:attribute value（attribute未提及不输出;数值转为string类型;注意key,value都要去重）}
    }
}
1. 输出一个JSON对象，不要多余的信息,JSON中所有字段都要去重。
2. 属性未提及则不抽取该属性; 属性有多个值，返回列表。
3. entity未提及则返回{}；
"""

def schema_optimizer(schema):
    #TODO 这里需要做一组实验
    # entity_type = schema['entity_type']
    # if entity_type == "实体":
    #     entity_type = "实体类型:结合最近的schema和input推理得到"
    #     schema['entity_type'] = entity_type
    attributes = schema['attributes']
    for attri, descri in attributes.items():
        if descri == 'NAN':
            attributes[attri] = attri
        if descri == "外文名" and '非中文' not in descri:
            attributes[attri] = "非中文名称，" + descri
    return schema

def create_uie_prompt_construct(info_fields):
    demo_fields = PROMPT_EXAMPLES['3']
    demo = f"这里有几个例子:\n{PROMPT_EXAMPLES['3']}\n\n{PROMPT_EXAMPLES['2']}\n"
    schema = schema_optimizer(info_fields[2])
    schema = json.dumps(info_fields[2], ensure_ascii=False)
    prompt = f"{pre_instruction}\n\n{demo}\nschema:{schema}\nInput:{info_fields[3]}\noutput(只输出JSON):"
    # print(prompt)
    return prompt


def create_repair_json_prompt(question):
    prompt = f"有段文本，帮我转为JSON对象。可能存在符号问题（比如括号不匹配，分隔符包含中文逗号，中文冒号）\n{question}"
    return prompt


