import sys
import json
from collections import defaultdict
import pprint
import random

# 把IEpile 数据格式转换为chatml
def iepile_generate_charml(input_file, output_file):
    fo = open(output_file, 'w')
    for line in open(input_file):
        fields = json.loads(line.strip())
        task = fields['task']
        source = fields['source']
        instructions = json.loads(fields['instruction'])

        schema = instructions['schema']
        instruction = instructions['instruction']
        input = instructions['input']
        output = fields['output']

        prompt = f"{instruction}\nschema:{schema}\ninput:{input}"
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role":"user", "content":prompt},{"role":"assistant", "content":output}]
        result = {"type":"chatml", "source":source, "messages":messages}
        fo.write(json.dumps(result,ensure_ascii=False) + "\n")
    fo.close()


def schema_structure():
    # TODO 个性化description, 同义词改写等都应该在这里
    schema_json = json.load(open('/root/autodl-tmp/InstructIE/schema_zh_detail.json', 'r'))
    result = {}
    for cate, values in schema_json.items():
        detail = {}
        attributes = values[1]
        relation_definition = [item.split('_') for item in values[0]]
        for rel in relation_definition:
            if rel[1] in attributes:
                detail[rel[1]] = rel[1]


        full_schema = {"entity_type":cate}
        full_schema['attributes'] = dict(detail)
        detail['schema'] = [full_schema]
        result[cate] = detail
        # pprint.pprint(detail)
    return result


# 输入数据分析, 相当于会有释义改写
def description_extract():
    attributes_descripter = defaultdict(set)
    entities_descripter = defaultdict(int)
    for line in open('../data/ccks2024复赛.json'):
        input_dict = json.loads(line.strip())
        instruction_dict = json.loads(input_dict['instruction'])
        schema = instruction_dict['schema'][0]

        details = {}
        entity_type = schema['entity_type']
        attributes = schema['attributes']
        entities_descripter[entity_type] += 1
        for attri, desc in attributes.items():
            attributes_descripter[attri].add(desc)
    # print(len(attributes_descripter))
    # 输出一些统计信息
    print(f"entities:{len(entities_descripter)}")
    print(f"attributes:{len(attributes_descripter)}")
    print("-------------entities---------------")
    pprint.pprint(entities_descripter)
    print("-------------attributes---------------")
    pprint.pprint(attributes_descripter)

    return attributes_descripter

# InstructIE 格式转换为chatml
def instructie_2_chatml(input_file, output_file):
    schema_dict = schema_structure()
    fo = open(output_file, 'w')
    for line in open(input_file):
        fields = json.loads(line.strip())
        input = fields['text']
        relations = fields['relation']
        if 'cate' in fields:
            cate = fields['cate']
        else:
            continue

        if cate not in ["人物", '生物', '作品', '医学', '组织', '地理地区']:
            continue
        
        #同一个head 要合并
        first_json = defaultdict(dict)
        #{人物:{xx:{}}}
        schema = defaultdict(dict)
        for relation in relations:
            head = relation['head']
            head_type = relation['head_type']
            schema['entity_type'] = head_type
            third_dict = first_json[head_type].get(head, {})
            relation_type = relation['relation']
            tail = relation['tail']
            if relation_type in third_dict:
                third_dict[relation_type].append(tail)
            else:
                third_dict[relation_type] = [tail]
            schema['attributes'][relation_type] = schema_dict[cate][relation_type]
            first_json[head_type][head] = third_dict

        schema = json.dumps([dict(schema)], ensure_ascii=False)
        first_json = dict(first_json)
        if random.random() <= 0.05 and cate in schema_dict:
            schema = json.dumps(schema_dict[cate]['schema'], ensure_ascii=False)


        if len(relations) > 0:
            instruction = "你是实体/属性抽取、知识结构化专家。根据输入实体类型(entity type)的schema描述，从文本中抽取出相应的实体及属性信息，并返回json对象。"
            prompt = f"{instruction}\nschema:{schema}\ninput:{input}"
            output = json.dumps(first_json, ensure_ascii=False)
            messages = [{"role": "system", "content": "你是Qwen,一名智能助手."}, {"role":"user", "content":prompt},{"role":"assistant", "content": output}]
            result = {"type":"chatml", "source":"self-made", "messages":messages}
            fo.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            # print(line)
            pass

    
    fo.close()
'''
TODO
1. schema添加无输出的定义
2. schema的长度变化
3. 需要对输出数据进行2次校验，过滤操作吗？
'''



def generate_ask4_format():
    schema = [{"entity_type": "菜品", "attributes": {"主要食材": "菜品的主要构成部分，通常是味道和营养的主要来源。", "辅材": "辅助食材，虽非主要但对菜品口感或外观有贡献。", "调料": "在烹饪中用于增添风味的物质, 如酱油、盐或香料。", "制作工艺": "描述制作某物品或食品的具体步骤或技术。", "口味": "描述菜品的风味特点，如酸、甜、咸、辣等。"}}]
    prompt = "你是一个图谱实体知识结构化专家。根据输入实体类型(entity type)的schema描述，从文本中抽取出相应的实体实例和其属性信息，不存在的属性不输出, 属性存在多值就返回列表，并输出为可解析的json格式。"    
    ID = 1000
    fo = open('caipin', 'w')
    for line in open('../data/human'):
        line = line.strip()
        instruction = {"instruction": prompt, "schema": schema, "input":line}
        fm = {"id":ID, "instruction": json.dumps(instruction, ensure_ascii=False) }
        ID += 1
        # pprint.pprint(fm)
        fo.write(json.dumps(fm, ensure_ascii=False) + "\n")

    fo.close()

if __name__ == '__main__':
    # instructie_2_chatml("/root/autodl-tmp/InstructIE/train_zh_plus.json", "/root/autodl-tmp/InstructIE/ccks/Instruct-chatml.jsonl")
    # schema_structure()
    description_extract()
    # generate_ask4_format()