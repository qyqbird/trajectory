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
        {attribute key:attribute value（attribute未提及不输出;数值转为string类型;True/False转换为是或否;注意key,value都要去重）}
    }
}
1.输出JSON对象，不要多余的信息,JSON中所有字段都要去重
2.未提及属性不输出
"""


EXAMPLE = """\nExample1:\nschema: [{\"entity_type\": \"运动员\", \"attributes\": {\"运动项目\": \"体育运动中的特定活动或比赛。\", \"主要奖项\": \"因卓越表现而获得的荣誉性奖项。\",  \"出生地\": \"人物出生的具体地点。\", \"死亡日期\": \"去世的时间。\"}}]
input: 叶乔波，女，1964年6月3日出生于吉林省长春市  。中国女子速滑运动员。北京冬奥组委运动员委员会委员  。 10岁进>入长春市业余体校速滑班，12岁入选八一速滑队。1991年首次夺得500米短道速滑世界冠军；1992年获第十六届冬奥会两枚银牌  ，为中国冬季项目实现冬奥会上奖牌零的突破。同年在挪威举行的世界短距离速滑锦标赛上，>获女子1000米速滑冠军，并夺得女子全能世界冠军，成为中国和亚洲第一个短距离速滑全能世界冠军；至1993年春季赛事结束，她共获得14个世界冠军，其中包括全部女子500米速滑金牌，创造了世界冰坛的“大满贯”战绩；1994年带伤夺得第17届冬奥会女子1000米速滑铜牌  ；冬奥会后因伤退役。1994年结束运动员生涯。 2000年清华大学经管学院毕业，少将（2006年晋升）。  2021年12月，由叶乔波等55位世界冠军共同唱响《我们北京见》MV发布。
output: "{\"运动员\": {\"叶乔波\": {\"运动项目\": \"速滑\", \"主要奖项\": [\"500米短道速滑世界冠军\", \"第十六届冬奥会两枚银牌\", \"女子1000米速滑冠军\", \"女子全能世界冠军\", \"全部女子50 >米速滑金牌\", \"第17届冬奥会女子1000米速滑铜牌\"],  \"出生地\": \"吉林省长春市\", \"死亡日期\":[]}}}\n\n
"""

def create_uie_prompt_construct(info_fields, source="default"):
    EXAMPLE = PROMPT_EXAMPLES[source][0]
    schema = json.dumps(info_fields[2], ensure_ascii=False)
    prompt = f"{pre_instruction}\nExample1:{EXAMPLE}\nQuestion:\nschema:{schema}\nInput:{info_fields[3]}\noutput(只输出JSON):"
    return prompt


def create_repair_json_prompt(question):
    prompt = f"有段文本，帮我转为JSON对象。可能存在符号问题（比如括号不匹配，分隔符包含中文逗号，中文冒号）\n{question}"
    return prompt


