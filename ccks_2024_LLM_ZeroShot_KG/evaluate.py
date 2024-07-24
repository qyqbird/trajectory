# coding=utf-8
import json
import sys
from copy import deepcopy
import pprint

class InputidsError(Exception):
    def __init__(self, message) :
        super().__init__(message)
        self.message=message

class JsonlenthError(Exception):
    def __init__(self, message) :
        super().__init__(message)
        self.message=message

class KGMetric():
    def __init__(self, match_mode="normal"):
        super().__init__()
        self.match_mode = match_mode
        self.pred_num = 0
        self.gold_num = 0
        self.tp = 0

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return round(a / b * 100, 2)

    @staticmethod
    def safe_div_(a, b):
        if b == 0.:
            return 0.
        else:
            return round(a / b, 2)

    def compute_f1(self):
        result = {}
        result['precision'] = self.safe_div(self.tp, self.pred_num)
        result['recall'] = self.safe_div(self.tp, self.gold_num)
        result['f1-score'] = self.safe_div_(2 * result['precision'] * result['recall'], result['precision'] + result['recall'])
        return result
    
    def kg2re(self, kgs):
        res = []
        for ent_type, ent_name, attris in kgs:
            for key, value in attris.items():
                if type(value) == list:
                    for it in value:
                        res.append((ent_name, key, it))
                else:
                    res.append((ent_name, key, value))
        return res

    def _count_instance_f1(self, gold_list, pred_list):
        pred_list = self.kg2re(pred_list)
        gold_list = self.kg2re(gold_list)

        gold_list = [tuple(it) for it in gold_list]
        pred_list = [tuple(it) for it in pred_list]
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
        dup_gold_list = deepcopy(gold_list)
        dup_pred_list = deepcopy(pred_list)
        
        self.pred_num += len(pred_list)
        self.gold_num += len(gold_list)

        pred_rel_types = set()
        for pred in pred_list:
            pred_rel_types.add(pred[1])
            if pred in dup_gold_list:
                self.tp += 1
                dup_gold_list.remove(pred)
                dup_pred_list.remove(pred)

    def count_instance_f1(self, gold_list, pred_list):
        self._count_instance_f1(gold_list, pred_list)

kg_metric = KGMetric()
id_mapper = {}

# 错误字典，这里只是示例
error_msg={
    1: "Wrong ids",
    2: "Wrong submit file lenth",
}

def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file)

def report_error_msg(detail, showMsg, out_p):
    error_dict=dict()
    error_dict['errorDetail']=detail
    error_dict['errorMsg']=showMsg
    error_dict['score']=0
    error_dict['scoreJson']={}
    error_dict['success']=False
    dump_2_json(error_dict,out_p)

def report_score(score, out_p):
    result = dict()
    result['success']=True
    result['score'] = score['f1-score']
    result['scoreJson'] = {'score':score['f1-score'],'precision':score['precision'],'recall':score['recall'],'f1-score':score['f1-score']}

    dump_2_json(result,out_p)

def check_format1(submit_path):
    with open(submit_path, 'r',encoding='utf-8') as reader:
        x = 0
        for line in reader:
            data = json.loads(line)
            if data['id'] not in id_mapper:
                raise InputidsError("Wrong ids")
            x += 1
        if(x!=1000):
            raise JsonlenthError("Wrong submit file lenth")

def post_process4(result, attribute_keys):
    try:      
        rst = json.loads(result)
    except json.decoder.JSONDecodeError:
        return False, []
    if type(rst) != dict:
        return False, []
    new_record = []
    for key, values in rst.items():  # entity_type
        if type(key) != str or type(values) != dict:
            continue
        for key1, values1 in values.items():   # entity, attributes
            if type(key1) != str or type(values1) != dict:
                continue
            attris = {}
            for key2, values2 in values1.items(): # key, value
                if key2 not in attribute_keys:
                    print(f"{key2}\t{values2}\n{result}")
                if type(values2) == list:
                    attri_value = []
                    for iit in values2:
                        if type(iit) != str:
                            continue
                        if iit == '无' or iit.lower() == 'nan':
                            continue
                        attri_value.append(iit)
                    if len(attri_value) > 0:
                        attris[key2] = attri_value
                elif type(values2) == str:
                    if values2 == '无' or values2.lower() == 'nan':
                        continue
                    attris[key2] = values2
                else:
                    pass
                    # print(f"{values2}\n{result}")
            new_record.append((key, key1, attris)) 
    return True, new_record

def convert_kg(outputs):
    kgs = []
    for entity_type, value1 in outputs.items():
        for entity_name, attributes in value1.items():
            kgs.append((entity_type, entity_name, attributes))
    return kgs

if __name__=="__main__":
    submit_path = "/Users/buring/study/competition/CCKS2024——大模型零样本知识抽取评测/submit.json"
    with open(submit_path, 'r',encoding='utf-8') as reader:
        for line in reader:
            # ori 解析，看是否能够通过
            data = json.loads(line)
            instruction = json.loads(data['instruction'])
            attribute_keys = []
            schemas = instruction['schema']
            for schema in schemas:
                entity_type = schema['entity_type']
                attributes = schema['attributes']
                attribute_keys.append(entity_type)
                attribute_keys.extend(attributes.keys())
            # print('\t'.join(attribute_keys))
            flag, pred = post_process4(data['output'], attribute_keys)

            # print(instruction['input'])
            # pprint.pprint(pred)
            # print('')

