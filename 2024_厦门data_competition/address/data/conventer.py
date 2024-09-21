import pandas as pd
import re
from collections import defaultdict
import pickle

ROAD_NAME = set()
CITY = '厦门市'
AREA = "思明区"
ROAD_SET = defaultdict(set)

def mark_slot(raw_text, entity, entity_type, slots):
    index = raw_text.index(entity)
    size = len(entity)
    if len(entity) == 1:
        slots[index] = 'S-' + entity_type
    elif len(entity) == 2:
        slots[index] = 'B-' + entity_type
        slots[index+1] = 'E-' + entity_type
    else:
        slots[index] = 'B-' + entity_type
        slots[index+size-1] = 'E-' + entity_type
        slots[index+1: index+size-1] = ['I-' + entity_type] * (size-2)
    
    return slots

# 一些重点小区等，关联路
def train_data_maker():
    with open('train.txt', 'w') as fo:
        df = pd.read_csv('初赛训练集.csv')
        for row in df.itertuples():
            raw_address = getattr(row, '非标地址').replace(' ', '').replace('號', '号').replace('$', '号').replace('#', '号')
            format_address = getattr(row,'对应标准地址').replace('福建省厦门市思明区', '')
            res = re.split('([a-zA-Z]?\d+室)', format_address)

            if len(res) == 1:
                road_name = res[0]
                house_num = None
            elif len(res) == 3:
                house_num = res[1]
                road_name = res[0]
            
            # res = re.split('(\d+号|\d+-\d+号|-\d+号)', road_name)
            res = re.split('(\d+.*号|-\d+号)', road_name)
            road_num = res[1]
            road_name = res[0]

            ROAD_SET['思明区'].add(road_name)
            slots = ['O'] * len(raw_address)
            if road_name in raw_address:
                mark_slot(raw_address, road_name, 'road_name', slots)
                # print(f"{raw_address}\t{road_name}\n{slots}")
            else:
                # print(f"{raw_address}\t{road_name}")
                pass

            if road_num in raw_address:
                mark_slot(raw_address, road_num, 'road_num', slots)
                # print(f"{raw_address}\t{road_num}\n{slots}")
            
            if house_num and house_num in raw_address:
                mark_slot(raw_address, house_num, 'house_num', slots)
                # print(f"{raw_address}\t{house_num}\n{slots}")
            fo.write(f'{raw_address}\t{format_address}\n')
            fo.write(' '.join(slots) + "\n")
    with open('kg.cpickle', 'wb') as kg:
        pickle.dump(ROAD_SET, kg)
    print(ROAD_SET)

if __name__ == '__main__':
    train_data_maker()