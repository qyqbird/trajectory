#coding:utf-8
import re
import os
import torch
import logging
import copy
import pandas as pd
logger = logging.getLogger(__name__)

FILTER = ['竞品手机','竞品电脑','竞品生态链','竞品手环','竞品手表','竞品耳机','竞品电视','竞品高管','竞品品牌']
def bert_tokenize_content(lines, tokenizer):
	tokens_line = []
	for line in lines:
		tokens = tokenizer.tokenize(line)
		result = []
		for token in tokens:
			if token.startswith('##'):
				token = token[2:]
			elif token == '[UNK]':
				token = ' '
			result.append(token)
		tokens_line.append(''.join(result))
	return tokens_line

def get_label2id_mapping(label_file):
	"""
	获取笔电标签体系
	:param label_file:
	:return:
	"""
	df_in = pd.read_csv(label_file).fillna("")
	category_label2id_table = {}
	for index, row in df_in.iterrows():
		category = row['cate_name'].strip()
		# 笔记本使用二级标签,其它品类使用三级标签
		third = str(row["failure_third_name"]).strip()
		second = str(row["failure_second_name"]).strip()
		first = str(row["failure_first_name"]).strip()
		third_id = str(row["failure_third_id"]).strip()
		second_id = str(row["failure_second_id"]).strip()
		first_id = str(row["failure_first_id"]).strip()

		target_dict = {}
		if category in category_label2id_table:
			target_dict = category_label2id_table[category]

		if category == "笔记本":
			target_dict[second] = [first, first_id, second, second_id, third, third_id]
		else:
			target_dict[third] = [first, first_id, second, second_id, third, third_id]
		category_label2id_table[category] = target_dict
	return category_label2id_table

def get_bizlabel_table(filename):
	dataframe = pd.read_excel(filename, sheet_name=None)	#指定为None，读取所有的sheet页面，否则只读取第一个sheet页面,返回是不一样的
	biz_label_config = {}

	for sheet_name, sheet in dataframe.items():
		aspects_opinions_label = {}	#{phone: {aspect:{opinion: value}}, "notebook":{}}
		merge_entity = set()
		for index, row in sheet.iterrows():
			aspects = row["aspect"]
			opinions = row["opinions"]
			label = str(row["label"]).strip()

			if not pd.isna(aspects)  and not pd.isna(opinions):
				aspects = str(aspects).strip().split(",")
				opinions = str(opinions).strip().split(",")
				for aspect in aspects:
					if aspect in aspects_opinions_label:
						for opinion in opinions:
							aspects_opinions_label[aspect][opinion] = label
					else:
						tmp = {}
						for opinion in opinions:
							tmp[opinion] = label
						aspects_opinions_label[aspect] = tmp
			elif not pd.isna(opinions):
				opinions = str(opinions).strip().split(",")
				if len(opinions) > 0:
					for opinion in opinions:
						tmp = {}
						tmp[opinion] = label
						aspects_opinions_label[opinion] = tmp
			elif label == "merge_key":
				merge_keys = str(aspects).strip().split(",")
				if len(merge_keys) > 0:
					merge_entity.update(merge_keys)
		print(f"{sheet_name}\tlabels:{len(aspects_opinions_label)}\tmerge_key:{len(merge_entity)}")
		print(f"{aspects_opinions_label}\nmerge_key:{merge_entity}")
		biz_label_config[sheet_name] = [aspects_opinions_label, merge_entity]

	return biz_label_config

def get_bizlabel_table(filename):
	dataframe = pd.read_excel(filename, sheet_name=None)	#指定为None，读取所有的sheet页面，否则只读取第一个sheet页面,返回是不一样的
	biz_label_config = {}

	for sheet_name, sheet in dataframe.items():
		aspects_opinions_label = {}	#{phone: {aspect:{opinion: value}}, "notebook":{}}
		merge_entity = set()
		for index, row in sheet.iterrows():
			aspects = row["aspect"]
			opinions = row["opinions"]
			label = str(row["label"]).strip()

			if not pd.isna(aspects)  and not pd.isna(opinions):
				aspects = str(aspects).strip().split(",")
				opinions = str(opinions).strip().split(",")
				for aspect in aspects:
					if aspect in aspects_opinions_label:
						for opinion in opinions:
							aspects_opinions_label[aspect][opinion] = label
					else:
						tmp = {}
						for opinion in opinions:
							tmp[opinion] = label
						aspects_opinions_label[aspect] = tmp
			elif not pd.isna(opinions):
				opinions = str(opinions).strip().split(",")
				if len(opinions) > 0:
					for opinion in opinions:
						tmp = {}
						tmp[opinion] = label
						aspects_opinions_label[opinion] = tmp
			elif label == "merge_key":
				merge_keys = str(aspects).strip().split(",")
				if len(merge_keys) > 0:
					merge_entity.update(merge_keys)
		print(f"{sheet_name}\tlabels:{len(aspects_opinions_label)}\tmerge_key:{len(merge_entity)}")
		print(f"{aspects_opinions_label}\nmerge_key:{merge_entity}")
		biz_label_config[sheet_name] = [aspects_opinions_label, merge_entity]
	return biz_label_config

#
# biz_label_config = get_bizlabel_table('aspect_fault_config.xlsx')
# print(biz_label_config)

'''
基本要用,空格切割句子了
'''
def detail_cut_sentence(content):
	result = []
	if len(content) == 0:
		return result
	if len(content) <= 78:
		result.append(content)
	else:
		bak = re.split("(，|,| )", content)
		ss_con = ""
		for ydx in range(0, len(bak), 2):
			sub = bak[ydx]
			if ydx + 1 < len(bak):
				sub += bak[ydx+1]
			ss_con += sub
			if len(ss_con) >= 66:
				result.append(ss_con)
				ss_con = ""
		if len(ss_con) > 0:
			result.append(ss_con)
	return result

#粗粒度切句子
def pre_cut_sentence(content):
	items = re.split("(。|？|！|；)", content.strip())
	result = []
	stringbuilder = ""
	for idx in range(0, len(items), 2):
		item = items[idx].strip()
		comma = ''
		if item == "":
			continue
		if idx+1 < len(items):
			comma = items[idx+1]

		bak = stringbuilder
		stringbuilder = stringbuilder + item + comma
		length = len(stringbuilder)
		if length < 60:
			continue
		elif length >= 60 and length <= 80:
			result.append(stringbuilder)
			stringbuilder = ""
		elif length >= 80:
			if len(bak) > 0:
				result.extend(detail_cut_sentence(bak))
			stringbuilder = item + comma
	if len(stringbuilder) > 0:
		result.extend(detail_cut_sentence(stringbuilder))
	result = [x.strip() for x in result]
	return result

url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
def clear_content(line):
	line = line.strip().lower().replace("”", '"').replace('“', '"').replace('➕', '+').\
		replace('—', '-').replace('’',"'").replace('…', '...')
	line = line.replace('[doge]','')
	line = line.replace('[舔屏]','')
	line = line.replace('[耐克嘴]','')
	line = line.replace('[打call]','')
	line = line.replace('\u200b','')	#<200b> 0宽断言特殊字符
	line = line.replace('[黑线]','')
	line = line.replace('[并不简单]','')
	line = line.replace('[裂开]','')
	line = line.replace('[失望]','')
	line = line.replace('💩','屎')
	line = line.replace('[衰]','').replace("[图片]", "").replace('image_emoticon', "")
	line = ' '.join(line.split())	#把多个空格替换为1个，避免分词出现少空格导致的异常
	urls = url_pattern.findall(line)
	for url in urls:
		line = line.replace(url, 'url')

	if line.startswith('"'):
		line = line[1:]
	if line.endswith('"'):
		line = line[:-1]
	#转发数据
	if re.search('回复@.*:', line):
		line = re.sub('回复@.*?:','', line)
	if re.search('@.* ', line):
		line = re.sub('@.*? ','', line)
	if re.search('@.*:', line):
		line = re.sub('@.*:','', line)

	if line.startswith("回复"):
		line = line[2:]
	if '¡评论配图' in line:
		line = line.replace('¡评论配图', '')

	line = line.strip().lower()
	if len(line) == 0:
		return []
	else:
		items = pre_cut_sentence(line)
		return items

def key_fault_insert(content, fault, extract_shouhou_map,map_info):
	if fault['label'] in extract_shouhou_map:
		target_label = extract_shouhou_map[fault['label']][fault['label']]
		target_label = label_rule_check(content, target_label, None, fault)
		map_info.append(target_label)
		return True
	return False

def entity_fault_insert(content, aspect, fault, extract_shouhou_map, map_info):
	if aspect['label'] in extract_shouhou_map and fault['label'] in extract_shouhou_map[aspect['label']]:
		target = label_rule_check(content, extract_shouhou_map[aspect['label']][fault['label']], aspect, fault)
		map_info.append(target)
		return 1
	else:
		return 0

def aspect_opinion_insert_delete(content, aspect, fault, extract_shouhou_map, map_info):
	if aspect['label'] in extract_shouhou_map and fault['label'] in extract_shouhou_map[aspect['label']]:
		target = label_rule_check(content, extract_shouhou_map[aspect['label']][fault['label']], aspect, fault)
		map_info.append(target)
	elif fault['label'] in extract_shouhou_map:
		map_info.append(extract_shouhou_map[fault['label']][fault['label']])

ZH_PUNCITON = re.compile(r"。|？|！")

# TODO: abc/ab 把标签放中间，然后过滤
def aspect_faults_match(content, line, extract_shouhou_map):
	map_info = []
	print(f"before abc:{content}\t{line}\t{map_info}")
	line = abc_pattern_merge(content, extract_shouhou_map, line, map_info)
	ab_pattern_merge(content, extract_shouhou_map, line, map_info)
	need_match_aspect = []
	need_match_opinion = None
	print(f"after abc:{content}\t{line}\t{map_info}")
	#2. 无交集部分线性搜索
	for item in line:
		if item['type'] == 'aspect':
			# print(f"{need_match_aspect}\t{need_match_opinion}\t{item}\t{map_info}")
			need_match_aspect.append(item)
			if need_match_opinion is not None:
				tmp_content = content[need_match_opinion['end']:item['start']]
				flag = -1
				if '。' not in tmp_content:
					if ',' in tmp_content or '，' in tmp_content or ' ' in tmp_content:
						if len(tmp_content) < 10:
							flag = entity_fault_insert(content, need_match_aspect[-1], need_match_opinion, extract_shouhou_map, map_info)
					elif len(tmp_content) < 16:
							flag = entity_fault_insert(content, need_match_aspect[-1], need_match_opinion, extract_shouhou_map, map_info)
				if flag > 0:
					need_match_aspect = []
					need_match_opinion = None
				elif flag <= 0:
					if flag == 0:
						logger.info(f"match error:{content}\t{item}\t{need_match_opinion}")
					key_fault_insert(content, need_match_opinion, extract_shouhou_map, map_info)
					need_match_opinion = None
		else:
			# print(f"{need_match_aspect}\t{need_match_opinion}\t{item}\t{map_info}\t{line}")
			if len(need_match_aspect) == 0:
				if need_match_opinion is not None:
					key_fault_insert(content, need_match_opinion, extract_shouhou_map, map_info)
				need_match_opinion = item
			else:
				tmp_content = content[need_match_aspect[-1]['end']:item['start']]
				if len(ZH_PUNCITON.findall(tmp_content)) > 1 or (('；' in tmp_content or '，' in tmp_content) and len(tmp_content) > 13) or len(tmp_content) > 21:
					#主语观点有距离
					need_match_aspect = []
					need_match_opinion = item
					continue

				if need_match_aspect[-1]['label'] in extract_shouhou_map and item['label'] in extract_shouhou_map[need_match_aspect[-1]['label']]:
					# 扬声器和听筒失效  解决扬声器的问题
					if len(need_match_aspect) >= 2 and (need_match_aspect[-1]['start'] - need_match_aspect[-2]['end']) <= 1:
						if need_match_aspect[-2]['label'] in extract_shouhou_map and item['label'] in extract_shouhou_map[need_match_aspect[-2]['label']] and '手机' not in need_match_aspect[-2]['label']:
							insert = extract_shouhou_map[need_match_aspect[-2]['label']][item['label']]
							map_info.append(insert)
					target_label = label_rule_check(content, extract_shouhou_map[need_match_aspect[-1]['label']][item['label']], need_match_aspect[-1], item)
					map_info.append(target_label)
				elif item['label'] in extract_shouhou_map:
					target_label = extract_shouhou_map[item['label']][item['label']]
					target_label = label_rule_check(content, target_label, None, item)
					map_info.append(target_label)
				need_match_aspect = []

	if need_match_opinion != None:
		if need_match_opinion['label'] in extract_shouhou_map:
			target_label = extract_shouhou_map[need_match_opinion['label']][need_match_opinion['label']]
			target_label = label_rule_check(content, target_label, None, need_match_opinion)
			map_info.append(target_label)
	map_info = filter(lambda x: len(x) > 0, map_info)
	map_info = list(set(map_info))
	return map_info


def abc_pattern_merge(content, extract_shouhou_map, line, map_info):
	cursor = 0
	while cursor + 2 < len(line):
		left, middle, right = line[cursor], line[cursor+1], line[cursor+2]
		if left['type'] == 'aspect' or middle['type'] == 'aspect' or right['type'] == 'aspect':
			if left['end'] >= middle['start'] and middle['end'] >= right['start']:
				if left['type'] == "opinion" and middle['type'] == "aspect" and right['type'] == "opinion":
					aspect_opinion_insert_delete(content, middle, left, extract_shouhou_map, map_info)
					aspect_opinion_insert_delete(content, middle, right, extract_shouhou_map, map_info)
					line.remove(left)
					line.remove(middle)
					line.remove(right)
				elif left['type'] == "aspect" and middle['type'] == "aspect" and right['type'] == "opinion":
					if left['label'] not in ["手机","小米手机","竞品手机"]:
						aspect_opinion_insert_delete(content, left, right, extract_shouhou_map, map_info)
					aspect_opinion_insert_delete(content, middle, right, extract_shouhou_map, map_info)
					line.remove(left)
					line.remove(middle)
					line.remove(right)
				elif left['type'] == "aspect" and middle['type'] == "opinion" and right['type'] == "opinion":
					aspect_opinion_insert_delete(content, left, middle, extract_shouhou_map, map_info)
					aspect_opinion_insert_delete(content, left, right, extract_shouhou_map, map_info)
					line.remove(left)
					line.remove(middle)
					line.remove(right)
				elif left['type'] == "opinion" and middle['type'] == "aspect" and right['type'] == "aspect":
					aspect_opinion_insert_delete(content, middle, left, extract_shouhou_map, map_info)
					aspect_opinion_insert_delete(content, right, left, extract_shouhou_map, map_info)
					line.remove(left)
					line.remove(middle)
					line.remove(right)
		cursor += 1
	return line

# aspect-opinion有交集，则直接merge 为label
def ab_pattern_merge(content, extract_shouhou_map, line, map_info):
	cursor = 0
	while cursor + 1 < len(line):
		start_item = line[cursor]
		end_item = line[cursor+1]
		if start_item['type'] == 'aspect' or end_item['type'] == 'aspect':
			if end_item['start'] <= start_item['end']:
				if start_item['type'] == "aspect" and end_item['type'] == "opinion":
					aspect_opinion_insert_delete(content, start_item, end_item, extract_shouhou_map, map_info)
					line.remove(start_item)
					line.remove(end_item)
				elif end_item['type'] == "aspect" and start_item['type'] == "opinion":
					aspect_opinion_insert_delete(content, end_item, start_item, extract_shouhou_map, map_info)
					line.remove(start_item)
					line.remove(end_item)
		cursor += 1
	return line

def post_rule_notebook(content, map_info):
	return map_info

def post_rule_pad(content, map_info):
	if '待机' in content:
		if '功耗不满意--七天无理由' in map_info:
			map_info.remove('功耗不满意--七天无理由')
			map_info.append('待机时耗电异常')
		elif '发热不满意--七天无理由' in map_info:
			map_info.remove('发热不满意--七天无理由')
			map_info.append('待机时手机发烫/高温/发热')

	if '前置' in content:
		map_info = [item.replace('后置', '前置') for item in map_info]

	return map_info

def post_rule_phone(content, map_info):
	# 后置处理规则
	if '主屏黑屏(有声音/震动/可打进电话)' in map_info:
		if '死机' in content or '黑屏关机' in content:
			map_info.remove("主屏黑屏(有声音/震动/可打进电话)")
	return map_info

def post_rule_bracket(content, map_info):
	# 后置处理规则
	if '黑屏关机' in content and '主屏黑屏(有声音/震动/可打进电话)' in map_info:
		map_info.remove("主屏黑屏(有声音/震动/可打进电话)")
	return map_info

def label_rule_check(content, biz_label, aspect, opinion):
	if aspect is None:
		start = max(0, opinion['start'] - 5)
		end = min(len(content), opinion['end'] + 5)
	else:
		start = aspect['start'] if aspect['start'] < opinion['start'] else opinion['start']
		end = aspect['end'] if aspect['end'] > opinion['end'] else opinion['end']
		start = max(0, start - 5)
		end = min(len(content), end + 5)
	search_content = content[start:end]
	# 后置处理规则
	if '前置' in search_content or '前摄' in content:
		biz_label = biz_label.replace('后置', '前置', 1).replace('后摄', '前摄', 1)
	elif '蓝牙耳机' in search_content:
		biz_label = biz_label.replace('耳机声音相关故障', '蓝牙声音相关故障', 1)

	if '通话' in search_content and '非通话' not in content:
		biz_label = biz_label.replace('非通话状态麦克无音', '通话时麦克风无声', 1)
		biz_label = biz_label.replace('非通话状态听筒无音', '通话时听筒无声', 1)
		biz_label = biz_label.replace('非通话状态听筒杂音', '通话时听筒杂音', 1)
		biz_label = biz_label.replace('非通话状态听筒声小', '通话时听筒声音小', 1)

	# 倾向于手机不充电
	if '无线充' in search_content:
		biz_label = biz_label.replace("手机不充电", "手机无线充电不充电", 1)
		biz_label = biz_label.replace("手机充电慢", "手机无线充电慢", 1)
		biz_label = biz_label.replace("充电时手机发烫/高温/发热", "手机无线充电时发热", 1)
	if '待机' in search_content:
		biz_label = biz_label.replace("使用其他应用时耗电快/续航差", "待机时耗电异常", 1)
		biz_label = biz_label.replace("功耗不满意--七天无理由", "待机时耗电异常", 1)
		biz_label = biz_label.replace("手机充电慢", "待机时手机发烫/高温/发热", 1)
		biz_label = biz_label.replace("充电时手机发烫/高温/发热", "待机/续航/耗电故障其他", 1)

	if '无信号' in content or '没有信号' in content or '没信号' in content:
		biz_label = biz_label.replace("信号差/格数少（2G/3G/4G/5G）", "无信号或无服务", 1)

	if '内屏' in search_content:
		biz_label = biz_label.replace("主屏外玻璃破损/碎裂", "主屏内屏损伤（外屏无损伤）", 1)
		biz_label = biz_label.replace("主屏显示故障其他", "主屏内屏损伤（外屏无损伤）", 1)
		biz_label = biz_label.replace("主屏触摸屏划伤", "主屏内屏损伤（外屏无损伤）", 1)
		biz_label = biz_label.replace("副屏屏幕破损/碎裂", "副屏屏幕破损（内屏）", 1)
		biz_label = biz_label.replace("副屏触摸屏划伤", "副屏屏幕破损（内屏）", 1)
	if '不跟手' in search_content or '误触' in search_content:
		biz_label = biz_label.replace("主屏触摸屏局部失灵", "主屏屏幕边缘误触")
		biz_label = biz_label.replace("副屏（小屏）触摸屏全屏失灵", "副屏（小屏）屏幕边缘误触")

	return biz_label


RULE_DICT = {"phone": post_rule_phone, "laptop": post_rule_notebook, "pad": post_rule_pad}
FILTER_DICT = {"phone": ['电脑','手环','平板','触控笔','睡眠','心率','计步','血氧','充电盒','表盘','表带','螺丝','office','硬盘','显卡','客服','安装','售后','包装','物流','发货','机顶盒','hdmi'],
			   "laptop": ['手环','平板','触控笔','睡眠','心率','计步','血氧','充电盒','表盘','表带','螺丝','客服','安装','售后','包装','物流','发货','机顶盒','手电筒'],
			    "pad": ['电脑','手环','睡眠','心率','计步','血氧','充电盒','表盘','表带','螺丝','硬盘','显卡','客服','安装','售后','包装','物流','发货','机顶盒','hdmi']}
def generate_bizlabelID(map_info, bizid_mapping):
	fir = list(map(lambda item: bizid_mapping[item][0] if item in bizid_mapping else None, map_info))
	fir_id = list(map(lambda item: bizid_mapping[item][1] if item in bizid_mapping else None, map_info))
	sec = list(map(lambda item: bizid_mapping[item][2] if item in bizid_mapping else None, map_info))
	sec_id = list(map(lambda item: bizid_mapping[item][3] if item in bizid_mapping else None, map_info))
	third = list(map(lambda item: bizid_mapping[item][4] if item in bizid_mapping else None, map_info))
	third_id = list(map(lambda item: bizid_mapping[item][5] if item in bizid_mapping else None, map_info))
	fir = list(filter(lambda x: x != None, fir))
	fir_id = list(filter(lambda x: x != None, fir_id))
	sec = list(filter(lambda x: x != None, sec))
	sec_id = list(filter(lambda x: x != None, sec_id))
	third = list(filter(lambda x: x != None, third))
	third_id = list(filter(lambda x: x != None, third_id))
	return fir, fir_id, sec, sec_id, third, third_id


def get_args(proj_name, biz_name, model_dir):
    return torch.load(os.path.join(proj_name, biz_name, model_dir, 'training_args.bin'))


