import pandas as pd

def get_desciption():
	sheet = pd.read_excel('data/客服知识库.xlsx', sheet_name='狗狗介绍', engine='openpyxl')
	descriptions = []
	for row in sheet.itertuples():
		desc = "狗名字：" + getattr(row, "名字")
		for column in sheet.columns[1:]:
			value = getattr(row, column)
			if type(value) == str:
				value = value.replace("\n", ";")
			if pd.notna(value):
				desc += "；" + column + "：" + str(value)
		descriptions.append(desc)
	descriptions[-3] = "小型犬：母犬在1岁时就是成年了、公犬要18个月才算成年犬。"
	return descriptions

def get_more_understand_dog():
	sheet = pd.read_excel('data/客服知识库.xlsx', sheet_name='更了解狗狗')
	descriptions = []
	for row in sheet.itertuples():
		desc = getattr(row, "问题")
		for column in ["原因", "解答"]:
			value = getattr(row, column)
			if type(value) == str:
				value = value.replace("\n", ";")
			if pd.notna(value):
				desc += " " + column + "：" + str(value)
		descriptions.append(desc)
	complentry = sheet.iloc[:,4]
	for con in complentry:
		if pd.notna(con):
			descriptions.append(con)
	return descriptions

def get_template_response():
	sheet = pd.read_excel('data/客服知识库.xlsx', sheet_name='通用语')
	descriptions = []
	for row in sheet.itertuples():
		desc = "话术场景：" + getattr(row, "分类")
		descriptions.append(desc + "\n回复话术：" + getattr(row, '设置话术1'))

		for key in ['回复话术2', '回复话术3', '回复话术4']:
			if pd.notna(getattr(row, key)):
				descriptions.append(desc + "\n回复话术:" + getattr(row, key))
	return descriptions

def get_review_response():
	sheet = pd.read_excel('data/客服知识库.xlsx', sheet_name='评价回复')
	descriptions = []
	for row in sheet.itertuples():
		desc = "产品类型：" + getattr(row, "产品类型")
		if type(getattr(row, "评价类型")) == str:
			desc = "产品类型：" + getattr(row, "产品类型") + "\n评价类型:" + getattr(row, "评价类型")
		if pd.notna(getattr(row, '评价回复1')):
			descriptions.append(desc + "\n评价回复1：" + getattr(row, '评价回复1'))
		if pd.notna(getattr(row, '评价回复2')):
			descriptions.append(desc + "\n评价回复2：" + getattr(row, '评价回复2'))

		if pd.notna(getattr(row, '评价回复3')):
			descriptions.append(desc + "\n评价回复3：" + getattr(row, '评价回复3'))
		if pd.notna(getattr(row, '评价回复4')):
			descriptions.append(desc + "\n评价回复4：" + getattr(row, '评价回复4'))
		descriptions.append(desc)
	return descriptions


def get_develiy_aftersale():
	sheet = pd.read_excel('data/客服知识库.xlsx', sheet_name='物流售后')
	descriptions = []
	for row in sheet.itertuples():
		desc = "问题：" + getattr(row, "问题")
		for column in sheet.columns[3:]:
			value = getattr(row, column)
			if type(value) == str:
				value = value.replace("\n", ";")
			if pd.notna(value):
				desc += "\n" + column + "：" + str(value)
		descriptions.append(desc)
	return descriptions

def get_toy_aftersale():
	sheet = pd.read_excel('data/客服知识库.xlsx', sheet_name='狗狗玩具售后')
	descriptions = []
	for row in sheet.itertuples():
		desc = "问题：" + getattr(row, "常见售后问题")
		for column in sheet.columns[1:]:
			value = getattr(row, column)
			if type(value) == str:
				value = value.replace("\n", ";")
			if pd.notna(value):
				desc += "\n" + column + "：" + str(value)
		descriptions.append(desc)
	return descriptions

def get_product_list():
	sheet = pd.read_excel('data/客服知识库.xlsx', sheet_name='产品清单')
	descriptions = []
	for row in sheet.itertuples():
		desc = "问题：" + getattr(row, "问题")
		for column in ['分类', '回复话术']:
			value = getattr(row, column)
			if type(value) == str:
				value = value.replace("\n", ";")
			if pd.notna(value):
				desc += "\n" + column + "：" + str(value)
		descriptions.append(desc)
	descriptions[-2] = '''退/换货地址：江苏省南京市雨花台区大周路39号\n联系人：xx售后\n联系电话：025-89637115'''
	descriptions[-1] = '''发票问题：确认收货后开电子发票'''
	return descriptions

def get_salepoint():
	sheet = pd.read_excel('data/客服知识库.xlsx', sheet_name='salepoint')
	descriptions = []
	for row in sheet.itertuples():
		desc = "分类：" + getattr(row, "分类") + "\nSPU产品名称：" + getattr(row, 'SPU产品名称') + "\n材质：" + getattr(row, '材质')
		salepoints = getattr(row, '卖点')
		salepoints = salepoints.split('\n')
		for salepoint in salepoints:
			con = desc + "\n卖点：" + salepoint
			descriptions.append(con)
	return descriptions


def get_customer_data():
	kg_types = [get_desciption, get_more_understand_dog, get_template_response, get_review_response,
				get_develiy_aftersale, get_toy_aftersale, get_product_list, get_salepoint]
	texts = []
	for func in kg_types:
		res = func()
		texts.extend(res)
	return texts

if __name__ == '__main__':
	get_desciption()
	# get_customer_data()



