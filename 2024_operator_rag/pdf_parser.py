from common_utils import timeit
# https://github.com/opendatalab/PDF-Extract-Kit/blob/main/README-zh_CN.md
# 表格是 latex 格式很复杂 我希望是csv 格式的数据

@timeit
def pdf_extract_kit():
	import json
	pdf = json.load(open('/root/PDF-Extract-Kit/output/AY01.json'))
	for ob in pdf:
		layout_dets = ob['layout_dets']
		for item in layout_dets:
			category_id = item['category_id']
			if category_id == 5:
				print(f"{item}")

