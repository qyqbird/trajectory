from common_utils import timeit

@timeit
def pdf_extract_kit():
	# https://github.com/opendatalab/PDF-Extract-Kit/blob/main/README-zh_CN.md
	# 表格是 latex 格式很复杂 我希望是csv 格式的数据
	import json
	pdf = json.load(open('/root/PDF-Extract-Kit/output/AY01.json'))
	for ob in pdf:
		layout_dets = ob['layout_dets']
		for item in layout_dets:
			category_id = item['category_id']
			if category_id == 5:
				print(f"{item}")


@timeit
def unstruct_pdf_extract():
	from langchain_community.document_loaders import UnstructuredPDFLoader
	import re
	loader = UnstructuredPDFLoader('data/A_document/AY01.pdf')	# 默认把所有页合在一起, mode='elements'  把每一行都单独一个单元
	pdf = loader.load()
	pdf.page_content = re.sub('中国联合网络通信股份有限公司 [0-9]{4} 年年度报告', '', pdf.page_content, re.MULTILINE)
	print(pdf)



@timeit
def markdown_pdf_extract():
	from langchain_community.document_loaders import UnstructuredMarkdownLoader
	import re
	loader = UnstructuredMarkdownLoader('data/A_document/AY01.pdf', mode="elements")	# 默认把所有页合在一起, mode='elements'  把每一行都单独一个单元
	pdf = loader.load()
	# pdf.page_content = re.sub('中国联合网络通信股份有限公司 [0-9]{4} 年年度报告', '', pdf.page_content, re.MULTILINE)
	print(pdf)

markdown_pdf_extract()