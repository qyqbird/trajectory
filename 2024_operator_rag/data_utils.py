from langchain_community.document_loaders import PyPDFLoader, PyPDFium2Loader, PyMuPDFLoader
import os
import time
from common_utils import timeit
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter,TokenTextSplitter
import pandas as pd

#1. PyPDFLoader
def load_data():
	start_time = time.time()
	pdf_container = []
	for filename in os.listdir('data/A_document'):
		# if filename.startswith('AT') or filename.startswith('AF') or filename.startswith('AW'):
		graph_path = "data/graphrag/" + filename.split('.')[0]
		loader = PyMuPDFLoader(f'data/A_document/{filename}')
		result = loader.load()

		# txt
		content = ""
		for page in result:	# page Document
			page_content = page.page_content.encode('utf-8', 'ignore').decode('utf-8').replace("\n", "")
			content += page_content
		content = content.replace("本文档为2024 CCF BDCI 比赛用语料的一部分。部分文档使用大语言模型改写生成，内容可能与现实情况不符，可能不具备现实意义，仅允许在本次比赛中使用。", "")
		with open(graph_path+".txt", 'w') as writer:
			writer.write(content)

		# 表格
		import tabula
		tables = tabula.read_pdf(f'data/A_document/{filename}', pages='all')	# 返回DataFrame
		for idx, table in enumerate(tables):
			csv_name = f"{graph_path}_table_{idx}.csv"
			table.to_csv(csv_name, index=False)

	consume = time.time() - start_time
	print(f"read PDF count:{len(pdf_container)} cousume time:{consume}")
	return pdf_container

@timeit
def pdf_parser(text_splitter):
	chunks = []
	for idx, filename in enumerate(os.listdir('data/A_document')):
		loader = PyPDFLoader(f'data/A_document/{filename}')
		result = loader.load()
		for doc in result:
			# print page_content UnicodeEncodeError: 'utf-8' codec can't encode characters in position 105-112: surrogates not allowed
			doc.page_content = doc.page_content.encode('utf-8', 'ignore').decode('utf-8')
		chunk = text_splitter.split_documents(result)
		chunks.extend(chunk)
		print(f"{idx}\t{filename}\tpage:{len(result)}\tchunks:{len(chunk)}")
	for idx, chunk in enumerate(chunks):
		print(f"{idx}\n{chunk}")
	return chunks


# 解析很快 3s  以text 返回的应该要差一些
@timeit
def parse_pdf(pdf_dir, text_splitter):
	import pymupdf
	chunks = []
	total_files = len([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
	print(f"开始解析PDF文件，共{total_files}个文件")
	for i, filename in enumerate(os.listdir(pdf_dir), 1):
		if filename.endswith('AY01.pdf'):
			pdf_path = os.path.join(pdf_dir, filename)
			print(f"正在处理第{i}/{total_files}个文件: {filename}")
			try:
				docs = pymupdf.open(pdf_path)
				content = ""
				for doc in docs:
					content += doc.get_text() + "\n"
				docs.close()
				chunk = text_splitter.split_text(content)
				chunks.extend(chunk)
				print(f"{idx}\t{filename}\tpage:{len(docs)}\tchunks:{len(chunk)}")
			except Exception as e:
				print(f"处理文件 {filename} 时出错: {str(e)}")

	for idx, chunk in enumerate(chunks):
		print(f"{idx}\n{chunk}")
	return chunks

@timeit
def pdfplumer_parser_pdf(pdf_dir, text_splitter):
	import pdfplumber
	remove_text = '本文档为2024CCFBDCI比赛用语料的一部分。部分文档使用大语言模型改写生成，内容可能与现实情况\n不符，可能不具备现实意义，仅允许在本次比赛中使用。\n'
	texts = []
	total_files = len([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
	print(f"开始解析PDF文件，共{total_files}个文件")
	for i, filename in enumerate(os.listdir(pdf_dir), 1):
		if filename.endswith('AY01.pdf'):
			pdf_path = os.path.join(pdf_dir, filename)
			print(f"正在处理第{i}/{total_files}个文件: {filename}")
			with pdfplumber.open(pdf_path) as pdf:
				content = ""
				for page in pdf.pages:
					content += page.extract_text()
				content = content.replace(remove_text, "")
				texts.append(content)
	
	return texts


@timeit
def table_parser():
	import fitz
	doc = fitz.open('data/A_document/AY01.pdf')
	for page in doc:
		if page.find_tables():
			tables = page.find_tables()
			for table in tables:
				df = table.to_pandas()
				print(df)
				df.to_csv("data/table.csv")
				df.to_excel('data/table.xlsx', index=False)

@timeit
def table_parser2(text_splitter):
	#https://github.com/ARTAvrilLavigne/ExtractFinancialStatement?tab=readme-ov-file
	import tabula
	tables = tabula.read_pdf('data/A_document/AY01.pdf', pages='all')	# 返回DataFrame
	# for index, row in tables[0].iterrows():
	# 	print(f'{row}')

	print(tables[2])
	# corresponding_profile(tables[1])
	# company_profile(tables[0], tables[1])

	# print(tables[1])
	# for table in tables:
	# 	print(table)
	# tabula.convert_into("data/A_document/AY01.pdf", "data/A_CSV/output.csv", output_format="csv", pages='all')

def corresponding_profile(table):
	infos = []
	for idx, column in enumerate(table.columns):
		if idx == 0 or idx == len(table.columns)-1:
			continue
		field_info = ""
		for rdx, row in table.iterrows():
			row_attri = table.iloc[rdx][0]
			if not column.startswith('Unnamed'):
				if rdx > 0:
					field_info += row_attri + ":" + table.iloc[rdx][idx] +";"
				else:
					field_info +=  ":" + table.iloc[rdx][idx] +";"
		field_info = column + ":" + field_info

		infos.append(field_info)
	return infos			

def company_profile(table):
	table.columns = table.iloc[0]
	table = table[1:].reset_index(drop=True)
	infos = []
	for column in table.columns:
		if column in table.iloc[0]:
			info = column + ":" + table.iloc[0][column]
			print(info)
			infos.append(info)
	return infos

@timeit
def unstruct_pdf_extract(text_splitter):
	'''
		主要缺点，对表格无能为力，后续处理很难
	'''
	from langchain_community.document_loaders import UnstructuredPDFLoader
	import re
	loader = UnstructuredPDFLoader('data/A_document/AY01.pdf')	# 默认把所有页合在一起, mode='elements'  把每一行都单独一个单元
	pdf = loader.load()
	for doc in pdf:
		doc.page_content = re.sub(r'\d+\n\n中国联合网络通信股份有限公司 [0-9]{4} 年年度报告\n\n', '', doc.page_content)	# 页眉 + 页码
		doc.page_content = doc.page_content.replace("\n\n", "")
	print(pdf)

	docs = text_splitter.split_documents(pdf)
	for idx, doc in enumerate(docs):
		print(f"{idx}\n{doc}")


''' TODO
1. 财报：页眉页脚删除
2. 表格如何切割
'''
if __name__ == '__main__':
	text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=18, separator='。', is_separator_regex=False, keep_separator='end')
	# parse_pdf('data/A_document', text_splitter)
	# table_parser2(text_splitter)
	# unstruct_pdf_extract(text_splitter)
	load_data()