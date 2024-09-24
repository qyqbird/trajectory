from langchain_community.document_loaders import PyPDFLoader, PyPDFium2Loader, PyMuPDFLoader, PDFPlumberLoader, UnstructuredPDFLoader
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
		if filename.startswith('AT') or filename.startswith('AF') or filename.startswith('AW'):
			loader = PyMuPDFLoader(f'data/A_document/{filename}')
			result = loader.load()
			content = ""
			for page in result:	# page Document
				page_content = page.page_content.encode('utf-8', 'ignore').decode('utf-8').replace("\n", "")
				content += page_content
			content = content.replace("本文档为2024CCFBDCI比赛用语料的一部分。部分文档使用大语言模型改写生成，内容可能与现实情况不符，可能不具备现实意义，仅允许在本次比赛中使用。", "")
			if len(content) > 0:
				pdf_container.append(content)
		else:
			# 年报中涉及到表格展示是有问题的  AY AZ
			loader = PyMuPDFLoader(f'data/A_document/{filename}')
			result = loader.load()
			content = ""
			for page in result:
				page_content = page.page_content.encode('utf-8', 'ignore').decode('utf-8').replace("\n", "")
				content += page_content
			content = content.replace("本文档为2024CCFBDCI比赛用语料的一部分。部分文档使用大语言模型改写生成，内容可能与现实情况不符，可能不具备现实意义，仅允许在本次比赛中使用。", "")
			if len(content) > 0:
					pdf_container.append(content)

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


# 解析很快 3s
@timeit
def parse_pdf(pdf_dir, text_splitter):
	import pymupdf
	texts = []
	total_files = len([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
	print(f"开始解析PDF文件，共{total_files}个文件")
	for i, filename in enumerate(os.listdir(pdf_dir), 1):
		if filename.endswith('AZ01.pdf'):
			pdf_path = os.path.join(pdf_dir, filename)
			print(f"正在处理第{i}/{total_files}个文件: {filename}")
			try:
				doc = pymupdf.open(pdf_path)
				content = ""
				for page in doc:
					content += page.get_text()
					print(content)
				doc.close()
				content = content.replace("\n", "")
				texts.append(content)
			except Exception as e:
				print(f"处理文件 {filename} 时出错: {str(e)}")
	print("PDF解析完成")
	return texts

def pdfplumer_parser_pdf(pdf_dir):
	import pdfplumber
	remove_text = '本文档为2024CCFBDCI比赛用语料的一部分。部分文档使用大语言模型改写生成，内容可能与现实情况\n不符，可能不具备现实意义，仅允许在本次比赛中使用。\n'
	texts = []
	total_files = len([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
	print(f"开始解析PDF文件，共{total_files}个文件")
	for i, filename in enumerate(os.listdir(pdf_dir), 1):
		if filename.endswith('.pdf'):
			pdf_path = os.path.join(pdf_dir, filename)
			print(f"正在处理第{i}/{total_files}个文件: {filename}")

			with pdfplumber.open(pdf_path) as pdf:
				content = ""
				for page in pdf.pages:
					content += page.extract_text() or ""
				content = content.replace(remove_text, "").replace("\n", "")
				texts.append(content)
	return ' '.join(texts)


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
	for table in tables:
		print(table.columns)
	# tabula.convert_into("data/A_document/AY01.pdf", "data/A_CSV/output.csv", output_format="csv", pages='all')


''' TODO
1. 财报：页眉页脚删除
2. 表格如何切割
'''
if __name__ == '__main__':
	text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=18, separator='。', is_separator_regex=False, keep_separator='end')
	# parse_pdf('data/A_document', text_splitter)
	# table_parser2(text_splitter)
