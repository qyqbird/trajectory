from langchain_community.document_loaders import PyPDFLoader, PyPDFium2Loader, PyMuPDFLoader
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import time

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

remove_text = '本文档为2024CCFBDCI比赛用语料的一部分。部分文档使用大语言模型改写生成，内容可能与现实情况\n不符，可能不具备现实意义，仅允许在本次比赛中使用。\n'
def parse_pdf(pdf_dir):
	import pymupdf
	texts = []
	total_files = len([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
	print(f"开始解析PDF文件，共{total_files}个文件")
	for i, filename in enumerate(os.listdir(pdf_dir), 1):
		if filename.endswith('.pdf'):
			pdf_path = os.path.join(pdf_dir, filename)
			print(f"正在处理第{i}/{total_files}个文件: {filename}")
			try:
				doc = pymupdf.open(pdf_path)
				for page in doc:
					text = page.get_text().replace('\n', '')
					texts.append(text)
				doc.close()
			except Exception as e:
				print(f"处理文件 {filename} 时出错: {str(e)}")
	print("PDF解析完成")
	return texts


def pdfplumer_parser_pdf(pdf_dir):
	import pdfplumber
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
				content = content.replace(remove_text, "")
				texts.append(content)
	return texts



if __name__ == '__main__':
	parse_pdf('data/A_document')
	# load_data()
	# load_inferencefile()
	# loader = PyPDFLoader(f'data/A_document/AZ06.pdf')
	# result = loader.load()
	#
	# content = ""
	# for page in result:
	# 	page_content = page.page_content.encode('utf-8', 'ignore').decode('utf-8')
	# 	content += page.page_content.replace("\n", "")
	# content = content.replace("本文档为2024CCFBDCI比赛用语料的一部分。部分文档使用大语言模型改写生成，内容可能与现实情况不符，可能不具备现实意义，仅允许在本次比赛中使用。", "")
	#
	# from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
	# # text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=18, separator='。|！', is_separator_regex=True, keep_separator='end')
	# text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=18, separators=['。', '！'], keep_separator='end')
	# chunks = text_splitter.split_text(content)

	# model_name = "bge-large-zh-v1.5"
	# model_kwargs = {'device': 'cpu'}
	# encode_kwargs = {'normalize_embeddings': True}
	# embedding = HuggingFaceBgeEmbeddings(
	# 	model_name=model_name,
	# 	model_kwargs=model_kwargs,
	# 	encode_kwargs=encode_kwargs,
	# 	query_instruction="为文本生成向量表示用于文本检索"
	# )
	# persis_dir = "data/database"
	# if not os.path.exists(persis_dir):
	# 	vectordb = Chroma.from_texts(chunks, embedding, persist_directory=persis_dir)
	# 	vectordb.persist()
	# else:
	# 	vectordb = Chroma(persist_directory=persis_dir, embedding_function=embedding)
	#
	# questions = load_inferencefile()
	# retriever = vectordb.as_retriever(search_kwargs={"k": 2})
	#
	# answer = []
	# embeddings = []
	# for row in questions.itertuples():
	# 	ques_id = getattr(row, 'ques_id')
	# 	question = getattr(row, 'question')
	# 	# result = retriever.invoke(question)
	# 	result = retriever.get_relevant_documents(question)
	# 	answer.append(result[0].page_content)
	# 	embeddings.append(embedding.embed_documents(result[0].page_content))
	# 	print(f"{ques_id}\t{question}\n{result[0].page_content}")
	# questions['answer'] = answer
	# questions['embedding'] = embeddings
	# questions.to_csv("data/demo.csv")