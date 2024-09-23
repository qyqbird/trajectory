from data_utils import load_data, parse_pdf, pdfplumer_parser_pdf, pdf_parser
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain_chroma import Chroma
import os
import pickle
import pandas as pd
import time
from ltp import StnSplit
import sys
from common_utils import get_embedding_tool, cut_sentence_with_quotation_marks

persis_dir = "data/database"
# source activate langchain
if __name__ == '__main__':
	read_pdf_flag = "0"
	if len(sys.argv) > 1:
		read_pdf_flag = sys.argv[1]
	embedding_tool = get_embedding_tool()	# 加载很耗时
	text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=18, separator='。', is_separator_regex=False, keep_separator='end')
	if read_pdf_flag == "1":
		# contents = parse_pdf('data/A_document')
		docs = pdf_parser(text_splitter)
		# contents = pdfplumer_parser_pdf('data/A_document')
		# text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=18, separators=['。', '！'], keep_separator='end')
		# text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=18, separators=['。', '！'], keep_separator='end')
		# chunks = cut_sentence_with_quotation_marks(contents)

		vector_start_time = time.time()
		vectordb = Chroma.from_documents(docs, embedding_tool, persist_directory=persis_dir)
		vector_time = time.time() - vector_start_time
		print(f"vector chunk count:{len(docs)} time cousume:{vector_time}s")

	vectordb = Chroma(persist_directory=persis_dir, embedding_function=embedding_tool)
	# retriever = vectordb.as_retriever(search_kwargs={"k": 1, "score_threshold":0.3})	# 调参还比较多
	retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 1, "score_threshold":0.3})

	questions = pd.read_csv('data/A_question.csv')
	answer = []
	embeddings = []
	for row in questions.itertuples():
		ques_id = getattr(row, 'ques_id')
		question = getattr(row, 'question')
		# result = retriever.invoke(question)
		result = retriever.invoke(question)
		answer.append(result[0].page_content)
		embeddings.append(embedding_tool.embed_query(result[0].page_content))
		print(f"{ques_id}\t{question}\n{result[0].page_content}")
	questions['answer'] = answer
	questions.to_csv("data/no_embedding_submit.csv", index=False)
	questions['embedding'] = embeddings
	questions.to_csv("data/submit.csv", index=False)
