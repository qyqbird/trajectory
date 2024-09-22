from common_utils import load_data, parse_pdf
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import os
import pickle
import pandas as pd

# source activate langchain
if __name__ == '__main__':
	contents = parse_pdf('data/A_document')
	# with open('contents.pickle', 'wb') as fo:
	# 	pickle.dump(contents, fo)
	# contents = pickle.load(open('contents.pickle', 'rb'))
	# text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=18, separator='。', is_separator_regex=False, keep_separator='end')
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=18, separators=['。', '！'], keep_separator='end')
	chunks = text_splitter.split_text(contents)

	for chunk in chunks:
		print(chunk)
	print(len(chunks))

	model_name = "/root/autodl-tmp/bge-large-zh-v1.5"
	model_kwargs = {'device': 'cuda'}
	encode_kwargs = {'normalize_embeddings': True}
	embedding = HuggingFaceBgeEmbeddings(
		model_name=model_name,
		model_kwargs=model_kwargs,
		encode_kwargs=encode_kwargs,
		query_instruction="为文本生成向量表示用于文本检索"
	)
	persis_dir = "data/database"
	if not os.path.exists(persis_dir):
		vectordb = Chroma.from_texts(chunks, embedding, persist_directory=persis_dir)
		vectordb.persist()
	else:
		vectordb = Chroma(persist_directory=persis_dir, embedding_function=embedding)

	questions = pd.read_csv('data/A_question.csv')
	retriever = vectordb.as_retriever(search_kwargs={"k": 1})

	answer = []
	embeddings = []
	for row in questions.itertuples():
		ques_id = getattr(row, 'ques_id')
		question = getattr(row, 'question')
		# result = retriever.invoke(question)
		result = retriever.get_relevant_documents(question)
		answer.append(result[0].page_content)
		embeddings.append(embedding.embed_query(result[0].page_content))
		print(f"{len(embeddings[-1])}\t{ques_id}\t{question}\n{result[0].page_content}")
	questions['answer'] = answer
	questions['embedding'] = embeddings
	questions.to_csv("data/submit.csv", index=False)