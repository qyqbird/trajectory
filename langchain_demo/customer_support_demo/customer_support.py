import os.path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from data_parser import get_customer_data
from model import DeepSeek 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PromptUtils import get_prompt
# conda activate langchain

# https://www.bilibili.com/video/BV18nYxecEMC?p=13&vd_source=48d52af8eec19b4d0b941c4ceee34422
class ChatDoc():
	def __init__(self):
		self.texts = get_customer_data()	# 外部知识库
		self.doc = self.split_sentence()
		# self.template = [("system", "你是一名电商主营狗狗类产品的客服。{context}"), ("human","您好!"),
		# 				 ("ai", "您好！"),
		# 				 ("human","{question}")]
		# self.prompt = ChatPromptTemplate.from_messages(self.template)
		self.retriever = self.embedding_vector().as_retriever()
		self.llm = DeepSeek()
		self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
		self.interface = ConversationalRetrievalChain.from_llm(self.llm, self.retriever, memory=self.memory)

	def split_sentence(self):
		text_splitter = RecursiveCharacterTextSplitter(chunk_size=320, chunk_overlap=45)
		docs = text_splitter.create_documents(self.texts)
		return docs

	def embedding_vector(self):
		from langchain_community.vectorstores import Chroma
		from langchain_community.embeddings import HuggingFaceBgeEmbeddings
		persis_dir = "data/database"
		model_kwargs = {'device': 'cpu'}
		encode_kwargs = {'normalize_embeddings': True}
		embeddings = HuggingFaceBgeEmbeddings(model_name='../../2024_operator_rag/bge-large-zh-v1.5',
											  model_kwargs=model_kwargs,
											  encode_kwargs=encode_kwargs)
		if not os.path.exists(persis_dir):
			vectordb = Chroma.from_documents(documents=self.doc, embedding=embeddings, persist_directory="data/database")
			# vectordb = Chroma.from_texts(self.texts, embeddings = embeddings)
			vectordb.persist()
			return vectordb
		else:
			# Now we can load the persisted database from disk, and use it as normal.
			vectordb = Chroma(persist_directory="data/database", embedding_function=embeddings)
			return vectordb

	def ask_and_rag(self, question):
		results = self.retriever.invoke(question)
		return results

	# 使用多重查询，多角度,提高文档检索的精度
	def ask_and_rag2(self, question):
		from langchain.retrievers import MultiQueryRetriever
		import logging
		logging.basicConfig()
		logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

		retriever_from_llm = MultiQueryRetriever.from_llm(retriever=self.retriever, llm=self.llm)
		results = retriever_from_llm.invoke(question)
		return results

	# 使用上下文压缩，不要立即返回文档，而是可以使用给定的查询的上下文对其压缩。既包含单文档压缩，也有整体的过滤
	# 也是利用大模型过滤
	def ask_and_rag3(self, question):
		from langchain.retrievers import ContextualCompressionRetriever
		from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, EmbeddingsFilter

		compresser = LLMChainExtractor.from_llm(self.llm)
		# fil = LLMChainFilter.from_llm(self.llm)	# 和上面的区别是，prompt template 不一样
		compresser_retriever = ContextualCompressionRetriever(retriever=self.retriever, compresser=compresser)
		return compresser_retriever.get_relevant_documents(question=question)

	def ask_and_rag_reorder(self, question):
		from langchain_community.document_transformers import LongContextReorder
		reordering = LongContextReorder()
		reordering.transform_documents(xx)

	def rag_parent_retriver(self, question):
		from langchain.retrievers import ParentDocumentRetriever
		ParentDocumentRetriever()

	# 在向量存储里使用大量边际相似性MMR和相似性打分
	def ask_and_rag4(self, question):
		db = self.embedding_vector()
		# retriever = db.as_retriever(search_type='mmr')	# 不是很好
		retriever = db.as_retriever(search_type='similarity_score_threshold', search_kwargs={"score_threshold":0.5, "k":1})
		return retriever.get_relevant_documents(question=question)

	def chat_with_doc(self, question):
		context = self.ask_and_rag2(question)
		_context = ''
		for i in context:
			_context += i.page_content

		# messages = self.prompt.from_messages(context=_context, question=question)
		prompt = get_prompt(_context, question)

		print(prompt)
		result = self.interface(prompt)
		# print(result['answer'])
		return result

if __name__ == '__main__':
	chatdoc = ChatDoc()
	questions = ['今天的天气如何？', '请问你的老板叫什么名字', '帮我介绍一下蝴蝶犬', '请问可以惩罚狗狗吗', '给猫买的玩具，玩一段时间不玩了,啥情况',
				 '很急，物流啥时候能到啊', '这个玩具球，如何选择大小呀', '我的这个球气味很大', '请问蛋壳不倒翁的特点是']

	for question in questions:
		print(f"问题：{question}")
		chatdoc.chat_with_doc(question)
	# while True:
	# 	question = input('请输入您的问题？')
	# 	print(f"您的问题是:{question}")
	# 	chatdoc.chat_with_doc(question)

