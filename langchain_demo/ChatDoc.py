from langchain.document_loaders import  Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
# conda activate langchain

# https://www.bilibili.com/video/BV18nYxecEMC?p=13&vd_source=48d52af8eec19b4d0b941c4ceee34422
class ChatDoc():
	def __init__(self):
		self.texts = []
		self.docs = None
		self.template = [("system", "你是一个文字处理工作者。{context}"), ("human","您好!"),
						 ("ai", "您好！"),
						 ("human","{question}")]
		self.prompt = ChatPromptTemplate.from_messages(self.template)

	def getFile(self):
		doc = self.doc
		loader = {"docx": Docx2txtLoader,
			"pdf": PyPDFLoader,
			"xlsx": UnstructuredExcelLoader
		}
		file_extensition = doc.split('/')[-1]
		loader_class = loader[file_extensition]
		text = loader_class.load()
		return text

	def split_sentence(self):
		full_text = self.getFile()
		text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
		texts = text_splitter.create_documents(full_text)
		self.texts = texts


	# 文本向量化
	def embedding_vector(self):
		embeddings = OpenAIEmbeddings()
		db = Chroma.from_documents(documents=self.texts, embeddings=embeddings)
		return db

	def ask_and_rag(self, question):
		db = self.embedding_vector()
		retriever = db.as_retriever()
		results = retriever.invoke(question)
		return results

	# 使用多重查询，多角度,提高文档检索的精度
	def ask_and_rag2(self, question):
		from langchain.retrievers import MultiQueryRetriever
		from langchain.chat_models import ChatOpenAI
		import logging
		logging.basicConfig()
		logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
		 
		db = self.embedding_vector()
		llm = ChatOpenAI()
		retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever, llm=llm)
		results = retriever_from_llm.invoke(question)
		return results

	# 使用上下文压缩，提升提高文档精度
	def ask_and_rag3(self, question):
		from langchain.llms import OpenAI
		from langchain.retrievers import ContextualCompressionRetriever
		from langchain.retrievers.document_compressors import LLMChainExtractor
		db = self.embedding_vector()
		retriever = db.as_retriever()
		llm = OpenAI(temperature=0)
		compresser = LLMChainExtractor.from_llm(llm)
		compresser_retriever = ContextualCompressionRetriever(retriever=retriever, compresser=compresser)
		return compresser_retriever.get_relevant_documents(question=question)

	# 在向量存储里使用大量边际相似性MMR和相似性打分
	def ask_and_rag4(self, question):
		db = self.embedding_vector()
		# retriever = db.as_retriever(search_type='mmr')	# 不是很好
		retriever = db.as_retriever(search_type='similarity_score_threshold',{"score_threold":0.5, k:1})
		return retriever.get_relevant_documents(question=question)

	def chat_with_doc(self, question):
		from langchain.chat_models import ChatOpenAI
		context = self.ask_and_rag3(question)
		_context = ''
		for i in context:
			_context += i.page

		messages = self.prompt.from_messages(context=_context, question=question)
		chat = ChatOpenAI(model="", temperature=0)
		return chat.invoke(messages)


ChatDoc.getFile()
chatdoc = ChatDoc()
chatdoc.chat_with_doc('')