from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

# demo url: https://zhuanlan.zhihu.com/p/668082024
# conda activate langchain
loader = TextLoader("./藜麦.txt")
documents = loader.load()
# print(documents)


# 文档分割
# splitter 很多的：CharacterTextSplitter 需要制定separator, 还包括PDF 抽取，用大模型多模态
text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=0, separator='\n')
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=0)
documents = text_splitter.split_documents(documents)
# print(len(documents))
# for doc in documents:
    # print(doc)

model_name = "moka-ai/m3e-base"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction="为文本生成向量表示用于文本检索"
            )

db = Chroma.from_documents(documents, embedding)
db.similarity_search("藜一般在几月播种？")

template = '''
        【任务描述】
        请根据用户输入的上下文回答问题，并遵守回答要求。

        【背景知识】
        {{context}}

        【回答要求】
        - 你需要严格根据背景知识的内容回答，禁止根据常识和已知信息回答问题。
        - 对于不知道的信息，直接回答“未找到相关答案”
        -----------
        {question}
        '''


from langchain import LLMChain
# from langchain_wenxin.llms import Wenxin
from model import Kimi
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = Kimi()
retriever = db.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
print(qa)
qa({"question": "藜怎么防治虫害？"})
print(qa)

