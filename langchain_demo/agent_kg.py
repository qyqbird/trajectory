from langchain.indexes import GraphIndexCreator
from langchain.chains import GraphQAChain
from langchain.graphs.networkx_graph import KnowledgeTriple
from model import Kimi
llm = Kimi()
index_creator = GraphIndexCreator(llm=llm)
f_index_creator = GraphIndexCreator(llm=llm)
final_graph = f_index_creator.from_text('')

texts = '''

'''

