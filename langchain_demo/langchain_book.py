from langchain_community.llms.fake import  FakeListLLM

response = ["要恢复手机出厂设置"]
llm = FakeListLLM(response=response)
