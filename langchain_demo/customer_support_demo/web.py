import gradio as gr
from customer_support import ChatDoc

chatdoc = ChatDoc()
def answer_question(question):
	# global chat_history result = chain({"question": question, "chat_history": chat_history})
	# chat_history.append((question, result['answer']))
	print(f"{question}")
	result = chatdoc.chat_with_doc(question)
	print(f"{result['answer']}")
	return result['answer']

iface = gr.Interface(fn= answer_question, inputs="text", outputs="text",
					title="XX卖家客服",
					description="您的专属客服")

iface.launch(share=True)
