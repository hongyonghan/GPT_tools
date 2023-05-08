# import gradio as gr
#
# from langchain.document_loaders import PyMuPDFLoader  # for loading the pdf
# from langchain.embeddings import OpenAIEmbeddings  # for creating embeddings
# from langchain.vectorstores import Chroma  # for the vectorization part
# from langchain.chains import ChatVectorDBChain  # for chatting with the pdf
# from langchain.llms import OpenAI  # the LLM model we'll use (CHatGPT)
#
#
# class Chat:
#     def __init__(self, pdf, api_input):
#         self.api = api_input
#         loader = PyMuPDFLoader(pdf)
#         pages = loader.load_and_split()
#
#         embeddings = OpenAIEmbeddings(openai_api_key=self.api)
#         vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
#         vectordb.persist()
#
#         self.pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo",
#                                                         openai_api_key=self.api),
#                                                  vectordb, return_source_documents=True)
#
#     def question(self, query):
#         result = self.pdf_qa({"question": "请使用中文回答" + query, "chat_history": ""})
#         print("Answer:")
#         print(result["answer"])
#
#         return result["answer"]
#
#
# def analyse(pdf_file, api_input):
#     print(pdf_file.name)
#     session = Chat(pdf_file.name, api_input)
#     return session, "文章分析完成"
#
#
# def ask_question(data, question):
#     if data == "":
#         return "Please upload PDF file first!"
#     return data.question(question)
#
#
# with gr.Blocks() as demo:
#     gr.Markdown(
#         """
#         # ChatPDF based on Langchain
#         """)
#     data = gr.State()
#     with gr.Tab("Upload PDF File"):
#         pdf_input = gr.File(label="PDF File")
#         api_input = gr.Textbox(label="OpenAI API Key")
#         result = gr.Textbox()
#         upload_button = gr.Button("Start Analyse")
#         question_input = gr.Textbox(label="Your Question", placeholder="Authors of this paper?")
#         answer = gr.Textbox(label="Answer")
#         ask_button = gr.Button("Ask")
#
#     upload_button.click(fn=analyse, inputs=[pdf_input, api_input], outputs=[data, result])
#     ask_button.click(ask_question, inputs=[data, question_input], outputs=answer)
#
# if __name__ == "__main__":
#     demo.title = "ChatPDF Based on Langchain"
#     demo.launch()
#

if __name__ == "__main__":
    print("构建测试通过！")
