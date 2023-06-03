from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ChatVectorDBChain # for chatting with the pdf
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)


class Chat:
    def __init__(self,pdf):
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()
        print(pages[0].page_content)

        embeddings = OpenAIEmbeddings(openai_api_key="MKiWJvBdv0HGDaMBZIn5T3BlbkFJUteJ7DY9FzlnXBF0uKD0")
        vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                        persist_directory="db")
        vectordb.persist()

        self.pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo",openai_api_key="MKiWJvBdv0HGDaMBZIn5T3BlbkFJUteJ7DY9FzlnXBF0uKD0"),
                                            vectordb, return_source_documents=True)


    def question(self,query):
        query = "请用中文概括一下本文内容"
        result = self.pdf_qa({"question": query, "chat_history": ""})
        print("Answer:")
        print(result["answer"])
        
        return result["answer"]

