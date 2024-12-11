import sys
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import RePhraseQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class PDFRAGSystem:
    def __init__(self, pdf_files):
        self.pdf_files = pdf_files
        self.documents = []
        self.vector_store = None
        self.qa_chain = None
        self.save_path = './vector_store'
        self.embeddings = OpenAIEmbeddings()

    def load_and_split_documents(self, pdf_dir):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=['。', '\n', '\r\n']
            # length_function=len,
            # is_separator_regex=False
        )

        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(pdf_dir, pdf_file))
                documents = loader.load_and_split(text_splitter)
                self.documents.extend(documents)

    def create_vector_store(self):
        self.vector_store = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )
        self.vector_store.save_local(self.save_path)

    def create_qa_chain(self):
        system_template = """
参考情報を元に、ユーザーからの質問にできるだけ正確に答えてください。
参考情報に含まれない情報を答えることはできません。
{context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "{question}")
        ])

        db = FAISS.load_local(
            folder_path=self.save_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(
            search_kwargs={'k': 5}
        )

        hyde_template = """
ユーザーの質問は以下の通りです。
質問: {question}
回答: """

        hyde_prompt = ChatPromptTemplate.from_template(hyde_template)

        llm = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0
        )

        re_phrase_retriever = RePhraseQueryRetriever.from_llm(
            retriever=retriever,
            prompt=hyde_prompt,
            llm=llm
        )

        self.qa_chain = (
            {"context": re_phrase_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def query(self, question):
        if not self.qa_chain:
            raise ValueError("QAチェーンが初期化されていません")

        return self.qa_chain.invoke(question)

    def prepare(self):
        self.load_and_split_documents('pdfs')
        self.create_vector_store()

    def process(self, question):
        self.create_qa_chain()
        return self.query(question)

def main():
    pdf_files = [
        "Architectural_Design_Service_Contract.pdf",
        "Call_Center_Operation_Service_Contract.pdf",
        "Consulting_Service_Contract.pdf",
        "Content_Production_Service_Contract_(Request_Form).pdf",
        "Customer_Referral_Contract.pdf",
        "Draft_Editing_Service_Contract.pdf",
        "Graphic_Design_Production_Service_Contract.pdf",
        "M&A_Advisory_Service_Contract_(Preparatory_Committee).pdf",
        "M&A_Intermediary_Service_Contract_SME_M&A_[Small_and_Medium_Enterprises].pdf",
        "Manufacturing_Sales_Post-Safety_Management_Contract.pdf",
        "software_development_outsourcing_contracts.pdf",
        "Technical_Verification_(PoC)_Contract.pdf",
    ]

    rag_system = PDFRAGSystem(pdf_files)

    if len(sys.argv) > 1:
        question = sys.argv[1]
        answer = rag_system.process(question)
        print(f"質問: {question}")
        print(f"回答: {answer}")

    else:
        rag_system.prepare()

if __name__ == "__main__":
    main()