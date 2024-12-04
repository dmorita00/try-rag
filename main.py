import sys
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class PDFRAGSystem:
    def __init__(self, pdf_files):
        self.pdf_files = pdf_files
        self.documents = []
        self.vector_store = None
        self.qa_chain = None

    def load_and_split_documents(self, pdf_dir):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=3
        )

        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(pdf_dir, pdf_file))
                documents = loader.load_and_split(text_splitter)
                self.documents.extend(documents)

    def create_vector_store(self):
        embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=embeddings
        )

    def create_qa_chain(self):
        retriever = self.vector_store.as_retriever(
            search_kwargs={'k': 3}  # 上位3つの関連文書を取得
        )

        llm = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever
        )

    def query(self, question):
        if not self.qa_chain:
            raise ValueError("QAチェーンが初期化されていません")

        return self.qa_chain.invoke(question)

    def process(self, question):
        self.load_and_split_documents('pdfs')
        self.create_vector_store()
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
        print("Please provide a question as a command-line argument.")
        sys.exit(1)

if __name__ == "__main__":
    main()