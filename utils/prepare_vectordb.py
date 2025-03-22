from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

def extract_pdf_text(pdfs):
    """
    Trích xuất văn bản từ tài liệu PDF

    Tham số:
    - pdfs (list): Danh sách tài liệu PDF

    Trả về:
    - docs: Danh sách văn bản được trích xuất từ tài liệu PDF
    """
    docs = []
    for pdf in pdfs:
        pdf_path = os.path.join("docs", pdf)
        # Tải văn bản từ PDF và mở rộng danh sách tài liệu
        docs.extend(PyPDFLoader(pdf_path).load())
    return docs

def get_text_chunks(docs):
    """
    Chia văn bản thành các đoạn nhỏ

    Tham số:
    - docs (list): Danh sách tài liệu văn bản

    Trả về:
    - chunks: Danh sách các đoạn văn bản nhỏ
    """
    # Kích thước đoạn được cấu hình để xấp xỉ với giới hạn mô hình 2048 token
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800, separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_vectorstore(pdfs, from_session_state=False):
    """
    Tạo hoặc lấy một kho vector từ tài liệu PDF

    Tham số:
    - pdfs (list): Danh sách tài liệu PDF
    - from_session_state (bool, optional): Cờ chỉ ra việc tải từ trạng thái phiên hay không. Mặc định là False

    Trả về:
    - vectordb hoặc None: Kho vector được tạo hoặc lấy ra. Trả về None nếu tải từ trạng thái phiên và cơ sở dữ liệu không tồn tại
    """
    load_dotenv()
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if from_session_state and os.path.exists("Vector_DB - Documents"):
        # Lấy kho vector từ cơ sở dữ liệu hiện có
        vectordb = Chroma(persist_directory="Vector_DB - Documents", embedding_function=embedding)
        return vectordb
    elif not from_session_state:
        docs = extract_pdf_text(pdfs)
        chunks = get_text_chunks(docs)
        # Tạo kho vector từ các đoạn và lưu nó vào thư mục Vector_DB - Documents
        vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="Vector_DB - Documents")
        return vectordb
    return None
