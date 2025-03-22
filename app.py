import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI with API key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector embeddings from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create a Q&A chain with a Vietnamese-optimized prompt"""
    prompt_template = """
    Bạn là trợ lý AI thông minh chuyên trả lời câu hỏi dựa trên thông tin từ tài liệu. 
    Hãy trả lời bằng tiếng Việt một cách chính xác, rõ ràng và đầy đủ.
    
    Nội dung tài liệu:
    {context}
    
    Câu hỏi: 
    {question}
    
    Trả lời:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Process user question and generate response from PDF content"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        st.write("Trả lời:", response["output_text"])
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {str(e)}")
        st.info("Vui lòng tải lên tài liệu PDF và nhấn 'Xử lý tài liệu' trước khi đặt câu hỏi.")

def main():
    """Main application function"""
    st.set_page_config(page_title="Chat PDF Tiếng Việt", layout="wide")
    
    st.title("🇻🇳 Trò chuyện với tài liệu PDF bằng Gemini AI")
    st.markdown("---")
    
    with st.sidebar:
        st.header("📁 Quản lý tài liệu")
        pdf_docs = st.file_uploader(
            "Tải lên tài liệu PDF của bạn", 
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        process_button = st.button("Xử lý tài liệu")
        
        if process_button and pdf_docs:
            with st.spinner("Đang xử lý tài liệu..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                
                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                get_vector_store(text_chunks)
                
                st.success("Xử lý tài liệu hoàn tất! Bạn có thể đặt câu hỏi ngay bây giờ.")
        
        st.markdown("---")
        st.markdown("### Hướng dẫn sử dụng:")
        st.info(
            "1. Tải lên tài liệu PDF\n"
            "2. Nhấn 'Xử lý tài liệu'\n"
            "3. Đặt câu hỏi về nội dung tài liệu"
        )
    
    # Chat interface
    st.header("💬 Đặt câu hỏi về tài liệu của bạn")
    user_question = st.text_input("Nhập câu hỏi của bạn về tài liệu PDF:")
    
    if user_question:
        user_input(user_question)
    
    # Display sample questions
    with st.expander("Các câu hỏi mẫu"):
        st.markdown("""
        - Nội dung chính của tài liệu là gì?
        - Tóm tắt thông tin quan trọng nhất trong tài liệu.
        - Giải thích khái niệm X được đề cập trong tài liệu.
        - Các điểm chính trong phần Y của tài liệu là gì?
        """)

if __name__ == "__main__":
    main()
