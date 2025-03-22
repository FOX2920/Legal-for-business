import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import traceback

# Kiểm tra và cấu hình API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY không được tìm thấy trong biến môi trường. Vui lòng cấu hình API key.")

# Cấu hình API key nếu có
if api_key:
    genai.configure(api_key=api_key)

# Hàm để trích xuất văn bản từ PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Kiểm tra nếu văn bản được trích xuất
                    text += page_text
        except Exception as e:
            st.error(f"Lỗi khi đọc file {pdf.name}: {str(e)}")
    return text

# Hàm để chia nhỏ văn bản
def get_text_chunks(text):
    if not text.strip():
        raise ValueError("Không có nội dung văn bản được trích xuất từ PDF")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    if not chunks:
        raise ValueError("Không thể chia nhỏ văn bản thành các đoạn")
    return chunks

# Hàm để tạo vector store
def get_vector_store(text_chunks):
    if not api_key:
        raise ValueError("API key chưa được cấu hình")
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return True

# Hàm để tạo chuỗi hội thoại
def get_conversational_chain():
    prompt_template = """
    Bạn là một trợ lý AI thông minh và thân thiện, giúp trả lời câu hỏi dựa trên thông tin từ tài liệu PDF.
    Hãy trả lời bằng tiếng Việt rõ ràng, dễ hiểu và thân thiện.
    Nếu không biết câu trả lời, hãy thành thật nói rằng bạn không tìm thấy thông tin đó trong tài liệu.
    
    Ngữ cảnh:\n {context}\n
    Câu hỏi: \n{question}\n
    Câu trả lời:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Lỗi khi tạo chuỗi hội thoại: {str(e)}")
        raise

# Hàm kiểm tra xem vector store đã được tạo chưa
def is_vector_store_ready():
    return os.path.exists("faiss_index")

# Hàm xử lý đầu vào của người dùng
def user_input(user_question):
    if not is_vector_store_ready():
        return "Vui lòng tải lên và xử lý tài liệu PDF trước khi đặt câu hỏi!"
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        if not docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu. Vui lòng thử câu hỏi khác hoặc tải lên tài liệu có thông tin liên quan."
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        return response["output_text"]
    except Exception as e:
        st.error(f"Lỗi chi tiết: {traceback.format_exc()}")
        return f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn: {str(e)}"

def main():
    st.set_page_config(page_title="Trợ lý tài liệu PDF", page_icon="📚")
    
    # Tạo tiêu đề và hiệu ứng
    st.markdown("""
    <h1 style='text-align: center; color: #1E88E5;'>Trợ lý Tài Liệu PDF 📚</h1>
    <p style='text-align: center; font-size: 18px;'>Tải lên tài liệu PDF và đặt câu hỏi bằng tiếng Việt</p>
    """, unsafe_allow_html=True)
    
    # Khởi tạo các biến session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = False
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Tạo sidebar
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>Tải lên tài liệu</h2>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Tải lên tài liệu PDF của bạn", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Xử lý tài liệu"):
            if not pdf_docs:
                st.error("Vui lòng tải lên ít nhất một tài liệu PDF!")
            else:
                with st.spinner("Đang xử lý tài liệu..."):
                    try:
                        # Reset trạng thái xử lý
                        st.session_state.processed_files = False
                        
                        # Xử lý tài liệu
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("Không thể trích xuất văn bản từ các tài liệu PDF đã tải lên. Vui lòng kiểm tra lại tài liệu.")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            success = get_vector_store(text_chunks)
                            if success:
                                st.session_state.processed_files = True
                                st.success(f"Hoàn thành! Đã xử lý {len(pdf_docs)} tài liệu PDF. Bạn có thể đặt câu hỏi ngay bây giờ.")
                    except Exception as e:
                        st.error(f"Lỗi khi xử lý tài liệu: {str(e)}")
        
        # Hiển thị trạng thái xử lý
        if st.session_state.processed_files:
            st.success("Tài liệu đã được xử lý và sẵn sàng để truy vấn.")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center;'>
            <p>Trợ lý sử dụng mô hình Gemini 1.5 Pro</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Thêm nút xóa lịch sử chat
        if st.button("Xóa lịch sử chat"):
            st.session_state.messages = []
            st.success("Đã xóa lịch sử chat!")
    
    # Hiển thị lịch sử chat bằng st.chat_message
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Sử dụng st.chat_input
    user_question = st.chat_input("Nhập câu hỏi của bạn ở đây...")
    
    if user_question:
        # Hiển thị tin nhắn người dùng
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Thêm tin nhắn người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Kiểm tra và xử lý câu hỏi
        with st.chat_message("assistant"):
            if not st.session_state.processed_files and not is_vector_store_ready():
                response = "Vui lòng tải lên và xử lý tài liệu PDF trước khi đặt câu hỏi!"
                st.warning(response)
            else:
                with st.spinner("Đang tìm câu trả lời..."):
                    response = user_input(user_question)
                    st.markdown(response)
            
            # Thêm câu trả lời vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
