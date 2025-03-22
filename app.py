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

# Tải biến môi trường từ file .env
load_dotenv()

# Cấu hình API key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Hàm để trích xuất văn bản từ PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Hàm để chia nhỏ văn bản
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Hàm để tạo vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

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
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Hàm xử lý đầu vào của người dùng
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        return response["output_text"]
    except Exception as e:
        return f"Xin lỗi, đã xảy ra lỗi: {str(e)}"

def main():
    st.set_page_config(page_title="Trợ lý tài liệu PDF", page_icon="📚")
    
    # Tạo tiêu đề và hiệu ứng
    st.markdown("""
    <h1 style='text-align: center; color: #1E88E5;'>Trợ lý Tài Liệu PDF 📚</h1>
    <p style='text-align: center; font-size: 18px;'>Tải lên tài liệu PDF và đặt câu hỏi bằng tiếng Việt</p>
    """, unsafe_allow_html=True)
    
    # Tạo sidebar
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>Tải lên tài liệu</h2>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Tải lên tài liệu PDF của bạn", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Xử lý tài liệu"):
            if not pdf_docs:
                st.error("Vui lòng tải lên ít nhất một tài liệu PDF!")
            else:
                with st.spinner("Đang xử lý tài liệu..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Hoàn thành! Bạn có thể đặt câu hỏi ngay bây giờ.")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center;'>
            <p>Trợ lý sử dụng mô hình Gemini 1.5 Pro</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Container chính cho chat
    chat_container = st.container()
    
    # Khởi tạo lịch sử chat nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with chat_container:
            if message["role"] == "user":
                st.markdown(f"""
                <div style='background-color: #EAEAEA; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                    <p><strong>Bạn:</strong> {message["content"]}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #E3F2FD; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                    <p><strong>Trợ lý:</strong> {message["content"]}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Phần nhập câu hỏi
    user_question = st.text_input("Nhập câu hỏi của bạn ở đây...", key="question_input")
    
    if st.button("Gửi"):
        if not user_question:
            st.warning("Vui lòng nhập câu hỏi!")
        else:
            try:
                # Thêm tin nhắn người dùng vào lịch sử
                st.session_state.messages.append({"role": "user", "content": user_question})
                
                # Kiểm tra xem đã xử lý tài liệu chưa
                try:
                    with st.spinner("Đang tìm câu trả lời..."):
                        response = user_input(user_question)
                        
                        # Thêm câu trả lời vào lịch sử
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Làm mới trang để hiển thị tin nhắn mới
                        st.experimental_rerun()
                except Exception:
                    st.error("Vui lòng tải lên và xử lý tài liệu PDF trước khi đặt câu hỏi!")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {str(e)}")

if __name__ == "__main__":
    main()
