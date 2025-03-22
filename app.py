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
def process_user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        # Tạo generator để mô phỏng streaming
        def response_generator():
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            return response["output_text"]
        
        return response_generator()
    except Exception as e:
        return f"Xin lỗi, đã xảy ra lỗi: {str(e)}"

def main():
    st.set_page_config(page_title="Trợ lý PDF Chat", page_icon="📚")
    
    # Tạo layout với sidebar và khu vực chat chính
    with st.sidebar:
        st.title("Trợ lý PDF Chat 📚")
        st.markdown("---")
        
        # Tải lên tài liệu
        pdf_docs = st.file_uploader("Tải lên tài liệu PDF", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Xử lý tài liệu"):
            if not pdf_docs:
                st.error("Vui lòng tải lên ít nhất một tài liệu PDF!")
            else:
                with st.spinner("Đang xử lý tài liệu..."):
                    # Xử lý tài liệu
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.pdf_processed = True
                    st.success("Hoàn thành! Bạn có thể đặt câu hỏi ngay bây giờ.")
        
        st.markdown("---")
        st.caption("Được hỗ trợ bởi Gemini 1.5 Pro")
    
    # Khu vực chat chính
    st.title("Chat với Tài liệu PDF")
    
    # Khởi tạo lịch sử chat trong session state nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Hiển thị tất cả tin nhắn từ lịch sử
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Xử lý đầu vào từ người dùng
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        if not st.session_state.pdf_processed:
            st.error("Vui lòng tải lên và xử lý tài liệu PDF trước khi đặt câu hỏi!")
            return
        
        # Hiển thị tin nhắn của người dùng
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Thêm tin nhắn của người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Xử lý câu hỏi và hiển thị câu trả lời
        with st.chat_message("assistant"):
            try:
                with st.spinner("Đang tìm câu trả lời..."):
                    response = process_user_input(prompt)
                    st.markdown(response)
                    
                    # Thêm câu trả lời vào lịch sử
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Đã xảy ra lỗi: {str(e)}"
                st.error(error_message)
                # Thêm thông báo lỗi vào lịch sử
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()
