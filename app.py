import streamlit as st
import pandas as pd
import numpy as np
import faiss
import time
import os
import re
import random
import google.generativeai as genai
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

# Cấu hình API Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Thiết lập tiêu đề ứng dụng
st.set_page_config(
    page_title="Chatbot Tư Vấn Pháp Lý",
    page_icon="⚖️",
    layout="wide"
)

# Hàm để tạo embeddings từ Google's Gemini
def get_embeddings(texts, model="models/embedding-001"):
    if not texts:
        return []
    
    embeddings = []
    # Xử lý từng văn bản để tạo embedding
    for text in texts:
        if not text or text.isspace():
            # Thêm vector zero nếu text rỗng
            embeddings.append(np.zeros(768))
            continue
        
        try:
            # Tạo embedding thông qua Gemini API
            embedding = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            # Lấy giá trị embedding
            embedding_values = embedding["embedding"]
            embeddings.append(embedding_values)
        except Exception as e:
            st.error(f"Lỗi khi tạo embedding: {str(e)}")
            # Thêm vector zero trong trường hợp lỗi
            embeddings.append(np.zeros(768))
    
    return embeddings

# Hàm để chia nhỏ văn bản thành các đoạn
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    if not text or text.isspace():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

# Hàm để tạo hoặc tải FAISS index
def create_or_load_faiss_index(df, force_recreate=False):
    index_file = "legal_docs_faiss.index"
    mapping_file = "chunk_mapping.csv"
    
    if os.path.exists(index_file) and os.path.exists(mapping_file) and not force_recreate:
        # Tải index đã có sẵn
        st.info("Đang tải FAISS index có sẵn...")
        index = faiss.read_index(index_file)
        chunk_mapping = pd.read_csv(mapping_file)
        return index, chunk_mapping
    
    st.info("Đang tạo FAISS index mới...")
    # Tạo các chunks từ nội dung văn bản
    all_chunks = []
    doc_ids = []
    chunk_texts = []
    
    with st.spinner("Đang xử lý văn bản thành các đoạn nhỏ..."):
        for i, row in df.iterrows():
            content = row.get('content', '')
            if not content or content.isspace():
                continue
                
            doc_id = i
            chunks = chunk_text(content)
            
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                doc_ids.append(doc_id)
                chunk_texts.append(chunk)
    
    # Tạo embeddings cho tất cả các chunks
    with st.spinner("Đang tạo embeddings cho các đoạn văn bản..."):
        embeddings = get_embeddings(all_chunks)
    
    # Tạo FAISS index
    with st.spinner("Đang xây dựng FAISS index..."):
        dimension = len(embeddings[0])  # Kích thước vector embedding
        index = faiss.IndexFlatL2(dimension)
        embeddings_np = np.array(embeddings).astype('float32')
        index.add(embeddings_np)
    
    # Lưu index và mapping
    faiss.write_index(index, index_file)
    
    # Tạo và lưu mapping giữa chunk index và document
    chunk_mapping = pd.DataFrame({
        'doc_id': doc_ids,
        'chunk_text': chunk_texts
    })
    chunk_mapping.to_csv(mapping_file, index=False)
    
    return index, chunk_mapping

# Hàm tìm kiếm các đoạn văn bản liên quan
def search_similar_chunks(query, index, chunk_mapping, df, top_k=5):
    # Tạo embedding cho câu hỏi
    query_embedding = get_embeddings([query])[0]
    query_embedding_np = np.array([query_embedding]).astype('float32')
    
    # Tìm kiếm các chunks gần nhất
    distances, indices = index.search(query_embedding_np, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunk_mapping):
            doc_id = chunk_mapping.iloc[idx]['doc_id']
            chunk_text = chunk_mapping.iloc[idx]['chunk_text']
            
            # Lấy thông tin văn bản từ doc_id
            if doc_id < len(df):
                doc_title = df.iloc[doc_id]['title']
                doc_link = df.iloc[doc_id]['link']
                doc_date = df.iloc[doc_id]['create date']
                
                results.append({
                    'text': chunk_text,
                    'title': doc_title,
                    'link': doc_link,
                    'date': doc_date,
                    'score': float(distances[0][i])
                })
    
    return results

# Hàm để tạo câu trả lời từ Gemini dựa trên các chunks tìm được
def generate_response(query, relevant_chunks, df):
    if not relevant_chunks:
        return "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu pháp lý của chúng tôi."
    
    # Tạo prompt cho Gemini
    context = "\n\n".join([f"Tài liệu: {chunk['title']}\nNgày ban hành: {chunk['date']}\nNội dung: {chunk['text']}" 
                           for chunk in relevant_chunks])
    
    prompt = f"""Bạn là trợ lý tư vấn pháp lý cho doanh nghiệp Việt Nam. Sử dụng thông tin dưới đây để trả lời câu hỏi của người dùng.
    
Câu hỏi: {query}

Thông tin tham khảo:
{context}

Hãy trả lời dựa trên thông tin pháp lý đã cung cấp. Nếu không có đủ thông tin, hãy nói rõ là bạn không có thông tin đầy đủ.
Trả lời ngắn gọn, rõ ràng và dẫn chiếu đến các điều khoản cụ thể nếu có. Đánh số các điểm chính và đề xuất hành động cụ thể nếu có thể.
Cuối cùng, thêm phần "Nguồn tham khảo" liệt kê các văn bản pháp lý đã sử dụng để trả lời.
"""

    try:
        # Tạo trả lời từ Gemini
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Lỗi khi tạo câu trả lời: {str(e)}")
        return f"Đã xảy ra lỗi khi tạo câu trả lời. Chi tiết lỗi: {str(e)}"

# Hàm chính để tải dữ liệu và thiết lập ứng dụng
def main():
    # Thiết lập giao diện
    st.title("⚖️ Chatbot Tư Vấn Pháp Lý Cho Doanh Nghiệp")
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .message-user {
        background-color: #dcf8c6;
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
    }
    .message-bot {
        background-color: #f1f0f0;
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: left;
        width: fit-content;
        max-width: 80%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Thanh bên (sidebar) cho cài đặt
    with st.sidebar:
        st.header("Cài đặt")
        
        # Tùy chọn tạo lại index
        recreate_index = st.checkbox("Tạo lại FAISS index", value=False)
        
        st.header("Thông tin")
        st.info("""
        Chatbot này cung cấp thông tin tư vấn pháp lý dựa trên cơ sở dữ liệu các văn bản pháp luật Việt Nam. 
        
        Lưu ý: Thông tin chỉ mang tính chất tham khảo. Vui lòng tham khảo ý kiến của chuyên gia pháp lý cho các quyết định quan trọng.
        """)
    
    # Tải dữ liệu
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv("documents_with_content.csv", encoding="utf-8-sig")
            return df
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu: {str(e)}")
            # Tạo DataFrame rỗng với các cột cần thiết
            return pd.DataFrame(columns=["title", "link", "create date", "last update", "content"])
    
    df = load_data()
    
    # Khởi tạo FAISS index
    if 'faiss_index' not in st.session_state or 'chunk_mapping' not in st.session_state or recreate_index:
        with st.spinner("Đang khởi tạo hệ thống tìm kiếm..."):
            index, chunk_mapping = create_or_load_faiss_index(df, force_recreate=recreate_index)
            st.session_state['faiss_index'] = index
            st.session_state['chunk_mapping'] = chunk_mapping
    
    # Khởi tạo chat history nếu chưa có
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi là trợ lý tư vấn pháp lý cho doanh nghiệp. Bạn có câu hỏi gì về các vấn đề pháp lý cần tư vấn không?"}
        ]

    # Hiển thị tin nhắn trò chuyện
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ô nhập liệu cho người dùng
    if prompt := st.chat_input("Nhập câu hỏi pháp lý của bạn..."):
        # Thêm câu hỏi vào lịch sử
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Tạo câu trả lời
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("⏳ Đang tìm kiếm thông tin pháp lý liên quan...")
            
            # Tìm kiếm các đoạn văn bản liên quan
            relevant_chunks = search_similar_chunks(
                prompt, 
                st.session_state['faiss_index'], 
                st.session_state['chunk_mapping'], 
                df, 
                top_k=5
            )
            
            # Hiển thị các thông tin tìm được (debug)
            if st.sidebar.checkbox("Hiển thị kết quả tìm kiếm chi tiết", value=False):
                st.sidebar.subheader("Các đoạn văn bản liên quan:")
                for i, chunk in enumerate(relevant_chunks):
                    with st.sidebar.expander(f"{i+1}. {chunk['title'][:50]}..."):
                        st.write(f"**Nguồn:** {chunk['title']}")
                        st.write(f"**Ngày ban hành:** {chunk['date']}")
                        st.write(f"**Link:** {chunk['link']}")
                        st.write(f"**Nội dung:**\n{chunk['text']}")
            
            # Tạo câu trả lời
            with st.spinner("Đang tạo câu trả lời..."):
                response = generate_response(prompt, relevant_chunks, df)
                # Cập nhật câu trả lời
                response_placeholder.markdown(response)
                # Thêm vào lịch sử
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
