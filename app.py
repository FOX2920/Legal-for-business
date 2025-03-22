import streamlit as st
import os
from utils.save_docs import save_docs_to_vectordb
from utils.session_state import initialize_session_state_variables
from utils.prepare_vectordb import get_vectorstore
from utils.chatbot import chat

class ChatApp:
    """
    Một ứng dụng Streamlit để trò chuyện với tài liệu PDF

    Lớp này đóng gói chức năng để tải lên tài liệu PDF, xử lý chúng,
    và cho phép người dùng trò chuyện với tài liệu bằng chatbot. Nó xử lý việc khởi tạo
    cấu hình Streamlit và các biến trạng thái phiên, cũng như giao diện người dùng cho việc tải lên tài liệu
    và tương tác trò chuyện
    """
    def __init__(self):
        """
        Khởi tạo lớp ChatApp

        Phương thức này đảm bảo sự tồn tại của thư mục 'docs', thiết lập cấu hình trang Streamlit,
        và khởi tạo các biến trạng thái phiên
        """
        # Đảm bảo thư mục docs tồn tại
        if not os.path.exists("docs"):
            os.makedirs("docs")

        # Cấu hình và khởi tạo trạng thái phiên
        st.set_page_config(page_title="Trò chuyện với PDF 📚", page_icon="📚")
        st.title("Trò chuyện với tài liệu PDF 📚")
        initialize_session_state_variables(st)
        self.docs_files = st.session_state.processed_documents

    def run(self):
        """
        Chạy ứng dụng Streamlit để trò chuyện với PDF

        Phương thức này xử lý giao diện người dùng để tải lên tài liệu, mở khóa trò chuyện khi tài liệu được tải lên,
        và khóa trò chuyện cho đến khi tài liệu được tải lên
        """
        upload_docs = os.listdir("docs")
        # Giao diện thanh bên cho việc tải lên tài liệu
        with st.sidebar:
            st.subheader("Tài liệu của bạn")
            if upload_docs:
                st.write("Tài liệu đã tải lên:")
                for doc in upload_docs:
                    st.text(f"📄 {doc}")
            else:
                st.info("Chưa có tài liệu nào được tải lên.")
            
            st.subheader("Tải lên tài liệu PDF")
            pdf_docs = st.file_uploader("Chọn tài liệu PDF và nhấn vào 'Xử lý tài liệu'", type=['pdf'], accept_multiple_files=True)
            if pdf_docs:
                save_docs_to_vectordb(pdf_docs, upload_docs)

        # Mở khóa trò chuyện khi tài liệu được tải lên
        if self.docs_files or st.session_state.uploaded_pdfs:
            # Kiểm tra xem có tài liệu mới được tải lên để cập nhật biến vectordb trong trạng thái phiên không
            if len(upload_docs) > st.session_state.previous_upload_docs_length:
                with st.spinner("Đang cập nhật cơ sở dữ liệu..."):
                    st.session_state.vectordb = get_vectorstore(upload_docs, from_session_state=True)
                    st.session_state.previous_upload_docs_length = len(upload_docs)
                    st.success("Cơ sở dữ liệu đã được cập nhật!")
            
            st.session_state.chat_history = chat(st.session_state.chat_history, st.session_state.vectordb)

        # Khóa trò chuyện cho đến khi tài liệu được tải lên
        if not self.docs_files and not st.session_state.uploaded_pdfs:
            st.info("Hãy tải lên tệp PDF để bắt đầu trò chuyện. Bạn có thể tiếp tục tải lên nhiều tệp để trò chuyện, và nếu bạn cần thoát, bạn sẽ không cần tải lại các tệp này khi quay lại.")
            st.markdown("""
            ### Hướng dẫn sử dụng:
            1. Tải lên một hoặc nhiều tệp PDF từ thanh bên trái
            2. Nhấn nút "Xử lý tài liệu" để phân tích tài liệu
            3. Đặt câu hỏi về nội dung của tài liệu trong khung chat
            4. Xem nguồn thông tin được sử dụng trong thanh bên trái
            """)

if __name__ == "__main__":
    app = ChatApp()
    app.run()
