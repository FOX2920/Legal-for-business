import streamlit as st
import os
from utils.prepare_vectordb import get_vectorstore

def save_docs_to_vectordb(pdf_docs, upload_docs):
    """
    Lưu tài liệu PDF đã tải lên vào thư mục 'docs' và tạo hoặc cập nhật kho vector

    Tham số:
    - pdf_docs (list): Danh sách tài liệu PDF đã tải lên
    - upload_docs (list): Danh sách tên của các tài liệu đã tải lên trước đó
    """
    # Lọc xem tệp có phải là tệp mới hay không. Nếu là tệp mới, nút xử lý sẽ xuất hiện
    new_files = [pdf for pdf in pdf_docs if pdf.name not in upload_docs]
    new_files_names = [pdf.name for pdf in new_files]
    if new_files and st.button("Xử lý tài liệu"):
        # Lặp qua các tệp mới và lưu chúng vào thư mục docs
        for pdf in new_files:
            pdf_path = os.path.join("docs", pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(pdf.getvalue())
            st.session_state.uploaded_pdfs.extend(pdf_docs)
        # Hiển thị thông báo đang xử lý
        with st.spinner("Đang xử lý..."):
            # Tạo hoặc cập nhật kho vector với các tài liệu mới tải lên
            get_vectorstore(new_files_names)
            st.success(f"Đã tải lên thành công {len(new_files)} tệp vào thư mục 'docs'.")
