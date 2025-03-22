import os
from utils.prepare_vectordb import get_vectorstore

def initialize_session_state_variables(st):
    """
    Khởi tạo các biến trạng thái phiên cho ứng dụng Streamlit

    Tham số:
    - st (streamlit.delta_generator.DeltaGenerator): Đối tượng DeltaGenerator của Streamlit được sử dụng để hiển thị các phần tử
    """
    # Lấy danh sách tài liệu đã tải lên
    upload_docs = os.listdir("docs")
    # Danh sách các biến trạng thái phiên để khởi tạo
    variables_to_initialize = ["chat_history", "uploaded_pdfs", "processed_documents", "vectordb", "previous_upload_docs_length"]
    # Lặp qua các biến và khởi tạo chúng nếu không có trong trạng thái phiên
    for variable in variables_to_initialize:
        if variable not in st.session_state:
            if variable == "processed_documents":
                # Đặt thành tên của các tệp có trong thư mục docs
                st.session_state.processed_documents = upload_docs
            elif variable == "vectordb":
                # Đặt thành None nếu đây là lần đầu tiên ứng dụng được khởi tạo. Nếu không, đặt thành cơ sở dữ liệu vector đã tồn tại
                st.session_state.vectordb = get_vectorstore(upload_docs, from_session_state=True)
            elif variable == "previous_upload_docs_length":
                # Đặt thành số lượng tài liệu trong thư mục docs khi khởi động ứng dụng
                st.session_state.previous_upload_docs_length = len(upload_docs)
            else:
                st.session_state[variable] = []
