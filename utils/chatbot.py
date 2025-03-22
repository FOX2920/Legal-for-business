import streamlit as st
from collections import defaultdict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

def get_context_retriever_chain(vectordb):
    """
    Tạo một chuỗi truy xuất ngữ cảnh để tạo phản hồi dựa trên lịch sử trò chuyện và cơ sở dữ liệu vector

    Tham số:
    - vectordb: Cơ sở dữ liệu vector được sử dụng để truy xuất ngữ cảnh

    Trả về:
    - retrieval_chain: Chuỗi truy xuất ngữ cảnh để tạo phản hồi
    """
    # Tải biến môi trường (lấy khóa API cho các mô hình)
    load_dotenv()
    # Khởi tạo mô hình, thiết lập bộ truy xuất và prompt cho chatbot
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, convert_system_message_to_human=True)
    retriever = vectordb.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một chatbot hỗ trợ. Bạn sẽ nhận được một câu hỏi kèm theo lịch sử trò chuyện và nội dung được truy xuất từ cơ sở dữ liệu vector dựa trên câu hỏi của người dùng. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng thông tin từ cơ sở dữ liệu, hạn chế sử dụng kiến thức riêng của bạn. Nếu vì lý do nào đó bạn không biết câu trả lời cho câu hỏi, hoặc câu hỏi không thể trả lời vì không có ngữ cảnh, hãy yêu cầu người dùng cung cấp thêm chi tiết. Không tự tạo ra câu trả lời. Hãy trả lời câu hỏi từ ngữ cảnh này: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    # Tạo chuỗi để tạo phản hồi và một chuỗi truy xuất
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, chain)
    return retrieval_chain

def get_response(question, chat_history, vectordb):
    """
    Tạo phản hồi cho câu hỏi của người dùng dựa trên lịch sử trò chuyện và cơ sở dữ liệu vector

    Tham số:
    - question (str): Câu hỏi của người dùng
    - chat_history (list): Danh sách các tin nhắn trò chuyện trước đó
    - vectordb: Cơ sở dữ liệu vector được sử dụng để truy xuất ngữ cảnh

    Trả về:
    - response: Phản hồi được tạo ra
    - context: Ngữ cảnh liên quan đến phản hồi
    """
    chain = get_context_retriever_chain(vectordb)
    response = chain.invoke({"input": question, "chat_history": chat_history})
    return response["answer"], response["context"]

def chat(chat_history, vectordb):
    """
    Xử lý chức năng trò chuyện của ứng dụng

    Tham số:
    - chat_history (list): Danh sách các tin nhắn trò chuyện trước đó
    - vectordb: Cơ sở dữ liệu vector được sử dụng để truy xuất ngữ cảnh

    Trả về:
    - chat_history: Lịch sử trò chuyện đã cập nhật
    """
    user_query = st.chat_input("Hỏi một câu hỏi:")
    if user_query is not None and user_query != "":
        # Tạo phản hồi dựa trên câu hỏi của người dùng, lịch sử trò chuyện và kho vector
        response, context = get_response(user_query, chat_history, vectordb)
        # Cập nhật lịch sử trò chuyện. Mô hình sử dụng tối đa 10 tin nhắn trước đó để kết hợp vào phản hồi
        chat_history = chat_history + [HumanMessage(content=user_query), AIMessage(content=response)]
        # Hiển thị nguồn của phản hồi trên thanh bên
        with st.sidebar:
            st.subheader("Nguồn thông tin:")
            metadata_dict = defaultdict(list)
            for metadata in [doc.metadata for doc in context]:
                metadata_dict[metadata['source']].append(metadata['page'])
            for source, pages in metadata_dict.items():
                st.write(f"Tài liệu: {source}")
                st.write(f"Trang: {', '.join(map(str, pages))}")
    # Hiển thị lịch sử trò chuyện
    for message in chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Người dùng"):
            st.write(message.content)
    return chat_history
