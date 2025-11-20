import os
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from pipeline.Dense_Retrieval.retrieval_by_dense_vector import DenseVectorRetriever
from pipeline.Dense_Rerank.retrieval_by_dense_rerank import DenseRerank


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "legal_clauses_AITeamVN"
TOP_K = 3


def setup_gemini():
    if not GEMINI_API_KEY:
        raise EnvironmentError("Missing GEMINI_API_KEY in environment or .env")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash")


@st.cache_resource(show_spinner=False)
def get_dense_retriever():
    return DenseVectorRetriever(collection_name=COLLECTION_NAME)


@st.cache_resource(show_spinner=False)
def get_rerank_retriever():
    return DenseRerank(collection_name=COLLECTION_NAME)


@st.cache_resource(show_spinner=False)
def get_llm():
    return setup_gemini()


def format_context(docs: List[Dict]) -> str:
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        article = doc.get("article_id") or doc.get("article") or "N/A"
        clause = doc.get("clause_id") or doc.get("clause") or "N/A"
        content = doc.get("content") or ""
        blocks.append(f"[{idx}] Điều {article} - Khoản {clause}:\n{content}")
    return "\n\n".join(blocks)


def stream_llm(question: str, docs: List[Dict]):
    context = format_context(docs)
    prompt = f"""Bạn là trợ lý pháp lý tiếng Việt. Hãy tóm tắt súc tích dựa trên top 3 kết quả, trích dẫn điều/khoản nếu có.
Câu hỏi người dùng: {question}

Các đoạn tham khảo (top 3):
{context}

Chỉ dựa trên nội dung tham khảo, tránh suy diễn ngoài văn bản."""
    model = get_llm()
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


def render_sources(docs: List[Dict]):
    st.markdown("**Trích xuất từ các tài liệu**")
    for doc in docs:
        with st.expander(f"Điều {doc.get('article_id','N/A')} | Khoản {doc.get('clause_id','N/A')}"):
            st.write(doc.get("content", ""))


def do_retrieval(question: str, use_rerank: bool) -> Optional[Tuple[List[Dict], str]]:
    try:
        if use_rerank:
            retriever = get_rerank_retriever()
            docs = retriever.rerank(question, top_k=TOP_K, candidate_pool=20)
            mode_label = "Dense + Rerank"
        else:
            retriever = get_dense_retriever()
            docs = retriever.search(question, top_k=TOP_K)
            mode_label = "Dense"
    except Exception as exc:
        st.error(f"Lỗi khi tìm kiếm: {exc}")
        return None

    if not docs:
        st.warning("Không tìm thấy kết quả phù hợp.")
        return None
    return docs, mode_label


def main():
    st.set_page_config(
        page_title="Hỏi đáp pháp luật",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp {
            color: #ffffff;
            background: radial-gradient(circle at 20% 20%, #0f172a, #0b1220 40%, #050a16);
        }
        .stMarkdown, .stTextInput, .stTextArea, .stButton button, .stSidebar {
            color: #ffffff;
        }
        textarea, input, .stTextArea textarea, .stTextInput input {
            background: #111827 !important;
            color: #ffffff !important;
        }
        .stButton button {
            background: #2563eb;
            color: #ffffff;
            border: 1px solid #3b82f6;
        }
        h1, h2, h3, h4, h5, h6, label {
            color: #ffffff !important;
        }
        .st-expander, .st-expander span, .st-expander div {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h1 style="text-align: center;">⚖️ Trợ Lí Pháp Luật</h1>',
        unsafe_allow_html=True,
    )
    st.sidebar.header("Cấu hình tìm kiếm")
    use_rerank = st.sidebar.toggle("Bật Rerank (FlagEmbedding)", value=False)
    st.sidebar.markdown(
        "- **Dense**: truy vấn nhanh bằng ngữ nghĩa.\n"
        "- **Dense + Rerank**: mở rộng truy vấn, sắp xếp lại bằng reranker."
    )

    history = st.session_state.setdefault("chat_history", [])

    # Hiển thị lịch sử chat
    for turn in history:
        with st.chat_message(turn["role"]):
            if turn["role"] == "assistant" and turn.get("mode"):
                st.caption(f"Chế độ: {turn['mode']}")
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and turn.get("docs"):
                render_sources(turn["docs"])

    prompt = st.chat_input("Hỏi bất kỳ điều gì về pháp luật...")

    if prompt:
        # Hiển thị message người dùng vừa nhập
        with st.chat_message("user"):
            st.markdown(prompt)
        history.append({"role": "user", "content": prompt})

        # Assistant placeholder
        with st.chat_message("assistant"):
            st.caption(f"Chế độ: {'Dense + Rerank' if use_rerank else 'Dense'}")
            st.markdown("Đối với câu hỏi của bạn, mời bạn tham khảo các điều luật dưới đây:")
            answer_placeholder = st.empty()
            status_placeholder = st.empty()

            retrieval = do_retrieval(prompt, use_rerank)
            if retrieval:
                docs, mode_label = retrieval
                bullets = [
                    f"- Điều {doc.get('article_id','N/A')}, Khoản {doc.get('clause_id','N/A')}\n"
                    for doc in docs
                ]
                st.markdown("\n".join(bullets))

                # Streaming answer
                full_answer = ""
                status_placeholder.info("⏳ Đang tổng hợp câu trả lời từ các nguồn...")
                for piece in stream_llm(prompt, docs):
                    full_answer += piece
                    answer_placeholder.markdown(full_answer)
                status_placeholder.empty()

                render_sources(docs)
                history.append(
                    {
                        "role": "assistant",
                        "content": full_answer,
                        "docs": docs,
                        "mode": mode_label,
                    }
                )
            else:
                answer_placeholder.markdown("Không tìm thấy kết quả phù hợp hoặc có lỗi xảy ra.")
                history.append(
                    {
                        "role": "assistant",
                        "content": "Không tìm thấy kết quả phù hợp hoặc có lỗi xảy ra.",
                        "docs": [],
                        "mode": "Dense + Rerank" if use_rerank else "Dense",
                    }
                )

        st.rerun()


if __name__ == "__main__":
    main()
