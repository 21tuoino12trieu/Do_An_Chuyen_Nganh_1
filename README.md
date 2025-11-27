# Vietnamese Legal Retrieval System (Hệ thống Tra cứu & Hỏi đáp Pháp luật)

Dự án xây dựng hệ thống tìm kiếm và hỏi đáp (RAG) chuyên biệt cho văn bản pháp luật Việt Nam. Hệ thống kết hợp các kỹ thuật tìm kiếm ngữ nghĩa (Dense Retrieval) và mô hình ngôn ngữ lớn (LLM) để cung cấp câu trả lời chính xác, có trích dẫn nguồn luật cụ thể.

## Mục tiêu

*   **Tìm kiếm thông minh:** Vượt qua giới hạn của tìm kiếm từ khóa (BM25) bằng cách sử dụng tìm kiếm vector (Dense Retrieval) để hiểu ngữ nghĩa câu hỏi.
*   **Độ chính xác cao:** Tích hợp kỹ thuật Reranking (sắp xếp lại kết quả) để tối ưu hóa độ liên quan của các điều luật được tìm thấy.
*   **Trợ lý ảo:** Cung cấp giao diện Chatbot (Streamlit) sử dụng Google Gemini để tổng hợp câu trả lời tự nhiên từ các văn bản luật tìm được.
*   **Đánh giá & So sánh:** Hệ thống bao gồm bộ công cụ benchmark để so sánh hiệu quả giữa BM25, Dense Retrieval và Dense + Rerank.

## Các Mô hình Embedding

Hệ thống hỗ trợ và đã thử nghiệm với các mô hình embedding sau:

1.  **AITeamVN (AI Team Việt Nam):**
    *   Mô hình được fine-tune chuyên biệt cho pháp luật tiếng Việt.
    *   Kiến trúc Contrastive Learning, tối ưu hóa cho các cặp câu hỏi - điều luật.
    *   **Hiệu năng:** Cao nhất trong các thử nghiệm (Precision@1 ~0.654).

2.  **Jina Embeddings v3:**
    *   Mô hình đa ngôn ngữ (hỗ trợ tốt tiếng Việt), context dài lên đến 8192 tokens.
    *   Phù hợp cho các văn bản dài và truy vấn đa ngôn ngữ.

3.  **Qwen3-Embedding-0.6B:**
    *   Mô hình từ Alibaba Cloud, pre-trained trên dữ liệu đa ngôn ngữ quy mô lớn.
    *   Hiệu năng tốt cho triển khai on-premise.

4.  **vn_dcm_embedding (Data-centric Movement):**
    *   Mô hình general purpose cho tiếng Việt, sử dụng làm baseline để so sánh.

## Hệ thống Đánh giá & Metrics

Dự án sử dụng bộ metrics tiêu chuẩn trong Information Retrieval để đánh giá hiệu quả:

*   **Precision@K:** Tỷ lệ tài liệu liên quan trong top K kết quả trả về.
*   **Recall@K:** Tỷ lệ tài liệu liên quan tìm được trên tổng số tài liệu liên quan có trong dữ liệu.
*   **MRR (Mean Reciprocal Rank):** Đánh giá thứ hạng của kết quả đúng đầu tiên (càng cao càng tốt).
*   **MAP (Mean Average Precision):** Độ chính xác trung bình, tính đến thứ hạng của tất cả các kết quả đúng.
*   **NDCG (Normalized Discounted Cumulative Gain):** Đánh giá chất lượng xếp hạng có tính đến mức độ liên quan (0: không liên quan, 1: cùng văn bản, 2: đúng điều khoản).

## Cài đặt

### Yêu cầu hệ thống
*   Python 3.10 trở lên
*   Docker (tùy chọn)
*   Tài khoản Qdrant (Cloud hoặc Local)
*   API Key Google Gemini

### Các bước cài đặt

1.  **Clone dự án:**
    ```bash
    git clone <repository_url>
    cd Do_An_Chuyen_Nganh_1
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Cấu hình môi trường:**
    Tạo file `.env` tại thư mục gốc và điền các thông tin sau:
    ```env
    GEMINI_API_KEY=your_gemini_api_key
    QDRANT_URL=your_qdrant_url
    QDRANT_API_KEY=your_qdrant_api_key
    ```

## Sử dụng

### 1. Khởi chạy Ứng dụng Hỏi đáp (Main App)
Giao diện chính cho người dùng cuối để hỏi đáp pháp luật.
```bash
streamlit run streamlit_app.py
```
Truy cập: `http://localhost:8501`
Minh họa trực tuyến: [https://doanchuyennganh1.streamlit.app/](https://doanchuyennganh1.streamlit.app/)

### 2. Khởi chạy Dashboard Đánh giá (Experiments Dashboard)
Xem biểu đồ so sánh hiệu quả giữa các phương pháp (F1, MAP, NDCG...).
```bash
streamlit run experiments_dashboard_app.py
```

### 3. Đánh chỉ mục dữ liệu (Indexing)
Nếu chạy lần đầu hoặc dữ liệu thay đổi, bạn cần vector hóa và đẩy dữ liệu lên Qdrant:
```bash
python indexing/embedding_by_AITeam.py
```

### 4. Chạy với Docker
```bash
docker build -t legal-assistant .
docker run -p 8501:8501 --env-file .env legal-assistant
```

## Cấu trúc dự án

```text
.
├── benchmark/                  # Các script và kết quả đánh giá hiệu năng (Metrics)
│   ├── experiments/            # Script chạy đánh giá cho từng phương pháp
│   └── results_of_experiments/ # Kết quả JSON (Precision, Recall, MRR...)
├── data/                       # Dữ liệu thô và đã qua xử lý
│   ├── Retrieval/              # Dữ liệu dùng cho indexing (documents, chunks)
│   └── Result_by_.../          # Kết quả truy vấn mẫu để chấm điểm
├── indexing/                   # Script tạo embedding và upload lên Qdrant
├── models/                     # Chứa weight của các model (AITeamVN, BGE-Reranker...)
├── pipeline/                   # Logic tìm kiếm cốt lõi
│   ├── Keyword_Retrieval/      # BM25
│   ├── Dense_Retrieval/        # Vector Search (Qdrant)
│   └── Dense_Rerank/           # Vector Search + Reranking
├── preprocessing/              # Script làm sạch và chuẩn hóa dữ liệu
├── prompt/                     # Các template prompt cho LLM
├── streamlit_app.py            # Ứng dụng chính (Chatbot)
├── experiments_dashboard_app.py# Ứng dụng Dashboard so sánh kết quả
└── requirements.txt            # Danh sách thư viện
```

## Luồng dữ liệu (Data Flow)

1.  **Preprocessing:** Dữ liệu thô (văn bản luật) -> `preprocessing/` -> JSONL chuẩn hóa (chia nhỏ thành các chunk/điều khoản).
2.  **Indexing:** JSONL -> `indexing/embedding_by_AITeam.py` -> Model Embedding -> Vector -> Lưu trữ tại **Qdrant Cloud**.
3.  **Retrieval (Khi người dùng hỏi):**
    *   Câu hỏi -> Model Embedding -> Vector Query.
    *   Gửi Vector Query -> Qdrant -> Trả về Top K văn bản liên quan (Dense Retrieval).
    *   *(Tùy chọn)* Top K văn bản -> Model Rerank -> Sắp xếp lại (Dense + Rerank).
4.  **Generation:**
    *   Câu hỏi + Văn bản đã tìm được (Context) -> **Google Gemini** -> Câu trả lời cuối cùng -> Hiển thị lên Streamlit.
5.  **Evaluation:**
    *   Chạy các script trong `benchmark/` để đo lường độ chính xác của từng pipeline (BM25, Dense, Rerank) dựa trên bộ dữ liệu test có sẵn.