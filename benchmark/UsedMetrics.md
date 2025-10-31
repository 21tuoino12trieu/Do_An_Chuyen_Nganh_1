# Các Độ Đo Đang Sử Dụng

## Danh sách nhanh

- Precision@K
- Recall@K
- HitRate@K
- MRR@K
- F1Score@K
- MAP@K
- NDCG@K

## Precision@K

### Mục đích

Precision@K đo mức độ chính xác của hệ thống truy hồi trong `K` kết quả đầu tiên.

### Định nghĩa

Giả sử:

- `q`: truy vấn (query).
- `L_q = [d_1, d_2, …, d_k]`: danh sách top-`k` tài liệu hệ thống trả về.
- `R_q`: tập tài liệu liên quan thực sự (ground truth) của truy vấn `q`.

Khi đó:\
`Precision@K = (số tài liệu đúng trong top-K) / K`

### Ví dụ

Truy vấn: “Trách nhiệm của đơn vị thẩm định trong xây dựng là gì?”

| ID tài liệu | Nội dung tóm tắt                            | Liên quan?           |
| ----------- | ------------------------------------------- | -------------------- |
| D1          | Điều 104 - Trách nhiệm của đơn vị thẩm định | ✅                    |
| D2          | Điều 105 - Trách nhiệm chủ đầu tư           | ✅ (liên quan nhẹ)    |
| D3          | Điều 120 - Quản lý chi phí đầu tư           | ❌                    |
| D4          | Điều 45 - Hồ sơ thiết kế kỹ thuật           | ❌                    |
| D5          | Điều 110 - Trách nhiệm nhà thầu             | ✅                    |
| D6          | Điều 60 - Báo cáo quyết toán                | ❌                    |

Hệ thống trả về: `L_q = [D3, D1, D5, D4, D2]`

| K | Các tài liệu top-K   | Số đúng trong top-K | Precision@K |
| - | -------------------- | ------------------- | ----------- |
| 1 | [D3]                 | 0                   | 0.00        |
| 2 | [D3, D1]             | 1                   | 0.50        |
| 3 | [D3, D1, D5]         | 2                   | 0.67        |
| 4 | [D3, D1, D5, D4]     | 2                   | 0.50        |
| 5 | [D3, D1, D5, D4, D2] | 3                   | 0.60        |

Diễn giải:

- Precision@3 = 0.67 ⇒ 67% kết quả đầu tiên là chính xác.
- Precision@5 = 0.60 ⇒ Trong top-5 có 60% tài liệu liên quan.

| Precision@K cao                                 | Precision@K thấp                               |
| ----------------------------------------------- | ---------------------------------------------- |
| Ít tài liệu sai ở top đầu                       | Nhiều tài liệu không liên quan                 |
| Người dùng xem vài kết quả đầu đã có đáp án     | Phải cuộn sâu mới thấy tài liệu đúng           |
| Phù hợp cho chatbot, RAG, QA, Search UX         | Cần cải thiện reranker hoặc embedding          |

## Recall@K

### Mục đích

Recall@K đo tỉ lệ tài liệu đúng mà hệ thống đã lấy được trong top-`K` so với tổng số tài liệu đúng thực tế.

### Định nghĩa

Giả sử:

- `R_q`: tập tài liệu liên quan thật sự (ground truth).
- `d_i`: tài liệu ở vị trí `i` trong danh sách kết quả.
- `K`: số kết quả đầu tiên được xét.

### Ví dụ

Truy vấn: “Trách nhiệm của đơn vị thẩm định là gì?”\
Ground truth: `R_q = {D1, D2, D5, D8}` (4 tài liệu).\
Hệ thống trả về top-5: `L_q = [D3, D1, D5, D9, D7]`.

| K | Các tài liệu top-K   | Số đúng tìm thấy | Recall@K |
| - | -------------------- | ---------------- | -------- |
| 1 | [D3]                 | 0                | 0.00     |
| 3 | [D3, D1, D5]         | 2                | 0.50     |
| 5 | [D3, D1, D5, D9, D7] | 2                | 0.50     |

Interpretation: Recall@3 = 0.5 ⇒ trong 4 tài liệu đúng, hệ thống tìm được 2.\
Tăng `K` thường làm Recall tăng nhưng Precision có thể giảm.\
(F1Score@K phía dưới cho thấy cách cân bằng Precision và Recall.)

## HitRate@K

### Mục đích

HitRate@K kiểm tra liệu trong top-`K` có ít nhất một tài liệu đúng hay không. Đây là độ đo dạng nhị phân (hit/miss).

### Công thức

```
HitRate@K(q) = 1 nếu tồn tại ít nhất một tài liệu đúng trong top-K
HitRate@K(q) = 0 nếu không có tài liệu đúng trong top-K
```

### Ví dụ

Truy vấn: “Điều kiện để nhà thầu được tham gia đấu thầu là gì?”\
Ground truth: `R_q = {D1, D4, D6}`.\
Hệ thống trả về top-5: `L_q = [D2, D3, D8, D1, D9]`.

| K | Các tài liệu top-K   | Có tài liệu đúng? | HitRate@K |
| - | -------------------- | ----------------- | --------- |
| 1 | [D2]                 | ❌                | 0         |
| 3 | [D2, D3, D8]         | ❌                | 0         |
| 5 | [D2, D3, D8, D1, D9] | ✅ (D1)           | 1         |

Diễn giải:

- HitRate@5 = 1 vì top-5 chứa ít nhất một tài liệu đúng.
- HitRate@3 = 0 vì top-3 chưa có tài liệu đúng.

## F1Score@K

### Mục đích

F1Score@K cân bằng giữa Precision@K và Recall@K. Đây là trung bình điều hòa, nhấn mạnh rằng hai độ đo phải cùng cao để điểm cuối cao.

### Công thức

```
F1Score@K = 2 * Precision@K * Recall@K / (Precision@K + Recall@K)
```

Nếu một trong hai (Precision hoặc Recall) bằng 0 thì F1Score@K = 0.

### Ví dụ

Với Precision@5 = 0.60 và Recall@5 = 0.50 ở ví dụ trên:\
`F1Score@5 = 2 * 0.60 * 0.50 / (0.60 + 0.50) ≈ 0.55`

F1Score@K phù hợp khi cần đánh giá hệ thống không chỉ trả đúng (precision) mà còn phủ đủ (recall) trong top-K.

## MAP@K (Mean Average Precision)

### Mục đích

MAP@K đánh giá chất lượng tổng thể của danh sách truy hồi bằng cách xét mức độ đúng đắn tại từng vị trí đúng trong top-K, sau đó lấy trung bình trên toàn bộ truy vấn.

### Định nghĩa

Với mỗi truy vấn `q`, Average Precision@K (AP@K) được tính như sau:

1. Duyệt qua từng vị trí `i` trong top-K.
2. Mỗi khi gặp một tài liệu đúng, lấy Precision@i.
3. Lấy trung bình Precision@i trên tổng số tài liệu đúng xuất hiện trong top-K.

Sau đó, MAP@K = trung bình của AP@K trên tất cả truy vấn.

### Ví dụ nhanh

Nếu một truy vấn có 3 tài liệu đúng xuất hiện ở các vị trí 2, 4, 6:

- Precision@2 = 1/2
- Precision@4 = 2/4
- Precision@6 = 3/6

`AP@K = (1/2 + 2/4 + 3/6) / 3 = 0.58` (giả sử K ≥ 6).\
MAP@K là trung bình AP@K của toàn bộ truy vấn.

## NDCG@K (Normalized Discounted Cumulative Gain)

### Mục đích

NDCG@K dùng cho các bài toán có mức độ liên quan nhiều cấp (ví dụ: rất liên quan, liên quan, không liên quan), chứ không chỉ đúng/sai. Thứ hạng cao được ưu tiên hơn nhờ hệ số chiết khấu.

### Công thức rút gọn

1. Tính DCG@K:

```
DCG@K = rel_1 + Σ_{i=2..K} (rel_i / log2(i))
```

Trong đó `rel_i` là độ liên quan tại vị trí `i`.\

2. Tính iDCG@K (DCG lý tưởng) bằng cách sắp xếp các tài liệu đúng theo thứ tự liên quan giảm dần.\
3. `NDCG@K = DCG@K / iDCG@K` (giới hạn trong [0, 1]).

### Ví dụ nhanh

Nếu top-3 có độ liên quan `[3, 2, 0]`:

- `DCG@3 = 3 + 2 / log2(2) + 0 / log2(3) = 3 + 2 = 5`
- Giả sử iDCG@3 = 3 + 2 + 1 = 6 ⇒ `NDCG@3 = 5 / 6 ≈ 0.83`

NDCG@K khuyến khích hệ thống đưa tài liệu rất liên quan lên đầu danh sách.
