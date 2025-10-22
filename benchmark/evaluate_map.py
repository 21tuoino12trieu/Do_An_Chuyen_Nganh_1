# MAP@K có thể hiểu là trung bình của Precision tại mỗi vị trí mà có kết quả đúng,
# sau đó lấy trung bình trên toàn bộ truy vấn.

# Ví dụ cụ thể:
# Giả sử 1 câu hỏi (query) có 3 đáp án đúng: {A, B, C}
# Hệ thống trả về Top-5 kết quả: [A, X, B, Y, C]

# | Vị trí (k) | Kết quả | Đúng? | Precision@k | Ghi chú       |
# | ---------- | ------- | ----- | ----------- | ------------- |
# | 1          | A       | ✅     | 1/1 = 1.00  | đúng đầu tiên |
# | 2          | X       | ❌     | —           | không tính    |
# | 3          | B       | ✅     | 2/3 = 0.67  | đúng thứ hai  |
# | 4          | Y       | ❌     | —           | không tính    |
# | 5          | C       | ✅     | 3/5 = 0.60  | đúng thứ ba   |

# AP=(1.00+0.67+0.60)/3=0.7567

from typing import List, Set
def map_at_k(retrieved: List[List[str]], ground_truth: List[Set[str]], k: int) -> float:
    """
    Tính Mean Average Precision (MAP) tại K cho một tập hợp truy vấn.

    Args:
        retrieved (List[List[str]]): Danh sách các danh sách kết quả được truy xuất cho mỗi truy vấn.
        ground_truth (List[Set[str]]): Danh sách các tập hợp câu trả lời đúng cho mỗi truy vấn.
        k (int): Số lượng kết quả hàng đầu để xem xét.

    Returns:
        float: MAP@K trung bình trên tất cả các truy vấn.
    """
    assert len(retrieved) == len(ground_truth), "Số lượng truy vấn và câu trả lời đúng phải bằng nhau."

    total_ap = 0.0
    total_queries = len(retrieved)

    for recs, true_answers in zip(retrieved, ground_truth):
        ap = 0.0
        correct_count = 0
        for rank, rec in enumerate(recs[:k], start=1):
            if rec in true_answers:
                correct_count += 1
                ap += correct_count / rank
        if correct_count > 0:
            ap /= correct_count
        total_ap += ap

    return total_ap / total_queries if total_queries > 0 else 0.0