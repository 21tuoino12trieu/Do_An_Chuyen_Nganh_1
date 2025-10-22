# HitRate@K đo xem hệ thống có “bắt trúng” ít nhất một kết quả đúng trong top-K hay không.
#   Nếu trong top-K kết quả, có ít nhất 1 phần tử đúng, → HitRate = 1.
#   Nếu tất cả đều sai, → HitRate = 0.
# Khi tính trung bình qua tất cả các câu hỏi (queries), ta được HitRate@K trung bình, 
# phản ánh tỷ lệ truy vấn mà hệ thống trả về ít nhất 1 kết quả đúng trong K kết quả đầu tiên.

# Ví dụ minh hoạ
# | Query | Answers (ground truth) | Top-K kết quả | Có trúng không? | HitRate |
# | ----- | ---------------------- | ------------- | --------------- | ------- |
# | Q1    | {A, B}                 | [A, X, Y]     | ✅ A trúng       | 1       |
# | Q2    | {C}                    | [X, Y, Z]     | ❌               | 0       |
# | Q3    | {D, E}                 | [E, B, F]     | ✅ E trúng       | 1       |

from typing import List, Set

def hitrate_at_k(retrieved: List[List[str]], ground_truth: List[Set[str]], k: int) -> float:
    """
    Tính HitRate@K cho một tập hợp truy vấn.

    Args:
        retrieved (List[List[str]]): Danh sách các danh sách kết quả được truy xuất cho mỗi truy vấn.
        ground_truth (List[Set[str]]): Danh sách các tập hợp câu trả lời đúng cho mỗi truy vấn.
        k (int): Số lượng kết quả hàng đầu để xem xét.

    Returns:
        float: HitRate@K trung bình trên tất cả các truy vấn.
    """
    assert len(retrieved) == len(ground_truth), "Số lượng truy vấn và câu trả lời đúng phải bằng nhau."

    hit_count = 0
    total_queries = len(retrieved)

    for recs, true_answers in zip(retrieved, ground_truth):
        top_k_recs = recs[:k]
        if any(ans in top_k_recs for ans in true_answers):
            hit_count += 1

    return hit_count / total_queries if total_queries > 0 else 0.0