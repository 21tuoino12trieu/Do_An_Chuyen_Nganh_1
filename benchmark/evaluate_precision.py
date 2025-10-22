# Precision đo tỷ lệ các kết quả mà hệ thống trả về là đúng.
# Tức là, trong tất cả các văn bản mà hệ thống cho là liên quan, bao nhiêu văn bản thật sự liên quan?
# Ví dụ minh họa:
    
# | Query | Ground Truth (đúng) | Top-5 hệ thống trả về | Kết quả đúng trong Top-5 | Precision@5 |
# | ----- | ------------------- | --------------------- | ------------------------ | ----------- |
# | Q1    | {A, B, C}           | [A, X, Y, B, Z]       | {A, B}                   | 2/5 = 0.4   |
# | Q2    | {D}                 | [E, F, D, G, H]       | {D}                      | 1/5 = 0.2   |
# | Q3    | {J, K, L}           | [M, N, O, P, Q]       | ∅                        | 0/5 = 0.0   |

# Mean_Precision@5=(0.4+0.2+0.0)/3=0.2

from typing import List, Set

def precision_at_k(retrieved: List[List[str]], ground_truth: List[Set[str]], k: int) -> float:
    """
    Tính Precision@K cho một tập hợp truy vấn.

    Args:
        retrieved (List[List[str]]): Danh sách các danh sách kết quả được truy xuất cho mỗi truy vấn.
        ground_truth (List[Set[str]]): Danh sách các tập hợp câu trả lời đúng cho mỗi truy vấn.
        k (int): Số lượng kết quả hàng đầu để xem xét.

    Returns:
        float: Precision@K trung bình trên tất cả các truy vấn.
    """
    assert len(retrieved) == len(ground_truth), "Số lượng truy vấn và câu trả lời đúng phải bằng nhau."

    total_precision = 0.0
    total_queries = len(retrieved)

    for recs, true_answers in zip(retrieved, ground_truth):
        top_k_recs = recs[:k]
        correct_count = sum(1 for ans in top_k_recs if ans in true_answers)
        precision = correct_count / k
        total_precision += precision

    return total_precision / total_queries if total_queries > 0 else 0.0