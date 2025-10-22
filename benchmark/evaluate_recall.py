Recall@K là độ đo song song và bổ sung cho Precision@K, 
giúp bạn hiểu được mức độ bao phủ (coverage) của hệ thống truy hồi.
Ví dụ minh hoạ:
    
| Query | Ground Truth (đúng) | Top-5 hệ thống trả về | # đúng trong Top-5 | Recall@5 |
| ----- | ------------------- | --------------------- | ------------------ | -------- |
| Q1    | {A, B, C}           | [A, X, Y, B, Z]       | 2/3                | 0.667    |
| Q2    | {D}                 | [E, F, D, G, H]       | 1/1                | 1.0      |
| Q3    | {J, K, L, M}        | [J, K, N, O, P]       | 2/4                | 0.5      |

Mean_Recall@5=(0.667+1.0+0.5)/3=0.722

from typing import List, Set
def recall_at_k(retrieved: List[List[str]], ground_truth: List[Set[str]], k: int) -> float:
    """
    Tính Recall@K cho một tập hợp truy vấn.

    Args:
        retrieved (List[List[str]]): Danh sách các danh sách kết quả được truy xuất cho mỗi truy vấn.
        ground_truth (List[Set[str]]): Danh sách các tập hợp câu trả lời đúng cho mỗi truy vấn.
        k (int): Số lượng kết quả hàng đầu để xem xét.

    Returns:
        float: Recall@K trung bình trên tất cả các truy vấn.
    """
    assert len(retrieved) == len(ground_truth), "Số lượng truy vấn và câu trả lời đúng phải bằng nhau."

    total_recall = 0.0
    total_queries = len(retrieved)

    for recs, true_answers in zip(retrieved, ground_truth):
        top_k_recs = recs[:k]
        correct_count = sum(1 for ans in top_k_recs if ans in true_answers)
        recall = correct_count / len(true_answers) if len(true_answers) > 0 else 0.0
        total_recall += recall

    return total_recall / total_queries if total_queries > 0 else 0.0