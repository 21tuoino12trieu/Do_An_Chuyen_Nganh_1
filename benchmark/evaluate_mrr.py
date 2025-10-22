# MRR tập trung vào vị trí xuất hiện của kết quả đúng đầu tiên.
# Ví dụ minh hoạ:

# | Query | Kết quả đúng đầu tiên ở vị trí | Reciprocal Rank | Diễn giải           |
# | ----- | ------------------------------ | --------------- | ------------------- |
# | Q1    | 1                              | 1/1 = **1.0**   | Đúng ngay đầu tiên  |
# | Q2    | 3                              | 1/3 = **0.333** | Đúng ở vị trí thứ 3 |
# | Q3    | 5                              | 1/5 = **0.2**   | Đúng ở vị trí thứ 5 |
# | Q4    | Không có                       | 0               | Không bắt trúng     |

# MRR=(1.0+0.333+0.2+0.0)/4=0.383

from typing import List, Set

def mean_reciprocal_rank(retrieved: List[List[str]], ground_truth: List[Set[str]]) -> float:
    """
    Tính Mean Reciprocal Rank (MRR) cho một tập hợp truy vấn.

    Args:
        retrieved (List[List[str]]): Danh sách các danh sách kết quả được truy xuất cho mỗi truy vấn.
        ground_truth (List[Set[str]]): Danh sách các tập hợp câu trả lời đúng cho mỗi truy vấn.

    Returns:
        float: MRR trung bình trên tất cả các truy vấn.
    """
    assert len(retrieved) == len(ground_truth), "Số lượng truy vấn và câu trả lời đúng phải bằng nhau."

    total_reciprocal_rank = 0.0
    total_queries = len(retrieved)

    for recs, true_answers in zip(retrieved, ground_truth):
        reciprocal_rank = 0.0
        for rank, rec in enumerate(recs, start=1):
            if rec in true_answers:
                reciprocal_rank = 1.0 / rank
                break
        total_reciprocal_rank += reciprocal_rank

    return total_reciprocal_rank / total_queries if total_queries > 0 else 0.0