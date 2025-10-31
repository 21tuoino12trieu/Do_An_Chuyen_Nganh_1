from typing import List
import math


def f1_score(precision: float, recall: float) -> float:
    """
    Tính F1-score từ Precision và Recall.

    Args:
        precision (float): Giá trị Precision.
        recall (float): Giá trị Recall.

    Returns:
        float: F1-score.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def hitrate_at_k(
    retrieved: List[List[str]], ground_truth: List[List[str]], k: int
) -> float:
    """
    Tính HitRate@K cho một tập hợp truy vấn.

    Args:
        retrieved (List[List[str]]): Danh sách các danh sách kết quả được truy xuất cho mỗi truy vấn.
        ground_truth (List[List[str]]): Danh sách các tập hợp câu trả lời đúng cho mỗi truy vấn.
        k (int): Số lượng kết quả hàng đầu để xem xét.

    Returns:
        float: HitRate@K trung bình trên tất cả các truy vấn.
    """
    assert len(retrieved) == len(
        ground_truth
    ), "Số lượng truy vấn và câu trả lời đúng phải bằng nhau."

    hit_count = 0
    total_queries = len(retrieved)

    for recs, true_answers in zip(retrieved, ground_truth):
        top_k_recs = recs[:k]
        if any(ans in top_k_recs for ans in true_answers):
            hit_count += 1

    return hit_count / total_queries if total_queries > 0 else 0.0


def map_at_k(
    retrieved: List[List[str]], ground_truth: List[List[str]], k: int
) -> float:
    """
    Tính Mean Average Precision (MAP) tại K cho một tập hợp truy vấn.

    Args:
        retrieved (List[List[str]]): Danh sách các danh sách kết quả được truy xuất cho mỗi truy vấn.
        ground_truth (List[List[str]]): Danh sách các tập hợp câu trả lời đúng cho mỗi truy vấn.
        k (int): Số lượng kết quả hàng đầu để xem xét.

    Returns:
        float: MAP@K trung bình trên tất cả các truy vấn.
    """
    assert len(retrieved) == len(
        ground_truth
    ), "Số lượng truy vấn và câu trả lời đúng phải bằng nhau."

    total_ap = 0.0
    total_queries = len(retrieved)

    for recs, true_answers in zip(retrieved, ground_truth):
        ap = 0.0
        correct_count = 0
        for rank, rec in enumerate(recs[:k], start=1):
            if rec in true_answers:
                correct_count += 1
                ap += correct_count / rank

        if len(true_answers) > 0:
            ap /= len(true_answers)
        else:
            ap = 0.0

        total_ap += ap

    return total_ap / total_queries if total_queries > 0 else 0.0


def mrr_at_k(
    retrieved: List[List[str]], ground_truth: List[List[str]], k: int
) -> float:
    """
    Tính Mean Reciprocal Rank (MRR) cho một tập hợp truy vấn.

    Args:
        retrieved (List[List[str]]): Danh sách các danh sách kết quả được truy xuất cho mỗi truy vấn.
        ground_truth (List[List[str]]): Danh sách các tập hợp câu trả lời đúng cho mỗi truy vấn.

    Returns:
        float: MRR trung bình trên tất cả các truy vấn.
    """
    assert len(retrieved) == len(
        ground_truth
    ), "Số lượng truy vấn và câu trả lời đúng phải bằng nhau."

    total_reciprocal_rank = 0.0
    total_queries = len(retrieved)

    for recs, true_answers in zip(retrieved, ground_truth):
        reciprocal_rank = 0.0
        for rank, rec in enumerate(recs[:k], start=1):
            if rec in true_answers:
                reciprocal_rank = 1.0 / rank
                break
        total_reciprocal_rank += reciprocal_rank

    return total_reciprocal_rank / total_queries if total_queries > 0 else 0.0


def _split_prefix(doc_id: str) -> str:
    """Trả về phần trước dấu # cuối cùng (để xác định cùng văn bản + điều)."""
    return doc_id.rsplit("#", 1)[0] if "#" in doc_id else doc_id


def relevance_level(pred_doc: str, gt_doc: str) -> int:
    """
    Gán nhãn 0/1/2 theo quy tắc:
    - 2: giống hệt
    - 1: cùng văn bản + cùng điều, khác khoản/điểm
    - 0: còn lại
    """
    if pred_doc == gt_doc:
        return 2
    if _split_prefix(pred_doc) == _split_prefix(gt_doc):
        return 1
    return 0


def build_relevance_list(pred: List[str], gt: List[str], k: int) -> List[int]:
    """
    Tạo list relevance theo thứ tự hệ thống trả về, cắt ở top-k.
    Mỗi phần tử = max relevance giữa tài liệu dự đoán và mọi ground truth.
    """
    rels: List[int] = []
    for doc in pred[:k]:
        rels.append(max(relevance_level(doc, g) for g in gt))
    return rels


def dcg_at_k(rels: List[int], k: int) -> float:
    """Tính DCG@K: (2^rel - 1) / log2(i+1)."""
    dcg = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def idcg_at_k(rels: List[int], k: int) -> float:
    """Tính IDCG@K = DCG của dãy relevance sắp xếp giảm dần."""
    sorted_rels = sorted(rels, reverse=True)
    return dcg_at_k(sorted_rels, k)


def ndcg_at_k(
    retrieved: List[List[str]], ground_truth: List[List[str]], k: int
) -> float:
    """
    Tính NDCG@K trung bình cho nhiều truy vấn, theo quy tắc gán nhãn pháp luật 0-1-2.

    Args:
        retrieved (List[List[str]]): Danh sách các danh sách kết quả được truy xuất cho mỗi truy vấn.
        ground_truth (List[List[str]]): Danh sách các danh sách câu trả lời đúng cho mỗi truy vấn.
        k (int): Số lượng kết quả hàng đầu để xem xét.

    Returns:
        float: Giá trị NDCG@K trung bình trên tất cả truy vấn.
    """
    assert len(retrieved) == len(
        ground_truth
    ), "Số lượng truy vấn và câu trả lời đúng phải bằng nhau."

    total_ndcg = 0.0
    total_queries = len(retrieved)

    for preds, gts in zip(retrieved, ground_truth):
        rels = build_relevance_list(preds, gts, k)
        dcg = dcg_at_k(rels, k)
        idcg = idcg_at_k(rels, k)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        total_ndcg += ndcg

    return total_ndcg / total_queries if total_queries > 0 else 0.0


def precision_at_k(
    retrieved: List[List[str]], ground_truth: List[List[str]], k: int
) -> float:
    """
    Tính Precision@K cho một tập hợp truy vấn.

    Args:
        retrieved (List[List[str]]): Danh sách các danh sách kết quả được truy xuất cho mỗi truy vấn.
        ground_truth (List[List[str]]): Danh sách các tập hợp câu trả lời đúng cho mỗi truy vấn.
        k (int): Số lượng kết quả hàng đầu để xem xét.

    Returns:
        float: Precision@K trung bình trên tất cả các truy vấn.
    """
    assert len(retrieved) == len(
        ground_truth
    ), "Số lượng truy vấn và câu trả lời đúng phải bằng nhau."

    total_precision = 0.0
    total_queries = len(retrieved)

    for recs, true_answers in zip(retrieved, ground_truth):
        top_k_recs = recs[:k]
        correct_count = sum(1 for ans in top_k_recs if ans in true_answers)
        precision = correct_count / k
        total_precision += precision

    return total_precision / total_queries if total_queries > 0 else 0.0


def recall_at_k(
    retrieved: List[List[str]], ground_truth: List[List[str]], k: int
) -> float:
    """
    Tính Recall@K cho một tập hợp truy vấn.

    Args:
        retrieved (List[List[str]]): Danh sách các danh sách kết quả được truy xuất cho mỗi truy vấn.
        ground_truth (List[List[str]]): Danh sách các tập hợp câu trả lời đúng cho mỗi truy vấn.
        k (int): Số lượng kết quả hàng đầu để xem xét.

    Returns:
        float: Recall@K trung bình trên tất cả các truy vấn.
    """
    assert len(retrieved) == len(
        ground_truth
    ), "Số lượng truy vấn và câu trả lời đúng phải bằng nhau."

    total_recall = 0.0
    total_queries = len(retrieved)

    for recs, true_answers in zip(retrieved, ground_truth):
        top_k_recs = recs[:k]
        correct_count = sum(1 for ans in top_k_recs if ans in true_answers)
        recall = correct_count / len(true_answers) if len(true_answers) > 0 else 0.0
        total_recall += recall

    return total_recall / total_queries if total_queries > 0 else 0.0
