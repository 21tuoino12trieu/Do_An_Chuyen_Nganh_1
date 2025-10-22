# F1-score là trung bình điều hòa của Precision và Recall.
# Nó cung cấp một cái nhìn tổng quan về hiệu suất của hệ thống, đặc biệt khi
# cần cân bằng giữa việc tìm kiếm đúng và việc tìm kiếm đầy đủ các kết quả đúng.
# Công thức: F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

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