from pyvi import ViUtils
from underthesea import word_tokenize
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

def preprocess_query(raw_query: str):
    """Normalize a user query to reduce noise before encoding."""

    if not isinstance(raw_query, str):
        logger.error("Query must be a string, received %s", type(raw_query))
        raise TypeError("Query must be a string")

    trimmed = raw_query.strip()
    if not trimmed:
        logger.warning("Query is empty after trimming whitespace")
        raise ValueError("Query is empty after trimming whitespace")

    normalized = unicodedata.normalize("NFC", trimmed)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"([!?.,])\1+", r"\1", normalized)
    
text = "Bộ Kế hoạch, và Đầu tư chịu trách nhiệm thẩm định các nội dung sau đây."
print(word_tokenize(preprocess_query(text)))