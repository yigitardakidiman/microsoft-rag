import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from database import get_all_chunks
from foundry_client import client, EMBEDDING_MODEL

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_a = np.linalg.norm(v1)
    norm_b = np.linalg.norm(v2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def get_top_chunks(query, top_k=2, score_threshold=0.58):
    """
    Sorguya en benzer top_k adet metin parçasını döndürür.
    score_threshold (varsayılan 0.58) üzerindeki alakalı parçalar filtrelenir.
    Döndürülen her öğe: {'content': str, 'source': str, 'score': float}
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        query_vector = response.data[0].embedding
    except Exception as e:
        print(f"Sorgu embedding hatası: {e}")
        return []

    all_chunks = get_all_chunks()
    if not all_chunks:
        return []
    
    scored_chunks = []
    for chunk_data in all_chunks:
        chunk_vector = chunk_data["embedding"]
        score = cosine_similarity(query_vector, chunk_vector)
        if score >= score_threshold:
            scored_chunks.append({
                "content": chunk_data["content"],
                "source": chunk_data.get("source", "Bilinmiyor"),
                "score": float(score)
            })
        
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_chunks[:top_k]
