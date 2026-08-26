import numpy as np
import logging
from config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

def compute_cosine_similarity(vec1, vec2):
    """
    Compute Cosine Similarity between two 512-D vectors:
    sim(u, v) = (u . v) / (||u|| * ||v||)
    """
    u = np.asarray(vec1, dtype=np.float32)
    v = np.asarray(vec2, dtype=np.float32)
    
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0
        
    dot_product = np.dot(u, v)
    similarity = dot_product / (norm_u * norm_v)
    # Clip float precision errors to [-1.0, 1.0]
    return float(np.clip(similarity, -1.0, 1.0))

def match_face_embedding(query_embedding, stored_embeddings_records, threshold=SIMILARITY_THRESHOLD, strategy='max'):
    """
    Compare a query embedding against all stored database embeddings.
    
    Multi-Sample Aggregation Strategy:
    - 'max': Uses the maximum cosine similarity among all samples for each registered person.
    - 'mean': Uses the average cosine similarity across all samples for each registered person.
    
    Returns:
        dict: {
            'matched': bool,
            'person_id': int or None,
            'name': str ('Unknown' if no match),
            'similarity': float (0.0 to 1.0),
            'threshold': float,
            'status': 'MATCH' or 'UNKNOWN',
            'person_similarities': list of dicts with scores per person
        }
    """
    if not stored_embeddings_records:
        return {
            'matched': False,
            'person_id': None,
            'name': 'Unknown',
            'similarity': 0.0,
            'threshold': threshold,
            'status': 'UNKNOWN',
            'person_similarities': [],
            'message': 'No registered persons found in database.'
        }
        
    # Group stored embeddings by person_id & name
    person_scores = {}
    
    for rec in stored_embeddings_records:
        p_id = rec['person_id']
        p_name = rec['name']
        db_emb = rec['embedding']
        
        sim = compute_cosine_similarity(query_embedding, db_emb)
        
        if p_id not in person_scores:
            person_scores[p_id] = {
                'person_id': p_id,
                'name': p_name,
                'similarities': []
            }
        person_scores[p_id]['similarities'].append(sim)
        
    # Aggregate scores per person
    aggregated_results = []
    for p_id, data in person_scores.items():
        sims = data['similarities']
        if strategy == 'max':
            score = max(sims)
        elif strategy == 'mean':
            score = sum(sims) / len(sims)
        else:
            score = max(sims)
            
        aggregated_results.append({
            'person_id': p_id,
            'name': data['name'],
            'similarity': float(score),
            'samples': len(sims)
        })
        
    # Sort by similarity descending
    aggregated_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    top_match = aggregated_results[0]
    top_score = top_match['similarity']
    
    if top_score >= threshold:
        return {
            'matched': True,
            'person_id': top_match['person_id'],
            'name': top_match['name'],
            'similarity': top_score,
            'threshold': threshold,
            'status': 'MATCH',
            'person_similarities': aggregated_results
        }
    else:
        return {
            'matched': False,
            'person_id': None,
            'name': 'Unknown',
            'similarity': top_score,
            'threshold': threshold,
            'status': 'UNKNOWN',
            'person_similarities': aggregated_results
        }
