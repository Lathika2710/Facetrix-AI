import numpy as np
import logging
from services.face_detector import detect_faces

logger = logging.getLogger(__name__)

def extract_face_embedding(image_bgr, face_info=None):
    """
    Extract 512-D ArcFace feature embedding from an image.
    
    If `face_info` is passed (from `detect_faces`), extracts embedding directly.
    Otherwise runs detection first.
    
    Returns:
        np.ndarray float32 of shape (512,) normalized to unit L2 length.
    """
    if face_info is None:
        faces = detect_faces(image_bgr)
        if not faces:
            raise ValueError("No face detected in the provided image.")
        if len(faces) > 1:
            logger.warning("Multiple faces detected (%d); selecting largest face.", len(faces))
            # Sort by bounding box area (largest area first)
            faces.sort(key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]), reverse=True)
        face_info = faces[0]
        
    embedding = face_info.get('embedding')
    if embedding is None:
        raise ValueError("Failed to extract ArcFace feature embedding for face.")
        
    # Ensure unit L2 normalization
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
        
    return embedding.astype(np.float32)
