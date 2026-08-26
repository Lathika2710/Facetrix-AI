import cv2
import numpy as np
import logging
from config import MODEL_NAME, DET_SIZE, MIN_FACE_SIZE

logger = logging.getLogger(__name__)

# Global model instance
_INSIGHTFACE_APP = None

def get_face_analyzer():
    """Lazy initialization of InsightFace FaceAnalysis engine."""
    global _INSIGHTFACE_APP
    if _INSIGHTFACE_APP is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            logger.info("Initializing InsightFace model: %s...", MODEL_NAME)
            _INSIGHTFACE_APP = FaceAnalysis(
                name=MODEL_NAME,
                providers=['CPUExecutionProvider']
            )
            _INSIGHTFACE_APP.prepare(ctx_id=0, det_size=DET_SIZE)
            logger.info("InsightFace model loaded successfully.")
        except Exception as e:
            logger.error("Error loading InsightFace model: %s", str(e))
            raise RuntimeError(f"Failed to load face recognition engine: {e}")
            
    return _INSIGHTFACE_APP

def detect_faces(image_bgr):
    """
    Detect all faces in an OpenCV BGR image regardless of resolution or aspect ratio.
    
    Returns:
        list of dicts containing:
        - 'bbox': [x1, y1, x2, y2] (integers)
        - 'kps': 5 facial landmark points [[x, y], ...]
        - 'landmarks_106': 106-point 2D landmark array if available
        - 'det_score': float detection confidence score
        - 'embedding': 512D normed float32 embedding
        - 'face_obj': raw InsightFace Face object
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid or empty image provided.")
        
    app = get_face_analyzer()
    raw_faces = app.get(image_bgr)
    
    valid_faces = []
    
    for face in raw_faces:
        bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
        w_box = max(0, bbox[2] - bbox[0])
        h_box = max(0, bbox[3] - bbox[1])
        
        # Filter out faces that are too small
        if w_box < MIN_FACE_SIZE or h_box < MIN_FACE_SIZE:
            logger.warning("Ignored face smaller than threshold (%dx%d)", w_box, h_box)
            continue
            
        kps = face.kps.tolist() if face.kps is not None else []
        det_score = float(face.det_score)
        
        # Extract 106-point landmark array if computed
        lmk106 = None
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            lmk106 = face.landmark_2d_106.astype(int).tolist()
        elif hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
            lmk106 = face.landmark_3d_68[:, :2].astype(int).tolist()
            
        # Extract 512D embedding if available
        normed_emb = getattr(face, 'normed_embedding', None)
        if normed_emb is None and getattr(face, 'embedding', None) is not None:
            raw_emb = face.embedding
            norm = np.linalg.norm(raw_emb)
            normed_emb = raw_emb / norm if norm > 0 else raw_emb
            
        valid_faces.append({
            'bbox': bbox,
            'kps': kps,
            'landmarks_106': lmk106,
            'det_score': det_score,
            'embedding': normed_emb,
            'face_obj': face
        })
        
    return valid_faces
