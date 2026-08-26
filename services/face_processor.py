import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Standard 112x112 target landmark coordinates for ArcFace alignment
# (Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner)
REFERENCE_FACIAL_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_and_crop_face(image_bgr, kps_5pt, crop_size=(112, 112)):
    """
    Perform 5-point facial landmark affine transformation to align the face
    horizontally and normalize size into a standard 112x112 chip.
    """
    if kps_5pt is None or len(kps_5pt) != 5:
        logger.warning("5-point landmarks not provided; skipping affine alignment.")
        return cv2.resize(image_bgr, crop_size)
        
    src_pts = np.array(kps_5pt, dtype=np.float32)
    dst_pts = REFERENCE_FACIAL_LANDMARKS_112
    
    tfm, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if tfm is None:
        logger.warning("Affine transformation estimation failed; fallback to simple crop.")
        return cv2.resize(image_bgr, crop_size)
        
    aligned_face_bgr = cv2.warpAffine(
        image_bgr, tfm, crop_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
    )
    return aligned_face_bgr

def draw_visualizations(image_bgr, faces_data):
    """
    Draw clean visual bounding boxes, 5-point facial landmarks, identity badges,
    and aspect ratio metadata directly onto the original image.
    
    faces_data: list of dicts with keys 'bbox', 'kps', 'identity', 'similarity', 'status'
    """
    canvas = image_bgr.copy()
    img_h, img_w = canvas.shape[:2]
    aspect_ratio_str = f"Dimensions: {img_w}x{img_h} (Aspect Ratio {img_w/img_h:.2f}:1)"
    
    # Header banner with aspect ratio info
    cv2.rectangle(canvas, (0, 0), (img_w, 32), (18, 22, 33), -1)
    cv2.putText(
        canvas, aspect_ratio_str, (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA
    )
    
    for face in faces_data:
        bbox = face['bbox']  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        
        identity = face.get('identity', 'Unknown')
        similarity = face.get('similarity', 0.0)
        is_match = face.get('status', '') == 'MATCH'
        
        # Color coding: Green for Match, Blue/Red for Unknown
        box_color = (46, 204, 113) if is_match else (52, 152, 219)
        bg_color = (39, 174, 96) if is_match else (41, 128, 185)
        
        # Draw sleek corner bounding box
        thickness = 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
        
        # Draw corner accents
        corner_len = min(20, (x2 - x1) // 4)
        # Top-Left
        cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), box_color, thickness + 2)
        cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), box_color, thickness + 2)
        # Bottom-Right
        cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), box_color, thickness + 2)
        cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), box_color, thickness + 2)
        
        # Draw 5-point facial landmarks (Solid White Node Circles)
        kps = face.get('kps', [])
        for pt in kps:
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(canvas, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 6, (0, 0, 0), 1, cv2.LINE_AA)
            
        # Identity label badge
        label = f"{identity} ({similarity*100:.1f}%)" if identity != "Unknown" else "Unknown"
        (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        lbl_y1 = max(0, y1 - lbl_h - 12)
        lbl_y2 = y1
        cv2.rectangle(canvas, (x1, lbl_y1), (x1 + lbl_w + 16, lbl_y2), bg_color, -1)
        cv2.putText(
            canvas, label, (x1 + 8, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        
    return canvas
