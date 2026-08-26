import os
import cv2
import numpy as np
import logging
from config import SIMILARITY_THRESHOLD
from database import init_db, add_person, add_face_embedding, get_all_embeddings
from services.face_detector import detect_faces
from services.face_embedding import extract_face_embedding
from services.face_matcher import match_face_embedding
from create_synthetic_testset import generate_test_dataset, TESTSET_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('Evaluation')

def run_evaluation():
    """
    Run quantitative evaluation of the Face Recognition pipeline.
    Calculates Accuracy, Precision, Recall, F1-Score, FAR, FRR,
    and aspect ratio recognition breakdown.
    """
    init_db()
    
    # Generate synthetic test images if test directory is empty
    if not os.path.exists(TESTSET_DIR) or not os.listdir(TESTSET_DIR):
        generate_test_dataset()
        
    print("\n" + "="*70)
    print("      AI FACE RECOGNITION SYSTEM - PERFORMANCE EVALUATION REPORT      ")
    print("="*70)
    
    person_dirs = [d for d in os.listdir(TESTSET_DIR) if os.path.isdir(os.path.join(TESTSET_DIR, d))]
    
    if len(person_dirs) < 2:
        logger.error("Not enough person directories in test set for evaluation.")
        return
        
    # Split dataset into Registration (Gallery) vs Query (Probe)
    registered_persons = person_dirs[:2]  # e.g., Person_A, Person_B
    unknown_persons = person_dirs[2:]     # e.g., Person_C
    
    print(f"[*] Registered Identity Classes: {registered_persons}")
    print(f"[*] Unknown / Unregistered Classes: {unknown_persons}")
    
    # 1. Registration Phase
    db_embeddings_cache = []
    
    for p_name in registered_persons:
        p_dir = os.path.join(TESTSET_DIR, p_name)
        images = [f for f in os.listdir(p_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Use first 3 images of different aspect ratios for registration
        reg_images = images[:3]
        person_id = add_person(p_name)
        
        for img_name in reg_images:
            img_path = os.path.join(p_dir, img_name)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            
            faces = detect_faces(img_bgr)
            if faces:
                emb = extract_face_embedding(img_bgr, face_info=faces[0])
                add_face_embedding(person_id, emb)
                
    stored_db_records = get_all_embeddings()
    print(f"[*] Database populated with {len(stored_db_records)} embeddings across {len(registered_persons)} persons.\n")
    
    # 2. Query Phase across all test images
    tp = 0  # True Positives: Registered person correctly matched as registered identity
    fp = 0  # False Positives: Unknown person falsely matched OR wrong identity matched
    tn = 0  # True Negatives: Unknown person correctly identified as Unknown
    fn = 0  # False Negatives: Registered person falsely identified as Unknown
    
    ar_stats = {}  # Aspect ratio performance tracking
    
    for p_name in person_dirs:
        p_dir = os.path.join(TESTSET_DIR, p_name)
        images = [f for f in os.listdir(p_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        is_registered = p_name in registered_persons
        
        for img_name in images:
            img_path = os.path.join(p_dir, img_name)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            
            h, w = img_bgr.shape[:2]
            ratio = w / h
            
            # Determine Aspect Ratio category name
            if abs(ratio - 16/9) < 0.1:
                ar_cat = "16:9 Landscape"
            elif abs(ratio - 4/3) < 0.1:
                ar_cat = "4:3 Standard"
            elif abs(ratio - 1.0) < 0.1:
                ar_cat = "1:1 Square"
            elif ratio < 0.8:
                ar_cat = "Portrait (3:4/9:16)"
            else:
                ar_cat = f"Custom ({ratio:.2f}:1)"
                
            if ar_cat not in ar_stats:
                ar_stats[ar_cat] = {'correct': 0, 'total': 0}
                
            faces = detect_faces(img_bgr)
            if not faces:
                print(f"[!] No face detected in query image: {img_name}")
                continue
                
            query_emb = extract_face_embedding(img_bgr, face_info=faces[0])
            match_res = match_face_embedding(query_emb, stored_db_records, threshold=SIMILARITY_THRESHOLD)
            
            predicted_name = match_res['name']
            
            ar_stats[ar_cat]['total'] += 1
            
            if is_registered:
                if predicted_name == p_name:
                    tp += 1
                    ar_stats[ar_cat]['correct'] += 1
                elif predicted_name == 'Unknown':
                    fn += 1
                else:
                    # Wrong identity matched
                    fp += 1
            else:
                # Target is Unknown
                if predicted_name == 'Unknown':
                    tn += 1
                    ar_stats[ar_cat]['correct'] += 1
                else:
                    fp += 1

    # 3. Calculate Global Metrics
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Acceptance Rate
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Rejection Rate
    
    # Print Confusion Matrix & Overall Metrics
    print("----------------------------------------------------------------------")
    print("                      CONFUSION MATRIX & METRICS                      ")
    print("----------------------------------------------------------------------")
    print(f"  True Positives  (TP) : {tp:<5} | False Positives (FP) : {fp}")
    print(f"  True Negatives  (TN) : {tn:<5} | False Negatives (FN) : {fn}")
    print("----------------------------------------------------------------------")
    print(f"  Overall Accuracy     : {accuracy * 100:.2f}%")
    print(f"  Precision            : {precision * 100:.2f}%")
    print(f"  Recall (Sensitivity) : {recall * 100:.2f}%")
    print(f"  F1-Score             : {f1_score * 100:.2f}%")
    print(f"  False Acceptance Rate: {far * 100:.2f}% (FAR)")
    print(f"  False Rejection Rate : {frr * 100:.2f}% (FRR)")
    print("----------------------------------------------------------------------\n")

    # 4. Aspect Ratio Breakdown Table
    print("----------------------------------------------------------------------")
    print("            RECOGNITION ACCURACY BY IMAGE ASPECT RATIO                ")
    print("----------------------------------------------------------------------")
    print(f"  {'Aspect Ratio Category':<25} | {'Samples':<8} | {'Accuracy':<10}")
    print("----------------------------------------------------------------------")
    for ar_cat, data in ar_stats.items():
        cat_acc = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0.0
        print(f"  {ar_cat:<25} | {data['total']:<8} | {cat_acc:.2f}%")
    print("----------------------------------------------------------------------\n")

if __name__ == '__main__':
    run_evaluation()
