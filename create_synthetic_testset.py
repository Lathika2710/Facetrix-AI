import os
import cv2
import numpy as np
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestsetGenerator")

TESTSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_dataset')
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_faces')

# Standard OpenCV & InsightFace test benchmark face images
FACE_URLS = {
    'Person_A': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
    'Person_B': 'https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/data/images/t1.jpg',
    'Person_C': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/smarties.png'
}

def place_face_in_canvas(face_img, target_width, target_height):
    """
    Embed a face crop inside an outer canvas frame of arbitrary aspect ratio and resolution.
    Applies letterboxing, scaling, and padding to simulate varied camera frames.
    """
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[:, :] = (35, 39, 47)  # Studio background gradient
    
    fh, fw = face_img.shape[:2]
    
    # Scale face to fit comfortably within target canvas (55% of canvas area)
    scale = min((target_width * 0.55) / fw, (target_height * 0.55) / fh)
    new_w = max(50, int(fw * scale))
    new_h = max(50, int(fh * scale))
    
    resized_face = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center face in canvas frame
    start_x = (target_width - new_w) // 2
    start_y = (target_height - new_h) // 2
    
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_face
    return canvas

def generate_test_dataset():
    """
    Generate evaluation test dataset across multiple aspect ratio configurations:
    - 16:9 Landscape (1920x1080, 1280x720)
    - 4:3 Standard (1024x768, 800x600)
    - 1:1 Square (1000x1000, 500x500)
    - Portrait 3:4 / 9:16 (600x800, 540x960)
    - Ultrawide 21:9 (1680x720)
    """
    os.makedirs(TESTSET_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    aspect_ratios = [
        ('16:9', 1920, 1080),
        ('16:9', 1280, 720),
        ('4:3', 1024, 768),
        ('4:3', 800, 600),
        ('1:1', 1000, 1000),
        ('1:1', 500, 500),
        ('Portrait', 600, 800),
        ('Portrait', 540, 960),
        ('Ultrawide', 1680, 720)
    ]
    
    print("[*] Fetching base face benchmark images from GitHub raw data...")
    base_faces = {}
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for p_name, url in FACE_URLS.items():
        cache_path = os.path.join(CACHE_DIR, f"{p_name}.jpg")
        if not os.path.exists(cache_path):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as resp:
                    arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        cv2.imwrite(cache_path, img)
                        logger.info(f"Downloaded reference face for {p_name}")
            except Exception as e:
                logger.warning(f"Could not download {url}: {e}")
                
        if os.path.exists(cache_path):
            base_faces[p_name] = cv2.imread(cache_path)

    print(f"[*] Generating test dataset in '{TESTSET_DIR}'...")
    manifest = []
    
    for p_name, face_img in base_faces.items():
        p_dir = os.path.join(TESTSET_DIR, p_name)
        os.makedirs(p_dir, exist_ok=True)
        
        for idx, (ar_name, width, height) in enumerate(aspect_ratios):
            canvas_img = place_face_in_canvas(face_img, width, height)
            fname = f"{p_name}_ar_{ar_name.replace(':', '_')}_{width}x{height}_{idx}.jpg"
            fpath = os.path.join(p_dir, fname)
            cv2.imwrite(fpath, canvas_img)
            manifest.append({
                'person': p_name,
                'path': fpath,
                'aspect_ratio_name': ar_name,
                'width': width,
                'height': height
            })
            
    print(f"[*] Generated {len(manifest)} evaluation images across {len(aspect_ratios)} aspect ratio configurations.")
    return manifest

if __name__ == '__main__':
    generate_test_dataset()
