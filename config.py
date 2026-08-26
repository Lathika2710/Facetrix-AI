import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, 'database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'database.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
for directory in [DATABASE_DIR, UPLOAD_FOLDER, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Face Recognition Settings
# InsightFace default model pack: 'buffalo_s' (fast CPU friendly) or 'buffalo_l'
MODEL_NAME = 'buffalo_s'
DET_SIZE = (640, 640)  # RetinaFace detection input size

# Cosine similarity threshold for ArcFace vectors
# ArcFace 512D normalized embeddings generally use 0.40 - 0.50 threshold for cosine similarity
SIMILARITY_THRESHOLD = 0.45

# Minimum face bounding box size (pixels) to filter out tiny background faces
MIN_FACE_SIZE = 40

# Web Security & Upload Settings
SECRET_KEY = os.urandom(24).hex()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max file upload
