import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IS_VERCEL = os.environ.get('VERCEL') == '1' or 'VERCEL' in os.environ

if IS_VERCEL:
    DATABASE_DIR = '/tmp'
    DATABASE_PATH = '/tmp/database.db'
    UPLOAD_FOLDER = '/tmp/uploads'
    MODELS_DIR = '/tmp/models'
    os.environ['HOME'] = '/tmp'
    os.environ['USERPROFILE'] = '/tmp'
else:
    DATABASE_DIR = os.path.join(BASE_DIR, 'database')
    DATABASE_PATH = os.path.join(DATABASE_DIR, 'database.db')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
for directory in [DATABASE_DIR, UPLOAD_FOLDER, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Face Recognition Settings
MODEL_NAME = 'buffalo_s'
DET_SIZE = (640, 640)  # RetinaFace detection input size

# Cosine similarity threshold for ArcFace vectors
SIMILARITY_THRESHOLD = 0.45

# Minimum face bounding box size (pixels) to filter out tiny background faces
MIN_FACE_SIZE = 40

# Web Security & Upload Settings
SECRET_KEY = os.urandom(24).hex()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max file upload
