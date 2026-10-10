"""
Configuration file for Face Recognition App
Adjust these parameters based on your needs and hardware
"""

# Path settings
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
OUTPUT_DIR = "output"
ENCODINGS_DIR = "encodings"
ENCODINGS_FILE = "encodings/encodings.pkl"
MODEL_DIR = "models"

# Face recognition settings
# Tolerance: lower = stricter matching, higher = more permissive
# Range: 0.4-0.7, default 0.6
TOLERANCE = 0.4
FACE_MATCH_THRESHOLD = 0.45
MIN_RECOGNITION_FACE_WIDTH = 60
MIN_RECOGNITION_FACE_HEIGHT = 60
FACE_RECOGNITION_DIAGNOSTICS = False

# Model type for face detection
# "hog" = faster (10-30 FPS), suitable for CPU
# "cnn" = slower (1-5 FPS) but more accurate, requires GPU
MODEL = "hog"

# Webcam settings
PROCESS_EVERY_N_FRAMES = 5  # Run expensive face inference every fifth frame
FRAME_WIDTH = 320  # Reduced resolution for real-time processing
FRAME_HEIGHT = 240
CAMERA_FPS = 30
DETECTOR_SIZE = 320
DETECTION_CONFIDENCE = 0.6
TRACKER_REFRESH_FRAMES = 5

# Optional age, gender, and emotion inference
AGE_DETECTION_ENABLED = True
GENDER_DETECTION_ENABLED = True
EMOTION_DETECTION_ENABLED = True
SHOW_AGE = True
SHOW_GENDER = True
SHOW_EMOTION = True
AGE_MODEL_PATH = "models/age-gender-recognition-retail-0013.xml"
GENDER_MODEL_PATH = AGE_MODEL_PATH
EMOTION_MODEL_PATH = "models/emotion_model.onnx"
AGE_GENDER_EMOTION_EVERY_N_FRAMES = 5

# Display settings
SHOW_FACE_LABELS = True
LABEL_FONT_SIZE = 0.6
LABEL_THICKNESS = 2
BOX_COLOR = (0, 255, 0)  # Green (BGR format)
TEXT_COLOR = (0, 255, 0)  # Green
DEBUG_MODE = False

# Emotion Challenge game settings
EMOTION_GAME_ROUNDS = 5
EMOTION_GAME_ROUND_TIME = 5.0
EMOTION_GAME_THRESHOLD = 0.70
EMOTION_GAME_REQUIRED_STREAK = 3
EMOTION_GAME_SUCCESS_DELAY = 1.2
EMOTION_GAME_FAILURE_DELAY = 1.2
GAME_OVER_RESULT_TIME = 3.0
AI_INTRO_TIME = 2.0
AI_COUNTDOWN_TIME = 5.0
AI_ANALYZING_TIME = 1.0
AI_STABILIZATION_SAMPLES = 10
