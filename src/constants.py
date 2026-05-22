import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
DATA_DIR = os.path.join(ROOT_DIR, "Food-Calorie-Estimation-2")

CSV_PATH = os.path.join(DATA_DIR, "_classes.csv")
IMAGE_TRAIN_FOLDER = os.path.join(DATA_DIR, "train")
IMAGE_TEST_FOLDER = os.path.join(DATA_DIR, "valid")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
CLASSES_PATH = os.path.join(ARTIFACTS_DIR, "classes.pkl")

# Model config
TRAIN_SPLIT = 12514
IMAGE_SIZE = (128,128)
BATCH_SIZE = 32
EPOCHS=15
step_size=8
LEARNING_RATE = 0.001
NUM_CLASSES = 36

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
