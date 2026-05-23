import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.constants import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_PATH,
    CLASSES_PATH,
)
from src.utils.helper import load_model, load_object
from src.logger import logger
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        self.model       = None
        self.class_names = None
        self.device      = torch.device("cpu")

        self._load_artifacts()

        # ✅ IMAGE_SIZE ka tuple ya integer dono handle hoga
        if isinstance(IMAGE_SIZE, (tuple, list)):
            resize_to = (int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1]))
        else:
            resize_to = (int(IMAGE_SIZE), int(IMAGE_SIZE))

        self.transform = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_artifacts(self) -> None:
        try:
            if self.model is None:
                logger.info("Loading model from: %s", MODEL_PATH)
                self.model = load_model(MODEL_PATH)
                self.model.to(self.device)
                self.model = self.model.half()   # float16 — RAM ~50% kam
                self.model.eval()
                logger.info("Model loaded in half precision (float16).")

            if self.class_names is None:
                logger.info("Loading class names from: %s", CLASSES_PATH)
                self.class_names = load_object(CLASSES_PATH)
                logger.info("Loaded %d class names.", len(self.class_names))

        except Exception as e:
            logger.error("Artifact loading failed: %s", e)
            raise CustomException(e, sys)

    def predict(self, image_path: str, threshold: float = 0.5):
        try:
            logger.info("Running inference on: %s", image_path)

            image  = Image.open(image_path).convert("RGB")
            tensor = self.transform(image)
            tensor = tensor.half().unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                scores = torch.sigmoid(logits)

            scores_np = scores.squeeze().cpu().float().numpy()

            predicted_labels = [
                self.class_names[i]
                for i, score in enumerate(scores_np)
                if score >= threshold
            ]

            top5_idx = scores_np.argsort()[::-1][:5]
            top5 = [
                (self.class_names[i], float(scores_np[i]))
                for i in top5_idx
            ]

            logger.info("Labels above threshold: %s", predicted_labels)
            return predicted_labels, top5

        except Exception as e:
            logger.error("Prediction error for %s: %s", image_path, e)
            raise CustomException(e, sys)