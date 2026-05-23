import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import sys

from src.constants import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_PATH,
    CLASSES_PATH
)

from src.utils.helper import load_model, load_object
from src.logger import logger
from src.exception import CustomException

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


class PredictPipeline:

    def __init__(self):

        self.model = None
        self.class_names = None

        # FORCE CPU FOR RENDER
        self.device = torch.device("cpu")

        # LOAD ONCE
        self._load_artifacts()

    def _load_artifacts(self):

        if self.model is None:
            logger.info("Loading model...")

            self.model = load_model(MODEL_PATH)

            self.model.to(self.device)

            self.model.eval()

        if self.class_names is None:

            logger.info("Loading class names...")

            self.class_names = load_object(CLASSES_PATH)

    def predict(self, image_path: str, threshold: float = 0.5):

        try:

            logger.info(f"Predicting: {image_path}")

            img = Image.open(image_path).convert("RGB")

            img = img.resize(IMAGE_SIZE)

            img = np.array(img, dtype=np.float32) / 255.0

            img = torch.tensor(img)

            img = img.permute(2, 0, 1)

            img = img.unsqueeze(0)

            img = _transform(img)

            img = img.to(self.device).float()

            with torch.no_grad():

                outputs = torch.sigmoid(
                    self.model(img)
                )

                predicted = (
                    outputs > threshold
                ).float().squeeze()

            predicted_labels = [

                self.class_names[i]

                for i, v in enumerate(
                    predicted.cpu().numpy()
                )

                if v == 1.0
            ]

            scores = outputs.squeeze().cpu().numpy()

            top5_idx = scores.argsort()[::-1][:5]

            top5 = [

                (
                    self.class_names[i],
                    float(scores[i])
                )

                for i in top5_idx
            ]

            return predicted_labels, top5

        except Exception as e:

            raise CustomException(e, sys)