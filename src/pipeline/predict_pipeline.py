import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_PATH, CLASSES_PATH
from src.utils.helper import load_model, load_object
from src.logger import logger


class PredictPipeline:

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = None
        self.class_names = None
        self._load_artifacts()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_artifacts(self):
        logger.info("Loading model...")
        self.model = load_model(MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Loading class names...")
        self.class_names = load_object(CLASSES_PATH)

    def predict(self, image_path, threshold=0.5):

        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = torch.sigmoid(self.model(img))

        outputs = outputs.squeeze().cpu().numpy()

        predicted = (outputs > threshold)

        labels = [
            self.class_names[i]
            for i, v in enumerate(predicted)
            if v == True
        ]

        top5_idx = outputs.argsort()[::-1][:5]
        top5 = [(self.class_names[i], float(outputs[i])) for i in top5_idx]

        return labels, top5