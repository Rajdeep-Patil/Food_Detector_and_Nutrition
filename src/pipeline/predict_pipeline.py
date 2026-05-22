import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import sys
from src.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_PATH, CLASSES_PATH
from src.utils.helper import load_model, load_object
from src.logger import logger
from src.exception import CustomException

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class PredictPipeline:
    """
    Loads the saved model.pkl and runs inference on a single image.
    Returns a list of predicted class labels.
    """

    def __init__(self):
        self.model        = None
        self.class_names  = None
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_artifacts(self):
        if self.model is None:
            self.model = load_model(MODEL_PATH)
            self.model.to(self.device)
            self.model.eval()
        if self.class_names is None:
            self.class_names = load_object(CLASSES_PATH)

    def predict(self, image_path: str, threshold: float = 0.5):
        """
        Args:
            image_path: path to a single food image
            threshold:  sigmoid threshold for multi-label decision

        Returns:
            List of predicted food labels (strings)
        """
        logger.info(f"Running prediction on: {image_path}")
        try:
            self._load_artifacts()

            img = Image.open(image_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # 1×C×H×W
            img = _transform(img).to(self.device).float()

            with torch.no_grad():
                outputs    = torch.sigmoid(self.model(img))          # (1, 36)
                predicted  = (outputs > threshold).float().squeeze() # (36,)

            predicted_labels = [
                self.class_names[i]
                for i, v in enumerate(predicted.cpu().numpy())
                if v == 1.0
            ]

            # Confidence scores (top-5)
            scores = outputs.squeeze().cpu().numpy()
            top5_idx = scores.argsort()[::-1][:5]
            top5 = [(self.class_names[i], round(float(scores[i]) * 100, 2)) for i in top5_idx]

            logger.info(f"Predicted labels: {predicted_labels}")
            return predicted_labels, top5

        except Exception as e:
            raise CustomException(e, sys)
