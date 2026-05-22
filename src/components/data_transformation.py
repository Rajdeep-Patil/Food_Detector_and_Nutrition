import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import sys
from src.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.logger import logger
from src.exception import CustomException

# ─── Transform pipeline ────────────────────────────────────────────────────────
custom_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class CustomDataset(Dataset):
    """
    PyTorch Dataset that loads images from disk paths,
    resizes them, converts to tensor, and applies normalization.
    """

    def __init__(self, image_paths, labels, transform=custom_transform,
                target_size=IMAGE_SIZE):
        self.image_paths = image_paths
        self.labels      = torch.tensor(labels, dtype=torch.float32)
        self.transform   = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        try:
            img = Image.open(self.image_paths[index]).convert("RGB")
            img = img.resize(self.target_size)
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.tensor(img).permute(2, 0, 1)   # HWC → CHW
            img = self.transform(img)
            return img, self.labels[index]
        except Exception as e:
            raise CustomException(e, sys)
