import os
import pickle
import torch
from src.logger import logger
from src.exception import CustomException
from sklearn.metrics import accuracy_score
import sys

def save_object(file_path: str, obj):
    """Save any Python object using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    """Load a pickle object from disk."""
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from: {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)

def save_model(file_path: str, model):
    """Save a PyTorch model's state dict wrapped in pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_model(file_path: str):
    """Load a pickled PyTorch model."""
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from: {file_path}")
        return model
    except Exception as e:
        raise CustomException(e, sys)

# ── Training function ─────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss    = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs    = model(X)
            loss       = criterion(outputs, y)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc
