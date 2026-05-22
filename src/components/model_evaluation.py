import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
from src.constants import BATCH_SIZE
from src.logger import logger
from src.exception import CustomException


class ModelEvaluation:
    """
    Evaluates a trained multi-label model on a test dataset
    and logs / prints standard classification metrics.
    """

    def initiate_model_evaluation(self, model, test_dataset, device):
        logger.info("Starting Model Evaluation...")
        try:
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=0)
            model.eval()
            all_predictions, all_labels = [], []

            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device).float(), y.to(device).float()
                    outputs   = torch.sigmoid(model(X))
                    predicted = (outputs > 0.5).float()
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            accuracy  = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average="micro", zero_division=0)
            recall    = recall_score(all_labels, all_predictions, average="micro", zero_division=0)
            f1        = f1_score(all_labels, all_predictions, average="micro", zero_division=0)

            metrics = {
                "accuracy":  round(float(accuracy),  4),
                "precision": round(float(precision), 4),
                "recall":    round(float(recall),    4),
                "f1_score":  round(float(f1),        4),
            }

            for k, v in metrics.items():
                logger.info(f"{k.capitalize()}: {v}")
                print(f"{k.capitalize()}: {v}")

            logger.info("Model Evaluation Completed!")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)
