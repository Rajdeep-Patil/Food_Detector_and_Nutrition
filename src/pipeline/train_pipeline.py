import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import CustomDataset, custom_transform
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.logger import logger
from src.exception import CustomException

# Re-export for backward-compat imports (e.g. from src.pipeline import train_pipeline)
from src.constants import IMAGE_TRAIN_FOLDER   # noqa: F401


def run_training_pipeline():
    """
    Orchestrates the full training workflow:
    Ingest → Transform → Train → Evaluate
    """
    logger.info("========== Training Pipeline Started ==========")
    try:
        # 1. Data Ingestion
        ingestion = DataIngestion()
        X_train, X_test, y_train, y_test, class_names = ingestion.initiate_data_ingestion()

        # 2. Data Transformation (wrap in Dataset objects)
        train_dataset = CustomDataset(X_train, y_train, custom_transform)
        test_dataset  = CustomDataset(X_test,  y_test,  custom_transform)

        # 3. Model Training  → saves model.pkl + classes.pkl
        trainer = ModelTrainer()
        model, device = trainer.initiate_model_training(
            train_dataset, test_dataset, class_names=class_names
        )

        # 4. Model Evaluation
        evaluator = ModelEvaluation()
        metrics = evaluator.initiate_model_evaluation(model, test_dataset, device)

        logger.info("========== Training Pipeline Completed ==========")
        return metrics

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
