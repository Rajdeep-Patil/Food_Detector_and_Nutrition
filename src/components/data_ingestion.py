import pandas as pd
import os
import sys
from src.constants import CSV_PATH, IMAGE_TRAIN_FOLDER, TRAIN_SPLIT
from src.logger import logger
from src.exception import CustomException


class DataIngestion:
    """
    Reads the _classes.csv, builds absolute image paths,
    and splits data into train / test sets.
    """

    def __init__(self, csv_path: str = CSV_PATH):
        self.csv_path = csv_path

    def initiate_data_ingestion(self):
        logger.info("Starting Data Ingestion...")
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Dataset loaded: {df.shape[0]} rows, columns: {list(df.columns)}")

            filenames = df["filename"].tolist()
            X_data = [os.path.join(IMAGE_TRAIN_FOLDER, fname) for fname in filenames]

            X_train = X_data[:TRAIN_SPLIT]
            X_test  = X_data[TRAIN_SPLIT + 1:]   # +1 to skip index 4172 (off-by-one safety)

            y_data  = df.iloc[:, 1:].values       # all columns except 'filename'
            y_train = y_data[:TRAIN_SPLIT]
            y_test  = y_data[TRAIN_SPLIT + 1:]

            # Save class names for later use (prediction labels)
            self.class_names = list(df.columns[1:])

            logger.info(
                f"Data Ingestion complete | "
                f"Train: {len(X_train)} | Test: {len(X_test)} | Classes: {len(self.class_names)}"
            )
            return X_train, X_test, y_train, y_test, self.class_names

        except Exception as e:
            raise CustomException(e, sys)
