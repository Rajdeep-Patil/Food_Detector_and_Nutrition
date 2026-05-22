"""
main.py — Run this file to train the model from scratch.
After training completes, artifacts/model.pkl and artifacts/classes.pkl
will be saved, and you can start the Flask app with: python app.py
"""
from src.pipeline.train_pipeline import run_training_pipeline

if __name__ == "__main__":
    metrics = run_training_pipeline()
    print("\n===== Final Evaluation Metrics =====")
    for k, v in metrics.items():
        print(f"  {k.capitalize():12s}: {v}")