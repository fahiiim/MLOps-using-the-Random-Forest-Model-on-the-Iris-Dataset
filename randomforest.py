from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow setup
MLFLOW_TRACKING_URI = "mlruns"  # Local directory for MLflow tracking
EXPERIMENT_NAME = "iris_random_forest"

def load_and_preprocess_data():
    """Load and preprocess the Iris dataset."""
    try:
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        
        # Validate data
        assert not X.isnull().any().any(), "Missing values found in features"
        assert not y.isnull().any(), "Missing values found in target"
        assert X.shape[0] == y.shape[0], "Feature and target dimensions mismatch"
        
        return X, y, data.target_names
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train RandomForest model and evaluate performance."""
    try:
        # Model parameters
        params = {
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": None,
            "min_samples_split": 2
        }
        
        # Initialize and train model
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        return clf, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }, y_pred, params
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def main():
    try:
        # Set up MLflow tracking
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        # Load and preprocess data
        X, y, target_names = load_and_preprocess_data()
        logger.info("Data loaded successfully")
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train and evaluate model
        clf, metrics, y_pred, params = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        
        # Print results
        logger.info("\nModel Performance:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")
        
        # Log results with MLflow
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model with signature
            signature = infer_signature(X_train, clf.predict(X_train))
            mlflow.sklearn.log_model(clf, "random_forest_model", signature=signature)
            
            # Log feature importance plot
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': clf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nFeature Importance:")
            logger.info(feature_importance)
            
            logger.info("Model and metrics logged in MLflow")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()