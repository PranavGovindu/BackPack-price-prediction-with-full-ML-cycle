import os
import yaml
import json
import pickle
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_evaluation")

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
    return params

def create_directories():
    """Create necessary directories if they don't exist"""
    Path("metrics").mkdir(parents=True, exist_ok=True)
    logger.info("Created metrics directory")

def load_model(model_path):
    """Load the trained model"""
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_preprocessor(preprocessor_path):
    """Load the saved preprocessor"""
    logger.info(f"Loading preprocessor from {preprocessor_path}")
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    return preprocessor

def load_data(train_path, test_path):
    """Load featured data"""
    logger.info(f"Loading data from {train_path} and {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def prepare_data(train_df, test_df, preprocessor):
    """Prepare data for evaluation"""
    logger.info("Preparing data for evaluation")
    
    # Extract features and target from training data
    X_train = train_df.drop(['Price', 'id'], axis=1, errors='ignore')
    y_train = train_df['Price']
    
    # Extract features from test data
    X_test = test_df.drop(['id'], axis=1, errors='ignore')
    
    # Transform data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, y_train, X_test_processed

def evaluate_model(model, X, y, dataset_name="train"):
    """Evaluate model performance on given dataset"""
    logger.info(f"Evaluating model on {dataset_name} dataset")
    
    predictions = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    logger.info(f"{dataset_name} RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return {
        f"{dataset_name}_rmse": rmse,
        f"{dataset_name}_mae": mae,
        f"{dataset_name}_r2": r2
    }

def generate_feature_importance(model, preprocessor, X_train):
    """Generate feature importance analysis"""
    logger.info("Generating feature importance analysis")
    
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return {}
    
    # Get feature names
    cat_cols = preprocessor.transformers_[1][1].get_feature_names_out().tolist()
    num_cols = X_train.columns[preprocessor.transformers_[0][2]].tolist()
    
    # Map importances to features
    feature_importances = model.feature_importances_
    
    importance_dict = {}
    total_features = len(num_cols) + len(cat_cols)
    
    if len(feature_importances) == total_features:
        # Simple case - direct mapping
        all_features = num_cols + cat_cols
        for feature, importance in zip(all_features, feature_importances):
            importance_dict[feature] = float(importance)
    else:
        # More complex case - need to handle transformed features
        logger.warning("Feature count mismatch. Using simplified feature importance.")
        importance_dict = {f"feature_{i}": float(importance) for i, importance in enumerate(feature_importances)}
    
    # Sort by importance
    sorted_importances = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Get top 10 features
    top_features = {k: sorted_importances[k] for k in list(sorted_importances)[:10]}
    
    return top_features

if __name__ == "__main__":
    logger.info("Starting model evaluation process")
    
    # Load parameters
    params = load_params()
    mlflow_params = params['mlflow']
    
    # Create directories
    create_directories()
    
    # Load model and preprocessor
    model = load_model(os.path.join("models", "trained_model.pkl"))
    preprocessor = load_preprocessor(os.path.join("artifacts", "preprocessor.pkl"))
    
    # Load data
    train_df, test_df = load_data(
        os.path.join("data", "featured", "train_featured.csv"),
        os.path.join("data", "featured", "test_featured.csv")
    )
    
    # Prepare data
    X_train_processed, y_train, X_test_processed = prepare_data(train_df, test_df, preprocessor)
    
    # Start MLflow run for logging evaluation results
    mlflow.set_experiment(mlflow_params['experiment_name'])
    
    with mlflow.start_run(run_name=f"{mlflow_params['run_name']}_evaluation"):
        # Evaluate on training data
        train_metrics = evaluate_model(model, X_train_processed, y_train, "train")
        
        # Log metrics to MLflow
        for metric_name, metric_value in train_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Generate feature importance
        feature_importances = generate_feature_importance(model, preprocessor, train_df.drop(['Price', 'id'], axis=1, errors='ignore'))
        
        # Log feature importances
        for feature, importance in feature_importances.items():
            mlflow.log_metric(f"importance_{feature.replace(' ', '_')}", importance)
        
        # Generate test predictions (if applicable)
        test_predictions = model.predict(X_test_processed)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Price': test_predictions
        })
        
        # Save submission file
        submission_path = os.path.join("metrics", "predictions.csv")
        submission.to_csv(submission_path, index=False)
        mlflow.log_artifact(submission_path)
        logger.info(f"Saved predictions to {submission_path}")
        
        # Save evaluation metrics
        metrics = {
            **train_metrics,
            "top_features": feature_importances
        }
        
        metrics_path = os.path.join("metrics", "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        mlflow.log_artifact(metrics_path)
        logger.info(f"Saved evaluation metrics to {metrics_path}")
    
    logger.info("Model evaluation completed successfully")
