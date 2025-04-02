import os
import yaml
import pickle
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_training")

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
    return params

def create_directories():
    """Create necessary directories if they don't exist"""
    Path("models").mkdir(parents=True, exist_ok=True)
    logger.info("Created models directory")

def load_preprocessed_data(data_path):
    """Load preprocessed data from CSV file"""
    logger.info(f"Loading featured data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")
    return df

def load_preprocessor(preprocessor_path):
    """Load the saved preprocessor"""
    logger.info(f"Loading preprocessor from {preprocessor_path}")
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    return preprocessor

def prepare_train_validation_data(df, preprocessor, test_size=0.2, random_state=42):
    """Prepare training and validation data"""
    logger.info("Preparing training and validation data")
    
    # Split features and target
    X = df.drop(['Price', 'id'], axis=1, errors='ignore')
    y = df['Price']
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Transform data using preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    logger.info(f"Train features shape: {X_train_processed.shape}")
    logger.info(f"Validation features shape: {X_val_processed.shape}")
    
    return X_train_processed, X_val_processed, y_train, y_val, X, y

def train_model(X_train, y_train, model_params):
    """Train the model with the given parameters"""
    logger.info("Training Random Forest model")
    
    model = RandomForestRegressor(
        n_estimators=model_params['n_estimators'],
        max_depth=model_params['max_depth'],
        min_samples_split=model_params['min_samples_split'],
        min_samples_leaf=model_params['min_samples_leaf'],
        max_features=model_params['max_features'],
        bootstrap=model_params['bootstrap'],
        random_state=model_params.get('random_state', 42)
    )
    
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    return model

def evaluate_model(model, X, y):
    """Evaluate model performance"""
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    
    return rmse, mae, r2, preds

def perform_cross_validation(model, X, y, cv=5):
    """Perform cross-validation on the model"""
    logger.info(f"Performing {cv}-fold cross-validation")
    
    cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, 
        X, 
        y, 
        cv=cv_obj,
        scoring='neg_root_mean_squared_error'
    )
    
    cv_rmse = -np.mean(cv_scores)
    cv_rmse_std = np.std(cv_scores)
    
    logger.info(f"Cross-validation RMSE: {cv_rmse:.4f} (±{cv_rmse_std:.4f})")
    
    return cv_rmse, cv_rmse_std

def save_model(model, output_path):
    """Save the trained model to disk"""
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {output_path}")

if __name__ == "__main__":
    logger.info("Starting model training process")
    
    # Load parameters
    params = load_params()
    model_params = params['model']['hyperparameters']
    evaluation_params = params['evaluation']
    mlflow_params = params['mlflow']
    
    # Create directories
    create_directories()
    
    # Load featured data
    train_df = load_preprocessed_data(os.path.join("data", "featured", "train_featured.csv"))
    
    # Load preprocessor
    preprocessor = load_preprocessor(os.path.join("artifacts", "preprocessor.pkl"))
    
    # Prepare data
    X_train_processed, X_val_processed, y_train, y_val, X_full, y_full = prepare_train_validation_data(
        train_df, 
        preprocessor,
        test_size=params['preprocessing']['test_size'],
        random_state=params['preprocessing']['random_state']
    )
    
    # Set up MLflow tracking
    mlflow.set_experiment(mlflow_params['experiment_name'])
    
    with mlflow.start_run(run_name=mlflow_params['run_name']):
        # Log model parameters
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        
        # Train model
        model = train_model(X_train_processed, y_train, model_params)
        
        # Evaluate on validation set
        # Evaluate on validation set
        val_rmse, val_mae, val_r2, _ = evaluate_model(model, X_val_processed, y_val)
        logger.info(f"Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        
        # Cross-validation
        cv_rmse, cv_rmse_std = perform_cross_validation(
            model, 
            X_train_processed, 
            y_train,
            cv=evaluation_params['cv_folds']
        )
        
        # Log metrics
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("cv_rmse_std", cv_rmse_std)
        
        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            mlflow.log_param("top_feature_index", int(np.argmax(importances)))
            mlflow.log_param("top_feature_importance", float(np.max(importances)))
        
        # Log the model
        mlflow.sklearn.log_model(
            model, 
            artifact_path=mlflow_params['artifact_path'],
            registered_model_name=mlflow_params['model_name'] if mlflow_params['register_model'] else None
        )
        
        # Save the run ID
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")
    
    # Save model locally
    save_model(model, os.path.join("models", "trained_model.pkl"))
    
    logger.info("Model training and logging completed successfully")
