import os
import yaml
import pickle
import logging
import mlflow
import mlflow.sklearn
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_pull")

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
    return params

def create_directories():
    """Create necessary directories if they don't exist"""
    Path("models").mkdir(parents=True, exist_ok=True)
    logger.info("Created models directory")

def pull_model_by_alias(model_name, alias):
    """Pull model from MLflow registry using alias"""
    logger.info(f"Pulling model {model_name} with alias '{alias}'")
    
    try:
        model_uri = f"models:/{model_name}@{alias}"
        logger.info(f"Loading model from URI: {model_uri}")
        
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Successfully loaded model with alias '{alias}'")
        return model
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        return None

def save_model(model, output_path):
    """Save the pulled model to disk"""
    logger.info(f"Saving model to {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    logger.info("Starting model pull process")
    
    # Create directories
    create_directories()
    
    # Load parameters
    params = load_params()
    mlflow_params = params['mlflow']
    
    # Pull the production model
    model = pull_model_by_alias(
        mlflow_params['model_name'],
        mlflow_params['model_production_alias']
    )
    
    if model:
        # Save the model locally
        save_model(model, os.path.join("models", "production_model.pkl"))
        logger.info("Production model pulled and saved successfully")
    else:
        # Try pulling the staging model as fallback
        logger.info("Attempting to pull staging model as fallback")
        model = pull_model_by_alias(
            mlflow_params['model_name'],
            mlflow_params['model_staging_alias']
        )
        
        if model:
            save_model(model, os.path.join("models", "production_model.pkl"))
            logger.info("Staging model pulled and saved successfully as fallback")
        else:
            logger.error("Failed to pull any model from the registry")
    
    logger.info("Model pull process completed")
