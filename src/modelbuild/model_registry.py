import os
import yaml
import json
import logging
import mlflow
import mlflow.sklearn
from pathlib import Path
import dagshub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_registry")

logger.info("Initializing DagsHub connection")
dagshub.init(repo_owner='PranavGovindu', repo_name='st-1', mlflow=True)
logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
    return params

def load_metrics(metrics_path):
    """Load evaluation metrics from JSON file"""
    logger.info(f"Loading metrics from {metrics_path}")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def get_latest_run_id(experiment_name, run_name):
    """Get the latest MLflow run ID for the given experiment and run name"""
    logger.info(f"Finding latest run for experiment: {experiment_name}, run_name: {run_name}")
    
    mlflow.set_experiment(experiment_name)
    
    # Get all runs in the experiment
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attribute.start_time DESC"]
    )
    
    if len(runs) == 0:
        logger.warning(f"No runs found for experiment {experiment_name} with run_name {run_name}")
        return None
    
    latest_run_id = runs.iloc[0]['run_id']
    logger.info(f"Found latest run ID: {latest_run_id}")
    
    return latest_run_id

def register_model(run_id, model_name, artifact_path):
    """Register model from the given run to the MLflow Model Registry"""
    logger.info(f"Registering model {model_name} from run {run_id}")
    
    try:
        model_details = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{artifact_path}",
            name=model_name
        )
        logger.info(f"Model registered with version: {model_details.version}")
        return model_details
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        return None

def create_model_alias(model_name, version, alias):
    """Create an alias for the model version in the registry"""
    logger.info(f"Creating alias '{alias}' for {model_name} version {version}")
    
    try:
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(model_name, alias, version)
        logger.info(f"Successfully set alias '{alias}' for version {version}")
        return True
    except Exception as e:
        logger.error(f"Error creating alias: {e}")
        return False

def save_model_info(model_details, metrics, output_path):
    """Save model registration details to a file"""
    logger.info(f"Saving model info to {output_path}")
    
    model_info = {
        "name": model_details.name,
        "version": model_details.version,
        "creation_timestamp": model_details.creation_timestamp,
        "last_updated_timestamp": model_details.last_updated_timestamp,
        "metrics": metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    logger.info(f"Model info saved to {output_path}")

if __name__ == "__main__":
    logger.info("Starting model registry process")
    
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    
    
    params = load_params()
    mlflow_params = params['mlflow']
    
    metrics = load_metrics(os.path.join("metrics", "evaluation_metrics.json"))
    
    # Get the latest run ID
    run_id = get_latest_run_id(
        mlflow_params['experiment_name'],
        mlflow_params['run_name']
    )
    
    if run_id:
        # Register the model
        model_details = register_model(
            run_id,
            mlflow_params['model_name'],
            mlflow_params['artifact_path']
        )
        
        if model_details:
            # Create aliases
            create_model_alias(
                mlflow_params['model_name'],
                model_details.version,
                mlflow_params['model_staging_alias']
            )
            
            if metrics.get('train_mae', 0) > 30: 
                logger.info(f"Model meets production criteria with R² = {metrics.get('train_r2')}")
                create_model_alias(
                    mlflow_params['model_name'],
                    model_details.version,
                    mlflow_params['model_production_alias']
                )
                logger.info(f"Model promoted to production")
            
            # Save model info
            save_model_info(
                model_details,
                metrics,
                os.path.join("artifacts", "model_info.json")
            )
    else:
        logger.warning("No run ID found. Creating a test run to verify DagsHub connectivity")
        with mlflow.start_run(run_name="dagshub_test_run"):
            mlflow.log_param('test_parameter', 'value')
            mlflow.log_metric('test_metric', 1)
            logger.info(f"Created test run with URI: {mlflow.get_tracking_uri()}")
    
    logger.info("Model registry process completed")
