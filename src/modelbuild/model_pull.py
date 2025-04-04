import os
import sys
import mlflow
import joblib
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_pull")

# Constants
PARAMS_FILE = "params.yaml"
OUTPUT_DIR_NAME = "models"
OUTPUT_MODEL_FILENAME = "production_model.pkl"

# --- Local Helper Functions (Replaced import from src.utils) ---
def load_yaml(filepath):
    """Loads a YAML file.
    Args:
        filepath (str or Path): Path to the YAML file.
    Returns:
        dict: Parsed YAML content.
    Raises:
        SystemExit: If file cannot be loaded or parsed.
    """
    try:
        with open(filepath, "r") as f:
            content = yaml.safe_load(f)
            logger.info(f"Loaded YAML from {filepath}")
            return content
    except Exception as e:
        logger.error(f"CRITICAL Error loading/parsing YAML file '{filepath}': {e}")
        sys.exit(1)

def create_directories(dir_path):
    """Creates directories if they don't exist.
    Args:
        dir_path (Path): Path object for the directory.
    Raises:
        SystemExit: If directory creation fails.
    """
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {dir_path}")
    except OSError as e:
        logger.error(f"CRITICAL Error creating directory {dir_path}: {e}")
        sys.exit(1)
# --- End Local Helper Functions ---

def set_mlflow_credentials(tracking_uri):
    """Sets MLflow tracking URI and handles DagsHub authentication."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        
        # Check if using DagsHub and attempt to set credentials via env vars
        if "dagshub.com" in tracking_uri:
            logger.info("Using DagsHub. Ensure DAGSHUB_USERNAME and DAGSHUB_TOKEN environment variables are set.")
            # MLflow automatically picks up DAGSHUB_USERNAME and DAGSHUB_TOKEN
            # No need to explicitly set os.environ['MLFLOW_TRACKING_USERNAME'] etc.
        else:
            # Handle other potential auth methods if needed (e.g., .env file for non-DagsHub)
            pass
        logger.info("MLflow tracking URI set, proceeding with model pull")
        return True
    except Exception as e:
        logger.error(f"Failed to set MLflow tracking URI or credentials: {e}")
        return False

def pull_model_by_alias(model_name, alias):
    """
    Attempts to pull a model from MLflow using its name and alias.
    Returns the loaded model object or None if unsuccessful.
    """
    if not model_name or not alias:
        logger.error("Model name or alias is empty. Cannot pull model.")
        return None

    model_uri = f"models:/{model_name}@{alias}"
    logger.info("Trying to pull model")
    logger.info(f"Using model URI: {model_uri}")
    
    try:
        logger.info("Attempting to load model directly using MLflow client")
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        logger.info(f"Successfully pulled model '{model_name}' with alias '{alias}'.")
        return model
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow Error loading model '{model_uri}': {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading model '{model_uri}': {e}")
        return None

def save_model(model, output_path):
    """Saves the model object to the specified path using joblib."""
    try:
        joblib.dump(model, output_path)
        logger.info(f"Model successfully saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save model to {output_path}: {e}")
        return False

if __name__ == "__main__":
    logger.info("--- Starting Model Pull Process ---")
    load_dotenv()  # Load environment variables from .env file, if present

    # Load parameters
    try:
        params = load_yaml(PARAMS_FILE)
        mlflow_config = params['mlflow']
        model_name = mlflow_config.get('model_name')
        prod_alias = mlflow_config.get('model_production_alias') 
        staging_alias = mlflow_config.get('model_staging_alias')
        tracking_uri = mlflow_config.get('tracking_uri')
        logger.info(f"Loaded parameters from {PARAMS_FILE}")
    except Exception as e:
        logger.error(f"Error loading parameters from {PARAMS_FILE}: {e}")
        sys.exit(1)

    # Set MLflow Tracking URI and Credentials
    if not tracking_uri:
        logger.error("MLFLOW_TRACKING_URI not specified in params.yaml.")
        sys.exit(1)
    if not set_mlflow_credentials(tracking_uri):
        sys.exit(1)

    # Define output path
    OUTPUT_DIR = Path(OUTPUT_DIR_NAME)
    create_directories(OUTPUT_DIR)
    output_model_path = OUTPUT_DIR / OUTPUT_MODEL_FILENAME
    
    model = None
    # --- Restored Original Logic: Try Production, then Staging Alias ---
    # try to get production model if alias is specified
    if prod_alias:
        logger.info(f"Trying to get PRODUCTION model: '{model_name}' with alias '{prod_alias}'")
        model = pull_model_by_alias(model_name, prod_alias)
     
    # if production not available or failed, try staging if alias is specified
    if model is None and staging_alias:
        logger.warning(f"Couldn't get production model. Trying STAGING model instead.")
        logger.info(f"Trying to get STAGING model: '{model_name}' with alias '{staging_alias}'")
        model = pull_model_by_alias(model_name, staging_alias)
    # --- End Modification ---
    
    if model is not None:
        if save_model(model, output_model_path):
            logger.info("Model successfully pulled and saved.")
        else:
            logger.error("Model pulled but failed to save.")
            sys.exit(1)
    else:
        logger.error("Couldn't get any model from MLflow.")
        sys.exit(1)

    logger.info("Model pull done")
