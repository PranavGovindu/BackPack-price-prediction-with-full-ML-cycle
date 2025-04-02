import os
import yaml
import logging
import boto3
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("data_ingestion")

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
    return params

def create_directories():
    """Create necessary directories if they don't exist"""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    logger.info("Created raw data directory")

def download_from_s3(bucket_name, file_key, output_path):
    """Download a file from S3"""
    try:
        logger.info(f"Downloading {file_key} from S3 bucket {bucket_name}")
        s3_client = boto3.client('s3', region_name=params['data']['region'])
        s3_client.download_file(bucket_name, file_key, output_path)
        logger.info(f"Successfully downloaded {file_key} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}")
        return False

def extract_file_key(s3_path):
    """Extract file key from S3 path"""
    if s3_path.startswith('s3://'):
        path = s3_path.replace('s3://', '')
        parts = path.split('/', 1)
        if len(parts) > 1:
            return parts[1]
    
    # If not a full s3 path, assume it's just the filename
    return os.path.basename(s3_path) 

if __name__ == "__main__":
    logger.info("Starting data ingestion process")
    
    # Load parameters
    params = load_params()
    
    # Create directories
    create_directories()
    
    # Get S3 bucket details
    bucket_name = params['data']['bucket_name']
    region = params['data']['region']
    
    # Download training data
    train_path = params['data']['train_path']
    train_key = extract_file_key(train_path)
    train_output_path = os.path.join("data", "raw", "train.csv")
    download_from_s3(bucket_name, train_key, train_output_path)
    
    # Download test data
    test_path = params['data']['test_path']
    test_key = extract_file_key(test_path)
    test_output_path = os.path.join("data", "raw", "test.csv")
    download_from_s3(bucket_name, test_key, test_output_path)
    
    # Verify downloaded files
    try:
        train_df = pd.read_csv(train_output_path)
        test_df = pd.read_csv(test_output_path)
        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Test data shape: {test_df.shape}")
        logger.info("Data ingestion completed successfully")
    except Exception as e:
        logger.error(f"Error verifying downloaded files: {e}")
