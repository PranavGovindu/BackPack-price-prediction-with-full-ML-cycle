import logging
import os
import sys
from datetime import datetime

# Get the root directory
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
log_dir_path = os.path.join(root_dir, 'logs')
os.makedirs(log_dir_path, exist_ok=True)

# Create log file name with timestamp
log_file_name = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
log_file_path = os.path.join(log_dir_path, log_file_name)

def configure_logger():
    """
    Configure logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)  
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Configure and get logger
logger = configure_logger()
