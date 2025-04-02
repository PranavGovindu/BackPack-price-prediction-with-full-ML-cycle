import os
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("data_preprocessing")

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
    return params

def create_directories():
    """Create necessary directories if they don't exist"""
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    logger.info("Created processed data directory")

def load_data(train_path, test_path):
    """Load raw data from CSV files"""
    logger.info(f"Loading data from {train_path} and {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    return train_df, test_df

def identify_column_types(df):
    """Identify numerical and categorical columns"""
    num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    cat_cols = [col for col in df.columns if df[col].dtype in ['object', 'category']]
    
    # Filter out the target and id columns from features
    num_features = [col for col in num_cols if col not in ['id', 'Price']]
    
    logger.info(f"Identified {len(num_features)} numerical features")
    logger.info(f"Identified {len(cat_cols)} categorical features")
    
    return num_features, cat_cols

def handle_missing_values(df, num_features, cat_cols):
    """Handle missing values in the dataframe"""
    logger.info("Handling missing values")
    df_copy = df.copy()
    
    missing_counts = df_copy.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Found {missing_counts.sum()} missing values")
        logger.info(f"Missing values by column: {missing_counts[missing_counts > 0].to_dict()}")
    
    # Fill numerical columns with median
    for col in num_features:
        if col in df_copy.columns and df_copy[col].isnull().sum() > 0:
            logger.info(f"Filling missing values in {col} with median")
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    # Fill categorical columns with mode
    for col in cat_cols:
        if col in df_copy.columns and df_copy[col].isnull().sum() > 0:
            logger.info(f"Filling missing values in {col} with mode")
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
    
    return df_copy

def handle_outliers(df, columns, strategy='winsorize'):
    """Handle outliers in specified columns using the specified strategy"""
    logger.info(f"Handling outliers using {strategy} strategy")
    df_copy = df.copy()
    
    if strategy == 'remove':
        # Remove outliers approach
        logger.info("Using removal strategy for outliers")
        mask = pd.Series(True, index=df_copy.index)
        
        for col in columns:
            if col in df_copy.columns:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_mask = (df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)
                outliers_count = (~col_mask).sum()
                logger.info(f"Identified {outliers_count} outliers in {col}")
                mask = mask & col_mask
        
        logger.info(f"Removed {(~mask).sum()} rows containing outliers")
        return df_copy[mask]
    
    elif strategy == 'winsorize':
        # Winsorization - caps outliers instead of removing
        logger.info("Using winsorization strategy for outliers")
        for col in columns:
            if col in df_copy.columns:
                Q1 = df_copy[col].quantile(0.05)  # Using 5% instead of 25% for more conservative capping
                Q3 = df_copy[col].quantile(0.95)  # Using 95% instead of 75%
                
                # Count outliers
                outliers_count = ((df_copy[col] < Q1) | (df_copy[col] > Q3)).sum()
                logger.info(f"Capping {outliers_count} outliers in {col}")
                
                # Cap values
                df_copy[col] = df_copy[col].clip(lower=Q1, upper=Q3)
    
    return df_copy

if __name__ == "__main__":
    logger.info("Starting data preprocessing process")
    
    # Load parameters
    params = load_params()
    
    # Create directories
    create_directories()
    
    # Load data
    train_df, test_df = load_data(
        os.path.join("data", "raw", "train.csv"),
        os.path.join("data", "raw", "test.csv")
    )
    
    # Identify column types
    num_features, cat_cols = identify_column_types(train_df)
    
    # Handle missing values if specified
    if params['preprocessing']['handle_missing']:
        train_df = handle_missing_values(train_df, num_features, cat_cols)
        test_df = handle_missing_values(test_df, num_features, cat_cols)
    
    # Handle outliers if specified
    if params['preprocessing']['handle_outliers']:
        outlier_columns = params['preprocessing']['outlier_columns']
        outlier_strategy = params['preprocessing']['outlier_strategy']
        
        # Only apply outlier handling to training data
        train_df = handle_outliers(train_df, outlier_columns, strategy=outlier_strategy)
    
    # Save processed data
    train_output_path = os.path.join("data", "processed", "train_preprocessed.csv")
    test_output_path = os.path.join("data", "processed", "test_preprocessed.csv")
    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    logger.info(f"Saved processed train data to {train_output_path}")
    logger.info(f"Saved processed test data to {test_output_path}")
    logger.info("Data preprocessing completed successfully")
