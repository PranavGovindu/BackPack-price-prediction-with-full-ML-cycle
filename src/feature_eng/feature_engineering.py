import os
import yaml
import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("feature_engineering")

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
    return params

def create_directories():
    """Create necessary directories if they don't exist"""
    Path("data/featured").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    logger.info("Created featured data and artifacts directories")

def load_preprocessed_data(train_path, test_path):
    """Load preprocessed data from CSV files"""
    logger.info(f"Loading preprocessed data from {train_path} and {test_path}")
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

def perform_feature_engineering(df, config):
    """Apply feature engineering transformations based on config parameters"""
    logger.info("Starting feature engineering process")
    df_copy = df.copy()
    features_added = 0
    
    # TEXT PROCESSING FOR CATEGORICAL FEATURES
    if config['create_text_features'] and 'Brand' in df_copy.columns:
        logger.info("Creating text-based features")
        df_copy['Brand_First_Word'] = df_copy['Brand'].astype(str).apply(lambda x: x.split()[0] if x else x)
        features_added += 1

    # INTERACTION FEATURES
    if config['create_interaction_features']:
        logger.info("Creating interaction features")
        for col in ['Material', 'Size', 'Style']:
            if 'Brand' in df_copy.columns and col in df_copy.columns:
                df_copy[f'Brand_{col}'] = df_copy['Brand'].astype(str) + '_' + df_copy[col].astype(str)
                features_added += 1

        if 'Style' in df_copy.columns and 'Material' in df_copy.columns:
            df_copy['Style_Material'] = df_copy['Style'].astype(str) + '_' + df_copy['Material'].astype(str)
            features_added += 1

    # BINARY FEATURES
    if config['create_binary_features']:
        logger.info("Creating binary features")
        for col in ['Laptop Compartment', 'Waterproof']:
            if col in df_copy.columns:
                df_copy[f'Is_{col.replace(" ", "_")}'] = df_copy[col].map({'Yes': 1, 'No': 0})
                features_added += 1

    # BINNING AND GROUPING
    if config['create_binning_features']:
        logger.info("Creating binning and grouping features")
        if 'Compartments' in df_copy.columns:
            df_copy['Compartments_Category'] = pd.cut(
                df_copy['Compartments'], bins=[0, 1, 3, 5, 10, np.inf], 
                labels=['Single', 'Few', 'Moderate', 'Many', 'Very Many']
            )
            features_added += 1

        if 'Size' in df_copy.columns:
            size_counts = df_copy['Size'].value_counts()
            rare_sizes = size_counts[size_counts < 10].index
            df_copy['Size_Group'] = df_copy['Size'].apply(lambda x: 'Rare Size' if x in rare_sizes else x)
            features_added += 1

    # RATIO FEATURES
    if config['create_ratio_features']:
        logger.info("Creating ratio features")
        if 'Weight Capacity (kg)' in df_copy.columns:
            df_copy['Weight_Capacity_Ratio'] = df_copy['Weight Capacity (kg)'] / df_copy['Weight Capacity (kg)'].max()
            features_added += 1

        if 'Weight Capacity (kg)' in df_copy.columns and 'Compartments' in df_copy.columns:
            df_copy['Weight_to_Compartments'] = df_copy['Weight Capacity (kg)'] / (df_copy['Compartments'] + 1)
            features_added += 1

    # LOG TRANSFORMATIONS FOR SKEWED FEATURES
    if config['create_log_transforms']:
        logger.info("Creating log transform features")
        if 'Weight Capacity (kg)' in df_copy.columns:
            df_copy['Log_Weight_Capacity'] = np.log1p(df_copy['Weight Capacity (kg)'])
            features_added += 1

    # AGGREGATION FEATURES
    if config['create_aggregation_features']:
        logger.info("Creating aggregation features")
        if 'Brand' in df_copy.columns and 'Weight Capacity (kg)' in df_copy.columns:
            df_copy['Brand_Avg_Capacity'] = df_copy.groupby('Brand')['Weight Capacity (kg)'].transform('mean')
            df_copy['Capacity_vs_Brand_Avg'] = df_copy['Weight Capacity (kg)'] / (df_copy['Brand_Avg_Capacity'] + 0.001)
            features_added += 2

    logger.info(f"Added {features_added} new engineered features")
    return df_copy

def create_preprocessor(num_features, cat_features):
    """Create a column transformer for preprocessing features"""
    logger.info("Creating feature preprocessing pipeline")
    
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ], remainder='drop')
    
    return preprocessor

if __name__ == "__main__":
    logger.info("Starting feature engineering process")
    
    # Load parameters
    params = load_params()
    feature_eng_params = params['feature_engineering']
    
    # Create directories
    create_directories()
    
    # Load preprocessed data
    train_df, test_df = load_preprocessed_data(
        os.path.join("data", "processed", "train_preprocessed.csv"),
        os.path.join("data", "processed", "test_preprocessed.csv")
    )
    
    # Apply feature engineering
    train_featured = perform_feature_engineering(train_df, feature_eng_params)
    test_featured = perform_feature_engineering(test_df, feature_eng_params)
    
    # Identify column types after feature engineering
    final_num_cols, final_cat_cols = identify_column_types(train_featured)
    
    # Create preprocessor
    preprocessor = create_preprocessor(final_num_cols, final_cat_cols)
    
    # Save the preprocessor for later use
    with open(os.path.join("artifacts", "preprocessor.pkl"), 'wb') as f:
        pickle.dump(preprocessor, f)
    logger.info("Saved preprocessor to artifacts/preprocessor.pkl")
    
    # Save featured data
    train_output_path = os.path.join("data", "featured", "train_featured.csv")
    test_output_path = os.path.join("data", "featured", "test_featured.csv")
    
    train_featured.to_csv(train_output_path, index=False)
    test_featured.to_csv(test_output_path, index=False)
    
    logger.info(f"Saved featured train data to {train_output_path}")
    logger.info(f"Saved featured test data to {test_output_path}")
    logger.info("Feature engineering completed successfully")
