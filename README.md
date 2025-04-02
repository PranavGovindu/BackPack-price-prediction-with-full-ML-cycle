# Backpack Price Prediction MLOps Pipeline

This repository contains an end-to-end MLOps pipeline for predicting backpack prices using Random Forest regression. The pipeline is managed with DVC and includes model tracking and registry with MLflow.

## Project Structure

```bash
├── artifacts/             # Model artifacts and metadata
├── data/                  # Data directories
│   ├── raw/               # Raw data from S3
│   ├── processed/         # Preprocessed data
│   └── featured/          # Feature engineered data
├── metrics/               # Model evaluation metrics
├── models/                # Trained model files
├── src/                   # Source code
│   ├── data_ingestion.py  # Data loading from S3
│   ├── data_preprocessing.py  # Data cleaning and preprocessing
│   ├── feature_engineering.py  # Feature creation and transformation
│   ├── model_training.py  # Model training with MLflow tracking
│   ├── model_evaluation.py  # Model evaluation and metrics logging
│   ├── model_registry.py  # MLflow model registry management
│   └── model_pull.py      # Pull model from registry
├── dvc.yaml               # DVC pipeline configuration
├── params.yaml            # Pipeline parameters
└── README.md              # Project documentation
```

## Pipeline Overview

1. **Data Ingestion**: Download training and testing data from S3
2. **Data Preprocessing**: Clean data, handle missing values and outliers
3. **Feature Engineering**: Create and transform features to improve model performance
4. **Model Training**: Train Random Forest regressor with hyperparameter tuning
5. **Model Evaluation**: Calculate performance metrics and log them to MLflow
6. **Model Registry**: Register the model in MLflow and create aliases
7. **Model Pull**: Pull the best model from MLflow registry for deployment

## Setup and Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -e .
   ```
3. Set up AWS credentials for S3 access
4. Configure MLflow tracking server (if using remote tracking)

## Running the Pipeline

Run the complete pipeline with:
```bash
dvc repro
```
Or run individual stages:
```bash
dvc repro data_ingestion
```

## Pipeline Parameters

All configurable parameters are stored in `params.yaml`. You can modify these parameters to customize the pipeline behavior.

## MLflow Integration

- Models are tracked and registered in MLflow
- Performance metrics are logged for each model
- Model versions are managed with aliases ('production' and 'staging')
- Best model can be pulled from the registry using aliases

## Feature Engineering

The pipeline implements several feature engineering techniques:
- Text-based features
- Interaction features
- Binary encoding
- Binning and grouping
- Ratio features
- Log transformations
- Aggregation features

## Model Evaluation

The model is evaluated using multiple metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)
Cross-validation is used during training to ensure model robustness.
