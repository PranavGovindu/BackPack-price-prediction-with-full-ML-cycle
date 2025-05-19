# Backpack Price Prediction MLOps Pipeline

This repository implements a comprehensive MLOps pipeline for predicting backpack prices using Random Forest regression. The pipeline integrates modern MLOps practices including automated data versioning with DVC, experiment tracking and model registry with MLflow, and a modular, production-ready codebase.

## Project Overview

The project aims to predict backpack prices based on various features like brand, material, capacity, and other specifications. It demonstrates end-to-end MLOps practices from data ingestion to model deployment.

### Key Features

- Automated data versioning and pipeline orchestration with DVC
- Experiment tracking and model registry with MLflow
- Robust feature engineering pipeline
- Cross-validation and comprehensive model evaluation
- Production and staging model versioning
- AWS S3 integration for data storage
- Logging and monitoring capabilities

## Technical Architecture

```bash
├── artifacts/             # Model artifacts and metadata
│   ├── model_info.json   # Model metadata and performance metrics
│   └── preprocessor.pkl  # Saved feature preprocessing pipeline
├── data/                 # Data directories (DVC tracked)
│   ├── raw/              # Raw data from S3
│   ├── processed/        # Cleaned and preprocessed data
│   └── featured/         # Feature engineered data
├── metrics/              # Model evaluation metrics
│   └── evaluation_metrics.json  # Detailed model performance metrics
├── models/               # Model files
│   ├── trained_model.pkl        # Latest trained model
│   └── production_model.pkl     # Production deployed model
├── src/                  # Source code
│   ├── data_processing/          # Data processing modules
│   │   ├── data_ingestion.py     # S3 data fetching
│   │   └── data_preprocessing.py  # Data cleaning
│   ├── feature_eng/              # Feature engineering
│   │   └── feature_engineering.py # Feature creation
│   ├── modelbuild/               # Model operations
│   │   ├── model_training.py     # Training with MLflow
│   │   ├── model_evaluation.py   # Metrics calculation
│   │   ├── model_registry.py     # MLflow registry management
│   │   └── model_pull.py         # Production model deployment
│   └── logger/                   # Logging configuration
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # Configuration parameters
└── requirements.txt      # Project dependencies
```

## Pipeline Stages

### 1. Data Ingestion

- Sources data from AWS S3 bucket
- Configurable data paths and bucket settings
- Automated data versioning with DVC

### 2. Data Preprocessing

- Handles missing values using median/mode imputation
- Outlier detection and treatment (winsorization/removal)
- Configurable preprocessing parameters
- Data validation and quality checks

### 3. Feature Engineering

Implements multiple feature creation strategies:

- **Text Features**: Brand name processing
- **Interaction Features**: Brand-Material, Style-Material combinations
- **Binary Features**: Laptop compartment, Waterproof indicators
- **Binning Features**: Compartment categories, Size grouping
- **Ratio Features**: Weight capacity ratios
- **Log Transformations**: For skewed numerical features
- **Aggregation Features**: Brand-level statistics

### 4. Model Training

- Random Forest Regressor implementation
- Hyperparameter configuration:
  - n_estimators: 235
  - max_depth: 20
  - min_samples_split: 5
  - min_samples_leaf: 10
  - max_features: log2
  - bootstrap: True
- MLflow experiment tracking
- Cross-validation with configurable folds

### 5. Model Evaluation

Comprehensive evaluation metrics:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)
- Cross-validation scores
- Performance logging to MLflow

### 6. Model Registry

- Model versioning in MLflow
- Automatic staging/production promotion
- Version aliases for easy deployment
- Model metadata storage

### 7. Model Deployment

- Production model serving
- Version control with aliases
- Easy model rollback capability
- Deployment metadata tracking

## Setup and Installation

1. Clone this repository:

   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Configure AWS credentials:

   ```bash
   export AWS_ACCESS_KEY_ID='your-access-key'
   export AWS_SECRET_ACCESS_KEY='your-secret-key'
   ```

4. Configure MLflow tracking:

   ```bash
   export MLFLOW_TRACKING_URI='your-mlflow-server'
   export DAGSHUB_USERNAME='your-username'
   export DAGSHUB_TOKEN='your-token'
   ```

## Usage

### Running the Pipeline

Complete pipeline execution:

```bash
dvc repro
```

Individual stage execution:

```bash
dvc repro [stage-name]
```

Available stages:

- data_ingestion
- data_preprocessing
- feature_engineering
- model_training
- model_evaluation
- model_registry
- model_pull

### Configuration

All pipeline parameters are configurable in `params.yaml`:

- Data paths and S3 settings
- Preprocessing parameters
- Feature engineering options
- Model hyperparameters
- Evaluation settings
- MLflow configuration

### Model Tracking

Access MLflow UI for experiment tracking:

```bash
mlflow ui
```

View experiment results:

- Training metrics
- Model parameters
- Artifacts and metadata
- Version history

## Performance Monitoring

The pipeline includes comprehensive logging and monitoring:

- Training metrics tracking
- Data quality monitoring
- Pipeline execution logs
- Model performance tracking

## Work in Progress

The following features are currently under development:

### Docker Containerization

- Containerization of the ML pipeline components
- Multi-stage Docker builds for optimized image sizes
- Development and production Docker configurations
- Docker Compose setup for local development
- Container registry integration (Amazon ECR)

### Kubernetes Deployment

- Kubernetes manifests for deploying the ML pipeline
- Helm charts for managing deployments
- Amazon EKS cluster configuration
- Horizontal Pod Autoscaling (HPA) setup
- Resource requests and limits optimization
- Monitoring with Prometheus and Grafana
- CI/CD pipeline for Kubernetes deployments

### Planned Infrastructure

- Amazon EKS cluster setup
- Load balancing with AWS ALB
- Auto-scaling configurations
- Secrets management with AWS Secrets Manager
- Logging with Amazon CloudWatch
- Monitoring with Amazon Managed Prometheus

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
