"""
Train XGBoost model on penguins dataset with proper preprocessing and evaluation.
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, accuracy_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_penguins_data() -> pd.DataFrame:
    """
    Load the penguins dataset from seaborn.
    
    Returns:
        pd.DataFrame: The penguins dataset with missing values dropped
    """
    logger.info("Loading penguins dataset from seaborn")
    penguins = sns.load_dataset("penguins")
    
    # Drop rows with missing values
    penguins_clean = penguins.dropna()
    logger.info(f"Dataset loaded with {len(penguins_clean)} rows after dropping missing values")
    
    return penguins_clean


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Preprocess the penguins dataset with one-hot encoding and label encoding.
    
    Args:
        df: Raw penguins dataframe
        
    Returns:
        Tuple containing:
        - X: Features dataframe with one-hot encoded categorical variables
        - y: Target variable (species) with label encoding
        - label_encoder: Fitted label encoder for species
    """
    logger.info("Starting data preprocessing")
    
    # Separate features and target
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Apply one-hot encoding to categorical features (sex and island)
    X_encoded = pd.get_dummies(X, columns=['sex', 'island'], drop_first=False)
    logger.info(f"Applied one-hot encoding. Features shape: {X_encoded.shape}")
    logger.info(f"Feature columns: {list(X_encoded.columns)}")
    
    # Apply label encoding to target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info(f"Applied label encoding to species. Classes: {label_encoder.classes_}")
    
    return X_encoded, pd.Series(y_encoded), label_encoder


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with parameters to prevent overfitting.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        xgb.XGBClassifier: Trained XGBoost model
    """
    logger.info("Training XGBoost model")
    
    # Initialize XGBoost with parameters to prevent overfitting
    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    return model


def evaluate_model(model: xgb.XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series, label_encoder: LabelEncoder) -> None:
    """
    Evaluate the trained model on training and test sets.
    
    Args:
        model: Trained XGBoost model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        label_encoder: Label encoder for species names
    """
    logger.info("Evaluating model performance")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Log results
    logger.info(f"Training F1-score: {train_f1:.4f}")
    logger.info(f"Test F1-score: {test_f1:.4f}")
    logger.info(f"Training Accuracy: {train_accuracy:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Print detailed classification report
    species_names = label_encoder.classes_
    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=species_names))


def save_model(model: xgb.XGBClassifier, filepath: str) -> None:
    """
    Save the trained model to a JSON file.
    
    Args:
        model: Trained XGBoost model
        filepath: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    model.save_model(filepath)
    logger.info(f"Model saved to {filepath}")


def main() -> None:
    """
    Main function to execute the complete training pipeline.
    """
    logger.info("Starting penguins classification training pipeline")
    
    # Load data
    df = load_penguins_data()
    
    # Preprocess data
    X, y, label_encoder = preprocess_data(df)
    
    # Split data into training and test sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_train, y_train, X_test, y_test, label_encoder)
    
    # Save model
    model_path = "app/data/model.json"
    save_model(model, model_path)
    
    # Also save the feature columns and label encoder classes for consistency
    feature_info = {
        'feature_columns': list(X.columns),
        'species_classes': list(label_encoder.classes_)
    }
    
    import json
    info_path = "app/data/model_info.json"
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    logger.info(f"Model info saved to {info_path}")
    
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()