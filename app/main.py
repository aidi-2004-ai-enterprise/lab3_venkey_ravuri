from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from typing import Dict
import logging
import os
import json
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Penguin Classifier", version="1.0.0")

# Pydantic models for input validation
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

class PredictionResponse(BaseModel):
    species: str
    confidence: float
    probabilities: Dict[str, float]

# Global variables for model
model = None
feature_columns = None
species_classes = None

def load_model():
    """Load the XGBoost model and metadata."""
    global model, feature_columns, species_classes
    
    try:
        import xgboost as xgb
        
        model_path = "app/data/model.json"
        info_path = "app/data/model_info.json"
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}. Run 'python train.py' first.")
            return False
        
        # Load model
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info("XGBoost model loaded successfully")
        
        # Load metadata
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            feature_columns = model_info['feature_columns']
            species_classes = model_info['species_classes']
        else:
            # Fallback values
            feature_columns = [
                'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year',
                'sex_female', 'sex_male', 'island_Biscoe', 'island_Dream', 'island_Torgersen'
            ]
            species_classes = ['Adelie', 'Chinstrap', 'Gentoo']
            logger.warning("Using fallback metadata")
        
        logger.info(f"Model ready: {len(feature_columns)} features, {len(species_classes)} classes")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def create_feature_vector(penguin_data: PenguinFeatures) -> pd.DataFrame:
    """Convert penguin data to model input format."""
    # Access enum values correctly
    sex_value = penguin_data.sex.value if hasattr(penguin_data.sex, 'value') else penguin_data.sex
    island_value = penguin_data.island.value if hasattr(penguin_data.island, 'value') else penguin_data.island
    
    features = {
        'bill_length_mm': penguin_data.bill_length_mm,
        'bill_depth_mm': penguin_data.bill_depth_mm,
        'flipper_length_mm': penguin_data.flipper_length_mm,
        'body_mass_g': penguin_data.body_mass_g,
        'year': penguin_data.year,
        'sex_female': 1 if sex_value == 'female' else 0,
        'sex_male': 1 if sex_value == 'male' else 0,
        'island_Biscoe': 1 if island_value == 'Biscoe' else 0,
        'island_Dream': 1 if island_value == 'Dream' else 0,
        'island_Torgersen': 1 if island_value == 'Torgersen' else 0,
    }
    
    # Create DataFrame with all possible columns first
    df = pd.DataFrame([features])
    
    # Make sure we have all the columns the model expects
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Return columns in the correct order
    df = df[feature_columns]
    return df

@app.on_event("startup")
async def startup_event():
    """Load model when app starts."""
    logger.info("Starting up...")
    load_model()

@app.get("/")
def read_root():
    return {
        "message": "Penguin Species Classifier API", 
        "status": "API is running",
        "version": "1.0.0",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "features_count": len(feature_columns) if feature_columns else 0,
        "species_classes": species_classes if species_classes else [],
        "endpoints": ["/", "/health", "/predict", "/docs"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_species(penguin_data: PenguinFeatures):
    """Predict penguin species using the trained XGBoost model."""
    try:
        # Debug logging
        sex_value = penguin_data.sex.value if hasattr(penguin_data.sex, 'value') else penguin_data.sex
        island_value = penguin_data.island.value if hasattr(penguin_data.island, 'value') else penguin_data.island
        
        logger.info(f"Prediction request: {island_value} island, {sex_value} penguin")
        logger.debug(f"Raw input: sex={penguin_data.sex}, island={penguin_data.island}")
        
        if model is None:
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. Please run 'python train.py' first and restart the API."
            )
        
        # Create feature vector
        features = create_feature_vector(penguin_data)
        logger.debug(f"Feature vector shape: {features.shape}")
        logger.debug(f"Feature columns: {list(features.columns)}")
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Format response
        predicted_species = species_classes[prediction]
        confidence = float(max(probabilities))
        
        prob_dict = {
            species: float(prob) 
            for species, prob in zip(species_classes, probabilities)
        }
        
        logger.info(f"Predicted: {predicted_species} (confidence: {confidence:.4f})")
        
        return PredictionResponse(
            species=predicted_species,
            confidence=confidence,
            probabilities=prob_dict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(f"Error type: {type(e)}")
        # Add more debugging info
        if 'penguin_data' in locals():
            logger.error(f"Input data: {penguin_data}")
        if 'features' in locals():
            logger.error(f"Features created: {features.columns.tolist() if hasattr(features, 'columns') else 'No features'}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Test endpoint for debugging
@app.post("/test")
def test_prediction_format(penguin_data: PenguinFeatures):
    """Test endpoint to see how data is being processed."""
    try:
        features = create_feature_vector(penguin_data)
        return {
            "received_data": penguin_data.dict(),
            "model_loaded": model is not None,
            "expected_feature_columns": feature_columns,
            "created_features": list(features.columns) if hasattr(features, 'columns') else None,
            "feature_values": features.iloc[0].to_dict() if hasattr(features, 'iloc') else None,
            "species_classes": species_classes
        }
    except Exception as e:
        return {
            "error": str(e),
            "received_data": penguin_data.dict(),
            "model_loaded": model is not None,
            "expected_feature_columns": feature_columns,
            "species_classes": species_classes
        }

@app.get("/debug")
def debug_info():
    """Debug endpoint to check model status."""
    return {
        "model_loaded": model is not None,
        "feature_columns": feature_columns,
        "species_classes": species_classes,
        "model_type": str(type(model)) if model else None
    }