from fastapi import FastAPI, HTTPException
from pydantic_models import PredictionRequest, PredictionResponse
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import List
import joblib

app = FastAPI(
    title="Credit Risk API",
    description="API for predicting credit risk using ML model",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = "../../models/best_model.pkl"
model = None

@app.on_event("startup")
async def load_model():
    """Load model when API starts"""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Prediction API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk for a customer
    
    Returns risk probability and category (Low/High Risk)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Make prediction
        risk_probability = model.predict_proba(input_data)[0][1]
        risk_category = "High Risk" if risk_probability >= 0.5 else "Low Risk"
        
        return PredictionResponse(
            risk_probability=float(risk_probability),
            risk_category=risk_category
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert requests to DataFrame
        input_data = pd.DataFrame([req.dict() for req in requests])
        
        # Make predictions
        risk_probabilities = model.predict_proba(input_data)[:, 1]
        
        responses = []
        for prob in risk_probabilities:
            responses.append({
                "risk_probability": float(prob),
                "risk_category": "High Risk" if prob >= 0.5 else "Low Risk"
            })
        
        return responses
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)