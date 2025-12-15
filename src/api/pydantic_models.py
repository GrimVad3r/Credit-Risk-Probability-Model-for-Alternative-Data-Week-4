from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
    """Input data for prediction"""
    total_transaction_amount: float = Field(..., description="Total transaction amount")
    avg_transaction_amount: float = Field(..., description="Average transaction amount")
    transaction_count: int = Field(..., description="Number of transactions")
    std_transaction_amount: float = Field(..., description="Standard deviation of amounts")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of transaction")
    transaction_month: int = Field(..., ge=1, le=12, description="Month of transaction")
    # Add more fields as needed based on your model features
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_transaction_amount": 1500.50,
                "avg_transaction_amount": 150.05,
                "transaction_count": 10,
                "std_transaction_amount": 50.25,
                "transaction_hour": 14,
                "transaction_month": 3
            }
        }

class PredictionResponse(BaseModel):
    """Output prediction"""
    customer_id: Optional[str] = None
    risk_probability: float = Field(..., description="Probability of being high risk")
    risk_category: str = Field(..., description="Low Risk or High Risk")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST123",
                "risk_probability": 0.75,
                "risk_category": "High Risk"
            }
        }