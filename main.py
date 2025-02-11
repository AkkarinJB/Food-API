import os
import uvicorn
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import train_knn, recommend_food

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Load or Train Model
model_path = "./knn_model.joblib"

def load_or_train_model():
    """โหลดโมเดลจากไฟล์ หรือ train ใหม่หากไม่มีไฟล์หรือไฟล์เสียหาย"""
    if os.path.exists(model_path):
        try:
            print("Loading model from file...")
            model_data = joblib.load(model_path)
            if not isinstance(model_data, tuple) or len(model_data) != 3:
                print("Invalid model format. Retraining...")
                return train_knn_and_save()
            knn_model, food_df, selected_features = model_data
            print("Model loaded successfully!")
            return knn_model, food_df, selected_features
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")
            return train_knn_and_save()
    
    return train_knn_and_save()

def train_knn_and_save():
    print("Training new model...")
    knn_model, food_df, selected_features = train_knn()
    joblib.dump((knn_model, food_df, selected_features), model_path)
    print("Model trained and saved successfully!")
    return knn_model, food_df, selected_features

# โหลดโมเดลเมื่อเริ่มต้น
knn_model, food_df, selected_features = load_or_train_model()

if knn_model is None or food_df is None or selected_features is None:
    raise RuntimeError("Failed to load or train model.")

# API Request Model
class FoodRequest(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    activity_level: str
    
    carbohydrates: float
    protein: float
    calories:float

    recommendations: int = 6

@app.post("/recommend")
def recommend_food_api(request: FoodRequest):
    if knn_model is None:
        raise HTTPException(status_code=500, detail="Model not available")
    return recommend_food(knn_model, food_df, selected_features, request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
