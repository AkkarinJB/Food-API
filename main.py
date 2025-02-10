import os
import uvicorn
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import train_knn, recommend_food

# Initialize FastAPI app
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
model_path = "./knn_model.pkl"
try:
    with open(model_path, "rb") as f:
        knn_model, food_df, selected_features = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Training model...")
    knn_model, food_df, selected_features = train_knn()
    with open(model_path, "wb") as f:
        pickle.dump((knn_model, food_df, selected_features), f)

# Health Check API
@app.get("/health")
def health_check():
    return {"status": "API is running", "model_loaded": knn_model is not None}

# Calculate BMR & TDEE
def calculate_calories(age, gender, weight, height, activity_level):
    activity_multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9
    }

    if activity_level not in activity_multipliers:
        raise HTTPException(status_code=400, detail="Invalid activity level. Choose from: 'sedentary', 'light', 'moderate', 'active', 'very active'")

    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    return round(bmr * activity_multipliers[activity_level])

# User Input Model
class UserInput(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    activity_level: str
    carbohydrates: float
    protein: float
    recommendations: int = 6

@app.post("/recommend")
def get_recommendation(user_input: UserInput):
    try:
        # คำนวณ Daily Calories ก่อน
        daily_calories = calculate_calories(
            user_input.age, user_input.gender, user_input.weight,
            user_input.height, user_input.activity_level
        )

        # ส่ง daily_calories ไปที่ recommend_food()
        recommended_foods = recommend_food(knn_model, food_df, selected_features, user_input, daily_calories)

        return {
            "recommended_foods": recommended_foods,
            "daily_calories": daily_calories
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
