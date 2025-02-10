import os
import uvicorn
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, FastAPI is running!"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000")) 
    uvicorn.run(app, host="0.0.0.0", port=port)

#(แก้ ติด CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# โหลดข้อมูล
food_df = pd.read_csv("data/pred_food.csv")

#Data Cleaning
columns_to_fill = ["Glycemic Index", "Calories", "Carbohydrates", "Protein", "Fat", "Fiber Content"]
for col in columns_to_fill:
    median_value = food_df[col].median()
    food_df[col] = food_df[col].replace(0, median_value)

# Scaling Data
scaler = MinMaxScaler()
food_df[columns_to_fill] = scaler.fit_transform(food_df[columns_to_fill])

# Feature Selection
X_food = food_df[["Calories", "Carbohydrates", "Protein", "Fat", "Fiber Content"]]
y_food = food_df["Glycemic Index"]
selector = SelectKBest(score_func=f_regression, k=3)
X_selected = selector.fit_transform(X_food, y_food)
selected_features = X_food.columns[selector.get_support()]

#Train KNN Model
knn_model = KNeighborsRegressor(n_neighbors=20, weights='distance')
knn_model.fit(X_selected, y_food)

#Calculate BMR & TDEE
def calculate_calories(age, gender, weight, height, activity_level):
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_multipliers = {
        "sedentary": 1.2,  # นั่งทำงาน ไม่ออกกำลังกาย
        "light": 1.375,    # ออกกำลังกายเล็กน้อย (1-3 วัน/สัปดาห์)
        "moderate": 1.55,  # ออกกำลังกายปานกลาง (3-5 วัน/สัปดาห์)
        "active": 1.725,   # ออกกำลังกายหนัก (6-7 วัน/สัปดาห์)
        "very active": 1.9 # นักกีฬา ออกกำลังกายหนักมาก
    }
    
    tdee = bmr * activity_multipliers.get(activity_level, 1.2)  #TDEE
    return round(tdee)

# 🎯 อัปเดต API `/recommend` ให้รองรับข้อมูลสุขภาพ
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
def recommend_food(user_input: UserInput):
    print(f"Received Data: {user_input}")  # Debug ค่าที่ได้รับจาก Frontend

    daily_calories = calculate_calories(user_input.age, user_input.gender, user_input.weight, user_input.height, user_input.activity_level)
    
    print(f"Calculated TDEE: {daily_calories}")  # Debug ค่า TDEE

    user_array = np.array([[daily_calories, user_input.carbohydrates, user_input.protein]])
    pred_gi = knn_model.predict(user_array)[0]

    print(f"Predicted GI: {pred_gi}")  # Debug ค่า Glycemic Index

    food_df["Predicted GI Diff"] = abs(food_df["Glycemic Index"] - pred_gi)
    recommended_foods = food_df.nsmallest(user_input.recommendations, "Predicted GI Diff")

    response_data = {
        "recommended_foods": recommended_foods[["Food Name", "Glycemic Index", "Calories", "Carbohydrates", "Protein", "Fat", "Fiber Content"]].to_dict(orient="records"),
        "daily_calories": daily_calories
    }
    
    print(f"Response Data: {response_data}")  # Debug ค่าที่จะส่งกลับไปให้ Frontend
    return response_data
