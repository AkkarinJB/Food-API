import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

def train_knn():
    food_df = pd.read_csv("data/pred_food.csv")

    # Data Cleaning
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

    # Train Model
    knn_model = KNeighborsRegressor(n_neighbors=20, weights='distance')
    knn_model.fit(X_selected, y_food)

    # Save Model
    with open("knn_model.pkl", "wb") as f:
        pickle.dump((knn_model, food_df, selected_features), f)
    
    return knn_model, food_df, selected_features

def recommend_food(knn_model, food_df, selected_features, user_input, daily_calories):
    user_array = np.array([[daily_calories, user_input.carbohydrates, user_input.protein]])
    try:
        pred_gi = knn_model.predict(user_array)[0]
    except Exception as e:
        raise ValueError(f"Error in prediction: {str(e)}")

    food_df["Predicted GI Diff"] = abs(food_df["Glycemic Index"] - pred_gi)
    recommended_foods = food_df.nsmallest(user_input.recommendations, "Predicted GI Diff")

    return recommended_foods[["Food Name", "Glycemic Index", "Calories", "Carbohydrates", "Protein", "Fat", "Fiber Content"]].to_dict(orient="records")
