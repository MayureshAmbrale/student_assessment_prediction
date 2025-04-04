import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import category_encoders as ce
import xgboost as xgb

df = pd.read_csv(r"C:\Users\Mayuresh\OneDrive\Desktop\DSBDA mini project\final_model\processed_student_data.csv")

selected_features = ["IQ of Student", "Time Spent per Day", "Level of Course", "Material Level", "Earning Class", "Parent Occupation", "State"]
target_score = "Assessment Score"
target_level = "Level of Student"

df_encoded = df.copy()
target_encoder = ce.TargetEncoder(cols=["Parent Occupation"])
df_encoded["Parent Occupation"] = target_encoder.fit_transform(df_encoded["Parent Occupation"], df_encoded["Assessment Score"])

with open("target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

label_encoders = {}
for col in ["Level of Course", "Material Level", "Earning Class", "State"]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le 

X = df_encoded[selected_features]
y_score = df_encoded[target_score]
y_level = df_encoded[target_level]

label_encoder_level = LabelEncoder()
y_level_encoded = label_encoder_level.fit_transform(y_level)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train_score, y_test_score = train_test_split(X_scaled, y_score, test_size=0.2, random_state=42)
_, _, y_train_level, y_test_level = train_test_split(X_scaled, y_level_encoded, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train_score)

best_rf_model = grid_search.best_estimator_

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
xgb_model.fit(X_train, y_train_score)

level_model = RandomForestClassifier(random_state=42)
level_model.fit(X_train, y_train_level)

y_pred_score_rf = best_rf_model.predict(X_test)
y_pred_score_xgb = xgb_model.predict(X_test)
y_pred_level = level_model.predict(X_test)

mae_rf = mean_absolute_error(y_test_score, y_pred_score_rf)
mse_rf = mean_squared_error(y_test_score, y_pred_score_rf)
r2_rf = r2_score(y_test_score, y_pred_score_rf)

mae_xgb = mean_absolute_error(y_test_score, y_pred_score_xgb)
mse_xgb = mean_squared_error(y_test_score, y_pred_score_xgb)
r2_xgb = r2_score(y_test_score, y_pred_score_xgb)

accuracy = accuracy_score(y_test_level, y_pred_level)

print("\n📊 Assessment Score Model - RandomForest:")
print(f"MAE: {mae_rf}")
print(f"MSE: {mse_rf}")
print(f"R² Score: {r2_rf}\n")

print("📊 Assessment Score Model - XGBoost:")
print(f"MAE: {mae_xgb}")
print(f"MSE: {mse_xgb}")
print(f"R² Score: {r2_xgb}\n")

print("🎯 Curriculum Level Model:")
print(f"Accuracy: {accuracy}\n")

cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
cv_scores = cross_val_score(level_model, X_scaled, y_level_encoded, cv=cv, scoring='accuracy')

print("🎯 Curriculum Level Model - Cross-validation Accuracy Scores:")
print(cv_scores)
print(f"Average CV Accuracy: {cv_scores.mean()}\n")

if os.path.exists("encoders.pkl"):
    os.remove("encoders.pkl")
    print("🗑️ Old encoders.pkl removed. Saving new encoders...")

with open("encoders.pkl", "wb") as file:
    pickle.dump(label_encoders, file)

with open("label_encoders.pkl", "rb") as file:
    saved_encoders = pickle.load(file)

print("✅ Successfully Saved Encoders:", saved_encoders.keys())

with open("assessment_score_model.pkl", "wb") as file:
    pickle.dump(best_rf_model, file)

with open("curriculum_level_model.pkl", "wb") as file:
    pickle.dump(level_model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

with open("label_encoder_level.pkl", "wb") as file:
    pickle.dump(label_encoder_level, file)

print("✅ Models and Encoders Trained & Saved Successfully!")
