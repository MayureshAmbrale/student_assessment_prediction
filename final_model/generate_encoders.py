import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\Mayuresh\OneDrive\Desktop\DSBDA mini project\final_model\final_model\processed_student_data.csv")

label_encoders = {}

for col in ["Level of Course", "Earning Class", "Material Level", "State", "Parent Occupation"]:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("label_encoder_level.pkl", "wb") as f:
    pickle.dump(label_encoders["Level of Course"], f)

print("✅ Encoders saved successfully!")
