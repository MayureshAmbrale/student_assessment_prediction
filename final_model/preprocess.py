import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\Mayuresh\OneDrive\Desktop\DSBDA mini project\final_model\final_model\synthetic_student_data.csv")

df["Original IQ"] = df["IQ of Student"]
df["Original Assessment Score"] = df["Assessment Score"]

scaler = StandardScaler()
df[["IQ of Student", "Assessment Score", "Time Spent per Day"]] = scaler.fit_transform(df[["IQ of Student", "Assessment Score", "Time Spent per Day"]])

df.to_csv("processed_student_data.csv", index=False)

print("✅ Data Preprocessing Complete! Processed file saved.")
