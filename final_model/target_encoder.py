import pandas as pd
import pickle
from category_encoders import TargetEncoder

df = pd.read_csv(r"C:\Users\SAMIKSHA\Desktop\final_model\processed_student_data.csv")

y = df["Assessment Score"] 

te = TargetEncoder(cols=["Parent Occupation"])
df["Parent Occupation Encoded"] = te.fit_transform(df[["Parent Occupation"]], y)

with open("target_encoder_parent.pkl", "wb") as f:
    pickle.dump(te, f)

print("✅ TargetEncoder saved!")
