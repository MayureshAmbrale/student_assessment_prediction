import pickle
import numpy as np
import pandas as pd
import os

base_path = r"C:\Users\Mayuresh\OneDrive\Desktop\DSBDA mini project\final_model\final_model"

with open(os.path.join(base_path, "assessment_score_model.pkl"), "rb") as file:
    rf_model = pickle.load(file)

with open(os.path.join(base_path, "curriculum_level_model.pkl"), "rb") as file:
    level_model = pickle.load(file)

with open(os.path.join(base_path, "scaler.pkl"), "rb") as file:
    scaler = pickle.load(file)

with open(os.path.join(base_path, "label_encoders.pkl"), "rb") as file:
    label_encoders = pickle.load(file)

with open(os.path.join(base_path, "label_encoder_level.pkl"), "rb") as file:
    label_encoder_level = pickle.load(file)

test_data = np.array([[120, 3.5, "Advanced", "Intermediate", "High", "Engineer", "Maharashtra"]])

feature_names = [
    "IQ of Student", "Time Spent per Day", "Level of Course",
    "Material Level", "Earning Class", "Parent Occupation", "State"
]
test_df = pd.DataFrame(test_data, columns=feature_names)

for col in ["Level of Course", "Material Level", "Earning Class", "State", "Parent Occupation"]:
    if col in label_encoders:
        encoder = label_encoders[col]

        print(f"Processing column: {col}, Encoder Type: {type(encoder)}") 

        if hasattr(encoder, "classes_"): 
            known_labels = set(encoder.classes_)
            most_frequent_label = encoder.classes_[0] 

            test_df[col] = test_df[col].apply(lambda x: x if x in known_labels else most_frequent_label)
            test_df[col] = encoder.transform(test_df[col])

        elif isinstance(encoder, type(label_encoders["Parent Occupation"])):  
            test_df[col] = encoder.transform(test_df[[col]])  

        else:
            raise ValueError(f"Unhandled encoder type for {col}")

test_df = test_df.astype(float)

test_df = pd.DataFrame(scaler.transform(test_df), columns=feature_names)

predicted_score = rf_model.predict(test_df)[0]

predicted_level_encoded = level_model.predict(test_df)[0]
predicted_level = label_encoder_level.inverse_transform([predicted_level_encoded])[0]

print(f"Predicted Assessment Score: {predicted_score:.2f}")
print(f"Predicted Level: {predicted_level}")
