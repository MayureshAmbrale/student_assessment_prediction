import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Predictor", page_icon="📚")


try:
    with open(r"C:\Users\Mayuresh\OneDrive\Desktop\DSBDA mini project\final_model\assessment_score_model.pkl", "rb") as file:
        assessment_model = pickle.load(file)

    with open(r"C:\Users\Mayuresh\OneDrive\Desktop\DSBDA mini project\final_model\curriculum_level_model.pkl", "rb") as file:
        curriculum_model = pickle.load(file)

    with open(r"C:\Users\Mayuresh\OneDrive\Desktop\DSBDA mini project\final_model\label_encoders.pkl", "rb") as file:
        label_encoders = pickle.load(file)
        if not isinstance(label_encoders, dict):
            raise ValueError("label_encoders.pkl should contain a dictionary.")

    with open(r"C:\Users\Mayuresh\OneDrive\Desktop\DSBDA mini project\final_model\scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open(r"C:\Users\Mayuresh\OneDrive\Desktop\DSBDA mini project\final_model\label_encoder_level.pkl", "rb") as file:
        level_decoder = pickle.load(file)

    with open(r"C:\Users\Mayuresh\OneDrive\Desktop\DSBDA mini project\final_model\final_model\target_encoder_parent.pkl", "rb") as file:
        target_encoder = pickle.load(file)

except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading models or encoders: {e}")
    st.stop()

try:
    level_encoder = label_encoders.get("Level of Course")
    earning_encoder = label_encoders.get("Earning Class")
    material_encoder = label_encoders.get("Material Level")
    state_encoder = label_encoders.get("State")

    if not all([level_encoder, earning_encoder, material_encoder, state_encoder]):
        raise ValueError("Some required encoders are missing from the label_encoders dictionary.")

except Exception as e:
    st.error(f"❌ Encoder loading error: {e}")
    st.stop()

st.title("📊 Student Assessment & Curriculum Predictor")
st.write("Fill in the details below to get a predicted assessment score and recommended curriculum level.")


st.header("🔹 Student Information")
time_spent = st.number_input("Enter Time Spent per Day (hours):", min_value=0.0, max_value=24.0, value=6.0, step=0.1)
iq = st.number_input("Enter IQ of Student:", min_value=50, max_value=200, value=100, step=1)

st.header("🔹 Additional Information")
level_of_course = st.selectbox("Select Level of Course:", level_encoder.classes_)
earning_class = st.selectbox("Select Earning Class:", earning_encoder.classes_)
material_level = st.selectbox("Select Material Level:", material_encoder.classes_)
state = st.selectbox("Select State:", state_encoder.classes_)
parent_occupation = st.text_input("Enter Parent Occupation:")


if st.button('🚀 Predict Assessment Score and Curriculum Level'):
    try:
        if not parent_occupation:
            st.warning("⚠️ Please enter Parent Occupation.")
            st.stop()

        level_encoded = level_encoder.transform([level_of_course])[0]
        earning_encoded = earning_encoder.transform([earning_class])[0]
        material_encoded = material_encoder.transform([material_level])[0]
        state_encoded = state_encoder.transform([state])[0]

        parent_occupation_df = pd.DataFrame({'Parent Occupation': [parent_occupation]})
        occupation_encoded = target_encoder.transform(parent_occupation_df).iloc[0, 0]

        features = np.array([[iq, time_spent, level_encoded, material_encoded, earning_encoded, occupation_encoded, state_encoded]])
        scaled_input = scaler.transform(features)

        predicted_score = assessment_model.predict(scaled_input)[0]
        predicted_level = curriculum_model.predict(scaled_input)[0]
        decoded_level = level_decoder.inverse_transform([predicted_level])[0]

        st.success("✅ Prediction Successful!")
        st.subheader(f"📊 Predicted Assessment Score: **{predicted_score:.2f}**")
        st.subheader(f"🎯 Recommended Curriculum Level: **{decoded_level}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
