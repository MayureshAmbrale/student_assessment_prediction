import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
file_path = r"synthetic_student_data.csv"
df = pd.read_csv(file_path)

# Sidebar filters with better structure
st.sidebar.header("🎛️ Filter Student Data")

# Gender filter
selected_genders = st.sidebar.multiselect(
    "Select Gender(s)", 
    options=df["Gender"].unique().tolist(), 
    default=df["Gender"].unique().tolist()
)

# Course Level filter
selected_levels = st.sidebar.multiselect(
    "Select Course Level(s)", 
    options=df["Level of Course"].unique().tolist(), 
    default=df["Level of Course"].unique().tolist()
)

# Course Name filter
selected_courses = st.sidebar.multiselect(
    "Select Course(s)", 
    options=df["Course Name"].unique().tolist(), 
    default=df["Course Name"].unique().tolist()
)

# Assessment Score filter
score_min = int(df["Assessment Score"].min())
score_max = int(df["Assessment Score"].max())
score_range = st.sidebar.slider(
    "Assessment Score Range", 
    min_value=score_min, max_value=score_max, 
    value=(score_min, score_max)
)

# Time Spent per Day filter
time_min = float(df["Time Spent per Day"].min())
time_max = float(df["Time Spent per Day"].max())
time_range = st.sidebar.slider(
    "Time Spent per Day (hours)", 
    min_value=round(time_min, 1), max_value=round(time_max, 1), 
    value=(round(time_min, 1), round(time_max, 1))
)

# Apply filters
filtered_df = df[
    (df["Gender"].isin(selected_genders)) &
    (df["Level of Course"].isin(selected_levels)) &
    (df["Course Name"].isin(selected_courses)) &
    (df["Assessment Score"] >= score_range[0]) &
    (df["Assessment Score"] <= score_range[1]) &
    (df["Time Spent per Day"] >= time_range[0]) &
    (df["Time Spent per Day"] <= time_range[1])
]

# Dashboard Title
st.title("📊 Student Performance Dashboard")

# First Row of Plots
col1, col2, col3 = st.columns(3)
with col1:
    fig1 = px.histogram(filtered_df, x="Assessment Score", nbins=20, title="Assessment Score Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.box(filtered_df, x="Gender", y="IQ of Student", title="IQ Distribution by Gender")
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    fig3 = px.pie(filtered_df, names="Level of Student", title="Student Level Distribution")
    st.plotly_chart(fig3, use_container_width=True)

# Second Row of Plots
col4, col5, col6 = st.columns(3)
with col4:
    fig4 = px.scatter(filtered_df, x="Time Spent per Day", y="Assessment Score", color="Gender", title="Time Spent vs Score")
    st.plotly_chart(fig4, use_container_width=True)

with col5:
    fig5 = px.bar(filtered_df.groupby("Course Name")["Assessment Score"].mean().reset_index(),
                  x="Course Name", y="Assessment Score", title="Avg Score by Course")
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    fig6 = px.violin(filtered_df, y="IQ of Student", x="Level of Course", box=True, points="all",
                     title="IQ by Course Level")
    st.plotly_chart(fig6, use_container_width=True)
