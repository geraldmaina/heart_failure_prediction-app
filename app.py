# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px

# Load the dataset
url = "https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/data"
st.title("Heart Failure Prediction Data Analysis")

# Introduction
st.header("Introduction")
st.write(
    "This Streamlit app analyzes the Heart Failure Prediction dataset. "
    "The dataset contains various clinical features, and the goal is to predict "
    "whether a patient is at risk of heart failure."
)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    return data

df = load_data()

# Display raw data
st.subheader("Raw Data")
st.dataframe(df)

# Summary Statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Visualizations
st.header("Data Visualizations")

# Correlation Matrix
st.subheader("Correlation Matrix")
fig_corr = px.imshow(df.corr(), title="Correlation Matrix")
st.plotly_chart(fig_corr)

# Age Distribution
st.subheader("Age Distribution")
fig_age_dist = px.histogram(df, x="age", title="Age Distribution")
st.plotly_chart(fig_age_dist)

# Gender Distribution
st.subheader("Gender Distribution")
fig_gender_dist = px.pie(df, names="sex", title="Gender Distribution")
st.plotly_chart(fig_gender_dist)

# Heart Failure Events
st.subheader("Heart Failure Events")
fig_heart_failure = px.pie(df, names="DEATH_EVENT", title="Heart Failure Events")
st.plotly_chart(fig_heart_failure)

# Feature Selection
st.header("Feature Selection and Model Training")

# Select Features and Target
features = df.drop("DEATH_EVENT", axis=1)
target = df["DEATH_EVENT"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Model Evaluation
st.subheader("Model Evaluation")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_conf_matrix = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), x=["0", "1"], y=["0", "1"])
st.plotly_chart(fig_conf_matrix)

# Classification Report
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Conclusion
st.header("Conclusion")
st.write(
    "In this analysis, we explored the Heart Failure Prediction dataset, performed visualizations, "
    "and trained a Random Forest Classifier to predict heart failure events. "
    "The model achieved a certain accuracy, but further fine-tuning and feature engineering "
    "could enhance its performance."
)

# Recommendations
st.header("Recommendations")
st.write(
    "For a more comprehensive analysis and accurate predictions, it's recommended to:"
    "\n1. Explore feature engineering techniques."
    "\n2. Fine-tune the model hyperparameters."
    "\n3. Consider other machine learning algorithms."
    "\n4. Validate the model on a larger dataset."
)

