# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Sidebar
st.sidebar.title("Heart Failure Prediction")
age = st.sidebar.slider("Select Age", min_value=29, max_value=100, value=50)
ejection_fraction = st.sidebar.slider("Select Ejection Fraction", min_value=14, max_value=80, value=50)
serum_creatinine = st.sidebar.slider("Select Serum Creatinine", min_value=0.5, max_value=9.4, value=1.0)
serum_sodium = st.sidebar.slider("Select Serum Sodium", min_value=113, max_value=148, value=135)

# Features
features = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']
X = data[features]
y = data['DEATH_EVENT']
# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([[age, ejection_fraction, serum_creatinine, serum_sodium]])

# Main App
st.title("Heart Failure Prediction App")

# Introduction
st.write("This app predicts the likelihood of heart failure based on clinical data.")

# User Input
st.sidebar.header("User Input Features")
st.sidebar.write(f"Age: {age}")
st.sidebar.write(f"Ejection Fraction: {ejection_fraction}")
st.sidebar.write(f"Serum Creatinine: {serum_creatinine}")
st.sidebar.write(f"Serum Sodium: {serum_sodium}")

# Prediction
st.header("Prediction Result")
st.write("Outcome: ", "Heart Failure" if prediction[0] == 1 else "No Heart Failure")

# Data Analysis
st.header("Data Analysis")


# Pairplot
st.subheader("Pairplot")
pairplot = sns.pairplot(data, hue="DEATH_EVENT")
st.pyplot(pairplot)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
correlation_matrix = data.corr()
figure, ax = plt.subplots(figsize=(8, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
st.pyplot(figure)

#Additional visualizations
st.subheader("Additional Visualization")

#Boxplot for age and death event
st.write("Boxplot for Age and Death Event")
fig, ax = plt.subplots()
sns.boxplot(x="DEATH_EVENT", y="age", data=data, ax=ax)
st.pyplot(fig, clear_figure=True)


#Countplot for Death Event
st.write("Countplot for Death Event")
fig, ax = plt.subplots()
sns.countplot(x="DEATH_EVENT", data=data, ax=ax)
st.pyplot(fig, clear_figure=True)

#model evaluation metrics
st.header("Model Evaluation Metrics")

# Confusion Matrix
st.subheader("Confusion Matrix")
y_pred = model.predict(X_test)
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
st.pyplot(fig)

#Classification Report
st.subheader("Classification Report")
classification_rep = classification_report(y_test, y_pred)
st.text(classification_rep)

# Conclusion
st.title("Conclusion")
st.write("In this data analysis and prediction project, we utilized clinical data to predict the likelihood of heart failure. The machine learning model, based on features such as age, ejection fraction, serum creatinine, and serum sodium, showed promising results. The interactive Streamlit app allows users to input their information and receive predictions in real-time.")

# Acknowledgment
st.write("Acknowledgment: This project uses the Heart Failure Prediction dataset available on Kaggle.")