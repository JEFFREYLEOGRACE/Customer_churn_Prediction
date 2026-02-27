📊 Customer Churn Prediction – Telecom Industry
📌 Project Overview
Customer churn is one of the biggest challenges in the telecom industry. This project focuses on analyzing customer data and building a Machine Learning model to predict whether a customer is likely to leave (churn) or stay with the company.
The objective of this project is to:
•	Perform data cleaning and preprocessing
•	Conduct Exploratory Data Analysis (EDA)
•	Build and train a predictive model
•	Evaluate model performance
•	Provide actionable business insights
________________________________________
📂 Dataset Description
The dataset used in this project contains telecom customer information such as:
•	Customer demographics
•	Account information
•	Service subscriptions
•	Monthly and total charges
•	Tenure
•	Churn status (Target Variable)
Target Variable:
•	Churn
o	Yes → Customer left the company
o	No → Customer stayed
________________________________________
🛠️ Technologies Used
•	Python
•	Pandas – Data manipulation
•	NumPy – Numerical operations
•	Matplotlib – Data visualization
•	Scikit-learn – Machine Learning
•	Jupyter Notebook – Development environment

🔷 2️⃣ Complete Project Workflow
________________________________________
✅ Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
Libraries used:
•	Pandas → Data manipulation
•	NumPy → Numerical operations
•	Matplotlib/Seaborn → Visualization
•	Scikit-learn → Preprocessing & evaluation
•	TensorFlow/Keras → Deep Learning model
________________________________________
✅ Step 2: Load Dataset
df = pd.read_csv("Customer_churn.csv")
Dataset contains:
•	Customer demographics
•	Services subscribed
•	Monthly & total charges
•	Churn status (Target variable)
________________________________________
✅ Step 3: Data Cleaning
🔹 Remove Unnecessary Column
df.drop('customerID', axis='columns', inplace=True)
CustomerID is not useful for prediction.
________________________________________
🔹 Handle Missing Values
TotalCharges had blank spaces.
df1 = df[df.TotalCharges!=' ']
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
Converted to numeric and removed invalid rows.
________________________________________
✅ Step 4: Exploratory Data Analysis (EDA)
📊 Tenure vs Churn
Customers with low tenure are more likely to churn.
📊 Monthly Charges vs Churn
Higher monthly charges → Higher churn probability.
This gives business insight:
Long-term customers are loyal; high charges increase churn risk.
________________________________________
✅ Step 5: Data Preprocessing
🔹 Replace Service Labels
df1.replace('No internet service','No', inplace=True)
df1.replace('No phone service','No', inplace=True)
________________________________________
🔹 Convert Yes/No to 1/0
df1[col].replace({'Yes':1,'No':0}, inplace=True)
________________________________________
🔹 Encode Gender
df1['gender'].replace({'Female':1,'Male':0}, inplace=True)
________________________________________
🔹 One-Hot Encoding
df2 = pd.get_dummies(df1, columns=['InternetService','Contract','PaymentMethod'], dtype=int)
Categorical features converted into numeric format.
________________________________________
✅ Step 6: Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(...)
Scaling helps neural networks train faster and better.
________________________________________
✅ Step 7: Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
•	80% training
•	20% testing
________________________________________
✅ Step 8: Build ANN Model
model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
Architecture:
•	Input layer: 26 features
•	Hidden layer: 20 neurons (ReLU)
•	Output layer: 1 neuron (Sigmoid)
Loss Function:
binary_crossentropy
Optimizer:
adam
________________________________________
✅ Step 9: Model Training
model.fit(X_train, Y_train, epochs=100)
Model learns customer behavior patterns.
________________________________________
✅ Step 10: Evaluation
🔹 Accuracy
model.evaluate(X_test, Y_test)
🔹 Classification Report
classification_report(Y_test, y_pred)
🔹 Confusion Matrix
•	True Positive
•	True Negative
•	False Positive
•	False Negative
________________________________________
🔷 3️⃣ Business Interpretation
•	Customers with short tenure → High churn
•	High monthly charges → High churn
•	Contract type significantly affects churn
Month-to-month contracts → Highest churn
