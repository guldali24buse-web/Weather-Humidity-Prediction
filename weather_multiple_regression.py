#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:06:26 2026

@author: buseguldalii
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('https://bilkav.com/odev_tenis.csv')
print(data)

# Converting categorical variables into numerical matrices (One-Hot Encoding)
df_encoded = pd.get_dummies(data, columns=['outlook', 'windy', 'play'], drop_first=True, dtype=int)

# Separating the target variable (y) and independent variables (X)
X = df_encoded.drop('humidity', axis=1)
y = df_encoded['humidity']

# Splitting the data into Train (70%) and Temporary (30%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)

# Splitting the Temporary set into Test (15%) and Prediction (15%) sets
X_test, X_pred, y_test, y_pred = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Adding a constant to the independent variables for the OLS report
X_train_sm = sm.add_constant(X_train)
model_ols = sm.OLS(y_train, X_train_sm).fit()
print("--- Initial OLS Model Summary ---")
print(model_ols.summary())

# Removing statistically insignificant variables with high P-values (Backward Elimination)
columns_to_drop = ['outlook_sunny', 'temperature']
X_train = X_train.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)
X_pred = X_pred.drop(columns=columns_to_drop)

# Creating a new OLS model with the updated data
X_train_sm_new = sm.add_constant(X_train)
model_ols_2 = sm.OLS(y_train, X_train_sm_new).fit()
print("\n--- Optimized OLS Model Summary ---")
print(model_ols_2.summary())

# Training the model using the 70% Train set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Evaluating model performance using the Test set (R-Squared)
y_test_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_test_pred)
print(f"\nModel Test Performance (R-Squared): {r2:.4f}")

# Making final predictions on unseen data (Prediction set)
final_predictions = regressor.predict(X_pred)
comparison_df = pd.DataFrame({'Actual Humidity': y_pred.values, 'Model Prediction': final_predictions})

print("\n--- Final Prediction Comparison ---")
print(comparison_df)




