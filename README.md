# Weather-Humidity-Prediction
A machine learning model which I used Scikit-Learn and Statsmodels to predict humidity based on weather conditions.
# Weather Humidity Prediction using Multiple Linear Regression

## Overview
This project implements a Multiple Linear Regression model to predict humidity levels based on various weather conditions. It demonstrates an end-to-end machine learning pipeline, emphasizing statistical feature selection and data preprocessing.

## Methodology & Workflow

1. **Data Preprocessing:**
   
   Converted categorical variables (outlook, windy, play) into numerical matrices using **One-Hot Encoding**.
   Implemented `drop_first=True` to prevent the **Dummy Variable Trap** (Multicollinearity).
     
2. **Statistical Feature Selection (Backward Elimination):**

    <img width="584" height="238" alt="Ekran Resmi 2026-10-14 10 58 41" src="https://github.com/user-attachments/assets/d9078ab4-5530-44be-87bc-564c7e06d5cd" />


   Utilized the **Ordinary Least Squares (OLS)** method via the `statsmodels` library.
   Evaluated the p-values ($P>\vert{}t\vert{}$) of independent variables.
   Eliminated statistically insignificant variables (p-value > 0.05) such as `outlook_sunny` and `temperature` to optimize the model.
     
3. **Model Training & Evaluation:**

    <img width="539" height="118" alt="Ekran Resmi 2026-10-14 10 59 31" src="https://github.com/user-attachments/assets/3efb5dc4-0c79-4027-a311-5e9b3753ffa0" />


   Split the dataset into Training (70%), Testing (15%), and Validation (15%) sets.
   Trained the optimized model using `scikit-learn`'s `LinearRegression`.
   Measured model performance using the **R-Squared ($R^2$)** metric on the test set.
     
4. **Final Prediction:**
   
   Simulated a real-world scenario by feeding unseen validation data into the trained model and comparing predicted values with actual humidity levels.

## Technologies & Libraries Used

    <img width="429" height="87" alt="Ekran Resmi 2026-10-14 11 04 25" src="https://github.com/user-attachments/assets/d63a8b6e-dfe0-493e-8ed4-0da13705abec" />


* **Python 3.x**
* **Pandas & NumPy** (Data manipulation and matrix operations)
* **Statsmodels** (Statistical modeling and OLS reports)
* **Scikit-Learn** (Machine learning algorithms and evaluation metrics)

## Author
**Buse Güldalı**  
*Industrial & Mathematical Engineering*
