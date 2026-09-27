#House Price Prediction

## 📌 Project Overview

This project is part of my **Machine Learning Internship at IncodeVision**.

The objective of this task is to build a Machine Learning model that predicts **house prices** based on different property features such as area, number of bedrooms, bathrooms, stories, parking, furnishing status, and other factors.

A **Linear Regression** model was trained and evaluated to understand how well the selected features can predict house prices.

## 🎯 Objectives

- Load and understand the housing dataset
- Perform data preprocessing
- Check for missing values
- Encode categorical variables
- Split the dataset into training and testing sets
- Train a Linear Regression model
- Make predictions on the test data
- Evaluate model performance
- Visualize actual vs predicted house prices
- Save the trained Machine Learning model

## 📂 Dataset

The dataset contains **545 records and 13 columns**.

### Features

| Feature | Description |
|---|---|
| `area` | Area of the house |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `mainroad` | Whether the house is connected to the main road |
| `guestroom` | Whether the house has a guest room |
| `basement` | Whether the house has a basement |
| `hotwaterheating` | Whether hot water heating is available |
| `airconditioning` | Whether air conditioning is available |
| `parking` | Number of parking spaces |
| `prefarea` | Whether the house is in a preferred area |
| `furnishingstatus` | Furnishing status of the house |

### Target

`price` - The predicted house price.

## 🔧 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Linear Regression
- Data Preprocessing
- Machine Learning

## ⚙️ Project Workflow

### 1. Data Loading

The housing dataset was loaded using Pandas and inspected to understand its structure, columns, data types, and values.

### 2. Data Preprocessing

The dataset was checked for missing values and categorical features were converted into numerical form using encoding techniques.

### 3. Feature and Target Selection

The input features were separated from the target variable:

- **Features (X):** Property-related attributes
- **Target (y):** House price

### 4. Train-Test Split

The dataset was divided into training and testing sets.

- Training data: **436 records**
- Testing data: **109 records**

### 5. Model Training

A **Linear Regression** model from Scikit-learn was trained using the training dataset.

### 6. Prediction

The trained model was used to predict house prices for the test dataset.

### 7. Model Evaluation

The model was evaluated using:

- Mean Squared Error (MSE)
- R² Score

## 📊 Model Performance

The Linear Regression model achieved the following results:

| Metric | Result |
|---|---:|
| Mean Squared Error (MSE) | 1,754,318,687,330.6643 |
| R² Score | 0.6529 |
| R² Percentage | 65.29% |

The R² score of approximately **65.29%** indicates that the model explains around 65% of the variation in house prices within the test dataset.

## 📈 Visualization

An **Actual vs Predicted House Prices** visualization was created to compare the real house prices with the prices predicted by the model.

This visualization helps understand how closely the predictions match the actual values.

## 💾 Model Saving

The trained model was saved using `joblib` so that it can be reused later without retraining.

Example:

```python
joblib.dump(model, "house_price_model.pkl")
