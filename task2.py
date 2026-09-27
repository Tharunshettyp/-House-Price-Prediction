import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# STEP 1: LOAD DATASET
df = pd.read_csv("housing.csv")
print("First 5 rows:")
print(df.head())

# STEP 2: DATASET INFORMATION
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nStatistical Summary:")
print(df.describe())

# STEP 3: SEPARATE FEATURES AND TARGET
X = df.drop("price", axis=1)
y = df["price"]
print("\nFeatures Shape:")
print(X.shape)
print("\nTarget Shape:")
print(y.shape)

# STEP 4: ENCODE CATEGORICAL FEATURES
X = pd.get_dummies(X, drop_first=True)
print("\nFeatures After Encoding:")
print(X.head())
print("\nShape After Encoding:")
print(X.shape)

# STEP 5: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining Data Shape:")
print(X_train.shape)
print("\nTesting Data Shape:")
print(X_test.shape)
print("\nTraining Target Shape:")
print(y_train.shape)
print("\nTesting Target Shape:")
print(y_test.shape)

# STEP 6: TRAIN LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)
print("\nLinear Regression model trained successfully!")

# STEP 7: MAKE PREDICTIONS
y_pred = model.predict(X_test)
print("\nPredicted House Prices:")
print(y_pred[:10])

# STEP 8: MODEL EVALUATION
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# STEP 9: ACTUAL VS PREDICTED PRICES
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
plt.show()

# STEP 10: SAVE THE TRAINED MODEL
joblib.dump(model, "house_price_linear_regression.pkl")
print("\nModel saved successfully!")