
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Generate sample house data
np.random.seed(42)
n_samples = 200

area = np.random.randint(500, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 40, n_samples)

# Sample price formula with some random noise
price = (
    50000
    + area * 150
    + bedrooms * 10000
    + bathrooms * 15000
    - age * 1200
    + np.random.normal(0, 20000, n_samples)
)

# 2. Create a DataFrame
data = pd.DataFrame({
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Age": age,
    "Price": price
})

# 3. Separate features and target
X = data[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = data["Price"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("House Price Prediction Model")
print("-" * 35)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Mean Absolute Error: {mae:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# 7. Save the trained model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved as house_price_model.pkl")

# 8. Display sample predictions
sample = pd.DataFrame({
    "Area": [1500, 2500],
    "Bedrooms": [3, 4],
    "Bathrooms": [2, 3],
    "Age": [10, 5]
})

sample_predictions = model.predict(sample)

print("\nSample Predictions:")
for i, predicted_price in enumerate(sample_predictions):
    print(
        f"House {i + 1}: "
        f"${predicted_price:,.2f}"
    )
