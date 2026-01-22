import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# 🔹 1. Dataset Load karo (Ye CSV file se data load krr rha hain)
df = pd.read_csv("data/housing.csv")

# 🔹 2. Input features (X) aur Output target (y) split karo
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = df['price']

# 🔹 3. Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 4. Model create aur train karo
model = LinearRegression()
model.fit(X_train, y_train)

# 🔹 5. Prediction karo test data par
y_pred = model.predict(X_test)

# 🔹 6. Accuracy check karo
print("\n🔍 Model Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score (Accuracy):", r2_score(y_test, y_pred))

# 🔹 7. Model save karo models/ folder me
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("\n✅ Model trained and saved as models/model.pkl")

# 🔹 8. Custom input se prediction
def predict_price(area, bedrooms, bathrooms, stories, parking):
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
    prediction = model.predict(input_data)
    return prediction[0]

# 🔹 9. Example prediction
print("\n📌 Predicted Price:", predict_price(9000, 3, 2, 2, 1))