import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from pathlib import Path

def train_and_save_model():
    """Model train karne ka function proper structure ke saath"""
    
    # ✅ Path handle karo
    current_dir = Path(__file__).parent
    data_path = current_dir / 'data' / 'housing.csv'
    models_dir = current_dir / 'models'
    
    # Check karo data file exist karti hai
    if not data_path.exists():
        print(f"❌ Data file not found at {data_path}")
        return
    
    # 1. Dataset Load karo
    print("📂 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Features aur target split karo
    X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
    y = df['price']
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    # 4. Model create aur train
    print("🤖 Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5. Predictions
    y_pred = model.predict(X_test)
    
    # 6. Evaluation
    print("\n📊 Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):,.2f}")
    print(f"R2 Score (Accuracy): {r2_score(y_test, y_pred):.4f}")
    
    # 7. Model save
    os.makedirs(models_dir, exist_ok=True)
    model_path = models_dir / 'model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved at: {model_path}")
    
    return model

def predict_price(area, bedrooms, bathrooms, stories, parking, model=None):
    """Prediction function"""
    if model is None:
        # Agar model nahi diya toh load karo
        current_dir = Path(__file__).parent
        model_path = current_dir / 'models' / 'model.pkl'
        model = joblib.load(model_path)
    
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == "__main__":
    # Run training
    model = train_and_save_model()
    
    # Example prediction
    if model:
        print("\n📌 Example Prediction:")
        price = predict_price(9000, 3, 2, 2, 1, model)
        print(f"Predicted Price: ₹ {price:,.2f}")