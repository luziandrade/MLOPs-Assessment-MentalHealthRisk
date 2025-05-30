import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class StressModel:
    def __init__(self, model_path='model.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
        self.model = joblib.load(model_path)

    def load_data(self):
        """Load dataset and split into features/target."""
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=['stress_level'])
        y = df['stress_level']
        return train_test_split(X, y, test_size=0.2, random_state=42)


    def predict(self, features):
        """Make prediction using trained model."""
        feature_names = ['sleep_hours', 'exercise_hours', 'screen_time', 'social_interaction', 'age', 'work_hours']
        X_new = pd.DataFrame([features], columns=feature_names)
        return int(self.model.predict(X_new)[0])