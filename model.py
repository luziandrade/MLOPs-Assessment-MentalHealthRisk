# Import required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class StressModel:
    def __init__(self, data_path='health_dataset.csv'):
        """Initialize model and trigger training on instantiation."""
        self.data_path = data_path
        self.model = None
        self.train_model()

    def load_data(self):
        """Load dataset and split into features/target."""
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=['stress_level'])
        y = df['stress_level']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        """Train logistic regression model and evaluate performance."""
        X_train, X_test, y_train, y_test = self.load_data()
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(X_train, y_train)

        # Print accuracy metrics
        print(f"Train Accuracy: {self.model.score(X_train, y_train):.2f}")
        print(f"Test Accuracy: {self.model.score(X_test, y_test):.2f}")

    def predict(self, features):
        """Make prediction using trained model."""
        feature_names = ['sleep_hours', 'exercise_hours', 'screen_time', 'social_interaction', 'age', 'work_hours']
        X_new = pd.DataFrame([features], columns=feature_names)
        return int(self.model.predict(X_new)[0])

# if __name__ == "__main__":
#     model = StressModel()
#     sample_features = [6.5, 1.0, 8.0, 2.0, 30, 8]  
#     prediction = model.predict(sample_features)
#     print("Predicted Stress Level:", "High" if prediction == 1 else "Low")
