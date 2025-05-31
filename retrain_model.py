import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Loading the original training dataset that contains baseline data for training model.
df_train = pd.read_csv('health_dataset.csv')

# If this file exists, its data will be combined with the original training data.
if os.path.exists('prediction_inputs_log.csv'):
    df_new = pd.read_csv('prediction_inputs_log.csv')
    if set(df_new.columns) == set(df_train.columns):
        df_all = pd.concat([df_train, df_new], ignore_index=True)
        print(f"Combined original and new data: {len(df_all)} rows.")
    else:
        print("Warning: Columns in prediction_inputs_log.csv do not match health_dataset.csv. Using only original data.")
        df_all = df_train
else:
    df_all = df_train
    print("No prediction_inputs_log.csv found. Using only original data.")

# Saving the (potentially combined) data to a new CSV file named 'combined_training_data.csv'.
df_all.to_csv('combined_training_data.csv', index=False)

# These are the input variables that the model will use to make predictions.
feature_columns = ['sleep_hours', 'exercise_hours', 'screen_time', 'social_interaction', 'age', 'work_hours']
target_column = 'stress_level'

X = df_all[feature_columns]
y = df_all[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, 'model.pkl')
print("Model retrained and saved as model.pkl.")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Retrained model accuracy: {accuracy:.2f}")
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))
