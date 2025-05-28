import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df_train = pd.read_csv('health_dataset.csv')

if os.path.exists('prediction_inputs_log.csv'):
    df_new = pd.read_csv('prediction_inputs_log.csv')
    if set(df_new.columns) == set(df_train.columns):
        df_all = pd.concat([df_train, df_new], ignore_index=True)
        print(f"Combined original and new data: {len(df_all)} rows.")
    else:
        print("Warning: Columns in rediction_inputs_log.csv do not match health_dataset.csv. Using only original data.")
        df_all = df_train
else:
    df_all = df_train
    print("No rediction_inputs_log.csv found. Using only original data.")

df_all.to_csv('combined_training_data.csv', index=False)

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
