import os
import pandas as pd
from nannyml import UnivariateDriftCalculator

TRAIN_DATA_PATH = 'health_dataset.csv'
LOG_DATA_PATH = 'prediction_inputs_log.csv'
REPORT_PATH = 'drift_report.html'

required_columns = ['timestamp', 'sleep_hours', 'exercise_hours', 'screen_time',
                    'social_interaction', 'age', 'work_hours']

if not os.path.exists(LOG_DATA_PATH):
    pd.DataFrame(columns=required_columns).to_csv(LOG_DATA_PATH, index=False)

df_train = pd.read_csv(TRAIN_DATA_PATH)
df_log = pd.read_csv(LOG_DATA_PATH)

if 'timestamp' in df_log.columns:
    df_log = df_log.drop(columns=['timestamp'])

feature_columns = ['sleep_hours', 'exercise_hours', 'screen_time',
                   'social_interaction', 'age', 'work_hours']

calc = UnivariateDriftCalculator(
    column_names=feature_columns,
    timestamp_column_name=None,
    chunk_size=5
)
calc.fit(df_train[feature_columns])

results = calc.calculate(df_log[feature_columns])

results_df = results.data.copy()
results_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                      for col in results_df.columns]

print("Drift summary for each feature:")
print(results_df)

drift_detected = any('alert' in col and results_df[col].any() for col in results_df.columns)

if drift_detected:
    print("Drift detected!")
else:
    print("No significant drift detected.")

with open('detect_drift_output.log', 'w') as log_file:
    log_file.write("Drift detected" if drift_detected else "No drift detected")
