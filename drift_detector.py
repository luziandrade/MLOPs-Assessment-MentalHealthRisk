import os
import pandas as pd
from nannyml import UnivariateDriftCalculator
from nannyml.datasets import load_synthetic_binary_classification_dataset

log_path = 'prediction_inputs_log.csv'
required_columns = ['timestamp', 'sleep_hours', 'exercise_hours', 'screen_time', 'social_interaction', 'age', 'work_hours']

if not os.path.exists(log_path):
    pd.DataFrame(columns=required_columns).to_csv(log_path, index=False)

TRAIN_DATA_PATH = 'health_dataset.csv'           
LOG_DATA_PATH = 'prediction_inputs_log.csv'      
REPORT_PATH = 'drift_report.html'                

df_train = pd.read_csv(TRAIN_DATA_PATH)

df_log = pd.read_csv(LOG_DATA_PATH)
if 'timestamp' in df_log.columns:
    df_log = df_log.drop(columns=['timestamp'])


feature_columns = ['sleep_hours', 'exercise_hours', 'screen_time', 'social_interaction', 'age', 'work_hours']

calc = UnivariateDriftCalculator(
    column_names=feature_columns,
    timestamp_column_name=None,
    chunk_size=5
  
)
calc.fit(df_train[feature_columns])  

results = calc.calculate(df_log[feature_columns])  

print("Drift summary for each feature:")
print(results.data)

try:
    results.plot().write_html(REPORT_PATH)
    print(f"Drift report saved to {REPORT_PATH}")
except Exception as e:
    print("Could not save drift report:", e)

if 'drift' in results.data.columns and results.data['drift'].any():
    print("Drift detected! Consider triggering retraining.")
else:
    print("No significant drift detected.")

with open('detect_drift_output.log', 'w') as log_file:
    if 'drift' in results.data.columns and results.data['drift'].any():
        log_file.write("Drift detected")
    else:
        log_file.write("No drift detected")