import pandas as pd
import numpy as np

df = pd.read_csv('prediction_inputs_log.csv')

if 'screen_time' in df.columns:
    df['screen_time'] = df['screen_time'] + 5
if 'age' in df.columns:
    df['age'] = df['age'] + 20

df.to_csv('prediction_inputs_log.csv', index=False)
print("Completed")
