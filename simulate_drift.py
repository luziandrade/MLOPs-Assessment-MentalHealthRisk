import pandas as pd
import numpy as np

# Reading data from a CSV file named 'prediction_inputs_log.csv' into a pandas DataFrame.
df = pd.read_csv('prediction_inputs_log.csv')

df['screen_time'] = df['screen_time'] + 100   
df['sleep_hours'] = df['sleep_hours'] - 5   

print("Drift simulation applied:")
print(df[['screen_time', 'age']].describe())    

# Writing the modified DataFrame back to the same CSV file ('prediction_inputs_log.csv').
df.to_csv('prediction_inputs_log.csv', index=False)
print("Completed")
