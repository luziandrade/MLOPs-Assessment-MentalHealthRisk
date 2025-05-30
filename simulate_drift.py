import pandas as pd
import numpy as np

df = pd.read_csv('prediction_inputs_log.csv')

df['screen_time'] = df['screen_time'] + 10   
df['sleep_hours'] = df['sleep_hours'] - 3   

print("Drift simulation applied:")
print(df[['screen_time', 'age']].describe())    


df.to_csv('prediction_inputs_log.csv', index=False)
print("Completed")
