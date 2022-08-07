import pandas as pd
import json
import pickle
import numpy as np
import plotly.express as px

with open(r'C:\Users\raphs\Documents\University\msc\thesis\Thesis-code\run_times.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(b)

df = pd.DataFrame(b)
df['time'] = df['time']/ np.timedelta64(1, "s")
df = df.rename(columns = {'time':'time_seconds'})
print(df)
fig = px.scatter(df, x="samples", y="time_seconds",color ='method', title='Run time by method')
fig.show()