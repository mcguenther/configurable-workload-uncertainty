import pandas as pd

df = pd.read_csv('measurements.csv')

df = df[df['partition'] == 5]
print(df.shape)

df.to_csv('measurements.csv', index=False)
