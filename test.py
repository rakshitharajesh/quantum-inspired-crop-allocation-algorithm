import pandas as pd
df = pd.read_csv("data/mandi_prices.csv")
print(df['commodity'].unique())