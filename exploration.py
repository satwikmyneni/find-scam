import pandas as pd

df = pd.read_csv("data/train.csv")
print(df.shape)
print(df.columns)
print(df["fraudulent"].value_counts(normalize=True))
df.head()
