import pandas as pd

df = pd.read_csv("AB_NYC_2019.csv")
mean = df["price"].mean()
var = df["price"].to_numpy().var()
print(mean, var)
