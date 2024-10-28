import pandas as pd

data = {
    "method": ["Alice"],
    "auc": [25],
    "f1": [1],
    "precision": [10],
    "recall": [100],
}

df = pd.DataFrame(data)

print(df)