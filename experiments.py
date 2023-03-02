# %%

from data import Dataset
import numpy as np
import pandas as pd

# %%

data = Dataset.from_openml(976)

X = data.features.to_numpy()
y = data.labels.to_numpy()

# %%

df = pd.concat([pd.DataFrame(y, columns=["Label"]), pd.DataFrame(X)], axis=1)

df.loc[:,"Label"].value_counts()

# %%

df.groupby(by="Label").mean()

# %%
