# %%

import pickle
import pandas as pd

# %%

path = "results_stored/"

history = pickle.load(open(path + "history_976_restart_1_2023_03_18-03_36_39_PM.pkl", 'rb'))

# %%

test_scores = []
costs = []
budgets = []

for i in range(len(history)):
    test_scores.append(history[i][4]["test_score"])
    costs.append(history[i][2])
    budgets.append(history[i][4]["budget"])

df = pd.DataFrame({"Test score": test_scores, "Cost": costs, "Budget": budgets})

# %%

display(df)

# %%