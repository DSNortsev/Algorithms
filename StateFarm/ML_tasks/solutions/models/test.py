import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

cats = ['tuesday', 'saturday', 'thursday', 'sunday', 'wednesday', 'monday']

df = pd.DataFrame({'cat': ['tuesday', 'saturday', 'friday']})

df["cat"] = df["cat"].astype(CategoricalDtype(cats))

dummies = pd.get_dummies(df, prefix='x5', prefix_sep='_',  dummy_na=True)

# dummies = dummies.T.reindex(cats).T.fillna(0)

print(dummies)