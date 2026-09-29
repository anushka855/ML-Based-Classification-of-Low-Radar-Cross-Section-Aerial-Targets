import numpy as np
import pandas as pd
from feature_extraction import FEATURE_NAMES

data = np.load("radar_dataset.npz")

X = data["X"]
y = data["y"]
# scene_ids 
scene_ids = data["scene_ids"]

# Convert to dataframe
df = pd.DataFrame(X, columns=FEATURE_NAMES)
df["label"] = y

# Disable row/column truncation
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(df)