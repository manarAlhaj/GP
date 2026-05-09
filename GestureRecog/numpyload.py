import numpy as np
import pandas as pd

arr = np.load(r"C:\Users\manar\Desktop\GP\data\rest_s20260508_001.npy")

labels = [
    "thumb", "pointer", "middle", "ring", "pinky",
    "yaw_R", "pitch_R", "roll_R",
    "yaw_L", "pitch_L", "roll_L"
]

df = pd.DataFrame(arr, columns=labels)

print(df.head())


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print(df)