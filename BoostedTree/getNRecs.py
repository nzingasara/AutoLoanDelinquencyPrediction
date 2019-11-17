import numpy as np
import pandas as pd
import math

type_dict = {'loan_amnt': np.int64,
             'state': 'str',
             'annual_inc': np.float64,
             'int_rate': np.float64,
             'delinquent': np.float64,
             'disbursal_date': 'str',
             'credit_score': np.int64
             }

file_name="mortgage_data.csv"
n = 737560
ratio = 0.5
df = pd.read_csv(file_name, header=0, delimiter=",", dtype=type_dict)

# drop records that contain NaN
df.dropna(inplace=True)

# shuffle before getting subset
df = df.sample(frac=1, replace=False)

df0 = df[df['delinquent'] == 0]
df1 = df[df['delinquent'] == 1]

num_pos_samples = math.floor(n*ratio)

num_neg_samples = n - num_pos_samples

df0 = df0.head(n=num_neg_samples)
df1 = df1.head(n=num_pos_samples)

frames = [df0, df1]
df_new = pd.concat(frames)

df_new = df_new.sample(frac=1, replace=False)

df_new['annual_inc'] = df_new['annual_inc'].astype(int)
df_new['delinquent'] = df_new['delinquent'].astype(int)

df_new.to_csv(path_or_buf="mortgage_data_small_50_50_2.csv", sep=",", index=False)
