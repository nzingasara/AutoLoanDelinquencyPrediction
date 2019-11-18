#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime as dt

df = pd.read_csv('LoanStats3a_securev1.csv', encoding="ISO-8859-1")
df = df[df.purpose == 'car']
df = df[['loan_amnt', 'addr_state', 'annual_inc', 'int_rate', 'loan_status',
         'issue_d', 'fico_range_low', 'fico_range_high']]
df['int_rate'] = df['int_rate'].astype(str)
df['int_rate'] = df.int_rate.str.replace('%', '').astype(float)
df['int_rate'] /= 100
df['issue_d'] = pd.to_datetime(df.issue_d)
df['issue_d'] = df['issue_d'].dt.strftime('%Y-%m')
df['loan_status'] = np.where(df['loan_status'] == 'Charged Off', 1, 0)  # 1 = delinquent, 0 = not delinquent
df['Credit Score'] = (df['fico_range_high'] + df[fico_range_low]) / 2
df.columns = ['loan_amnt', 'state', 'annual_inc', 'int_rate',
              'delinquent', 'disbursal_date', 'fico_low', 'fico_high', 'Credit Score']  # fico_low and fico_high are the credit scores
df.to_csv('auto_data.csv', encoding='utf-8', index=False)
