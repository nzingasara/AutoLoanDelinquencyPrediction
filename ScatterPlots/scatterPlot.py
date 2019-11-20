import numpy as np
from util import load_data, initialize_plot, plot, save_clear_plt
import matplotlib.pyplot as plt

colors = ("green", "red")

# load data
df_data, uniques = load_data("auto_data.csv")

columns = df_data.columns

# splits the 0s and 1s
df_data_0s = df_data[df_data['delinquent']==0]
df_data_1s = df_data[df_data['delinquent']==1]

np_loan_amnt0s = df_data_0s['loan_amnt'].values
np_loan_amnt1s = df_data_1s['loan_amnt'].values

np_annual_inc0s = df_data_0s['annual_inc'].values
np_annual_inc1s = df_data_1s['annual_inc'].values

np_int_rate0s = df_data_0s['int_rate'].values
np_int_rate1s = df_data_1s['int_rate'].values

np_disbursal_timestamp0s = df_data_0s['disbursal_timestamp'].values
np_disbursal_timestamp1s = df_data_1s['disbursal_timestamp'].values

np_credit_score0s = df_data_0s['credit_score'].values
np_credit_score1s = df_data_1s['credit_score'].values

# loan amts vs annual income
fig, ax = initialize_plot("loan_amnt vs annual_inc", "loan_amt", "annual_inc")
plot(np_loan_amnt0s, np_annual_inc0s, "green", "non-delinquent", ax)
plot(np_loan_amnt1s, np_annual_inc1s, "red", "delinquent", ax)
save_clear_plt("loanAmntVsAnnualInc.png", ax, fig)

# loan amts vs int rates
fig, ax = initialize_plot("loan_amnt vs int_rate", "loan_amt", "int_rate")
plot(np_loan_amnt0s, np_int_rate0s, "green", "non-delinquent", ax)
plot(np_loan_amnt1s, np_int_rate1s, "red", "delinquent", ax)
save_clear_plt("loanAmntVsIntRate.png", ax, fig)

# loan amts vs disbursal timestamps
fig, ax = initialize_plot("loan_amnt vs disbursal_timestamp", "loan_amt", "disbursal_timestamp")
plot(np_loan_amnt0s, np_disbursal_timestamp0s, "green", "non-delinquent", ax)
plot(np_loan_amnt1s, np_disbursal_timestamp1s, "red", "delinquent", ax)
save_clear_plt("loanAmntVsDisbursalTimestamp.png", ax, fig)

# loan amts vs credit scores
fig, ax = initialize_plot("loan_amnt vs credit_score", "loan_amt", "credit_score")
plot(np_loan_amnt0s, np_credit_score0s, "green", "non-delinquent", ax)
plot(np_loan_amnt1s, np_credit_score1s, "red", "delinquent", ax)
save_clear_plt("loanAmntVsCreditScore.png", ax, fig)

# annual income vs int rates
fig, ax = initialize_plot("annual_inc vs int_rate", "annual_inc", "int_rate")
plot(np_annual_inc0s, np_int_rate0s, "green", "non-delinquent", ax)
plot(np_annual_inc1s, np_int_rate1s, "red", "delinquent", ax)
save_clear_plt("annualIncVsIntRate.png", ax, fig)

# annual income vs disbursal timestamps
fig, ax = initialize_plot("annual_inc vs disbursal_timestamp", "annual_inc", "disbursal_timestamp")
plot(np_annual_inc0s, np_disbursal_timestamp0s, "green", "non-delinquent", ax)
plot(np_annual_inc1s, np_disbursal_timestamp1s, "red", "delinquent", ax)
save_clear_plt("annualIncVsDisbursalTimestamp.png", ax, fig)

# annual income vs credit score
fig, ax = initialize_plot("annual_inc vs credit_score", "annual_inc", "credit_score")
plot(np_annual_inc0s, np_credit_score0s, "green", "non-delinquent", ax)
plot(np_annual_inc1s, np_credit_score1s, "red", "delinquent", ax)
save_clear_plt("annualIncVsCreditScore.png", ax, fig)

# int rate vs disbursal timestamps
fig, ax = initialize_plot("int_rate vs disbursal_timestamp", "int_rate", "disbursal_timestamp")
plot(np_int_rate0s, np_disbursal_timestamp0s, "green", "non-delinquent", ax)
plot(np_int_rate1s, np_disbursal_timestamp1s, "red", "delinquent", ax)
save_clear_plt("intRateVsDisbursalTimestamp.png", ax, fig)

# int rate vs credit scores
fig, ax = initialize_plot("int_rate vs credit_score", "int_rate", "credit_score")
plot(np_int_rate0s, np_credit_score0s, "green", "non-delinquent", ax)
plot(np_int_rate1s, np_credit_score1s, "red", "delinquent", ax)
save_clear_plt("intRateVsCreditScore.png", ax, fig)

# disbursal timestamps vs credit scores
fig, ax = initialize_plot("disbursal timestamp vs credit score", "disbursal_timestamp", "credit_score")
plot(np_disbursal_timestamp0s, np_credit_score0s, "green", "non-delinquent", ax)
plot(np_disbursal_timestamp1s, np_credit_score1s, "red", "delinquent", ax)
save_clear_plt("disbursalTimestampVsCreditScore.png", ax, fig)
