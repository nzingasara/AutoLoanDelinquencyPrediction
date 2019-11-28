import sklearn
from sklearn import *
from copy import deepcopy
from scipy import *
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


def get_data_stats(df):
    # get all unique labels
    unique_arr = df['delinquent'].unique()
    print("unique labels:")
    print(unique_arr)
    print("unique labels count:")
    print(len(unique_arr))

    # count the numbr of each label UCR #
    for label in unique_arr:
        count = df[df['delinquent'] == label]['delinquent'].count()
        print("%s: %ld" % (label, count))

    return unique_arr


def get_timestamp(date):
    if "-" in date:
        return datetime.datetime.strptime(date + "-01", "%Y-%m-%d").timestamp()
    else:
        return datetime.datetime.strptime(date + "01", "%Y%m%d").timestamp()


def load_data(file_name):
    type_dict = {'loan_amnt': np.float64,
                 'state': 'str',
                 'annual_inc': np.float64,
                 'int_rate': np.float64,
                 'delinquent': np.float64,
                 'disbursal_date': 'str',
                 'credit_score': np.float64
                 }

    df = pd.read_csv(file_name, header=0, delimiter=",", dtype=type_dict)

    print("df shape before dropping Nans:")
    print(df.shape[0])
    # drop records that contain NaN
    df.dropna(inplace=True)
    print("df shape after dropping Nans:")
    print(df.shape[0])

    # create new column representing converted dates into timestamps
    new_col = df["disbursal_date"].apply(get_timestamp)
    df.insert(df.shape[1], "disbursal_timestamp", new_col, False)

    # remove old disbursal_date column
    df.pop('disbursal_date')

    # shuffle stratified
    #df = df.sample(frac=1, replace=False)

    # print the stats of the data
    uniques = get_data_stats(df)

    return df, uniques


def fit_scaler(train):
    return StandardScaler().fit(train)


def scale_data(a_scaler, data):
    return a_scaler.transform(data)


def fit_normalizer(train):
    return Normalizer().fit(train)


def normalize_data(nmlzr, data):
    return nmlzr.transform(data)


def plot(hist_arr, n_bins, ax):
    ax.hist(hist_arr)#, bins=n_bins)


def initialize_plot(title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def save_clear_plt(png_file_name, ax, fig):
    #plt.tight_layout()
    ax.legend(loc="best")
    fig.savefig(png_file_name)
    fig.clf()


# returns histogram array and bin_edges
def get_histogram_arr(data_arr, n_bins):
    return np.histogram(data_arr, bins=n_bins)


if __name__ == "__main__":
    n_bins = 20

    # load mortgage and auto data into dataframes
    df_mort, mort_uniques = load_data("mortgage_data_small_50_50_2.csv")
    df_auto, auto_uniques = load_data("auto_data.csv")

    df_mort.pop("state")
    df_auto.pop("state")

    print("df shape:")
    print(df_mort.shape)

    column_names = df_mort.columns
    print("df columns:")
    print(column_names)

    print("df_mort before scaling:")
    print(df_mort)

    # scale the data and put into scaled copy
    sclr_mort = fit_scaler(df_mort)
    sclr_auto = fit_scaler(df_auto)

    mort_data_scl = scale_data(sclr_mort, df_mort)
    auto_data_scl = scale_data(sclr_auto, df_auto)

    print("sclr_mort mean")

    print("mort after scaling:")
    print(mort_data_scl)

    # normalize the data and put into a normalized copy
    nmlzr_mort = fit_normalizer(df_mort)
    nmlzr_auto = fit_normalizer(df_auto)

    mort_data_nml = normalize_data(nmlzr_mort, df_mort)
    auto_data_nml = normalize_data(nmlzr_auto, df_auto)

    print("mort normalized data:")
    print(mort_data_nml)

    #print("auto_data_nml type:")
    #print(auto_data_nml[:,0])
    #exit()

    print("data after loading shape:")
    print(mort_data_scl.shape)
    n_cols = mort_data_scl.shape[1]


    # todo: remove once done testing
    #col_nm="int_rate"
    #col_data = mort_data_scl[:, 2]
    #hist_arr = get_histogram_arr(col_data, n_bins)
    #fig, ax = initialize_plot("Mortgage int_rate histogram (scaled)", "", "")
    #plot(col_data, n_bins, ax)
    #save_clear_plt("Mortgage %s histogram (scaled)" % col_nm, ax, fig)

    #################################

    # for each column in scaled data, make a histogram
    for i in list(range(n_cols)):
        col_nm = column_names[i]
        col_data = mort_data_scl[:, i]
        #hist_arr = get_histogram_arr(col_data, n_bins)
        fig, ax = initialize_plot("Mortgage %s histogram (scaled)" % col_nm, "", "")
        plot(col_data, n_bins, ax)
        save_clear_plt("Mortgage %s histogram (scaled)" % col_nm, ax, fig)

    for i in list(range(n_cols)):
        col_nm = column_names[i]
        col_data = auto_data_scl[:, i]
        #hist_arr = get_histogram_arr(col_data, n_bins)
        fig, ax = initialize_plot("Auto %s histogram (scaled)" % col_nm, "", "")
        plot(col_data, n_bins, ax)
        save_clear_plt("Auto %s histogram (scaled)" % col_nm, ax, fig)

    # for each column in normalized data, make a histogram
    for i in list(range(n_cols)):
        col_nm = column_names[i]
        col_data = mort_data_nml[:, i]
        #hist_arr = get_histogram_arr(col_data, n_bins)
        fig, ax = initialize_plot("Mortgage %s histogram (normalized)" % col_nm, "", "")
        plot(col_data, n_bins, ax)
        save_clear_plt("Mortgage %s histogram (normalized)" % col_nm, ax, fig)

    for i in list(range(n_cols)):
        col_nm = column_names[i]
        col_data = auto_data_nml[:, i]
        #hist_arr = get_histogram_arr(col_data, n_bins)
        fig, ax = initialize_plot("Auto %s histogram (normalized)" % col_nm, "", "")
        plot(col_data, n_bins, ax)
        save_clear_plt("Auto %s histogram (normalized)" % col_nm, ax, fig)

