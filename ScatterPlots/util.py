import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np


def plot(x, y, color, label, ax):
    #ax.plot(x, y, label=label)
    ax.scatter(x, y, c=color, label=label)


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


def get_timestamp(date):
    if "-" in date:
        return datetime.datetime.strptime(date + "-01", "%Y-%m-%d").timestamp()
    else:
        return datetime.datetime.strptime(date + "01", "%Y%m%d").timestamp()


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
    df = df.sample(frac=1, replace=False)

    # print the stats of the data
    uniques = get_data_stats(df)

    return df, uniques
