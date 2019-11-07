import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import StratifiedShuffleSplit
import math
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def fit_scaler(train):
    return StandardScaler().fit(train)


def scale_data(a_scaler, data):
    return a_scaler.transform(data)


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


def get_f1_score(pre, rec):
    return (2.0 * pre * rec)/(pre + rec)


def get_f1_scores(precisions, recalls):
    f1_scores = []
    for i, pre in enumerate(precisions):
        f1_scores.append(get_f1_score(pre, recalls[i]))

    return f1_scores


def get_timestamp(date):
    if "-" in date:
        return datetime.datetime.strptime(date + "-01", "%Y-%m-%d").timestamp()
    else:
        return datetime.datetime.strptime(date + "01", "%Y%m%d").timestamp()


# use Pred_Label
def get_largest_cluster(df):
    mode_series = df['Pred_Label'].mode()
    return mode_series[0]


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


def get_train_test_split(df):
    #largest_cluster = get_largest_cluster(df)

    #df_train_X = df[df['Pred_Label'] == largest_cluster]
    #df_test_X = df[df['Pred_Label'] != largest_cluster]

    #####################################
    # separate into train and test
    num_rows = df.shape[0]
    train_num_rows = math.floor(num_rows * 0.8)
    df_train_X = df.iloc[:train_num_rows, :]
    df_test_X = df.iloc[train_num_rows:, :]

    # separate into two dataframes each, i.e. X, y
    df_train_y = df_train_X.pop('delinquent')
    df_test_y = df_test_X.pop('delinquent')
    #######################################

    # convert delinquent column to integer
    print("df_train_y in get_train_test_split:")
    print(df_train_y.head(n=5))
    df_train_y = df_train_y.astype(np.int32)
    df_test_y = df_test_y.astype(np.int32)

    return df_train_X, df_train_y, df_test_X, df_test_y




# modifies the dataframe by hashing the columns given into no_new_coils_per new columns, removes old cols
# return: the transformed dataframe
def hash_encoder(df, cols, no_new_cols_per):
    print("<hash> df rows: %ld" % df.shape[0])
    for col in cols:
        print("hashing col %s" % col)
        ce_hash = ce.HashingEncoder(cols=[col], n_components=no_new_cols_per)
        X = df[col]
        new_cols_df = ce_hash.fit_transform(X)
        print("new cols df rows: %ld" % new_cols_df.shape[0])
        df = df.drop(col, axis=1)
        for i in range(0, no_new_cols_per):
            placeholder_name = "col_%ld" % i
            new_col_name = "%s%s%ld" % (col, "_", i)
            #print("new_cols_df before rename:")
            #print(new_cols_df.head(n=1))
            new_cols_df = new_cols_df.rename(columns={placeholder_name: new_col_name})
            #print("new_cols_df after rename:")
            #print(new_cols_df.head(n=1))

        # append the new columns to the dataframe
        print("BEFORE concatting for col %s" % col)
        print("<hash> df rows: %ld" % df.shape[0])
        print("<hash> new cols rows: %ld" % new_cols_df.shape[0])
        df.reset_index(drop=True, inplace=True)
        new_cols_df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, new_cols_df], axis=1)
        print("concatting for col %s" % col)
        print("<hash> df rows: %ld" % df.shape[0])

    return df


def ordinal_enc(df, ordinal_mapping=None):
    used_labels = {}
    ordinal_labels = []

    if ordinal_mapping is not None:
        used_labels = ordinal_mapping

    raw_labels = df.values

    ordinal_index = 0

    for label in raw_labels:
        if label not in used_labels:
            used_labels[label] = ordinal_index
            ordinal_index = ordinal_index + 1
        ordinal_labels.append(used_labels[label])

    print("raw_labels:")
    print(raw_labels[:15])
    print("raw_labels length:")
    print(len(raw_labels))
    print("ordinal_labels:")
    print(ordinal_labels[:15])
    print("ordinal_labels length:")
    print(len(ordinal_labels))

    return np.asarray(ordinal_labels)


def map_unique(unique_labels):
    dic = {}
    for i, label in enumerate(unique_labels):
        dic[label] = i

    return dic


def one_hot_encode(values, num_classes):
    return np.eye(num_classes)[values]


def plot(x, y, label, ax):
    ax.plot(x, y, label=label)


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
