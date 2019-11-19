import sklearn
from sklearn import *
import subprocess
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import graphviz
from subprocess import check_call
from sklearn.tree._tree import TREE_LEAF
from copy import deepcopy
import scipy
from scipy import *
import pandas as pd
import numpy as np
import datetime
import category_encoders as ce
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


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
    df = df.sample(frac=1, replace=False)

    # print the stats of the data
    uniques = get_data_stats(df)

    return df, uniques


# description: load the csv data file
# inputs: data_file (string), header (integer), target_names (list of strings)
# returns: the Bunch object holding the loaded data
# Remarks 1: if a header in the csv file, pass 1 for header
# Remarks 2: if header is 'None', target_names will be set to 'None' regardless of what was passed into this function
def convert_to_bunch(data_df, labels_name):

    X_df = data_df
    y_df = X_df.pop(labels_name)

    feature_names = X_df.columns.values

    bunch = sklearn.datasets.base.Bunch(data=X_df.values, target=y_df.values, feature_names=feature_names,
                                        target_names=None)
    # print(bunch)

    # todo: remove this. Only for testing
    #bunch = load_iris()

    #bunch.data = np.float32(bunch.data)

    # todo: remove this. Will shuffle while doing train test split in main.py
    #bunch.data, bunch.target = shuffle(bunch.data, bunch.target)
    # print("bunch data after loading:")
    # print(bunch.data)
    # print("bunch target after loading:")
    # print(bunch.target)
    # print(bunch)
    return bunch, X_df, y_df


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


# create a random forest classifier object without fitting
def create_random_forest_classifier(**kwargs):
    return RandomForestClassifier(**kwargs)


# fits the given classifier to the data given
def fit_random_forest_classifier(X, y, clf=None, **kwargs):
    # if no classifier passed, create one
    if clf is None:
        clf = create_random_forest_classifier(**kwargs)

    # fit
    #start = time()
    clf = clf.fit(X, y)
    #time_fit = (time() - start)

    #print(" DT time to train: ")
    #print(time_fit)
    return clf


def plot_max_depth_learning_curve(title, x_label, y_label, X, y, scoring_metric, kfolds=3,
                             train_size=1.0):

    print("*** In plot_max_depth_learning_curve now!!!***")
    train_sizes = [train_size]
    wall_clock_times = []
    iterations_list = []
    train_mean_scores = []
    test_mean_scores = []
    # for loop add max_iter range to hyper_params for plotting
    max_depth_list = np.logspace(0, 6, num=7, base=2.0)
    #max_depth_list = [32, 64]

    hyper_params_cpy = {"max_features":'auto', "oob_score": True, "n_estimators":100}#deepcopy(hyper_params)

    criterion_list = ["entropy"]

    color_arr = [("r", "g"), ("b", "m")]

    fig, ax = initialize_plot(title, x_label, y_label)

    best_mean_test_score = -1
    hyper_params_best = {}

    i = 0
    wall_clock_times = [[], []]
    # show how max depth affects DT with both gini and entropy criterions
    for i, crit in enumerate(criterion_list):
        colors = color_arr[i]
        hyper_params_cpy["criterion"] = crit
        iterations_list = []
        train_mean_scores = []
        test_mean_scores = []

        for iteration in max_depth_list:
            hyper_params_cpy["max_depth"] = iteration

            # create the neural net with the hyper parameter dictionary
            estimator = create_random_forest_classifier(**hyper_params_cpy)
            print("estimator:")
            print(estimator)

            #start = time()
            train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=kfolds, scoring=scoring_metric,
                                                                    train_sizes=train_sizes)
            #wall_clock_times[i].append(time() - start)

            # get the mean test score
            train_score_mean = np.mean(train_scores, axis=1)
            test_score_mean = np.mean(test_scores, axis=1)

            print("~~~~~~~~~~~~~~~~~~")
            print("max depth:")
            print(iteration)
            print("train_mean:")
            print(train_score_mean)
            print("test_score_mean:")
            print(test_score_mean)

            # store test score in lists
            iterations_list.append(iteration)
            train_mean_scores.append(train_score_mean)
            test_mean_scores.append(test_score_mean)

            if test_score_mean > best_mean_test_score:
                best_mean_test_score = test_score_mean
                hyper_params_best = deepcopy(hyper_params_cpy)
                print("best hyperparams so far:")
                print(hyper_params_best)

        # plot
        plot(iterations_list, train_mean_scores, "training_%s" % crit, ax)
        plot(iterations_list, test_mean_scores, "test_%s" % crit, ax)

    save_clear_plt("%s_%s.png" % (title, crit), ax, fig)

    # plot the wall clock times
    #init_plot_data("DT training and testing wall clock time function of max depth(%s)" % criterion_list[0], "max_depth",
                   #"wall clock time", iterations_list, wall_clock_times[0], None, None, y_lim=None)
    #init_plot_data("DT training and testing wall clock time function of max depth(%s)" % criterion_list[1], "max_depth",
                   #"wall clock time", iterations_list, wall_clock_times[1], None, None, y_lim=None)

    #print("*** Completed plot_max_depth_learning_curve now!!!***")
    return wall_clock_times, iterations_list, hyper_params_best


if __name__ == "__main__":
    # list of columns to hash encode
    cols_to_hash = ['state']
    no_new_cols_per = 6

    # load the data
    df_data, uniques = load_data("mortgage_data_small_50_50_2.csv")

    print("df_data before hashing:")
    print(df_data)
    df_data = hash_encoder(df_data, cols_to_hash, no_new_cols_per)
    print("df_data:")
    print(df_data)

    df_auto_data, auto_uniques = load_data("auto_data.csv")
    df_auto_data = hash_encoder(df_auto_data, cols_to_hash, no_new_cols_per)

    data_object, X_df_, y_df_ = convert_to_bunch(df_data, "delinquent")

    data_object_auto, X_auto_df_, y_auto_df_ = convert_to_bunch(df_auto_data, "delinquent")

    print("X_df_ columns:")
    print(X_df_.columns)

    # separate data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_df_, y_df_, test_size=0.2, shuffle=True, stratify=y_df_)
    print("X_train:")
    print(X_train)
    print("y_train")
    print(y_train)
    print("X_test:")
    print(X_test)
    print("y_test:")
    print(y_test)

    wall_clock_times, iterations_list, hyper_params_best = plot_max_depth_learning_curve("Random Forest Accuracy Score as Function of Max Depth", "max depth", "accuracy score", X_df_,
                                  y_df_, 'accuracy', kfolds=5, train_size=1.0)

    print("best hyperparams:")
    print(hyper_params_best)

    # hyperparams to change: criterion = 'entropy'
    #rf_clf = fit_random_forest_classifier(data_object.data, data_object.target, random_state=0, **hyper_params_best)
    rf_clf = fit_random_forest_classifier(X_train, y_train, random_state=0, **hyper_params_best)
    final_train_score_mortgage = rf_clf.score(X_train, y_train)
    final_test_score_mortgage = rf_clf.score(X_test, y_test)
    print("final train score mortgage:")
    print(final_train_score_mortgage)
    print("final test score mortgage:")
    print(final_test_score_mortgage)

    print("feature importances:")
    print(rf_clf.feature_importances_)

    # get the f1 score on 0s (non-delinquent) and 1s (delinquent)
    # won't use accuracy since the auto loan data has imbalance in 0s and 1s
    auto_test_predictions = rf_clf.predict(X_auto_df_)

    print("np.unique(auto_test_predictions):")
    print(np.unique(auto_test_predictions))
    print("number of predictions auto:")
    print(len(auto_test_predictions))
    print("# 0s in auto predictions:")
    print(len(auto_test_predictions[auto_test_predictions == 0]))
    print("# 1s in auto predictions:")
    print(len(auto_test_predictions[auto_test_predictions == 1]))

    print("type(auto_test_predictions):")
    print(type(auto_test_predictions))
    print("auto test f1 score for 0s:")
    print(f1_score(y_auto_df_, auto_test_predictions, pos_label=0))
    print("auto test f1 score for 1s:")
    print(f1_score(y_auto_df_, auto_test_predictions, pos_label=1))
