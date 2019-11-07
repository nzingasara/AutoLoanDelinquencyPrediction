import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import math
import util

# todo: add header to data
# todo: make sure file name is still same after Thomas finishes rerun

# global variables
max_tree_depth_list = [2,3,4,5,6]
n_trees_list = [100, 150, 200, 400, 800, 1600]
learn_rate_list = [0.1, 0.3, 0.5]

classes = ["Not Delinquent", "Delinquent"]


# input function for training
def make_input_func(X, y, n_epochs, batch_size):
    def input_func():
        ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
        ds = ds.repeat(n_epochs)
        ds = ds.batch(batch_size)
        return ds
    return input_func


def make_boosted_tree_classifier(feature_columns, max_depth, n_trees, learning_rate):
    return tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=1, model_dir='./btc_model/',
                                               max_depth=max_depth, n_trees=n_trees, learning_rate=learning_rate)


def get_preds(est, input_fn):
    return [elem["class_ids"][0] for elem in est.predict(input_fn=input_fn)]


# todo: put mortgage_data back
#file_name = "mortgage_data.csv"
#file_name = "mortgagePractice.csv"
file_name = "mortgage_data_small.csv"

df_train_X, df_train_y, df_test_X, df_test_y, feature_cols = util.load_prep_data(file_name)

# train and test input functions
train_in = make_input_func(df_train_X, df_train_y, 1, len(df_train_y))
test_in = make_input_func(df_test_X, df_test_y, 1, len(df_train_y))

#########
#print("train # rows: %ld" % df_train_y.shape[0])
#print("df_train_y head (5):")
#print(df_train_y.head(n=5))
#print("train label counts:")
#print(df_train_y.value_counts(dropna=False))
#print("test # rows: %ld" % df_test_y.shape[0])
#print("df_test_y head (5):")
#print(df_test_y.head(n=5))
#print("test label counts:")
#print(df_test_y.value_counts(dropna=False))
#########

# at this point, all initial data loading and processing is done

# arrays for storing info for plotting
f1_scores_list = []
f1_train_list = []

############################
for tree_depth in max_tree_depth_list:
    print("processing tree depth %ld" % tree_depth)
    fig, ax = util.initialize_plot("Random Boosted Forrest Performance with %ld Max Depth" % tree_depth, "n_trees", "F1 Score")
    for lr in learn_rate_list:
        # clear f1 scores list
        f1_scores_list = []
        f1_train_list = []
        for n_tree in n_trees_list:
            # create the estimator
            est = make_boosted_tree_classifier(feature_cols, tree_depth, n_tree, lr)

            # training
            est.train(train_in, max_steps=None)

            # evaluation
            results = est.evaluate(test_in)
            #clear_output()

            # predictions
            predictions = list(est.predict(test_in))
            probabilities = pd.Series([pred['probabilities'][1] for pred in predictions])

            preds = est.predict(test_in)

            # true labels
            predicted_labels = get_preds(est, test_in)

            # get the F1 score and store in array
            curr_prec = results['precision']
            curr_rec = results['recall']
            f1_train_list.append(util.f1_score_sklearn(df_train_y.tolist(), get_preds(est, train_in)))
            f1_scores_list.append(util.f1_score_sklearn(df_test_y.tolist(), predicted_labels))
            #output = tfa.metrics.f_scores.F1Score(num_classes=2)
            #print("F1 metric tf:")
            #print(output.update_state(df_test_y, predicted_labels))

            # confusion matrix
            util.make_confusion_matrix(classes, df_test_y, predicted_labels,
                                  "Delinquency Confusion Matrix (lr=%f, num trees=%ld, max depth=%ld)" % (lr, n_tree, tree_depth),
                                  "Confusion Matrix_lr=%f_numTrees=%ld_maxDepth=%ld.png" % (lr, n_tree, tree_depth))

            # ROC curve
            util.plot_roc_curve(df_test_y,probabilities, "ROC Curve (lr=%f, num trees=%ld, max depth=%ld)" % (lr, n_tree, tree_depth),
                                "ROCCurve_lr=%f_numTrees=%ld_maxDepth=%ld.png" % (lr, n_tree, tree_depth))


        # done with n_trees loop for this learning rate. Plot the values
        util.plot(n_trees_list, f1_train_list, "train learning rate = %f" % lr, ax)
        util.plot(n_trees_list, f1_scores_list, "test learning rate = %f" % lr, ax)

    # done with plotting lines for all learning rates for this tree depth. Save the plot and clear plt
    util.save_clear_plt("rbfPerfWith_%ld_MaxDepth.png" % tree_depth, ax, fig)
############################

# todo: fix save model code. These don't work
#est.export_saved_model('btc_model')
#tf.saved_model.save(est, "tmp/btc/1/")
