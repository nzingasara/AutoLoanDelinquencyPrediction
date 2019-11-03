import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import math
from sklearn.metrics import roc_curve
import util

# todo: add header to data
# todo: make sure file name is still same after Thomas finishes rerun


# input function for training
def make_input_func(X, y, n_epochs, batch_size):
    def input_func():
        ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
        ds = ds.repeat(n_epochs)
        ds = ds.batch(batch_size)
        return ds
    return input_func


# todo: put mortgage_data back
# file_name = "mortgage_data.csv"
file_name = "mortgagePractice.csv"

df_train_X, df_train_y, df_test_X, df_test_y, feature_cols = util.load_prep_data(file_name)

# train and test input functions
train_in = make_input_func(df_train_X, df_train_y, None, len(df_train_y))
test_in = make_input_func(df_test_X, df_test_y, 1, len(df_train_y))

# at this point, all initial data loading and processing is done

# create the estimator
est = tf.estimator.BoostedTreesClassifier(feature_cols, n_batches_per_layer=1, model_dir='.')

# training
est.train(train_in, max_steps=None)

# evaluation
results = est.evaluate(test_in)
#clear_output()
print("results:")
print(pd.Series(results))

# predictions
predictions = list(est.predict(test_in))
print("predictions:")
print(predictions)
probabilities = pd.Series([pred['probabilities'][1] for pred in predictions])

print("probabilities:")
print(probabilities)

preds = est.predict(test_in)
for elem in preds:
    print("another prob:")
    print(elem)
    print(elem["probabilities"])


# plot into roc curve
false_pos_rates, true_pos_rates, thresholds = roc_curve(df_test_y, probabilities)
plt.plot(false_pos_rates, true_pos_rates)
plt.title("ROC Curve")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.xlim(0,)
plt.ylim(0,)
plt.show()
