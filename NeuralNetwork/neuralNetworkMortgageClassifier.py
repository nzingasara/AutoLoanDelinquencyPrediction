import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.datasets import mnist
import util
import tensorflow as tf
import pandas as pd


def create_model(input_shape, layers_info):
    model = Sequential()
    input_shape_given = False
    for layer in layers_info:
        if not input_shape_given:
            model.add(Dense(layer[0], input_shape=input_shape, activation=layer[1]))
            input_shape_given = True
        else:
            model.add(Dense(layer[0], activation=layer[1]))

    return model


def count_labels(df):
    labels = df.values
    count0 = 0
    count1 = 0

    for label in labels:
        if label == 0:
            count0 = count0 + 1
        if label == 1:
            count1 = count1 + 1

    print("count0:")
    print(count0)
    print("count1:")
    print(count1)


def make_epoch_list(len):
    epoch_list = []
    for i in range(len):
        epoch_list.append(i)

    return epoch_list

# list of columns to hash encode
cols_to_hash = ['state']
no_new_cols_per = 6

# load the data
file_name = "mortgage_data_small_50_50_2.csv"
df, unique_labels = util.load_data(file_name)

print("df after loading (%ld rows):" % df.shape[0])
print(df.head(n=5))

# hash encode the data
df = util.hash_encoder(df, cols_to_hash, no_new_cols_per)
print("df after hashing (%ld rows):" % df.shape[0])
print(df.head(n=5))

# separate into train/test X/y splits
df_train_X, df_train_y, df_test_X, df_test_y = util.get_train_test_split(df)

print("df train X:")
print(df_train_X.head(n=5))
print('df train X rows:')
print(df_train_X.shape[0])

print("df train y head:")
print(df_train_y.head(n=5))
print("df test y head:")
print(df_test_y.head(n=5))

print("df_train_y label counts:")
count_labels(df_train_y)
print("df_test_y label counts:")
count_labels(df_test_y)

print("df test X:")
print(df_test_X)
print("df test X rows:")
print(df_test_X.shape[0])

print("df_train_X shape:")
print(df_train_X.shape)
print("df_test_X shape:")
print(df_test_X.shape)

# one hot encode the y data
print("df_train_y.values type:")
print(type(df_train_y.values))
print("df_train_y.values[:10]:")
print(df_train_y.values[:10])
train_y = util.one_hot_encode(df_train_y.values, 2)
test_y = util.one_hot_encode(df_test_y, 2)

print("train_y one hot:")
print(train_y[:10])

sclr = util.fit_scaler(df_train_X)
train_X = util.scale_data(sclr, df_train_X)
test_X = util.scale_data(sclr, df_test_X)
print("train_X AFTER scaling:")
print(train_X)
print("test_X AFTER scaling:")
print(test_X)

print("Before getting predictions, test X shape:")
print(test_X.shape)
print("Before getting predictions, test y shape:")
print(test_y.shape)

# get first 1000 and pop off to get predition list
pred_X = test_X[0:1000]
pred_y = test_y[0:1000]
test_X = np.delete(test_X, slice(0, 1000), 0)
test_y = np.delete(test_y, slice(0,1000), 0)


print("After getting predictions, test X shape:")
print(test_X.shape)
print("After getting predictions, test y shape:")
print(test_y.shape)

print("pred X shape:")
print(pred_X.shape)
print("pred y shape:")
print(pred_y.shape)

lr = 0.0001
###############################
input_shape = (11,)
leaky_relu = LeakyReLU(0.1)

layers_info_small1 = [(5, leaky_relu), (2, 'softmax')]
layers_info_small2 = [(10, leaky_relu), (2, 'softmax')]
layers_info_small3 = [(20, leaky_relu), (2, 'softmax')]
layers_info_small4 = [(40, leaky_relu), (2, 'softmax')]

layers_info1 = [(5, leaky_relu), (5, leaky_relu), (5, leaky_relu), (5, leaky_relu), (2, 'softmax')]
layers_info2 = [(10, leaky_relu), (10, leaky_relu), (10, leaky_relu), (10, leaky_relu), (2, 'softmax')]
layers_info3 = [(20, leaky_relu), (20, leaky_relu), (20, leaky_relu), (20, leaky_relu), (2, 'softmax')]
layers_info4 = [(20, leaky_relu), (20, leaky_relu), (20, leaky_relu), (20, leaky_relu), (20, leaky_relu),
                (20, leaky_relu), (20, leaky_relu), (20, leaky_relu), (20, leaky_relu), (20, leaky_relu),
                (20, leaky_relu), (20, leaky_relu), (20, leaky_relu), (20, leaky_relu), (20, leaky_relu),
                (20, leaky_relu), (20, leaky_relu), (20, leaky_relu), (20, leaky_relu), (20, leaky_relu), (2, 'softmax')]
layers_info5 = [(40, leaky_relu), (40, leaky_relu), (40, leaky_relu), (40, leaky_relu), (2, 'softmax')]

#layers_info_list = {"leakyRelu_5_node_5_layer_lr_%f" % lr: layers_info1,"leakyRelu_10_node_5_layer_lr_%f" % lr:layers_info2,
                    #"leakyRelu_20_node_5_layer_lr_%f" % lr: layers_info3, "leakyRelu_20_node_21_layer_lr_%f" % lr: layers_info4,
                    #"leakyRelu_40_node_5_layer_lr_%f" % lr: layers_info5}

layers_info_list = {"leakyRelu_5_node_2_layer_lr_%f_small" % lr: layers_info_small1,"leakyRelu_10_node_2_layer_lr_%f_small" % lr:layers_info_small2,
                    "leakyRelu_20_node_2_layer_lr_%f_small" % lr: layers_info_small3, "leakyRelu_40_node_2_layer_lr_%f_small" % lr: layers_info_small4}

fig_loss, ax_loss = util.initialize_plot("Neural Network loss lr=%f" % lr, "# epochs", "loss")
fig_f1, ax_f1 = util.initialize_plot("Neural Network F1 Score lr=%f" % lr, "# epochs", "F1 Score")
fig_acc, ax_acc = util.initialize_plot("Neural Network Accuracy lr=%f" % lr, "# epochs", "Accuracy")

for i, key in enumerate(layers_info_list):
    model = create_model(input_shape, layers_info_list[key])

    model.summary()

    #model = Sequential()

    # get actual input shape and enter it here
    #model.add(Dense(128, input_shape=(11,), activation='relu'))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(2, activation='softmax'))
    ###############################

    # check if classes are relatively equal or not. If not, change metric
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])

    print("train_X BEFORE fitting:")
    print(train_X[:5])
    print("train_y BEFORE fitting:")
    print(train_y[:5])
    print("test_X BEFORE fitting:")
    print(test_X[:5])
    print("test_y BEFORE fitting:")
    print(test_y[:5])
    print("number of unique vals in train_y:")
    print(np.unique(train_y, return_counts=True, axis=0))
    print("number of unique vals in test_y:")
    print(np.unique(test_y, return_counts=True, axis=0))
    # load in data and put in the x and y parts
    history = model.fit(x=train_X, y=train_y, epochs=50)
    print("history.history:")
    print(history.history)

    precisions = None
    recalls = None
    if i == 0:
        precisions = history.history['precision']
        recalls = history.history['recall']
    else:
        precisions = history.history['precision_%ld' % i]
        recalls = history.history['recall_%ld' % i]
    losses = history.history['loss']
    accuracies = history.history['accuracy']

    f1_scores = util.get_f1_scores(precisions, recalls)

    # put the test data in here
    results = model.evaluate(x=test_X, y=test_y)
    print("results evaluated:")
    print(results)

    # plot loss and f1 score
    util.plot(make_epoch_list(len(losses)), losses, key, ax_loss)
    util.plot(make_epoch_list(len(losses)), f1_scores, key, ax_f1)
    util.plot(make_epoch_list(len(accuracies)), accuracies, key, ax_acc)

    predictions = model.predict(pred_X)
    print("predictions:")
    print(predictions)
    print("pred_y:")
    print(pred_y)
    d = {'predictions': util.get_preds_from_probs(predictions), 'truth': util.get_preds_from_probs(pred_y)}
    pred_df = pd.DataFrame(d)
    pred_df.to_csv(path_or_buf="results_predictions_set_%s.csv" % key, sep=",", index=False)

util.save_clear_plt("nn_loss_lr_%f.png" % lr, ax_loss, fig_loss)
util.save_clear_plt("nn_f1_lr_%f.png" % lr, ax_f1, fig_f1)
util.save_clear_plt("nn_acc_lr_%f.png" % lr, ax_acc, fig_acc)
