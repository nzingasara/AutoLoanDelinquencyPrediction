import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.datasets import mnist
import util
import tensorflow as tf


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
file_name = "mortgage_data_small.csv"
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

###############################
input_shape = (11,)
layers_info1 = [(5, 'sigmoid'), (5, 'sigmoid'), (5, 'sigmoid'), (5, 'sigmoid'), (2, 'softmax')]
layers_info2 = [(5, 'relu'), (5, 'relu'), (5, 'relu'), (5, 'relu'), (2, 'softmax')]
layers_info3 = [(10, 'relu'), (10, 'relu'), (10, 'relu'), (10, 'relu'), (2, 'softmax')]
layers_info4 = [(20, 'relu'), (20, 'relu'), (20, 'relu'), (20, 'relu'), (2, 'softmax')]

layers_info_list = {"sigmoid_5_node_5_layer": layers_info1, "relu_5_node_5_layer": layers_info2,
                    "relu_10_node_5_layer": layers_info3, "relu_20_node_5_layer": layers_info4}

fig_loss, ax_loss = util.initialize_plot("Neural Network loss", "# epochs", "loss")
fig_f1, ax_f1 = util.initialize_plot("Neural Network F1 Score", "# epochs", "F1 Score")

for i, key in enumerate(layers_info_list):
    model = create_model(input_shape, layers_info1)

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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # load in data and put in the x and y parts
    history = model.fit(x=df_train_X.values, y=train_y, epochs=5)
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

    f1_scores = util.get_f1_scores(precisions, recalls)

    # put the test data in here
    results = model.evaluate(x=df_test_X.values, y=test_y)
    print("results evaluated:")
    print(results)

    # plot loss and f1 score
    util.plot(make_epoch_list(len(losses)), losses, key, ax_loss)
    util.plot(make_epoch_list(len(losses)), f1_scores, key, ax_f1)

util.save_clear_plt("nn_loss.png", ax_loss, fig_loss)
util.save_clear_plt("nn_f1.png", ax_f1, fig_f1)
