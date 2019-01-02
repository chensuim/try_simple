import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras import backend as K


def precision(y_true, y_pred):
    """Precision metric.
     Only computes a batch-wise average of precision.
    """
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    """
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_score(y_true, y_pred):
    """Computes the F score.
     The F1 score is harmonic mean of precision and recall.
     it is computed as a batch-wise average.
     This is can be used for multi-label classification. 
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1_score = 2 * p * r / (p+r)
    return f1_score


def tf_round(x):
    sub = tf.constant(1, dtype=x.dtype)
    return tf.round(x + sub)

"""
def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32")
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(tf.reduce_sum(y_true, axis=1), axis=0)
    sum_pred = tf.reduce_sum(tf.reduce_sum(y_pred, axis=1), axis=0)
    sum_correct = tf.reduce_sum(tf.reduce_sum(y_correct, axis=1), axis=0)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 2 * precision * recall / (precision + recall)
    return (f_score)


def recall(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32")
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(tf.reduce_sum(y_true, axis=1), axis=0)
    sum_correct = tf.reduce_sum(tf.reduce_sum(y_correct, axis=1), axis=0)
    recall_ = sum_correct / sum_true
    # recall_ = tf.where(tf.is_nan(recall_), tf.zeros_like(recall_), recall_)
    return (recall_)


def precision(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32")
    y_correct = y_true * y_pred
    sum_pred = tf.reduce_sum(tf.reduce_sum(y_pred, axis=1), axis=0)
    sum_correct = tf.reduce_sum(tf.reduce_sum(y_correct, axis=1), axis=0)
    precision_ = sum_correct / sum_pred
    #precision_ = tf.where(tf.is_nan(precision_), tf.zeros_like(precision_), precision_)
    return (precision_)
"""


class DataGen(keras.utils.Sequence):
    def __init__(self, lines, batch_size, num_classes):
        self._data = lines
        self._num_classes = num_classes
        self._l = int(np.ceil(float(len(lines)) / batch_size))
        self._batch_size = batch_size
        self._len_lines = len(lines)

    def on_epoch_end(self, *args, **kwargs):
        return

    def deal_line(self, _id):
        line = self._data[_id]
        breaks = line.split()
        labels = list()
        vec = list()
        for bb in breaks:
            if bb.startswith('__label__'):
                labels.append(int(bb[9:]))
            else:
                vec.append(float(bb))
        return (vec, labels)

    def __len__(self):
        return self._l

    def __getitem__(self, idx):
        f_ids = range(idx * self._batch_size, min((idx+1) * self._batch_size, self._len_lines))
        return self._get_data_by_ids(f_ids)

    def _get_data_by_ids(self, _ids):
        X, y = [], []
        for idx in _ids:
            X_i, y_i = self.deal_line(idx)
            y_i = np.sum(keras.utils.to_categorical(y_i, num_classes=self._num_classes), axis=0)
            X.append(X_i)
            y.append(y_i)
        X = np.array(X)
        y = np.array(y)
        return X, y


def get_model(num_class):
    model = Sequential()
    model.add(Dense(1024, input_dim=768, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_class, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score, precision, recall])
    print(model.summary())
    return model


def fit(model, train_gen, val_gen):
    model.fit_generator(generator=train_gen, steps_per_epoch=len(train_gen), validation_data=val_gen, validation_steps=len(val_gen), epochs=10, use_multiprocessing=False)
    model.save('./res/simple_model.h5')


def predict(model, test):
    loss_and_metrics = model.evaluate_generator(generator=test, steps=len(test))
    print(loss_and_metrics)


def main(num_classes, fn, train_ratio=0.85, val_ratio=0.1):
    num_classes = int(num_classes)
    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    with open(fn, 'r') as f:
        lines = f.readlines()
        train_num = int(len(lines) * train_ratio)
        val_num = int(len(lines) * val_ratio)
        train = lines[:train_num]
        val = lines[train_num:train_num+val_num] 
        test = lines[train_num+val_num:]
    train_gen = DataGen(train, 128, num_classes)
    test_gen = DataGen(test, 128, num_classes)
    val_gen = DataGen(val, 128, num_classes)
    print(len(train_gen), len(test_gen), len(val_gen))
    model = get_model(num_classes)
    print(model.metrics_names)
    fit(model, train_gen, val_gen)
    predict(model, test_gen)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(*sys.argv[1:]))
