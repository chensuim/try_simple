import keras
import numpy as np


class DataGen(keras.utils.Sequence):
    def __init__(self, data, batch_size, num_classes):
        self._data = list()
        self._num_classes = num_classes
        for line in data:
            breaks = line.split()
            labels = list()
            vec = list()
            for bb in breaks:
                if bb.startswith('__label__'):
                    labels.append(int(bb[9:]))
                else:
                    vec.append(float(bb))
            self._data.append((vec, labels))

        self._l = int(np.ceil(float(len(lines)) / batch_size))
        self._batch_size = batch_size
        self._len_lines = len(lines)

    def __len__(self):
        return self._l

    def __getitem__(self, idx):
        f_ids = range(idx * self._batch_size, min((idx+1) * self._batch_size, self._len_lines))
        return self._get_data_by_ids(f_ids)

    def _get_data_by_ids(self, _ids):
        X, y = [], []
        for idx in _ids:
            X_i, y_i = self._data[idx]
            y_i = np.sum(keras.utils.to_categorical(y_i, num_classes=self._num_classes), axis=0)
            X.append(X_i)
            y.append(y_i)
        X = np.array(X)
        y = np.array(y)
        return X, y


def get_model(num_class):
    model = Sequential()
    model.add(Dense(128, input_dim=728, activation='relu'))
    model.add(Dense(num_class, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def fit(model, train_gen, val_gen):
    model.fit_generator(generator=train_gen, validation_data=val_gen, epochs=30, use_multiprocessing=True, workers=6)
    model.save('./res/simple_model.h5')


def main(num_classes, fn, train_ratio=0.85, val_ratio=0.1):
    train = list()
    test = list()
    val = list()
    with open(fn, 'r') as f:
        r = random.random()
        lines = f.readlines()
        for line in lines:
            if r < train_ratio:
                train.append(line)
            elif r < train_ratio + val_ratio:
                val.append(line)
            else:
                test.append(line)
    train_gen = DataGen(train, 128, num_classes)
    teat_gen = DataGen(test, 128, num_classes)
    val_gen = DataGen(val, 128, num_classes)
    model = get_model(num_classes)
    fit(model, train_gen, val_gen)
    predict(model, teat_gen)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(*sys.argv[1:]))
