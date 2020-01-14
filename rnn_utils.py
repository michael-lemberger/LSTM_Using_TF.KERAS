import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential


def prep_data(train_siz, test_siz):

    cols = ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm', 'mode', 'centroid',
            'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx', 'label']

    df = pd.read_csv('voice.csv', header=None, names=cols)


    df = df.drop(df.index[0])

    # Encode class
    class_name = ['male', 'female']
    df['label_num'] = [class_name.index(class_str)
                            for class_str in df['label'].values]

    # Random Shuffle before split to train/test
    orig = np.arange(len(df))
    perm = np.copy(orig)
    np.random.shuffle(perm)
    data = df[
        ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm', 'mode', 'centroid',
         'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx', 'label_num']].values
    data[orig, :] = data[perm, :]

    # Split dataset
    trX = data[:train_siz, :-1]
    teX = data[train_siz:, :-1]
    trY = data[:train_siz, -1]
    teY = data[train_siz:, -1]
    trX3 = np.expand_dims(trX, axis=2)
    teX3 = np.expand_dims(trX, axis=2)
    return trX3, trY, teX3, teY


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prep_data(train_siz=1584, test_siz=1584)

    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.summary()

    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(lr=1e-3 , decay=1e-5)
    model.compile(loss='kullback_leibler_divergence', optimizer=opt, metrics=['Accuracy'])

    model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))
