from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization
from keras.utils import np_utils
import keras
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

nb_epochs = 2000
all_result_file = 'all_results.txt'

flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z',
'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics',
'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1',
'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols',
'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

#flist  = ['Adiac']
for each in flist:
    fname = each
    x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
    x_test, y_test = readucr(fname+'/'+fname+'_TEST')
    nb_classes = len(np.unique(y_test))
    batch_size = min(x_train.shape[0]/10, 16)

    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)


    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean)/(x_train_std)

    x_test = (x_test - x_train_mean)/(x_train_std)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    print(x_train.shape)

    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(LSTM(32))
    model.add(keras.layers.Dense(nb_classes, activation='softmax'))

    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', keras.metrics.categorical_accuracy])

    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                      patience=5, min_lr=0.0001)
    early_stop = EarlyStopping(monitor = 'val_acc', min_delta = 0, patience = 50, mode='max')

    hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
              verbose=1, validation_data=(x_test, Y_test), callbacks = [reduce_lr, early_stop])

    #Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    log.to_csv('./history/'+fname+'_BiDir_LSTM_all_history.csv')

    with open(all_result_file,"a") as f:
        f.write(fname+", BiDirLSTM"+", "+str(log.loc[log['loss'].idxmin]['loss'])+", "
                +str(log.loc[log['loss'].idxmin]['val_acc'])+"\n")

    # summarize history for accuracy
    plt.plot(log['acc'])
    plt.plot(log['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./history/'+fname+'_BiDirLSTM_model_accuracy.png')
    plt.close()
    # summarize history for loss
    plt.plot(log['loss'])
    plt.plot(log['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./history/'+fname+'_BiDirLSTM_model_loss.png')