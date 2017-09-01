# this script trains a model for "SINGLE" chord estimation based on the feature extraction results

import os
import sys
import numpy
import cPickle
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Merge, Lambda, Activation, LSTM
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, BatchNormalization
from keras.optimizers import Adadelta, Adam, RMSprop
from keras import losses
from keras import regularizers
from keras import optimizers
from keras import backend as K
from keras.utils import plot_model

from sklearn import preprocessing
from sklearn.externals import joblib
import h5py

from ..helpers import helpers

def robustscaler(X):
    nsamples = X.shape[0]
    nseg = X.shape[1]
    nfeature = X.shape[2]
    oldshape = X.shape
    newshape = (nsamples, nseg * nfeature)
    X = X.reshape(newshape)
    scaler = preprocessing.RobustScaler().fit(X)  # any other ideas?
    joblib.dump(scaler,'scaler.pkl')
    X = scaler.transform(X)
    X = X.reshape(oldshape)
    return X

def datasplit(X,Y,split):
    numpy.random.seed(6)
    nsamples = len(X)
    ranidxes = numpy.random.permutation(nsamples)
    X = X[ranidxes]
    Y = Y[ranidxes]
    train = (X[:int(split*nsamples)],Y[:int(split*nsamples)])
    test = (X[int(split*nsamples):],Y[int(split*nsamples):])
    return train,test


def load_h5py(root, datasets,traintype):
    # each dataset are in the shape of: (n_samples, n_seg, n_feature)

    for dataset in datasets:
        h5path = root + dataset + '.h5'
        print h5path

        with h5py.File(h5path, "r") as f:
            if traintype == 'cg':
                print "load chromagram data..."
                Xcg = f['Xcg'][:]
                Ycg = f['Ycg'][:]
                print 'Xcg.shape', Xcg.shape
                print 'Ycg.shape', Ycg.shape
            if traintype == 'cqt':
                print "loading cqtspec data..."
                Xcqt = f['Xcqt'][:]
                Ycqt = f['Ycqt'][:]
                print 'Xcqt.shape', Xcqt.shape
                print 'Ycqt.shape', Ycqt.shape

        try:
            if traintype == 'cg':
                Xcgall = numpy.concatenate((Xcgall,Xcg))
                Ycgall = numpy.concatenate((Ycgall,Ycg))
            if traintype == 'cqt':
                Xcqtall = numpy.concatenate((Xcqtall,Xcqt))
                Ycqtall = numpy.concatenate((Ycqtall,Ycqt))
        except:
            if traintype == 'cg':
                Xcgall = Xcg
                Ycgall = Ycg
            if traintype == 'cqt':
                Xcqtall = Xcqt
                Ycqtall = Ycqt

    if traintype == 'cg':
        print "nan to num..."
        Xcgall = numpy.nan_to_num(Xcgall)
        print 'Xcgall minmax:',numpy.min(Xcgall),numpy.max(Xcgall)
        print 'Ycgall minmax:', numpy.min(Ycgall), numpy.max(Ycgall)

        # print "standardizing..."
        # Xcgall = robustscaler(Xcgall)
        # print 'Xcgall minmax:', numpy.min(Xcgall), numpy.max(Xcgall)

        print "outputing..."
        print 'Xcgall.shape', Xcgall.shape
        print 'Ycgall.shape', Ycgall.shape

        return datasplit(Xcgall,Ycgall,0.8)

    if traintype == 'cqt':
        print "nan to num..."
        Xcqtall = numpy.nan_to_num(Xcqtall)
        print 'Xcqtall minmax:', numpy.min(Xcqtall), numpy.max(Xcqtall)
        print 'Ycqtall minmax:', numpy.min(Ycqtall), numpy.max(Ycqtall)

        # print "standardizing..."
        # Xcqtall = robustscaler(Xcqtall)
        # print 'Xcqtall minmax:', numpy.min(Xcqtall), numpy.max(Xcqtall)

        print "outputing..."
        print 'Xcqtall.shape', Xcqtall.shape
        print 'Ycqtall.shape', Ycqtall.shape

        return datasplit(Xcqtall, Ycqtall, 0.8)

def gtp(x):
    p1 = K.mean(x, axis=1)
    p2 = K.max(x, axis=1)
    # p3 = x.norm(2,axis=1)
    return K.concatenate([p1, p2], axis=1)

def gtp_output_shape(input_shape):  # make sure to compute this correctly
    shape = list(input_shape)
    return (None, shape[-1] * 2)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        "please provide two arguments: datasets(e.g. CJK), and traintype(cg or cqt)"
        quit()

    root = '../data/feature/'
    datasets = sys.argv[1]
    traintype = sys.argv[2]

    assert((traintype=='cg' or traintype =='cqt') and 'only cg or cqt is allowed to be the traintype')

    (x_train, y_train), (x_test, y_test) = load_h5py(root, datasets, traintype)

    print "data shape:"
    print 'x_train.shape', x_train.shape
    print 'y_train.shape', y_train.shape
    print 'x_test.shape', x_test.shape
    print 'y_test.shape', y_test.shape

    batch_size = 32
    epochs = 200
    dropoutrate = 0.25
    num_classes = helpers.nchords + 1 # plus the "N" chord
    modelname = datasets + '.' + traintype + '.model'
    modelplot = datasets + '.' + traintype + '.png'
    modelhistory = datasets + '.' + traintype + '.history'

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    nsample_train = x_train.shape[0]
    nsample_test = x_test.shape[0]
    nframe = x_train.shape[1]
    nbin = x_train.shape[2]


    # for fcnn
    # x_train = x_train.reshape(nsample_train,nframe*nbin) # flatten the samples
    # x_test = x_test.reshape(nsample_test, nframe * nbin) # flatten the samples

    # print x_train.shape, 'train shape x'
    # print x_test.shape, 'test shape x'

    # print y_train.shape, 'train shape y'
    # print y_test.shape, 'test shape y'

    # input_shape = (nframe * nbin,)
    # model = Sequential()
    # model.add(Dense(512, activation='relu', input_shape=input_shape))
    # model.add(Dropout(dropoutrate))
    #
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(dropoutrate))

    # for cnn or rnn
    print x_train.shape, 'train shape x'
    print x_test.shape, 'test shape x'

    print y_train.shape, 'train shape y'
    print y_test.shape, 'test shape y'
    input_shape = (nframe, nbin)
    model = Sequential()
    # model.add(LSTM(512,
    #                input_shape=input_shape,
    #                dropout=dropoutrate,
    #                recurrent_dropout=dropoutrate,
    #                return_sequences=True # if true, return the whole sequence, else return the last output
    #                ))
    model.add(LSTM(512,
                   input_shape=input_shape,
                   dropout=dropoutrate,
                   recurrent_dropout=dropoutrate,
                   #return_sequences=True # if true, return the whole sequence, else return the last output
                   ))

    model.add(Dense(num_classes, activation='softmax'))

    print ("--------------------------------")
    print ("Input Shape:" + str(input_shape))
    model.summary()

    # compile the model
    sgd = optimizers.SGD(lr=1e-6, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        # loss=keras.losses.kullback_leibler_divergence,
        metrics=['accuracy'],
        # optimizer=sgd
        # optimizer=keras.optimizers.Adam()
        # optimizer=keras.optimizers.Adadelta()
        # optimizer=keras.optimizers.RMSprop()
        # optimizer=keras.optimizers.Adagrad()
        optimizer=keras.optimizers.Adamax()
        # optimizer=keras.optimizers.Nadam()
    )

    checkPoint = keras.callbacks.ModelCheckpoint(modelname,
                                                 monitor='val_acc',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='auto',
                                                 period=1)

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                                                  min_delta=0,
                                                  patience=10,
                                                  verbose=1,
                                                  mode='auto')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        # callbacks=[checkPoint],
                        callbacks=[checkPoint, earlyStopping],
                        validation_data=(x_test, y_test)
                        )

    with open(modelhistory, 'wb') as f:
        cPickle.dump(
            [history.history['loss'], history.history['val_loss'], history.history['acc'], history.history['val_acc']],
            f, -1)

    model = keras.models.load_model(modelname)
    plot_model(model, to_file=modelplot)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
