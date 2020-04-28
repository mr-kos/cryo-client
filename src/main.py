import os
import sys
import csv
import numpy as np
import pandas as pd
import logging
from timeit import default_timer as timer

logging.basicConfig(filename="/root/shared/results/my_logs.log", level=logging.INFO, filemode="w")
log = logging.getLogger(__name__)

from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
log.info('Main modules were imported')

from model import tcnn_compile
from data import data_split, data_downloading, get_image_data

log.info('Custom modules were imported')
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def main():

    log.info('Starting program..')

    log.info('Current dir ' + os.getcwd())
    try:
        df = pd.read_csv('/src/fragments-train.csv')
    except Exception as e:
        log.exception('Exception during reading CSV-file')
        return 1
    log.info('CSV-file with data structure was readed')

    # READ DATAFRAME WITH TRAIN SAMPLES AND SPLIT IT BEFORE DOWNLOADING
    # AND RECEIVING IMAGE DATA
    try:
        splitted_data = data_split(df=df, samples_per_class=5)
    except Exception as e:
        log.exception('Error during splitting')
    try:
        data_downloading(splitted_data=splitted_data, fragments_per_sample=10)
    except Exception as e:
        log.exception('Error during data downloading')
    try:
        X_, Y_, class_dict = get_image_data(df)
    except:
        log.exception('Error during image data receiving')
    # X_, Y_, class_dict = get_image_data(df)
    log.info('Image data received')
    # Y_test = np.array(Y_['test'])


    # new_class_dict = {}
    # Yn_ = {}


    # for key in class_dict:
    #     new_class_dict[class_dict[key]] = key

    # num_classes = len(new_class_dict)


    # Fragments dataset
    # for key in X_:
    #     X_[key] = np.array(X_[key]) / 256
    #     X_[key] = X_[key].reshape(X_[key].shape[0], 256, 256, 1)
    #     X_[key] = np.float32(X_[key])
    # for key in Y_:
    #     Yn_[key] = np_utils.to_categorical(Y_[key], num_classes)

    # T-CNN(2) implementation

    # model = tcnn_compile(conv_layers_count=2, num_classes=num_classes)
    try:
        model = tcnn_compile(conv_layers_count=2, num_classes=6)
    except Exception as e:
        log.exception('Exception during model compiling')
        return 1

    # csv_logger = CSVLogger('tcnn_crio_frags.log')
    # early_stops = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    # model_ckpt = ModelCheckpoint(filepath='tcnn_crio.ckpt',
    #                          monitor='val_accuracy',
    #                          save_best_only=True,
    #                          verbose=1)

    # model.fit(X_['train'], Yn_['train'], batch_size=64, epochs=200,
    #                       validation_data=(X_['val'], Yn_['val']), verbose=1, shuffle=True,
    #                       callbacks=[csv_logger, early_stops, reduce_lr, model_ckpt])

    try:
        model.save("/root/shared/results/tcnn_crio_frags.h5")
    except Exception as e:
        log.exception('Exception during model saving')

    log.info('Model was saved!')

    log.info('FINISH')

if __name__ == "__main__":
    sys.exit(main())
