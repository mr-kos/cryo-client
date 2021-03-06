import os
import sys
import csv
import numpy as np
import pandas as pd
import logging
from timeit import default_timer as timer
import time

logging.basicConfig(filename="/root/shared/results/main.log", level=logging.INFO, filemode="w")
log = logging.getLogger(__name__)

from tensorflow.keras.models import load_model
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
log.info('Main modules were imported')

from data import data_split, data_downloading, get_image_data

log.info('Custom modules were imported')
seed = np.random.randint(1000)

def main():

    log.info('Starting program..')

    log.info('Current dir ' + os.getcwd())

    log.info('Seed = {}'.format(seed))

    download_status = False
    downloading_cycles = 0
    try:
        df = pd.read_csv('/src/fragments3-lite-train.csv')
    except Exception as e:
        log.exception('Exception during reading CSV-file')
        return 1
    log.info('CSV-file with data structure was readed')

    # READ DATAFRAME WITH TRAIN SAMPLES AND SPLIT IT BEFORE DOWNLOADING
    # AND RECEIVING IMAGE DATA
    log.info('Splitting data into train and val subsets...')
    try:
        splitted_data = data_split(df, samples_per_class=40,
                  split_koeffs=[1],
                  arrays_labels=['train', 'none'], seed=seed)
    except Exception as e:
        log.exception('Error during splitting')
        return 1
    while download_status == False:
        downloading_cycles += 1
        try:
            data_downloading(ip='83.149.249.48', splitted_data=splitted_data, fragments_per_sample=50, seed=seed)
            download_status = True
        except Exception as e:
            log.exception('Error during data downloading')
            if downloading_cycles > 5:
                log.info("Bad connection. Downloading stops!")
                return 1
            time.sleep(600)
    try:
        X_, Y_, class_dict = get_image_data(df)
    except:
        log.exception('Error during image data receiving')
        return 1

    log.info('Image data received')

    log.info('Start data processing..')
    try:
        new_class_dict = {}
        Yn_ = {}

        for key in class_dict:
            new_class_dict[class_dict[key]] = key

        num_classes = len(new_class_dict)

        for key in Y_:
            Yn_[key] = np_utils.to_categorical(Y_[key], num_classes)
    except Exception as e:
        log.exception('Data processing failed!')
        return 1
    # T-CNN(2) implementation
    log.info('Data processing was finished')
    try:
        model = load_model('/src/global_tcnn2_crio_3.h5')
    except Exception as e:
        log.exception('Exception during model loading')
        return 1

    log.info('Initializing model callbacks..')
    try:
        csv_logger = CSVLogger('/root/shared/results/tcnn_crio.log')
    except Exception as e:
        log.exception('Callbacks were not initialized')
        return 1

    log.info('Initializing completed')
    log.info('Fitting model..')
    try:
        model.fit(X_['train'], Yn_['train'],
                    batch_size=16, epochs=10,
                    verbose=1, shuffle=True,
                    callbacks=[csv_logger])
    except Exception as e:
        log.exception('Fitting was failed!')
        return 1

    log.info('Success!')

    model.save("/root/shared/results/global_tcnn2_crio_3.h5")

    log.info('Model was saved!')

    log.info('FINISH')

if __name__ == "__main__":
    sys.exit(main())
