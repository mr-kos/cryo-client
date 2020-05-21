import os
from timeit import default_timer as timer
from PIL import Image
from ftplib import FTP
import numpy as np
import time

import logging
log = logging.getLogger(__name__)

def data_split(df, samples_per_class=20, split_koeffs=[0.8, 0.1], arrays_labels=['train', 'val', 'test'], seed=7):

    class_list = list(df['additive'].unique())
    classes = {}

    result_dict = {}
    split_num = len(split_koeffs) + 1
    for koeff_idx in range(split_num):
        result_dict[ arrays_labels[koeff_idx] ] = []

    for item in class_list:
        classes[item] = list(df[ df['additive']==item ]['dir'].unique())

    for one_class in classes:
        samples = np.random.RandomState(seed=seed).permutation(classes[one_class])[:int(samples_per_class)]
        splits = np.array_split(samples,
                                [int( len(samples)*sum(split_koeffs[:koeff_idx+1]) )
                                 for koeff_idx, _ in enumerate(split_koeffs)])
        for split_idx, split in enumerate(splits):
            result_dict[arrays_labels[split_idx]] += list(split)

    return result_dict

def ftp_connect(ip):
    log.info('Connecting to FTP server...')
    ftp = FTP()
    connect = ftp.connect(ip)
    log.info(connect)
    con_log = ftp.login()
    log.info(con_log)
    con_mode = ftp.sendcmd('PASV')
    log.info(con_mode)
    log.info('Connection succeed!')
    return ftp
# Downloading data to server or client from ftp
# IN - data splitted on train and not_train parts : dict,
#      fragments count per each sample of image: int(0 to 6972),
#      only keys of data dict to be downloaded: list
# OUT - finish message (and data, downloaded on the client or server. path: '.../server_data/...')
def data_downloading(ip, splitted_data, fragments_per_sample=100, special_keys=[], seed=7):
    con_cycles = 0
    con_status = False
    while con_status == False and con_cycles <= 12:
        con_cycles += 1
        try:
            ftp = ftp_connect(ip)
            con_status = True
        except Exception as e:
            log.exception('Error during connection to server')
            time.sleep(300)

    log.info('Downloading train data from FTP server...')

    for sub_array in splitted_data:

        log.info('Sub array == '+ str(sub_array))

        if len(special_keys):
            if sub_array not in special_keys:
                continue

        path = '/src/server_data/' + sub_array

        os.makedirs(path, exist_ok=True)

        log.info('Dirs were made!')

        start = int(timer())
        exist_files = 0

        for one_dir in splitted_data[sub_array]:

            ftp.cwd('/dataset/dataset_fragments/' + str(one_dir))

            filenames = ftp.nlst()
            filenames = np.random.RandomState(seed=seed).permutation(filenames)

            for filename in filenames[:fragments_per_sample]:
                file_path = path + '/' + filename
                file_size = ftp.size(filename)
                if os.path.exists(file_path) and os.path.getsize(file_path) == file_size:
                    exist_files += 1
                    continue
                with open(path + '/' + filename, 'wb') as file:
                    ftp.retrbinary('RETR ' + filename, file.write)

        end = int(timer())
        hours = (end - start) // 60 // 60
        minutes = (end - start) // 60 - hours * 60
        seconds = (end - start) - (end - start) // 60 * 60

        log.info('Data downloaded in "' + path + '"')
        log.info('Files, that were already exist: count = {}'.format(exist_files))
        log.info('Time: {:d} h. {:d} m. {:d} s.'.format(hours, minutes, seconds))

    ftp.close()

    log.info('Time: %.2f' % ( (end - start) // 60), ' m. ')

    log.info('Data dowloaded successfully!')

# Getting data for model from images
# IN - dataframe with image ids and class names
# OUT - X_, Y_ and class dictionary
def get_image_data(df):

    log.info('Mining data from images...')

    main_path = '/src/server_data'
    sub_arrays = os.listdir(path=main_path)
    classes = list(df['additive'].unique())
    class_dict = {}
    X_, Y_ = {}, {}

    for index, item in enumerate(classes):
        class_dict[item] = index

    for sub_array in sub_arrays:
        path = main_path+'/'+sub_array
        images = os.listdir(path=path)
        X_[sub_array] = []
        Y_[sub_array] = []

        for image in images:
            im = np.array(Image.open(path+'/'+image).getdata(0), dtype=np.uint8)
            class_name = class_dict[list(df[df['id']==int(image[:-4])]['additive'])[0]]
            X_[sub_array].append(np.array(im / 256, dtype=np.float32))
            Y_[sub_array].append(class_name)

        X_[sub_array] = np.array(X_[sub_array], dtype=np.float32)
        X_[sub_array] = X_[sub_array].reshape(X_[sub_array].shape[0], 256, 256, 1)

    log.info('Images data received!')

    return X_, Y_, class_dict
