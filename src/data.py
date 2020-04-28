import os
from timeit import default_timer as timer
from PIL import Image
from ftplib import FTP
import numpy as np

import logging
log = logging.getLogger(__name__)

# Dataset splitting into train(with validation) and test/validation sub-arrays
# IN - dataframe : pd.DataFrame, samples per each class : int, koef : float(0 to 1) for train sub-array size.
# OUT - dict with keys 'train' and 'not_train', values - lists of directories of samples on the ftp-server
def data_split(df, samples_per_class=20, train_koef=0.7):

    log.info('Splitting data...')

    class_list = list(df['additive'].unique())
    classes = {}
    train_dirs, test_dirs = [], []

    for item in class_list:
        classes[item] = list(df[ df['additive']==item ]['dir'].unique())

    for one_class in classes:
        samples = np.random.permutation(classes[one_class])[:int(samples_per_class)]
        train, test = np.split(samples, [int(samples_per_class*train_koef)])
        train_dirs += list(train)
        test_dirs += list(test)

    log.info('Successfull splitting!')

    return {'train': train_dirs, 'not_train': test_dirs}

# Downloading data to server or client from ftp
# IN - data splitted on train and not_train parts : dict,
#      fragments count per each sample of image: int(0 to 6972),
#      only keys of data dict to be downloaded: list
# OUT - finish message (and data, downloaded on the client or server. path: '.../server_data/...')
def data_downloading(splitted_data, fragments_per_sample=100, special_keys=[]):

    log.info('Downloading train data from FTP server...')

    log.info('Connecting to FTP server...')
    ftp = FTP()
    ftp.connect('83.149.249.48')
    ftp.login()
    try:
        ftp.sendcmd('PORT 83,149,249,48,192,92')
    except:
        log.info('sendcmd PORT failed!')

    log.info('Connection succeed!')

    log.info('Downloading data...')

    for sub_array in splitted_data:
        if len(special_keys):
            if sub_array not in special_keys:
                continue

        path = '/src/server_data/' + sub_array

        os.makedirs(path, exist_ok=True)

        log.info('Dirs were made!')

        start = timer()

        for one_dir in splitted_data[sub_array]:

            ftp.cwd('/dataset/dataset_fragments/' + str(one_dir))

            filenames = ftp.nlst()
            filenames = np.random.permutation(filenames)

            for filename in filenames[:fragments_per_sample]:
                with open(path + '/' + filename, 'wb') as file:
                    ftp.retrbinary('RETR ' + filename, file.write)
        end = timer()

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
            im = list(Image.open(path+'/'+image).getdata(0))
            class_name = class_dict[df.iloc[int(image[:-4]) - 1, ]['additive']]
            X_[sub_array].append(im)
            Y_[sub_array].append(class_name)

    log.info('Images data received!')

    return X_, Y_, class_dict
