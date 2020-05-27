from tensorflow.keras.models import Model # basic class for specifying and training a neural network
from tensorflow.keras.layers import Input, MaxPooling2D, Dense, Dropout, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

import logging
log = logging.getLogger(__name__)

def tcnn_compile(conv_layers_count=2, num_classes=6):

    log.info('Creating tcnn model...')

    inp = Input(shape=(256, 256, 1))
    inp_norm = BatchNormalization()(inp)

    # Conv [96] -> Pool
    conv_1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
                    kernel_initializer='he_uniform',
                    activation="relu", padding="valid")(inp_norm)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_1)
    pool_1 = BatchNormalization()(pool_1)

    # Conv [256] -> Energy
    conv_2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
                    kernel_initializer='he_uniform',
                    activation="relu", padding="valid")(pool_1)

    if conv_layers_count == 2:
        conv_last = conv_2
        kernel_size = 27
    elif conv_layers_count == 3:
        pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_2)
        pool_2 = BatchNormalization()(pool_2)

            # Conv [64] -> Energy
        conv_last = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                    kernel_initializer='he_uniform',
                    activation="relu", padding="valid")(pool_2)
        kernel_size = 11

    conv_last = BatchNormalization()(conv_last)

    energy = AveragePooling2D(pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='valid')(conv_last)

    energy = BatchNormalization()(energy)

    # FC -> FC -> softmax
    flat = Flatten()(energy)
    hidden_1 = Dense(4096,
                     kernel_initializer='he_uniform',
                     activation='relu')(flat) # 4096 in origin
    drop_1 = Dropout(0.4)(hidden_1)
    hidden_2 = Dense(4096,
                     kernel_initializer='he_uniform',
                     activation='relu')(drop_1) # 4096 in origin
    drop_2 = Dropout(0.4)(hidden_2)

    out = Dense(num_classes,
                kernel_initializer='glorot_uniform',
                activation='softmax')(drop_2)

    # optimizer
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model = Model(inputs=inp, outputs=out)

    log.info('Model has been created!')
    log.info('Compiling model...')

    model.compile(loss='categorical_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])

    log.info('Model successfully compiled!')

    return model
