import os
import math
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from model2 import model_axial_coarse


def step_decay_schedule(initial_lr, decay_factor, step_size):

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)


def get_data(folder_path):
    data_path = os.path.join(folder_path, 'x_train.npy')
    label_path = os.path.join(folder_path, 'y_train.npy')
    print('data loading...')
    with open(data_path, 'rb') as f:
        train_data = np.load(f)
    with open(label_path, 'rb') as f:
        label_data = np.load(f).astype(np.float32)
    return train_data, label_data


def get_split_data(train_data, label_data, batch_size):
    train_step = math.ceil(int(label_data.shape[0] * 0.8) / batch_size)
    train_size = train_step * batch_size
    x_train = train_data[:train_size, :, :, :]
    y_train = label_data[:train_size, :]

    val_x = train_data[train_size:, :, :, :]
    val_y = label_data[train_size:, :]
    val_step = math.ceil(val_y.shape[0] / batch_size)
    val_size = val_step * batch_size
    if val_size > val_y.shape[0]:
        val_step -= 1
        val_size = val_step * batch_size
    val_x = val_x[:val_size, :, :, :]
    val_y = val_y[:val_size, :]
    del train_data, label_data
    return x_train, y_train, val_x, val_y, train_step, val_step


def main():
    folder_path = os.getcwd().replace('code', 'data')
    model_path = os.path.join(folder_path, 'model', 'model_axial_coarse-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5')
    
    # a strategy to use multiple GPU
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    devices = tf.config.experimental.list_physical_devices("GPU")
    print(f'found GPU devices: {devices}')

    # data loading and split into training & validation
    train_data, label_data = get_data(folder_path)
    batch_size = 50 * strategy.num_replicas_in_sync
    x_train, y_train, val_x, val_y, train_step, val_step = get_split_data(train_data, label_data, batch_size)
    print(x_train.shape, y_train.shape, val_x.shape, val_y.shape, train_step, val_step)

    # load the model in each GPU
    with strategy.scope():
        model = model_axial_coarse((256, 256, 3), 3)
        model.compile(optimizer=Adam(epsilon=1e-8),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

    lr_sched = step_decay_schedule(initial_lr=0.008, decay_factor=0.8, step_size=100)
    model_checkpoint = ModelCheckpoint(model_path,
                                       monitor='val_categorical_accuracy',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max')
    model.fit(x_train,
              y_train,
              steps_per_epoch=train_step,
              epochs=2500,
              verbose=1,
              callbacks=[model_checkpoint, lr_sched],
              validation_data=(val_x, val_y),
              validation_steps=val_step,
              shuffle=True)


if __name__ == '__main__':
    main()
