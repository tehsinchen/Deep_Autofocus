import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import efficientnet


def model_axial_fine(input_shape, category):
    model = efficientnet.EfficientNetB2(
        include_top=True,
        weights=None,
        pooling='avg',
        input_shape=input_shape,
        classes=category,
        classifier_activation='sigmoid'
    )
    model.summary()
    return model


if __name__ == '__main__':
    model_axial_fine((256, 256, 7), 7)

