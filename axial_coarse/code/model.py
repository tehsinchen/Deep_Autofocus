import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential


def model_axial_coarse(input_shape, category):
    model = tf.keras.applications.MobileNet(
        alpha=1.2,
        include_top=False,
        weights=None,
        input_shape=input_shape
    )
    x = layers.GlobalMaxPool2D()(model.layers[-1].output)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(category, activation="softmax")(x)
    model = Model(model.inputs, x)
    model.summary()
    return model


if __name__ == '__main__':
    model_axial_coarse((256, 256, 3), 3)

