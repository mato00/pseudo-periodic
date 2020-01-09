import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class ShiftConv1D(layers.layer):
    def __init__(self, filters, tau, fs,
                 kernel_size=3, strides=1, padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        assert tau >= 0 and fs > 0
        assert filters > 0 and isinstance(filters, int)
        super().__init__(**kwargs)

        self.filters = filters
        self.tau = tau
        self.fs = fs
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        self.shift = int(self.tau * self.fs)
        self.input_channels = int(input_shape[1] // self.shift)

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[1] // 2 >= self.kernel_size

        # channel last for Tensorflow
        assert K.image_data_format() == 'channels_last'

        self.kernel = self.add_weight(name='kernel',
                                      shape=(*self.kernel_size, self.input_channels, self.filters),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        super().build(input_shape)

    def call(self, inputs):
        left_shift = tf.roll(inputs, self.shift, axis=1)
        right_shift = tf.roll(inputs, (-1)*self.shift, axis=1)
        shift_inputs = tf.squeeze(tf.stack([inputs, left_shift, right_shift], axis=-2))
