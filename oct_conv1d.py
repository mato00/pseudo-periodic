import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from shift_conv1d import ShiftConv1D


class OctConv1D(layers.Layer):
    def __init__(self, filters, alpha, beta,
                 kernel_size=3, strides=1, padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):

        assert alpha >= 1
        assert beta >= 0 and beta <= 1
        assert filters > 0 and isinstance(filters, int)
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.filters = filters
        # optional values
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        # -> Low channels
        self.low_channels = int(self.filters * self.beta)
        # -> high channels
        self.high_channels = self.filters - self.low_channels

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3 and len(input_shape[1]) == 3
        # Assertion for high inputs
        assert input_shape[0][1] // self.alpha >= self.kernel_size
        # Assertion for low inputs
        assert input_shape[0][1] // input_shape[1][1] == self.alpha
        # channel last for Tensorflow
        assert K.image_data_format() == 'channels_last'

        high_in = int(input_shape[0][2])
        low_in = int(input_shape[1][2])

        # High -> High
        self.high_to_high_kernel = self.add_weight(name="high_to_high_kernel",
                                                   shape=(self.kernel_size, high_in, self.high_channels),
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.kernel_regularizer,
                                                   constraint=self.kernel_constraint)
        # High -> Low
        self.high_to_low_kernel = self.add_weight(name="high_to_low_kernel",
                                                  shape=(self.kernel_size, high_in, self.low_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        # Low -> High
        self.low_to_high_kernel = self.add_weight(name="low_to_high_kernel",
                                                  shape=(self.kernel_size, low_in, self.high_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        # Low -> Low
        self.low_to_low_kernel = self.add_weight(name="low_to_low_kernel",
                                                 shape=(self.kernel_size, low_in, self.low_channels),
                                                 initializer=self.kernel_initializer,
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)
        super().build(input_shape)

    def call(self, inputs):
        # Input = [X^H, X^L]
        assert len(inputs) == 2
        high_input, low_input = inputs
        # High -> High conv
        high_to_high = K.conv1d(high_input, self.high_to_high_kernel,
                                strides=self.strides, padding=self.padding,
                                data_format="channels_last")
        # High -> Low conv
        high_to_low = layers.AveragePooling1D(self.alpha, strides=self.alpha, padding="same")(high_input)
        high_to_low = K.conv1d(high_to_low, self.high_to_low_kernel,
                               strides=self.strides, padding=self.padding,
                               data_format="channels_last")
        # Low -> High conv
        low_to_high = K.conv1d(low_input, self.low_to_high_kernel,
                               strides=self.strides, padding=self.padding,
                               data_format="channels_last")
        low_to_high = K.repeat_elements(low_to_high, self.alpha, axis=1)
        # Low -> Low conv
        low_to_low = K.conv1d(low_input, self.low_to_low_kernel,
                               strides=self.strides, padding=self.padding,
                               data_format="channels_last")
        # Cross Add
        high_add = tf.concat([high_to_high, low_to_high], axis=-1)
        low_add = tf.concat([high_to_low, low_to_low], axis=-1)

        return [high_add, low_add]

    def compute_output_shape(self, input_shapes):
        high_in_shape, low_in_shape = input_shapes
        high_out_shape = (high_in_shape[: 2], self.high_channels)
        low_out_shape = (low_in_shape[: 2], self.low_channels)

        return [high_out_shape, low_out_shape]

    def get_config(self):
        base_config = super().get_config()
        out_config = {
            **base_config,
            "filters": self.filters,
            "alpha": self.alpha,
            "beta": self.beta,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
        }
        return out_config
