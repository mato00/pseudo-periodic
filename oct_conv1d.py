from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from shift_conv1d import ShiftConv1D


class OctConv1D(layers.Layer):
    def __init__(self, filters, alpha, beta, low_tau, high_tau, fs,
                 kernel_size=3, strides=1, padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):

        assert alpha >= 0 and alpha <= 1
        assert filters > 0 and isinstance(filters, int)
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.filters = filters
        self.low_tau = low_tau
        self.high_tau = high_tau
        self.fs = fs
        # optional values
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        # -> Low fs
        self.low_fs = int(self.fs / self.alpha)
        # -> High fs
        self.high_fs = self.fs

        # -> Low channels
        self.low_channels = int(self.filters * self.beta)
        # -> high channels
        self.high_channels = self.filters - self.low_channels

        # -> Low shift length
        self.low_shift = self.low_tau * fs / self.alpha
        # -> High shift length
        self.high_shift = self.high_tau * fs

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3 and len(input_shape[1]) == 3
        # Assertion for high inputs
        assert input_shape[0][1] // self.alpha >= self.kernel_size
        # Assertion for low inputs
        assert input_shape[0][1] // input_shape[1][1] == self.alpha
        # channel last for Tensorflow
        assert K.image_data_format() == 'channels_last'

        super().build(input_shape)

    def call(self, inputs):
        # Input = [X^H, X^L]
        assert len(inputs) == 2
        high_input, low_input = inputs
        # High -> High conv
        high_to_high = ShiftConv1D(self.high_channels,
                                   self.high_tau,
                                   self.high_fs,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding=self.padding)(high_input)
        # High -> Low conv
        high_to_low = ShiftConv1D(self.low_channels,
                                  self.high_tau,
                                  self.high_fs,
                                  self.kernel_size,
                                  strides=self.strides,
                                  padding=self.padding)(high_input)
        high_to_low = K.pool1d(high_to_low,
                               self.alpha,
                               strides=self.alpha,
                               pool_mode='max')
        # Low -> High conv
        low_to_high = ShiftConv1D(self.high_channels,
                                  self.low_tau,
                                  self.low_fs,
                                  self.kernel_size,
                                  strides=self.strides,
                                  padding=self.padding)(low_input)
        low_to_high = K.repeat_elements(low_to_high, self.alpha, axis=1)
        # Low -> Low conv
        low_to_low = ShiftConv1D(self.low_channels,
                                 self.low_tau,
                                 self.low_fs,
                                 self.kernel_size,
                                 strides=self.strides,
                                 padding=self.padding)(low_input)
        # Cross Add
        high_add = high_to_high + low_to_high
        low_add = high_to_low + low_to_low

        return [high_add, low_add]

    def compute_output_shape(self, input_shapes):
        high_in_shape, low_in_shape = input_shapes
        high_out_shape = (*high_in_shape[: 2], self.high_channels)
        low_out_shape = (*low_in_shape[: 2], self.low_channels)

        return [high_out_shape, low_out_shape]

    def get_config(self):
        base_config = super().get_config()
        out_config = {
            **base_config,
            "filters": self.filters,
            "alpha": self.alpha,
            "beta": self.beta,
            "low_tau": self.low_tau,
            "high_tau": self.high_tau,
            "fs": self.fs,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
        }
        return out_config
