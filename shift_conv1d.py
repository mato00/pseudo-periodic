from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class ShiftConv1D(layers.Layer):
    def __init__(self, filters, alpha, low_tau, high_tau, fs
                 kernel_size=3, strides=1, padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):

        assert alpha >= 0 and alpha <= 1
        assert filters > 0 and isinstance(filters, int)
        super().__init__(**kwargs)

        self.alpha = alpha
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

        # -> Low channels
        self.low_channels = int(self.filters * self.alpha)
        # -> high channels
        self.high_channels = self.filters - self.low_channels

        # -> Low shift length
        self.low_shift = self.low_tau * fs / 2
        # -> High shift length
        self.high_shift = self.high_tau * fs

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3 and len(input_shape[1]) == 3
        # Assertion for high inputs
        assert input_shape[0][1] // 2 >= self.kernel_size
        # Assertion for low inputs
        assert input_shape[0][1] // input_shape[0][1] == 2
        # channel last for Tensorflow
        assert K.image_data_format() == 'channels_last'
        # input_channels
        high_in = int(input_shape[0][2])
        low_in = int(input_shape[1][2])

        # High -> High
        self.high_to_high_kernel = self.add_weight(name='high_to_high_kernel',
                                                   shape=(*self.kernel_size, high_in, self.high_channels),
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.kernel_regularizer,
                                                   constraint=self.kernel_constraint)
        # High -> Low
        self.high_to_low_kernel = self.add_weight(name="high_to_low_kernel",
                                                  shape=(*self.kernel_size, high_in, self.low_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        # Low -> High
        self.low_to_high_kernel = self.add_weight(name="low_to_high_kernel",
                                                  shape=(*self.kernel_size, low_in, self.high_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        # Low -> Low
        self.low_to_low_kernel = self.add_weight(name="low_to_low_kernel",
                                                 shape=(*self.kernel_size, low_in, self.low_channels),
                                                 initializer=self.kernel_initializer,
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)

        super().build(input_shape)

    def call(self, inputs):
        # Input = [X^H, X^L]
        assert len(inputs) == 2
        high_input, low_input = inputs
