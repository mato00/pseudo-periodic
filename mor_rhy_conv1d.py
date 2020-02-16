import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow.keras.backend as K

from ncausal_conv import NCausalConv1D


class MorRhyConv1D(layers.Layer):
    def __init__(self, filters, tau_mor, tau_rhy, fs,
                 alpha=0.5,
                 beta=2,
                 dropout=0.2,
                 strides=1,
                 mor_dilation=1,
                 rhy_dilation=1,
                 **kwargs):

        assert alpha >= 0 and alpha <= 1
        assert beta >= 1
        assert filters > 0 and isinstance(filters, int)
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.fs = fs
        self.filters = filters
        # optional values
        self.dropout = dropout
        self.strides = strides
        self.mor_dilation = mor_dilation
        self.rhy_dilation = rhy_dilation

        # -> Low channels
        self.low_channels = int(self.filters * self.alpha)
        # -> high channels
        self.high_channels = self.filters - self.low_channels

        self.low_fs = self.fs / self.beta
        self.high_fs = self.fs

        self.mor_kernel_size = (int(tau_mor * self.high_fs) - 1 //
                                self.mor_dilation) + 1
        self.rhy_kernel_size = (int(tau_rhy * self.low_fs) - 1 //
                                self.rhy_dilation) + 1

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3 and len(input_shape[1]) == 3
        # Assertion for high inputs
        assert input_shape[0][1] >= self.mor_kernel_size
        # Assertion for low inputs
        assert input_shape[0][1] // input_shape[1][1] == self.beta
        # channel last for Tensorflow
        assert K.image_data_format() == 'channels_last'

        self.conv_high_high = NCausalConv1D(filters=self.high_channels,
                                            kernel_size=self.mor_kernel_size,
                                            dilation_rate=self.mor_dilation,
                                            activation=tf.nn.leaky_relu,
                                            )
        self.conv_low_high = NCausalConv1D(filters=self.high_channels,
                                           kernel_size=self.rhy_kernel_size,
                                           dilation_rate=self.rhy_dilation,
                                           activation=tf.nn.leaky_relu,
                                           )
        self.conv_high_low = NCausalConv1D(filters=self.low_channels,
                                           kernel_size=self.mor_kernel_size,
                                           dilation_rate=self.mor_dilation,
                                           activation=tf.nn.leaky_relu,
                                           )
        self.conv_low_low = NCausalConv1D(filters=self.low_channels,
                                          kernel_size=self.rhy_kernel_size,
                                          dilation_rate=self.rhy_dilation,
                                          activation=tf.nn.leaky_relu,
                                          )
        self.upsampling1d = layers.UpSampling1D(size=self.beta)
        self.averagepooling1d = layers.AveragePooling1D(pool_size=self.beta)

        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        self.layer_norm3 = layers.LayerNormalization()
        self.layer_norm4 = layers.LayerNormalization()

        self.dropout_high = layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.high_channels)])
        self.dropout_low = layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.low_channels)])

        super().build(input_shape)

    def call(self, inputs):
        # Input = [X^H, X^L]
        assert len(inputs) == 2
        high_input, low_input = inputs

        # High -> High conv
        high_to_high = self.layer_norm1(self.conv_high_high(high_input))
        high_to_high = self.dropout_high(high_to_high)
        # Low -> High conv
        low_to_high = self.layer_norm2(self.conv_low_high(low_input))
        low_to_high = self.dropout_low(low_to_high)
        low_to_high = self.upsampling1d(low_to_high)
        # High -> Low conv
        high_to_low = self.layer_norm3(self.conv_high_low(high_input))
        high_to_low = self.dropout_high(high_to_low)
        high_to_low = self.averagepooling1d(high_to_low)
        # Low -> Low conv
        low_to_low = self.layer_norm4(self.conv_low_low(low_input))
        low_to_low = self.dropout_low(low_to_low)

        # Cross Add
        high_add = layers.add([high_to_high, low_to_high])
        low_add = layers.add([high_to_low, low_to_low])

        high_add = tf.nn.leaky_relu(high_add)
        low_add = tf.nn.leaky_relu(low_add)

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
            "tau_mor": self.tau_mor,
            "tau_rhy": self.tau_rhy,
            "fs": self.fs,
            "alpha": self.alpha,
            "beta": self.beta,
            "dropout": self.dropout,
            "rhy_dilation": self.rhy_dilation,
            "mor_dilation": self.mor_dilation,
            "strides": self.strides,
        }
        return out_config


if __name__ == '__main__':
    visualize_model = True

    high_input = Input(shape=(5000, 1))
    low_input = Input(shape=(2500, 1))
    inputs = [high_input, low_input]

    xh, xl = MorRhyConv1D(filters=32,
                          tau_mor=0.5,
                          tau_rhy=2.0,
                          fs=500,
                          strides=1,
                          mor_dilation=9,
                          rhy_dilation=21)(inputs)
    x = [xh, xl]
    model = Model(inputs, x)
    model.summary()

    if visualize_model:
        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file='ecg_conv.png', show_shapes=True)
