import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow.keras.backend as K

from bi_gru import BiGRUCell
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
        # self.bi_gru = BiGRUCell(rnn_size=self.filters,
        #                         dropout=self.dropout)
        self.high_cross = layers.Conv1D(self.filters, 1, padding='same')
        self.low_cross = layers.Conv1D(self.filters, 1, padding='same')
        self.upsampling1d = layers.UpSampling1D(size=self.beta)
        self.averagepooling1d = layers.AveragePooling1D(pool_size=self.beta)

        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
        self.batch_norm3 = layers.BatchNormalization()
        self.batch_norm4 = layers.BatchNormalization()

        self.dropout_high = layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.high_channels)])
        self.dropout_low = layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.low_channels)])

        super().build(input_shape)

    def call(self, inputs, is_training=True):
        # Input = [X^H, X^L]
        assert len(inputs) == 2
        high_input, low_input = inputs

        # High -> High conv
        high_to_high = self.batch_norm1(self.conv_high_high(high_input), training=is_training)
        high_to_high = self.dropout_high(high_to_high, training=is_training)
        # Low -> High conv
        low_to_high = self.batch_norm2(self.conv_low_high(low_input), training=is_training)
        low_to_high = self.dropout_low(low_to_high, training=is_training)
        low_to_high = self.upsampling1d(low_to_high)
        # High -> Low conv
        high_to_low = self.batch_norm3(self.conv_high_low(high_input), training=is_training)
        high_to_low = self.dropout_high(high_to_low, training=is_training)
        high_to_low = self.averagepooling1d(high_to_low)
        # Low -> Low conv
        low_to_low = self.batch_norm4(self.conv_low_low(low_input), training=is_training)
        low_to_low = self.dropout_low(low_to_low, training=is_training)

        # Cross Concat
        if low_to_high.shape[1] != high_to_high.shape[1]:
            low_to_high = tf.pad(low_to_high, tf.constant([(0, 0), (1, 1), (0, 0)]) *
                                 (high_to_high.shape[1]-low_to_high.shape[1])//2)
        high_add = tf.concat([high_to_high, low_to_high], -1)
        low_add = tf.concat([high_to_low, low_to_low], -1)
        high_add = self.high_cross(high_add)
        low_add = self.low_cross(low_add)
        # low_add = self.bi_gru(low_add, training=is_training)

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
