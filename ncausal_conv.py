import tensorflow as tf
from tensorflow.keras import layers
import math


class NCausalConv1D(layers.Conv1D):
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 dilation_rate=1,
                 pad=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(NCausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.pad = pad

    def call(self, inputs):
        if self.pad is not None:
            inputs = tf.pad(inputs, tf.constant([(0, 0), (1, 1), (0, 0)]) * self.pad)
        else:
            pad = math.ceil((self.kernel_size[0]-1) * self.dilation_rate[0] / 2)
            inputs = tf.pad(inputs, tf.constant([(0, 0), (1, 1), (0, 0)]) * pad)

        return super(NCausalConv1D, self).call(inputs)


if __name__ == '__main__':
    from tensorflow.keras.models import Model
    from tensorflow.keras import Input

    visualize_model = True

    input = Input(shape=(5000, 1))

    output = NCausalConv1D(filters=64,
                           kernel_size=8,
                           strides=2,
                           pad=3,
                           activation=tf.nn.leaky_relu)(input)
    model = Model(input, output)
    model.summary()

    if visualize_model:
        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file='ncausal_conv.png', show_shapes=True)
