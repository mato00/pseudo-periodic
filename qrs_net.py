import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import Input

from ncausal_conv import NCausalConv1D
from mor_rhy_conv1d import MorRhyConv1D


class QRSNet(Model):
    def __init__(self):
        super(QRSNet, self).__init__()
        # 500Hz
        self.m_rconv1d_1 = MorRhyConv1D(filters=16,
                                        tau_mor=0.1,
                                        tau_rhy=0.5,
                                        fs=500,
                                        alpha=0.5,
                                        beta=2,
                                        dropout=0.2,
                                        rhy_dilation=8,
                                        mor_dilation=8
                                        )
        self.m_rconv1d_2 = MorRhyConv1D(filters=16,
                                        tau_mor=0.2,
                                        tau_rhy=1.0,
                                        fs=500,
                                        alpha=0.5,
                                        beta=2,
                                        dropout=0.2,
                                        rhy_dilation=16,
                                        mor_dilation=16
                                        )
        self.m_rconv1d_3 = MorRhyConv1D(filters=16,
                                        tau_mor=0.4,
                                        tau_rhy=2.0,
                                        fs=500,
                                        alpha=0.5,
                                        beta=2,
                                        dropout=0.2,
                                        rhy_dilation=32,
                                        mor_dilation=32
                                        )
        self.mconv1d_1 = NCausalConv1D(filters=16,
                                       kernel_size=16,
                                       strides=2,
                                       pad=7,
                                       activation=tf.nn.leaky_relu
                                       )
        self.rconv1d_1 = NCausalConv1D(filters=16,
                                       kernel_size=16,
                                       strides=2,
                                       pad=7,
                                       activation=tf.nn.leaky_relu
                                       )
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        # 250Hz
        self.m_rconv1d_4 = MorRhyConv1D(filters=64,
                                        tau_mor=0.1,
                                        tau_rhy=0.5,
                                        fs=250,
                                        alpha=0.5,
                                        beta=2,
                                        dropout=0.2,
                                        rhy_dilation=4,
                                        mor_dilation=4
                                        )
        self.m_rconv1d_5 = MorRhyConv1D(filters=64,
                                        tau_mor=0.2,
                                        tau_rhy=1.0,
                                        fs=250,
                                        alpha=0.5,
                                        beta=2,
                                        dropout=0.2,
                                        rhy_dilation=8,
                                        mor_dilation=8
                                        )
        self.m_rconv1d_6 = MorRhyConv1D(filters=64,
                                        tau_mor=0.4,
                                        tau_rhy=2.0,
                                        fs=250,
                                        alpha=0.5,
                                        beta=2,
                                        dropout=0.2,
                                        rhy_dilation=16,
                                        mor_dilation=16
                                        )
        self.mconv1d_2 = NCausalConv1D(filters=64,
                                       kernel_size=8,
                                       strides=2,
                                       pad=3,
                                       activation=tf.nn.leaky_relu
                                       )
        self.rconv1d_2 = NCausalConv1D(filters=64,
                                       kernel_size=8,
                                       strides=2,
                                       pad=3,
                                       activation=tf.nn.leaky_relu
                                       )
        self.layer_norm3 = layers.LayerNormalization()
        self.layer_norm4 = layers.LayerNormalization()
        # 125Hz, 125Hz -> 62.5Hz
        self.m_rconv1d_7 = MorRhyConv1D(filters=256,
                                        tau_mor=0.1,
                                        tau_rhy=0.5,
                                        fs=125,
                                        alpha=0.5,
                                        beta=2,
                                        dropout=0.2,
                                        rhy_dilation=2,
                                        mor_dilation=2
                                        )
        self.m_rconv1d_8 = MorRhyConv1D(filters=256,
                                        tau_mor=0.2,
                                        tau_rhy=1.0,
                                        fs=125,
                                        alpha=0.5,
                                        beta=2,
                                        dropout=0.2,
                                        rhy_dilation=4,
                                        mor_dilation=4
                                        )
        self.m_rconv1d_9 = MorRhyConv1D(filters=256,
                                        tau_mor=0.4,
                                        tau_rhy=2.0,
                                        fs=125,
                                        alpha=0.5,
                                        beta=2,
                                        dropout=0.2,
                                        rhy_dilation=8,
                                        mor_dilation=8
                                        )
        self.mconv1d_3 = NCausalConv1D(filters=256,
                                       kernel_size=4,
                                       strides=2,
                                       pad=1,
                                       activation=tf.nn.leaky_relu
                                       )
        self.rconv1d_3 = layers.Dense(256, activation=tf.nn.leaky_relu)
        self.layer_norm5 = layers.LayerNormalization()
        self.layer_norm6 = layers.LayerNormalization()
        # output
        self.out = layers.Dense(2, activation='sigmoid')

    def call(self, inputs):
        # 500Hz
        mor_input = inputs
        rhy_input = layers.AveragePooling1D(2)(inputs)
        mor_feat, rhy_feat = self.m_rconv1d_1([mor_input, rhy_input])
        mor_feat, rhy_feat = self.m_rconv1d_2([mor_feat, rhy_feat])
        mor_feat, rhy_feat = self.m_rconv1d_3([mor_feat, rhy_feat])
        mor_feat = self.mconv1d_1(mor_feat)
        mor_feat = self.layer_norm1(mor_feat)
        rhy_feat = self.rconv1d_1(rhy_feat)
        rhy_feat = self.layer_norm2(rhy_feat)
        # 250Hz
        mor_feat, rhy_feat = self.m_rconv1d_4([mor_feat, rhy_feat])
        mor_feat, rhy_feat = self.m_rconv1d_5([mor_feat, rhy_feat])
        mor_feat, rhy_feat = self.m_rconv1d_6([mor_feat, rhy_feat])
        mor_feat = self.mconv1d_2(mor_feat)
        mor_feat = self.layer_norm3(mor_feat)
        rhy_feat = self.rconv1d_2(rhy_feat)
        rhy_feat = self.layer_norm4(rhy_feat)
        # 125Hz, 125Hz -> 62.5Hz
        mor_feat, rhy_feat = self.m_rconv1d_7([mor_feat, rhy_feat])
        mor_feat, rhy_feat = self.m_rconv1d_8([mor_feat, rhy_feat])
        mor_feat, rhy_feat = self.m_rconv1d_9([mor_feat, rhy_feat])
        mor_feat = self.mconv1d_3(mor_feat)
        mor_feat = self.layer_norm5(mor_feat)
        rhy_feat = self.rconv1d_3(rhy_feat)
        rhy_feat = self.layer_norm6(rhy_feat)
        # Add mor and rhy
        rhy_feat = layers.add([mor_feat, rhy_feat])
        # out
        rhy_out = self.out(rhy_feat)

        return rhy_out


if __name__ == '__main__':
    visualize_model = True

    inputs = Input(shape=(5000, 1))
    qrs_net = QRSNet()

    rhy_out = qrs_net(inputs)
    # model = Model(inputs, rhy_out)
    qrs_net.summary()

    if visualize_model:
        from tensorflow.keras.utils import plot_model

        plot_model(qrs_net, to_file='qrs_net.png', show_shapes=True)
