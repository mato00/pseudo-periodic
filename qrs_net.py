import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import Input

from bi_gru import BiGRUCell
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
                                        beta=16,
                                        dropout=0.2,
                                        rhy_dilation=1,
                                        mor_dilation=8
                                        )
        self.m_rconv1d_2 = MorRhyConv1D(filters=16,
                                        tau_mor=0.2,
                                        tau_rhy=1.0,
                                        fs=500,
                                        alpha=0.5,
                                        beta=16,
                                        dropout=0.2,
                                        rhy_dilation=2,
                                        mor_dilation=16
                                        )
        self.m_rconv1d_3 = MorRhyConv1D(filters=16,
                                        tau_mor=0.4,
                                        tau_rhy=2.0,
                                        fs=500,
                                        alpha=0.5,
                                        beta=16,
                                        dropout=0.2,
                                        rhy_dilation=4,
                                        mor_dilation=32
                                        )
        self.mconv1d_1 = NCausalConv1D(filters=32,
                                       kernel_size=16,
                                       strides=2,
                                       pad=7,
                                       activation=tf.nn.leaky_relu
                                       )
        """
        self.rconv1d_1 = NCausalConv1D(filters=32,
                                       kernel_size=16,
                                       strides=2,
                                       pad=7,
                                       activation=tf.nn.leaky_relu
                                       )
        """
        self.bi_gru_1 = BiGRUCell(rnn_size=32,
                                  dropout=0.2)
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
        # 250Hz
        self.m_rconv1d_4 = MorRhyConv1D(filters=32,
                                        tau_mor=0.1,
                                        tau_rhy=0.5,
                                        fs=250,
                                        alpha=0.5,
                                        beta=8,
                                        dropout=0.2,
                                        rhy_dilation=1,
                                        mor_dilation=4
                                        )
        self.m_rconv1d_5 = MorRhyConv1D(filters=32,
                                        tau_mor=0.2,
                                        tau_rhy=1.0,
                                        fs=250,
                                        alpha=0.5,
                                        beta=8,
                                        dropout=0.2,
                                        rhy_dilation=2,
                                        mor_dilation=8
                                        )
        self.m_rconv1d_6 = MorRhyConv1D(filters=32,
                                        tau_mor=0.4,
                                        tau_rhy=2.0,
                                        fs=250,
                                        alpha=0.5,
                                        beta=8,
                                        dropout=0.2,
                                        rhy_dilation=4,
                                        mor_dilation=16,
                                        )
        self.mconv1d_2 = NCausalConv1D(filters=64,
                                       kernel_size=8,
                                       strides=2,
                                       pad=3,
                                       activation=tf.nn.leaky_relu
                                       )
        """
        self.rconv1d_2 = NCausalConv1D(filters=64,
                                       kernel_size=8,
                                       strides=2,
                                       pad=3,
                                       activation=tf.nn.leaky_relu
                                       )
        """
        self.bi_gru_2 = BiGRUCell(rnn_size=64,
                                  dropout=0.2)
        self.batch_norm3 = layers.BatchNormalization()
        self.batch_norm4 = layers.BatchNormalization()
        # 125Hz, 125Hz -> 62.5Hz
        self.m_rconv1d_7 = MorRhyConv1D(filters=64,
                                        tau_mor=0.1,
                                        tau_rhy=0.5,
                                        fs=125,
                                        alpha=0.5,
                                        beta=4,
                                        dropout=0.2,
                                        rhy_dilation=1,
                                        mor_dilation=2
                                        )
        self.m_rconv1d_8 = MorRhyConv1D(filters=64,
                                        tau_mor=0.2,
                                        tau_rhy=1.0,
                                        fs=125,
                                        alpha=0.5,
                                        beta=4,
                                        dropout=0.2,
                                        rhy_dilation=2,
                                        mor_dilation=4
                                        )
        self.m_rconv1d_9 = MorRhyConv1D(filters=64,
                                        tau_mor=0.4,
                                        tau_rhy=2.0,
                                        fs=125,
                                        alpha=0.5,
                                        beta=4,
                                        dropout=0.2,
                                        rhy_dilation=4,
                                        mor_dilation=8
                                        )
        self.mconv1d_3 = NCausalConv1D(filters=128,
                                       kernel_size=4,
                                       strides=4,
                                       pad=0,
                                       activation=tf.nn.leaky_relu
                                       )
        # self.rconv1d_3 = layers.Dense(128, activation=tf.nn.leaky_relu)
        self.bi_gru_3 = BiGRUCell(rnn_size=128,
                                  dropout=0.2)
        self.batch_norm5 = layers.BatchNormalization()
        self.batch_norm6 = layers.BatchNormalization()
        # output
        self.dense1 = layers.Dense(64, activation=tf.nn.leaky_relu)
        self.out = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True):
        # 500Hz
        mor_feat, rhy_feat = self.m_rconv1d_1(inputs, is_training=training)
        mor_feat, rhy_feat = self.m_rconv1d_2([mor_feat, rhy_feat], is_training=training)
        mor_feat, rhy_feat = self.m_rconv1d_3([mor_feat, rhy_feat], is_training=training)
        mor_feat = self.mconv1d_1(mor_feat)
        mor_feat = self.batch_norm1(mor_feat, training=training)
        # rhy_feat = self.rconv1d_1(rhy_feat)
        rhy_feat = self.bi_gru_1(rhy_feat, is_training=training)
        rhy_feat = self.batch_norm2(rhy_feat, training=training)
        # 250Hz
        mor_feat, rhy_feat = self.m_rconv1d_4([mor_feat, rhy_feat], is_training=training)
        mor_feat, rhy_feat = self.m_rconv1d_5([mor_feat, rhy_feat], is_training=training)
        mor_feat, rhy_feat = self.m_rconv1d_6([mor_feat, rhy_feat], is_training=training)
        mor_feat = self.mconv1d_2(mor_feat)
        mor_feat = self.batch_norm3(mor_feat, training=training)
        # rhy_feat = self.rconv1d_2(rhy_feat)
        rhy_feat = self.bi_gru_2(rhy_feat, is_training=training)
        rhy_feat = self.batch_norm4(rhy_feat, training=training)
        # 125Hz, 125Hz -> 62.5Hz
        mor_feat, rhy_feat = self.m_rconv1d_7([mor_feat, rhy_feat], is_training=training)
        mor_feat, rhy_feat = self.m_rconv1d_8([mor_feat, rhy_feat], is_training=training)
        mor_feat, rhy_feat = self.m_rconv1d_9([mor_feat, rhy_feat], is_training=training)
        mor_feat = self.mconv1d_3(mor_feat)
        mor_feat = self.batch_norm5(mor_feat, training=training)
        # rhy_feat = self.rconv1d_3(rhy_feat)
        rhy_feat = self.bi_gru_3(rhy_feat, is_training=training)
        rhy_feat = self.batch_norm6(rhy_feat, training=training)
        # Add mor and rhy
        rhy_feat = tf.concat([mor_feat, rhy_feat], -1)
        # out
        rhy_feat = self.dense1(rhy_feat)
        rhy_out = self.out(rhy_feat)

        return rhy_out


if __name__ == '__main__':
    visualize_model = True

    mor_inputs = Input(shape=(5000, 1))
    rhy_inputs = Input(shape=(312, 1))
    qrs_net = QRSNet()

    rhy_out = qrs_net([mor_inputs, rhy_inputs])
    # model = Model(inputs, rhy_out)
    qrs_net.summary()

    if visualize_model:
        from tensorflow.keras.utils import plot_model

        plot_model(qrs_net, to_file='qrs_net.png', show_shapes=True)
