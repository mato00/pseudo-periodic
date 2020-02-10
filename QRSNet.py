import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from oct_conv1d import OctConv1D


class QRSNet(Model):
    def __init__(self):
        super(QRSNet, self).__init__()

        self.octconv1d_1 = OctConv1D(filters=16,
                                     tau_mor=0.1,
                                     tau_rhy=0.5,
                                     fs=500,
                                     alpha=0.5,
                                     beta=2,
                                     dropout=0.2,
                                     padding='same',
                                     rhy_dilation=8,
                                     mor_dilation=8,
                                     downsample=False)
        self.octconv1d_2 = OctConv1D(filters=32,
                                     tau_mor=0.2,
                                     tau_rhy=1.0,
                                     fs=250,
                                     alpha=0.5,
                                     beta=2,
                                     dropout=0.2,
                                     padding='valid',
                                     rhy_dilation=8,
                                     mor_dilation=8,
                                     downsample=False)
        self.octconv1d_3 = OctConv1D(filters=128,
                                     tau_mor=0.4,
                                     tau_rhy=2.0,
                                     fs=125,
                                     alpha=0.5,
                                     beta=2,
                                     dropout=0.2,
                                     rhy_dilation=8,
                                     mor_dilation=8,
                                     downsample=False)
        self.octconv1d_4 = OctConv1D(filters=256,
                                     tau_mor=0.8,
                                     tau_rhy=4.0,
                                     fs=125,
                                     alpha=0.5,
                                     beta=2,
                                     dropout=0.2,
                                     rhy_dilation=16,
                                     mor_dilation=16,
                                     downsample=False)

        # self.conv1d_mor2rhy = layers.Conv1D(128, 3, strides=2, padding="valid")

        # self.out = layers.Dense(2, activation='sigmoid')

    def call(self, inputs):
        # Input
        mor, rhy = inputs



        # FC
        logits = self.out(x)
        return logits

if __name__ == '__main__':
    qrs_net = QRSNet()
