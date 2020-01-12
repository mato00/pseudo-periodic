from shift_conv1d import ShiftConv1D
from oct_conv1d import OctConv1D
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


class QRSNet(Model):
    def __init__(self):
        super(QRSNet, self).__init__()
        self.average_pool = layers.AveragePooling1D(8)

        self.mor_shiftconv1d_1 = ShiftConv1D(filters=16, tau=0.1, fs=500, kernel_size=5)
        self.mor_shiftconv1d_2 = ShiftConv1D(filters=16, tau=0.2, fs=500, kernel_size=5)
        self.mor_shiftconv1d_3 = ShiftConv1D(filters=16, tau=0.3, fs=500, kernel_size=5)
        self.mor_shiftconv1d_4 = ShiftConv1D(filters=16, tau=0.4, fs=500, kernel_size=5)
        self.mor_shiftconv1d_5 = ShiftConv1D(filters=16, tau=0.5, fs=500, kernel_size=5)

        self.rhy_shiftconv1d_1 = ShiftConv1D(filters=16, tau=0.4, fs=125, kernel_size=3)
        self.rhy_shiftconv1d_2 = ShiftConv1D(filters=16, tau=0.8, fs=125, kernel_size=3)
        self.rhy_shiftconv1d_3 = ShiftConv1D(filters=16, tau=1.2, fs=125, kernel_size=3)
        self.rhy_shiftconv1d_4 = ShiftConv1D(filters=16, tau=1.6, fs=125, kernel_size=3)
        self.rhy_shiftconv1d_5 = ShiftConv1D(filters=16, tau=2, fs=125, kernel_size=3)

        self.octconv1d_1 = OctConv1D(filters=32, alpha=8, beta=0.5, kernel_size=3)
        self.octconv1d_2 = OctConv1D(filters=32, alpha=8, beta=0.5, kernel_size=3)

        self.conv1d_1 = layers.Conv1D(16, 3, padding="same")
        self.conv1d_2 = layers.Conv1D(16, 3, padding="same")
        self.high_to_low = layers.AveragePooling1D(8)

        self.out = layers.Dense(2, activation='sigmoid')

    def call(self, input):
        # Input
        high = input
        low = self.average_pool(input)
        high, low = self.octconv1d_1([high, low])
        # Morphology Convolution
        high = self.mor_shiftconv1d_1(high)
        high = layers.BatchNormalization()(high)
        high = layers.LeakyReLU()(high)

        high = self.mor_shiftconv1d_2(high)
        high = layers.BatchNormalization()(high)
        high = layers.LeakyReLU()(high)

        high = self.mor_shiftconv1d_3(high)
        high = layers.BatchNormalization()(high)
        high = layers.LeakyReLU()(high)

        high = self.mor_shiftconv1d_4(high)
        high = layers.BatchNormalization()(high)
        high = layers.LeakyReLU()(high)

        high = self.mor_shiftconv1d_5(high)
        high = layers.BatchNormalization()(high)
        high = layers.LeakyReLU()(high)
        # Rhythm Convolution
        low = self.rhy_shiftconv1d_1(low)
        low = layers.BatchNormalization()(low)
        low = layers.LeakyReLU()(low)

        low = self.rhy_shiftconv1d_2(low)
        low = layers.BatchNormalization()(low)
        low = layers.LeakyReLU()(low)

        low = self.rhy_shiftconv1d_3(low)
        low = layers.BatchNormalization()(low)
        low = layers.LeakyReLU()(low)

        low = self.rhy_shiftconv1d_4(low)
        low = layers.BatchNormalization()(low)
        low = layers.LeakyReLU()(low)

        low = self.rhy_shiftconv1d_5(low)
        low = layers.BatchNormalization()(low)
        low = layers.LeakyReLU()(low)
        # Last Block
        high, low = self.octconv1d_2([high, low])
        high = layers.BatchNormalization()(high)
        high = layers.LeakyReLU()(high)
        low = layers.BatchNormalization()(low)
        low = layers.LeakyReLU()(low)

        high = self.conv1d_1(high)
        low = self.conv1d_2(low)
        high_to_low = self.high_to_low(high)

        x = tf.concat([low, high_to_low], axis=-1)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        # FC
        logits = self.out(x)
        return logits

if __name__ == '__main__':
    qrs_net = QRSNet()
