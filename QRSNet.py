from shift_conv1d import ShiftConv1D
from oct_conv1d import OctConv1D
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


class QRSNet(Model):
    def __init__(self):
        super(QRSNet, self).__init__()
        self.average_pool = layers.AveragePooling1D(4)
        self.leaky_relu = layers.LeakyRelu()
        self.batch_norm = layers.BatchNormalization()

        self.mor_shiftconv1d_1 = ShiftConv1D(filters=16, tau=0.1, fs=500, kernel_size=5)
        self.mor_shiftconv1d_2 = ShiftConv1D(filters=32, tau=0.2, fs=500, kernel_size=5)
        self.mor_shiftconv1d_3 = ShiftConv1D(filters=64, tau=0.3, fs=500, kernel_size=5)
        self.mor_shiftconv1d_4 = ShiftConv1D(filters=128, tau=0.4, fs=500, kernel_size=5)
        self.mor_shiftconv1d_5 = ShiftConv1D(filters=256, tau=0.5, fs=500, kernel_size=5)

        self.rhy_shiftconv1d_1 = ShiftConv1D(filters=16, tau=0.4, fs=125, kernel_size=3)
        self.rhy_shiftconv1d_2 = ShiftConv1D(filters=32, tau=0.8, fs=125, kernel_size=3)
        self.rhy_shiftconv1d_3 = ShiftConv1D(filters=64, tau=1.2, fs=125, kernel_size=3)
        self.rhy_shiftconv1d_4 = ShiftConv1D(filters=128, tau=1.6, fs=125, kernel_size=3)
        self.rhy_shiftconv1d_5 = ShiftConv1D(filters=256, tau=2, fs=125, kernel_size=3)

        self.octconv1d_1 = OctConv1D(filters=16, alpha=4, beta=0.5, kernel_size=3)
        self.octconv1d_2 = OctConv1D(filters=512, alpha=4, beta=0.5, kernel_size=3)

        self.conv1d_1 = layers.Conv1D(512, 3, padding="same")
        self.conv1d_2 = layers.Conv1D(512, 3, padding="same")
        self.low_to_high = layers.Lambda(lambda x: K.repeat_elements(x, 4, axis=1))

        self.out = layers.Dense(2)

    def call(self, input):
        # Input
        high = input
        low = self.average_pool(input)
        high, low = self.octconv1d_1([high, low])
        # Morphology Convolution
        high = self.mor_shiftconv1d_1(high)
        high = self.batch_norm(high)
        high = self.leaky_relu(high)

        high = self.mor_shiftconv1d_2(high)
        high = self.batch_norm(high)
        high = self.leaky_relu(high)

        high = self.mor_shiftconv1d_3(high)
        high = self.batch_norm(high)
        high = self.leaky_relu(high)

        high = self.mor_shiftconv1d_4(high)
        high = self.batch_norm(high)
        high = self.leaky_relu(high)

        high = self.mor_shiftconv1d_5(high)
        high = self.batch_norm(high)
        high = self.leaky_relu(high)
        # Rhythm Convolution
        low = self.rhy_shiftconv1d_1(low)
        low = self.batch_norm(low)
        low = self.leaky_relu(low)

        low = self.rhy_shiftconv1d_2(low)
        low = self.batch_norm(low)
        low = self.leaky_relu(low)

        low = self.rhy_shiftconv1d_3(low)
        low = self.batch_norm(low)
        low = self.leaky_relu(low)

        low = self.rhy_shiftconv1d_4(low)
        low = self.batch_norm(low)
        low = self.leaky_relu(low)

        low = self.rhy_shiftconv1d_5(low)
        low = self.batch_norm(low)
        low = self.leaky_relu(low)
        # Last Block
        high, low = self.octconv1d_2([high, low])
        high = self.batch_norm(high)
        high = self.leaky_relu(high)
        low = self.batch_norm(low)
        low = self.leaky_relu(low)

        high = self.conv1d_1(high)
        low = self.conv1d_2(low)
        low_to_high = self.low_to_high(low)

        x = layers.add()([high, low_to_high])
        x = self.batch_norm(x)
        x = self.leaky_relu(x)

        # FC
        logits = self.out(x)
        return logits

if __name__ == '__main__':
    qrs_net = QRSNet()
