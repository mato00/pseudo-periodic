import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


class BiGRUCell(layers.Layer):
    def __init__(self, rnn_size, dropout, **kwargs):
        assert rnn_size > 0 and isinstance(rnn_size, int)
        super().__init__(**kwargs)
        self.rnn_size = rnn_size
        self.dropout = dropout

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.grucell_0 = layers.GRU(self.rnn_size,
                                    dropout=self.dropout,
                                    return_sequences=True)
        self.grucell_1 = layers.GRU(self.rnn_size,
                                    dropout=self.dropout,
                                    return_sequences=True,
                                    go_backwards=True)
        self.bi_gru = layers.Bidirectional(self.grucell_0,
                                           backward_layer=self.grucell_1,
                                           merge_mode='sum')
        self.dropout = layers.Dropout(self.dropout,
                                      [tf.constant(1), tf.constant(1), tf.constant(self.rnn_size)])

        super().build(input_shape)

    def call(self, inputs, is_training=True):
        output = self.dropout(self.bi_gru(inputs), training=is_training)

        return output

    def compute_output_shape(self, input_shape):
        out_shape = (input_shape[: 2], self.rnn_size)

        return out_shape

    def get_config(self):
        base_config = super().get_config()
        out_config = {
            **base_config,
            "rnn_size": self.rnn_size,
            "dropout": self.dropout
        }

        return out_config


if __name__ == '__main__':
    visualize_model = True

    inputs = Input(shape=(1250, 1))

    outputs = BiGRUCell(rnn_size=8,
                         dropout=0.2)(inputs)
    model = Model(inputs, outputs)
    model.summary()

    if visualize_model:
        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file='bigru.png', show_shapes=True)
