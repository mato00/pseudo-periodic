import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.callbacks import LearningRateScheduler, History
from QRSNet import *

import pickle, os, time, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from prepare_dataset import SigDataset

def loss_func(labels, preds):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, preds))

def optimizer(lr):
    return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

def train_loop(data, labels, lr):
    with tf.GradientTape() as tape:
        preds = QRSNet()(data)

        loss = loss_func(labels, preds)
        # Get the gradient
        gradients = tape.gradient(loss, QRSNet().trainable_variables)
        # Update weights
        optimizer(lr).apply_gradients(zip(gradients, QRSNet().trainable_variables))

        return loss

# Train the model
def train_model(lr, batch_size, train_size, epoches, dataset):
    train_iters = train_size // batch_size

    start = time.time()
    for epoch in range(epoches):
        for step in range((epoch*train_iters), (epoch+1)*train_iters):
            batch_x, batch_y = dataset.inputs(is_training=True)
            loss = train_loop(batch_x, batch_y, lr)

            if step%10 == 0:
                logging.info('epoch {:}/{:}, step= {:}/{:}, loss={:.4f}'.format(epoch, epoches, step-(epoch*train_iters), train_iters, loss))

    logging.info('training done.')
    logging.info("{} seconds".format(time.time()-start))

if __name__ == '__main__':
    CV_PATH = './split_data/'
    LABEL_PATH = '/data/ECG/CPSC2019/aug_train/ref/'
    # LOG_PATH = './model_10s/log_cv1/'
    # MODEL_PATH = './model_10s/model_cv1/'
    SIG_LEN = 5000
    BATCH_SIZE = 20
    EPOCHES = 200
    LR = 0.1
    FOLD = 1
    TRAIN_SIZE = 1800

    dataset = SigDataset(batch_size=BATCH_SIZE, cv_path=CV_PATH, fold=FOLD, label_path=LABEL_PATH)
    train_model(LR, BATCH_SIZE, TRAIN_SIZE, EPOCHES, dataset)
