import os
import pickle
import time

import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.callbacks import LearningRateScheduler, History

from qrs_net import QRSNet
from prepare_dataset import SigDataset
from solve_cudnn_error import *

solve_cudnn_error()
model = QRSNet()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.BinaryAccuracy(name='train_acc')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.BinaryAccuracy(name='test_acc')

@tf.function
def loss_func(labels, preds):
    return tf.reduce_mean(tf.keras.losses.BinaryCrossentropy()(labels, preds))

def opt_func(lr):
    return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    # return tf.keras.optimizers.Adam(learning_rate=lr)

@tf.function
def train_step(data, labels, lr, optimizer):
    tf.keras.backend.set_learning_phase(True)
    with tf.GradientTape() as tape:
        logits = model(data, training=True)
        logits = tf.squeeze(logits)
        loss = loss_func(labels, logits)
    # Get the gradient
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, logits)

@tf.function
def test_step(data, labels):
    tf.keras.backend.set_learning_phase(False)
    logits = model(data, training=False)
    logits = tf.squeeze(logits)
    # labels_c = tf.reshape(labels, [tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]])
    # logits_c = tf.reshape(logits, [tf.shape(logits)[0]*tf.shape(logits)[1], tf.shape(logits)[2]])
    t_loss = loss_func(labels, logits)

    test_loss(t_loss)
    test_acc(labels, logits)

# Train the model
def train_model(lr, batch_size, train_size, epoches, dataset, model_path):
    optimizer = opt_func(lr)
    train_iters = train_size // batch_size
    start = time.time()
    ckpt_dir = os.path.dirname(model_path)
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    best_acc = 0
    if ckpt_path:
        model.load_weights(ckpt_path)
    for epoch in range(epoches):
        for step in range((epoch*train_iters), (epoch+1)*train_iters):
            batch_x, batch_y = dataset.inputs(is_training=True)
            train_step(batch_x, batch_y, lr, optimizer)
            if step%10 == 0:
                step_template = 'Epoch {}, Step {}/{}, Loss: {}, Acc: {}'
                print (step_template.format(epoch+1,
                                            step-(epoch*train_iters),
                                            train_iters,
                                            train_loss.result(),
                                            train_acc.result()*100))
        # test data
        test_x, test_y = dataset.inputs(is_training=False)
        test_step(test_x, test_y)
        test_template = 'Epoch {}, Test Loss: {}, Test Acc: {}'
        print (test_template.format(epoch+1,
                                    test_loss.result(),
                                    test_acc.result()*100))
        # save model
        if test_acc.result() >= best_acc:
            model.save_weights(model_path)
            best_acc = test_acc.result()
    print ('training done.')
    print ("{} seconds".format(time.time()-start))

if __name__ == '__main__':
    CV_PATH = './split_data/'
    LABEL_PATH = '/data/ECG/CPSC2019/aug_train/ref/'
    # LOG_PATH = './model_10s/log_cv1/'
    MODEL_PATH = './model/my_checkpoint'
    SIG_LEN = 5000
    BATCH_SIZE = 20
    EPOCHES = 500
    LR = 0.01
    FOLD = 1
    TRAIN_SIZE = 1800

    dataset = SigDataset(batch_size=BATCH_SIZE, cv_path=CV_PATH, fold=FOLD, label_path=LABEL_PATH)
    train_model(LR, BATCH_SIZE, TRAIN_SIZE, EPOCHES, dataset, MODEL_PATH)
