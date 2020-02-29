import numpy as np
import os

import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import ecg_preprocess as ep
from qrs_net import QRSNet
from solve_cudnn_error import *


solve_cudnn_error()

def qrs_detect(data, model, threshold):
    logits = model.predict(data)
    logits = np.squeeze(logits)
    preds = logits.copy()
    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0

    return logits, preds

if __name__ == '__main__':
    MODEL_PATH = './model/'
    CV_PATH = './split_data/'
    LABEL_PATH = '/data/ECG/CPSC2019/aug_train/ref/'
    PIC_PATH = '../pseudo_periodic_result/pic_result/'
    THRESHOLD = 0.5
    FOLD = 1

    if not os.path.exists(PIC_PATH):
        os.makedirs(PIC_PATH)

    model = QRSNet()
    ckpt_dir = os.path.dirname(MODEL_PATH)
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_path:
        model.load_weights(ckpt_path)

    test_set = open(os.path.join(CV_PATH, 'test_cv'+str(FOLD)+'.txt'), 'r').read().splitlines()
    test_data_list = np.asarray(test_set)
    mor_data = []
    rhy_data = []
    test_data = []
    test_labels = []
    for sample_path in test_data_list:
        sample_name = sample_path.split('/')[-1].split('.')[0]
        index = sample_name.split('_')[1]
        label_path = os.path.join(LABEL_PATH, 'R_'+index+'.npy')
        mor_sample = np.load(sample_path)
        rhy_sample = ep.downsample(mor_sample, 500, 250)
        label = np.load(label_path)
        up_label = np.array([[x] * 8 for x in label]).flatten()
        mor_data.append(mor_sample)
        rhy_data.append(rhy_sample)
        test_labels.append(up_label)
    mor_data = np.asarray(mor_data)
    rhy_data = np.asarray(rhy_data)
    mor_data = np.expand_dims(mor_data, -1)
    rhy_data = np.expand_dims(rhy_data, -1)
    test_data = [mor_data, rhy_data]
    test_labels = np.asarray(test_labels)
    preds = qrs_detect(test_data, model, THRESHOLD)
    # Draw picture
    for i, test_ecg in enumerate(mor_data):
        up_pred = np.array([[x] * 8 for x in preds[i]]).flatten()
        plt.figure(figsize=(30,6))
        plt.plot(up_pred, color='b')
        plt.plot(test_labels[i], color='r')
        plt.plot(test_ecg, color='k')
        plt.savefig(os.path.join(PIC_PATH, '{}.png'.format(str(i))))
        plt.close()
