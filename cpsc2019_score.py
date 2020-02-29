import math
import numpy as np
import os
import re

import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import preprocessing
import tensorflow as tf

from CPSC2019_challenge import *
import ecg_preprocess as ep
from qrs_net import QRSNet
from qrs_detect import *


def pp(data):
    x = np.max(data)
    if x>20:
        b = np.argwhere(data>20)
        for k in b[:,0]:
            if k>0:
                data[k] = data[k-1]
    return data

def load_ans(model, data_path, rpos_path, pic_path, fs):
    '''
    Please modify this function when you have to load model or any other parameters in CPSC2019_challenge()
    '''
    def is_mat(l):
        return l.endswith('.mat')
    ecg_files = list(filter(is_mat, os.listdir(data_path)))
    rpos_files = list(filter(is_mat, os.listdir(rpos_path)))

    # prediction
    HR_ref = []
    R_ref = []
    HR_ans = []
    R_ans = []
    for i, rpos_file in enumerate(rpos_files):
        index = re.split('[_.]', rpos_files[i])[1]
        print(index)
        ecg_file = 'data_' + index + '.mat'
        ref_path = os.path.join(rpos_path, rpos_files[i])
        ecg_path = os.path.join(data_path, ecg_file)

        ecg_data = sio.loadmat(ecg_path)['ecg'].squeeze()
        ecg_data = ep.pp(ecg_data)
        ecg_lp = ecg_data - ep.lowpass_filter(ecg_data, 0.1, 500)
        test_ecg = ecg_lp - ep.highpass_filter(ecg_lp, 45, 500)

        r_ref = sio.loadmat(ref_path)['R_peak'].squeeze()
        r_ref = r_ref[(r_ref >= 0.5*fs) & (r_ref <= 9.5*fs)]
        r_ref = np.unique(r_ref)

        ann_target = np.zeros([5000, ], dtype=np.int)
        ann_d = list(map(lambda x: int(round((x-1))), r_ref))
        for ann in ann_d:
            ann_target[ann-32: ann+48] += 1

        r_hr = np.array([loc for loc in r_ref if
                        (loc > 5.5 * fs and loc < len(ecg_data) - 0.5 * fs)])

        ecg_period = preprocessing.scale(test_ecg)
        mor_period = ecg_period
        rhy_period = mor_period - ep.highpass_filter(mor_period, 15, 500)
        rhy_period = ep.downsample(ecg_period, 500, 31.2)
        mor_period = np.expand_dims(mor_period, 0)
        mor_period = np.expand_dims(mor_period, -1)
        rhy_period = np.expand_dims(rhy_period, 0)
        rhy_period = np.expand_dims(rhy_period, -1)

        logits, preds = qrs_detect([mor_period, rhy_period], model, threshold=0.5)
        up_pred = np.array([[i] * 16 for i in preds]).flatten()
        up_logits = np.array([[i] * 16 for i in logits]).flatten()

        plt.figure(figsize=(30,6))
        plt.plot(up_logits, color='b')
        plt.plot(ann_target, color='r')
        plt.plot(ecg_period, color='k')
        plt.savefig(os.path.join(pic_path, '{}.png'.format(str(index))))
        plt.close()

        try:
            hr_ans, r_ans = CPSC2019_challenge(up_pred)
        except:
            hr_ans = 80
            r_ans = np.array([0])
        HR_ref.append(round( 60 * fs / np.mean(np.diff(r_hr))))
        R_ref.append(r_ref)
        HR_ans.append(hr_ans)
        R_ans.append(r_ans)

    return R_ref, HR_ref, R_ans, HR_ans, rpos_files

def score(r_ref, hr_ref, r_ans, hr_ans, fs_, thr_, rpos_files_):
    HR_score = 0
    record_flags = np.ones(len(r_ref))
    err_files = []
    for i in range(len(r_ref)):
        print (r_ref[i])
        print (r_ans[i])
        FN = 0
        FP = 0
        TP = 0

        if math.isnan(hr_ans[i]):
            hr_ans[i] = 0
        hr_der = abs(int(hr_ans[i]) - int(hr_ref[i]))
        if hr_der <= 0.02 * hr_ref[i]:
            HR_score = HR_score + 1
        elif hr_der <= 0.05 * hr_ref[i]:
            HR_score = HR_score + 0.75
        elif hr_der <= 0.1 * hr_ref[i]:
            HR_score = HR_score + 0.5
        elif hr_der <= 0.2 * hr_ref[i]:
            HR_score = HR_score + 0.25

        for j in range(len(r_ref[i])):
            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= thr_*fs_)[0]
            if j == 0:
                err = np.where((r_ans[i] >= 0.5*fs_ + thr_*fs_) & (r_ans[i] <= r_ref[i][j] - thr_*fs_))[0]
            elif j == len(r_ref[i])-1:
                err = np.where((r_ans[i] >= r_ref[i][j]+thr_*fs_) & (r_ans[i] <= 9.5*fs_ - thr_*fs_))[0]
            else:
                err = np.where((r_ans[i] >= r_ref[i][j]+thr_*fs_) & (r_ans[i] <= r_ref[i][j+1]-thr_*fs_))[0]

            FP = FP + len(err)
            if len(loc) >= 1:
                TP += 1
                FP = FP + len(loc) - 1
            elif len(loc) == 0:
                FN += 1

        if FN + FP > 1:
            record_flags[i] = 0
        elif FN == 1 and FP == 0:
            record_flags[i] = 0.3
        elif FN == 0 and FP == 1:
            record_flags[i] = 0.7

        if record_flags[i] != 1:
            # print(rpos_files_[i])
            err_files.append(rpos_files_[i])

    rec_acc = round(np.sum(record_flags) / len(r_ref), 4)
    hr_acc = round(HR_score / len(r_ref), 4)

    print( 'QRS_acc: {}'.format(rec_acc))
    print('HR_acc: {}'.format(hr_acc))
    print('Scoring complete.')

    return rec_acc, hr_acc, err_files

if __name__ == '__main__':
    MODEL_PATH = './model/'
    DATA_PATH = '/data/ECG/CPSC2019/debug/data/'
    RPOS_PATH = '/data/ECG/CPSC2019/debug/ref/'
    PIC_PATH = '../pseudo_periodic_result/debug_result/'
    THR= 0.075
    FS = 500

    if not os.path.exists(PIC_PATH):
        os.makedirs(PIC_PATH)

    model = QRSNet()
    ckpt_dir = os.path.dirname(MODEL_PATH)
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_path:
        model.load_weights(ckpt_path)

    R_ref, HR_ref, R_ans, HR_ans, rpos_files = load_ans(model, DATA_PATH, RPOS_PATH, PIC_PATH, FS)
    rec_acc, hr_acc, err_files = score(R_ref, HR_ref, R_ans, HR_ans, FS, THR, rpos_files)

    with open('score.txt', 'w') as score_file:
        print('Total File Number: %d\n' %(np.shape(HR_ans)[0]), file=score_file)
        print('R Detection Acc: %0.4f' %rec_acc, file=score_file)
        print('HR Detection Acc: %0.4f' %hr_acc, file=score_file)
        print(err_files, file=score_file)

        score_file.close()
