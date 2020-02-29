import os
import numpy as np
import random
import multiprocessing
from sklearn import preprocessing
import scipy.io as sio
import scipy.signal as signal
import ecg_preprocess as ep

def file_list(dirname, ext='.mat'):
    return list(filter(
        lambda filename: os.path.splitext(filename)[1] == ext,
        os.listdir(dirname)))

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

# statistic number of labels and transform labels to the most label each frame.
def label_produce(labels, slice_len):
    labels_new = np.zeros((500, ), dtype=np.int64)
    labels_new[np.rint(labels / slice_len).astype(np.int)] = 1

    labels_new = np.expand_dims(labels_new, -1)

    return labels_new

class SigDataset():
    def __init__(self, batch_size, cv_path, label_path, fold=1):
        self.batch_size = batch_size
        self.cv_path = cv_path
        self.label_path = label_path
        self.fold = fold

    def inputs(self, is_training=True):
        train_set = open(os.path.join(self.cv_path, 'train_cv'+str(self.fold)+'.txt'), 'r').read().splitlines()
        test_set = open(os.path.join(self.cv_path, 'test_cv'+str(self.fold)+'.txt'), 'r').read().splitlines()
        train_data_list = np.asarray(random.sample(train_set, self.batch_size))
        test_data_list = np.asarray(test_set)
        mor_data = []
        rhy_data = []
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        if is_training:
            for sample_path in train_data_list:
                sample_name = sample_path.split('/')[-1].split('.')[0]
                index = sample_name.split('_')[1]
                label_path = os.path.join(self.label_path, 'R_'+index+'.npy')
                mor_sample = np.load(sample_path)
                rhy_sample = mor_sample - ep.highpass_filter(mor_sample, 15, 500)
                rhy_sample = ep.downsample(rhy_sample, 500, 31.2)
                label = np.load(label_path)

                mor_data.append(mor_sample)
                rhy_data.append(rhy_sample)
                train_labels.append(label)

            mor_data = np.asarray(mor_data)
            rhy_data = np.asarray(rhy_data)
            mor_data = np.expand_dims(mor_data, -1)
            rhy_data = np.expand_dims(rhy_data, -1)
            train_data = [mor_data, rhy_data]
            train_labels = np.asarray(train_labels)

            return train_data, train_labels
        else:
            for sample_path in test_data_list:
                sample_name = sample_path.split('/')[-1].split('.')[0]
                index = sample_name.split('_')[1]
                label_path = os.path.join(self.label_path, 'R_'+index+'.npy')
                mor_sample = np.load(sample_path)
                rhy_sample = mor_sample - ep.highpass_filter(mor_sample, 15, 500)
                rhy_sample = ep.downsample(rhy_sample, 500, 31.2)
                label = np.load(label_path)

                mor_data.append(mor_sample)
                rhy_data.append(rhy_sample)
                test_labels.append(label)

            mor_data = np.asarray(mor_data)
            rhy_data = np.asarray(rhy_data)
            mor_data = np.expand_dims(mor_data, -1)
            rhy_data = np.expand_dims(rhy_data, -1)
            test_data = [mor_data, rhy_data]
            test_labels = np.asarray(test_labels)

            return test_data, test_labels

if __name__ == '__main__':
    CV_PATH = './split_data/'
    LABEL_PATH = '/data/ECG/CPSC2019/aug_train/ref/'

    # Train
    dataset = SigDataset(batch_size=20, cv_path=CV_PATH, label_path=LABEL_PATH)
    x, y = dataset.inputs(is_training=True)
    print(x.shape)
    print(y.shape)

    # Test
    '''
    sample_path = '/home/wearable/workspace/data/heart_sound/final/training_feat_4/a0001_sample_130.npy'
    label_path = '/home/wearable/workspace/data/heart_sound/final/training_label_4/a0001_label_130.npy'
    x, y = read_sample(sample_path, label_path, slice_len_=SLICE_LEN, num_classes_=NUM_CLASSES)
    print(x.shape)
    print(y.shape)
    '''
