import glob
import numpy as np
import os

def split_data(file_list, cv_path, fold=5):
    for i in range(fold):
        data_num = len(file_list)
        if i == fold-1:
            test_list = file_list[data_num*i//fold: ]
            train_list = file_list[: data_num*i//fold]
        else:
            test_list = file_list[data_num*i//fold: data_num*(i+1)//fold]
            train_list = file_list[: data_num*i//fold]
            train_list.extend(file_list[data_num*(i+1)//fold:])
        print('file len: {}'.format(len(file_list)))
        print('test len: {}'.format(len(test_list)))
        f_test = open(os.path.join(cv_path, 'test_cv'+str(i+1)+'.txt'), 'a+')
        count = 5
        for j in test_list:
            if count > 0:
                print(j)
                count -=1
            f_test.write(j+'\n')
        f_test.close()
        print('train len: {}'.format(len(train_list)))
        f_train = open(os.path.join(cv_path, 'train_cv'+str(i+1)+'.txt'), 'a+')
        count = 5
        for j in train_list:
            if count > 0:
                print(j)
                count -=1
            f_train.write(j+'\n')
        f_train.close()

def generate_cv_list(data_path, cv_path, fold_=5):
    ecg_list = glob.glob(data_path + '/*.npy')
    np.random.shuffle(ecg_list)
    split_data(ecg_list, cv_path, fold=fold_)

if __name__ == '__main__':
    CV_PATH = './split_data/'
    DATA_PATH = '/data/ECG/CPSC2019/aug_train/data/'

    if not os.path.exists(CV_PATH):
        os.makedirs(CV_PATH)

    generate_cv_list(DATA_PATH, CV_PATH, fold_=10)
