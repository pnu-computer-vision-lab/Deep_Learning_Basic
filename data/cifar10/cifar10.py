
import numpy as np
import cv2
import pickle
from PIL import Image


from torch import FloatTensor


def load_cifar10_data(filename):
    with open(filename, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        features = batch[b'data']
        labels = batch[b'labels']
        filenames = batch[b'filenames']
        return features, labels, filenames


def load_cifar10_meta_data(filename):
    with open(filename, 'rb') as file:
        # batch = pickle.load(file, encoding='bytes')
        # labels = batch[b'label_names']
        batch = pickle.load(file, encoding='utf-8')
        labels = batch['label_names']
        return labels


def load_train_data(data_path):
    data_path = rf'./data/{data_path}/'
    train_1, train_labels_1, train_filenames_1 = load_cifar10_data(data_path + 'data_batch_1')
    train_2, train_labels_2, train_filenames_2 = load_cifar10_data(data_path +'data_batch_2')
    train_3, train_labels_3, train_filenames_3 = load_cifar10_data(data_path +'data_batch_3')
    train_4, train_labels_4, train_filenames_4 = load_cifar10_data(data_path +'data_batch_4')
    train_5, train_labels_5, train_filenames_5 = load_cifar10_data(data_path +'data_batch_5')
    train_meta = load_cifar10_meta_data(data_path +'batches.meta')

    X_train = np.concatenate([train_1, train_2, train_3, train_4, train_5], axis=0)  # shape : (50000, 3072)
    y_train = np.concatenate([train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5], axis=0)
    # CIFAR-10은 HxWxC로 1-dimension 형태로 저장되어 있음(3072)
    # reshape(-1, 3, 32, 32)로 (sample num, channel, height, width)로 변경
    # 대부분의 Deep learning framework는 (sample num, height, width, channel)로 구성
    # transpose로 재배열
    # torch는 HxWxC인데.... 굳이 바꿔줄 필요가 있나..?
    # X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # shape (50000, 32, 32, 3)
    X_train = X_train.reshape(-1, 3, 32, 32)

    return FloatTensor(X_train), FloatTensor(y_train)


def cifar10_meta(index):
    train_meta = load_cifar10_meta_data('batches.meta')

    return train_meta[index]

def load_test_data():
    test, test_labels, test_filenames = load_cifar10_data('test_batch')
    # X_test = test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = test.reshape(-1, 3, 32, 32)
    y_test = np.array(test_labels)
    return X_test, y_test

if __name__ == '__main__':

    X_train, y_train = load_train_data()
    print(type(X_train), type(y_train))