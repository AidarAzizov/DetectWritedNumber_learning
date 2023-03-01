import tqdm
import math
import torch
import random
import torchvision

import json
import numpy as np
import matplotlib.pyplot as plt

import csv

from torchvision import transforms
from sklearn.metrics import confusion_matrix

from scipy.ndimage.measurements import center_of_mass
import cv2

from MNISTNet import MNISTNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

output_path = 'outputs/'
input_path = 'inputs/'

model_filename = output_path + 'model.pt'
check_dir = input_path + 'check'
test_dir = input_path + 'test'
data_info_name = output_path + 'data.json'
test_output_name = output_path + 'test.csv'
check_output_name = output_path + 'check.csv'


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def show_input(input_tensor, title=''):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = input_tensor.permute(1, 2, 0).numpy()
    image = std * image + mean
    plt.imshow(image.clip(0, 1))
    plt.title(title)
    plt.show()
    plt.pause(0.001)


def get_average(arr):
    return arr.sum() / len(arr)


def get_scores(predicted, actual):
    cnf_matrix = confusion_matrix(actual, predicted, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    axis_y_sum = cnf_matrix.sum(axis=0)
    axis_x_sum = cnf_matrix.sum(axis=1)

    TP = np.diag(cnf_matrix)
    FP = axis_y_sum - TP
    FN = axis_x_sum - TP
    TN = cnf_matrix.sum() - (FP + FN + TP)

    return TP, FP, FN, TN


def test(CNNnet, criterion, test_dataloader, trusted_threshold):
    assert len(test_dataloader.dataset.classes) == 10
    print('*' * 10, '  TEST start  ', '*' * 10)

    batch_size = test_dataloader.batch_size

    CNNnet.eval()

    true_positive = np.full((10), 1e-5)
    false_positive = np.full((10), 1e-5)
    false_negative = np.full((10), 1e-5)
    true_negative = np.full((10), 1e-5)
    running_loss = 0.0

    with open(test_output_name, 'w', newline='') as csvfile:
        fieldnames = ['feature_name', 'logit 0', 'logit 1', 'logit 2', 'logit 3', 'logit 4',
                      'logit 5', 'logit 6', 'logit 7', 'logit 8', 'logit 9', 'predicted', 'threshold_act', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for features, labels, sample_names in test_dataloader:
            features = features.to(device)
            labels = labels.to(device)

            probabilities = CNNnet(features)
            loss = criterion(probabilities, labels)
            predicted = probabilities.argmax(dim=1).numpy()

            running_loss += loss.item()
            actual = labels.numpy()

            for i in range(batch_size):
                predicted_class = predicted[i].item()
                threshold_act = '+' if probabilities[i][predicted_class] > trusted_threshold else '-'

                writer.writerow({'feature_name': sample_names[i],
                                 'logit 0': probabilities[i][0].item(),
                                 'logit 1': probabilities[i][1].item(),
                                 'logit 2': probabilities[i][2].item(),
                                 'logit 3': probabilities[i][3].item(),
                                 'logit 4': probabilities[i][4].item(),
                                 'logit 5': probabilities[i][5].item(),
                                 'logit 6': probabilities[i][6].item(),
                                 'logit 7': probabilities[i][7].item(),
                                 'logit 8': probabilities[i][8].item(),
                                 'logit 9': probabilities[i][9].item(),
                                 'predicted': predicted_class,
                                 'threshold_act': threshold_act,
                                 'label': actual[i]})

            TP, FP, FN, TN = get_scores(predicted=predicted, actual=actual)

            true_positive += TP
            false_positive += FP
            false_negative += FN
            true_negative += TN


    test_data_count = len(test_dataloader)

    precision = (true_positive) / (true_positive + false_positive)
    recall = (true_positive) / (true_positive + false_negative)
    acc = (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)
    f1 = 2 * precision * recall / (recall + precision)
    test_loss = running_loss / test_data_count

    avr_precision = get_average(precision)
    avr_recall = get_average(recall)
    avr_acc = get_average(acc)
    avr_f1 = get_average(f1)
    avr_macro_f1 = 2 * avr_precision * avr_recall / (avr_recall + avr_precision)

    print(f'Phase: test; Loss: {test_loss:.4f}, Acc: {avr_acc:.4f}, Pre: {avr_precision:.4f}, Rec: {avr_recall:.4f}, macro-avr F1: {avr_macro_f1:.4f}, avr F1: {avr_f1:.4f}')

    print('*' * 10, '  TEST end  ', '*' * 10, '\n')


def check(CNNnet, check_dataloader, trusted_threshold):
    print('*' * 10, '  CHECK start  ', '*' * 10)

    batch_size = check_dataloader.batch_size

    CNNnet.eval()

    with open(check_output_name, 'w', newline='') as csvfile:
        fieldnames = ['feature_name', 'logit 0', 'logit 1', 'logit 2', 'logit 3', 'logit 4',
                      'logit 5', 'logit 6', 'logit 7', 'logit 8', 'logit 9', 'predicted', 'threshold_act',
                      'is_real_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


        for features, labels, sample_names in check_dataloader:
            features = features.to(device)
            labels = labels.to(device)

            probabilities = CNNnet(features)
            predicted = probabilities.argmax(dim=1).numpy()

            for i in range(batch_size):
                predicted_class = predicted[i].item()

                threshold_act = '+' if probabilities[i][predicted_class] > trusted_threshold else '-'
                is_real_number = '+' if check_dataloader.dataset.classes[labels[i]] == 'numbers' else '-'

                writer.writerow({'feature_name': sample_names[i],
                                 'logit 0': probabilities[i][0].item(),
                                 'logit 1': probabilities[i][1].item(),
                                 'logit 2': probabilities[i][2].item(),
                                 'logit 3': probabilities[i][3].item(),
                                 'logit 4': probabilities[i][4].item(),
                                 'logit 5': probabilities[i][5].item(),
                                 'logit 6': probabilities[i][6].item(),
                                 'logit 7': probabilities[i][7].item(),
                                 'logit 8': probabilities[i][8].item(),
                                 'logit 9': probabilities[i][9].item(),
                                 'predicted': predicted_class,
                                 'threshold_act': threshold_act,
                                 'is_real_number': is_real_number})

            for i in range(batch_size):
                show_input(features[i], title=predicted[i])


    print('*' * 10, '  CHECK end  ', '*' * 10, '\n')


def get_best_shift(img):
    cy, cx = center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def to_tensor_stacking(x):
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).type(torch.FloatTensor)
    return x


def mnist_require_preprocessing(image):
    mask_threshold = torch.nn.Threshold(0.7, 0)
    mask_grayscale = transforms.Grayscale()

    gray = mask_threshold(image)

    if gray.shape[0] == 3:
        gray = mask_grayscale(gray)
    gray = gray.squeeze()
    gray = gray.numpy()

    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)
    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)
    rows, cols = gray.shape

    need_blur = False
    if rows > 103 and cols > 103:
        need_blur = True

    if need_blur:
        gray = cv2.resize(gray, (int(cols/3), int(rows/3)), cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))

    gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = get_best_shift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted

    image = gray * 255
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image = image.type(torch.FloatTensor)

    return image


if __name__ == '__main__':
    # print(torch.utils.cmake_prefix_path) #for C++ cmake
    model = torch.load(model_filename)

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(mnist_require_preprocessing)
    ])

    with open(data_info_name, 'r') as f:
        data = f.read()
        json_data = json.loads(data)
    trusted_threshold = float(json_data['trusted_threshold'])

    test_dataset = ImageFolderWithPaths(test_dir, transformations)
    dataset_size = len(test_dataset)
    batch_size = 50 if dataset_size > 50 else dataset_size

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    test(model, criterion=torch.nn.CrossEntropyLoss(), test_dataloader=test_dataloader,
         trusted_threshold=trusted_threshold)

    check_dataset = ImageFolderWithPaths(check_dir, transformations)
    dataset_size = len(check_dataset)
    batch_size = 50 if dataset_size > 50 else dataset_size

    check_dataloader = torch.utils.data.DataLoader(check_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    check(model, check_dataloader=check_dataloader, trusted_threshold=trusted_threshold)
