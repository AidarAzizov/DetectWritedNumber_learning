import os
import copy
import torch
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix
from MNISTNet import MNISTNet

import json

import scipy.stats as st

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

output_path = 'outputs/'
model_filename = output_path + 'model.pt'
val_image_name = output_path + 'validation.png'
data_info_name = output_path + 'data.json'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_percent = 0.90


from torchvision import transforms
train_transforms = transforms.Compose([
    transforms.RandomSolarize(0),
    transforms.ColorJitter(
        brightness=0.175,
        contrast=0.175,
        saturation=0.195,
        hue=(0.1, 0.25))
])

import torchvision.datasets
MNIST_train = torchvision.datasets.MNIST('./', download=False, train=True, transform=train_transforms)
MNIST_test = torchvision.datasets.MNIST('./', download=False, train=False)

train_data_count = int(len(MNIST_train.train_data) * train_percent)
train_label_count = int(len(MNIST_train.train_labels) * train_percent)

train_features = MNIST_train.train_data[0 : train_data_count]
test_features = MNIST_test.test_data
val_features = MNIST_train.train_data[train_data_count:]

train_labels = MNIST_train.train_labels[0 : train_label_count]
test_labels = MNIST_test.test_labels
val_labels = MNIST_train.train_labels[train_label_count:]

def mnist_dataset_imbalance():
    train_labels_count = len(train_labels)
    test_labels_count = len(test_labels)

    unique_train_labels = np.unique(train_labels.numpy(), return_counts=True)
    assert unique_train_labels[1].sum() == train_labels_count

    unique_test_labels = np.unique(test_labels.numpy(), return_counts=True)
    assert unique_test_labels[1].sum() == test_labels_count

    MNIST_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    print('Imbalance check:')
    for label in MNIST_labels:
        train_percent = (unique_train_labels[1][label] / train_labels_count) * 100
        test_percent = (unique_test_labels[1][label] / test_labels_count) * 100

        print(f'\tLabel {label:1d}:\t', end='')
        print(f'train - {train_percent:.2f}, test - {train_percent:.2f}')
# mnist_dataset_imbalance()


import matplotlib.pyplot as plt

def print_loss_acc(loss, acc):

    loss_x = range(0, len(loss))
    acc_x = range(0, len(acc))

    fig, (ax_acc, ax_loss) = plt.subplots(2)
    ax_acc.plot(acc_x, acc, 'tab:red')
    ax_loss.plot(loss_x, loss, 'tab:green')

    ax_acc.set_title('Acc')
    ax_loss.set_title('Loss')

    ax_acc.grid(axis='both')
    ax_loss.grid(axis='both')

    fig.savefig(val_image_name)

    plt.show()

def print_some_features(print_count=10):
    for i in range(print_count):
        image = train_features[i]
        label = train_labels[i].item()

        plt.imshow(image.clip(0, 1))
        plt.title(label)
        plt.show()
        plt.pause(0.001)
# print_some_features()


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

def train(CNNnet, epoches, batch_size, optimizer, criterion):

    val_loss = []
    val_acc = []

    start_train_time = datetime.now()

    best_model_wts = copy.deepcopy(CNNnet.state_dict())
    best_acc = 0.0

    print('*' * 10, '  TRAIN start  ', '*' * 10)
    print('-' * 10)
    for epoch in range(epoches):
        start_epoch_time = datetime.now()

        print('Epoch {}/{}'.format(epoch + 1, epoches))

        for phase in ['train', 'val']:

            true_positive = 0
            false_positive = 0
            false_negative = 0
            true_negative = 0
            running_loss = 0.0

            if phase == 'train':
                CNNnet.train()  # Set model to train mode
                features = train_features
                labels = train_labels
            else:
                CNNnet.eval()   # Set model to evaluate mode
                features = val_features
                labels = val_labels

            order = np.random.permutation(len(features))

            for start_index in range(0, len(features), batch_size):
                batch_indexes = order[start_index:start_index + batch_size]
                X_batch = features[batch_indexes].to(device)
                y_batch = labels[batch_indexes].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    preds = CNNnet(X_batch)
                    loss = criterion(preds, y_batch)

                    # backward + optimize only if in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                predicted = preds.argmax(dim=1).numpy()
                actual = y_batch.numpy()

                TP, FP, FN, TN = get_scores(predicted=predicted, actual=actual)
                true_positive += TP
                false_positive += FP
                false_negative += FN
                true_negative += TN

            epoch_precision = (true_positive) / (true_positive + false_positive)
            epoch_recall = (true_positive) / (true_positive + false_negative)
            epoch_loss = running_loss / (len(features) // batch_size)
            epoch_acc = (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)
            epoch_f1 = 2 * epoch_precision * epoch_recall / (epoch_recall + epoch_precision)

            avr_precision = get_average(epoch_precision)
            avr_recall = get_average(epoch_recall)
            avr_acc = get_average(epoch_acc)
            avr_f1 = get_average(epoch_f1)

            avr_macro_f1 = 2 * avr_precision * avr_recall / (avr_recall + avr_precision)

            # deep copy the model
            if phase == 'val' and avr_acc > best_acc:
                best_acc = avr_acc
                best_model_wts = copy.deepcopy(CNNnet.state_dict())
            if phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(avr_acc)

            print(f'Phase: {phase}; Loss: {epoch_loss:.4f}, Acc: {avr_acc:.4f}, Pre: {avr_precision:.4f}, Rec: {avr_recall:.4f}, macro-avr F1: {avr_macro_f1:.4f}, avr F1: {avr_f1:.4f}')

        end_epoch_time = datetime.now()
        print('Epoch time =', end_epoch_time - start_epoch_time)
        print('-' * 10)

    end_train_time = datetime.now()
    print('Trained about of ', end_train_time - start_train_time, ' time')
    print('*' * 10, '  TRAIN end  ', '*' * 10, '\n')

    CNNnet.load_state_dict(best_model_wts)

    return val_loss, val_acc

def test(CNNnet, criterion, batch_size):

    print('*' * 10, '  TEST start  ', '*' * 10)

    maximum_class_probabilities = []

    CNNnet.eval()

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    running_loss = 0.0

    test_data_count = len(test_features)
    order = np.random.permutation(test_data_count)
    for start_index in range(0, test_data_count, batch_size):

        batch_indexes = order[start_index:start_index + batch_size]
        features = test_features[batch_indexes].to(device)
        labels = test_labels[batch_indexes].to(device)

        preds = CNNnet(features)
        loss = criterion(preds, labels)

        running_loss += loss.item()
        predicted = preds.argmax(dim=1).numpy()
        actual = labels.numpy()

        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                maximum_class_probabilities.append(preds[i][predicted[i]].item())

        TP, FP, FN, TN = get_scores(predicted=predicted, actual=actual)

        true_positive += TP
        false_positive += FP
        false_negative += FN
        true_negative += TN

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

    res = st.norm.interval(alpha=0.95, loc=np.mean(maximum_class_probabilities), scale=st.sem(maximum_class_probabilities))
    trusted_threshold = res[0]

    prepared_json = json.dumps({"trusted_threshold": trusted_threshold})
    with open(data_info_name, 'w') as f:
        json.dump(json.loads(prepared_json), f, indent=2)

    print(f'Phase: test; Loss: {test_loss:.4f}, Acc: {avr_acc:.4f}, Pre: {avr_precision:.4f}, Rec: {avr_recall:.4f}, macro-avr F1: {avr_macro_f1:.4f}, avr F1: {avr_f1:.4f}, trusted threshold: {trusted_threshold:.4f}')

    print('*' * 10, '  TEST end  ', '*' * 10, '\n')


if __name__ == '__main__':
    os.makedirs(output_path, exist_ok=True)

    train_features = train_features.unsqueeze(1).float()
    test_features = test_features.unsqueeze(1).float()
    val_features = val_features.unsqueeze(1).float()

    mnist_net = MNISTNet().to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mnist_net.parameters(), lr=5e-4)

    val_loss, val_acc = train(CNNnet=mnist_net, epoches=1, batch_size=50, optimizer=optimizer, criterion=loss)
    print_loss_acc(loss=val_loss, acc=val_acc)

    torch.save(mnist_net, model_filename)
    # torch.save(mnist_net.state_dict(), './mnist_weights.pt')

    mnist_net = torch.load(model_filename)
    test(CNNnet=mnist_net, batch_size=50, criterion=loss)


