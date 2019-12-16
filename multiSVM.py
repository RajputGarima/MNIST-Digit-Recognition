import numpy as np
import pandas as pd
from svmutil import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import timeit
import sys

train_file = sys.argv[1]
test_file = sys.argv[2]
mode = sys.argv[3]


start = timeit.default_timer()
df_train = pd.read_csv(train_file, header=None)
df_test = pd.read_csv(test_file, header=None)


test = np.asarray(df_test)
train = np.asarray(df_train)


m = len(train)
p = len(test)
n = 784
full_train = np.zeros((m, n))
full_train_output = np.zeros((m, 1))
full_test = np.zeros((p, n))
full_test_output = np.zeros((p, 1))


def pre_process():
    for i in range(p):
        for j in range(n):
            if not test[i][j] == 0:
                full_test[i][j] = test[i][j]/255.0
    for i in range(p):
        full_test_output[i] = test[i][784]

    for i in range(m):
        for j in range(n):
            if not train[i][j] == 0:
                full_train[i][j] = train[i][j]/255.0
    for i in range(m):
        full_train_output[i] = train[i][784]


pre_process()


def toFloatList(a):
    b = []
    for i in a:
        b.append(i[0])
    return b


if mode == 'b' or mode == 'c':
    y = np.asarray(full_train_output).tolist()
    y = toFloatList(y)
    x = np.asarray(full_train).tolist()

    parameter = svm_parameter('-q -t 2 -g 0.05')
    problem = svm_problem(y, x)
    model = svm_train(problem, parameter)

    y1 = np.asarray(full_test_output).tolist()
    y1 = toFloatList(y1)
    tp = np.asarray(full_test)
    x1 = tp.tolist()
    stop = timeit.default_timer()
    p_lbl_test, p_acc_test, _ = svm_predict(y1, x1, model)
    p_lbl_train, p_acc_train, _ = svm_predict(y, x, model)

    print("Accuracy on train data: ")
    print(p_acc_train[0])
    print("Accuracy on test data: ")
    print(p_acc_test[0])
    print("Training time: ", stop - start)

# confusion matrix

def plot_confusion_matrix(y_actual, y_predicted,  classes, title,  cmap=plt.cm.Blues):
    cm = confusion_matrix(y_actual, y_predicted)
    # classes = classes[unique_labels(y_actual, y_predicted)]
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


class_names = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
if mode == 'c':
    plot_confusion_matrix(full_train_output, p_lbl_train, classes=class_names, title='Confusion matrix: Training data')
    plt.show()
    plot_confusion_matrix(full_test_output, p_lbl_test, classes=class_names, title='Confusion matrix: Testing data')
    plt.show()


# --(d)
if mode == 'd':
    x_train, x_test, train_lbl, test_lbl = train_test_split(full_train, full_train_output, test_size=0.10)

    validation_accuracy = []
    test_accuracy = []
    param = '-q -t 2 -g 0.05'

    y = np.asarray(train_lbl).tolist()
    y = toFloatList(y)
    x = np.asarray(x_train).tolist()
    problem = svm_problem(y, x)

    for c in [10 ** -5, 10 ** -3, 1, 5, 10]:
        c = " -c %f " % c
        params = param + c
        parameter = svm_parameter(params)
        model = svm_train(problem, parameter)
        y1 = np.asarray(test_lbl).tolist()
        y1 = toFloatList(y1)
        tp = np.asarray(x_test)
        x1 = tp.tolist()
        _, v_acc, _ = svm_predict(y1, x1, model)
        validation_accuracy.append(v_acc[0])
        y1 = np.asarray(full_test_output).tolist()
        y1 = toFloatList(y1)
        tp = np.asarray(full_test)
        x1 = tp.tolist()
        _, t_acc, _ = svm_predict(y1, x1, model)
        test_accuracy.append(t_acc[0])

    #print(validation_accuracy)
    #print(test_accuracy)

    C = [0.00001, 0.001, 1, 5, 10]

    #validation_acc = [9.35, 9.35, 97.25, 97.35, 97.35]
    #test_acc = [9.82, 9.82, 97.13, 97.23, 97.23]

    c_line, = plt.plot(C, validation_accuracy, label="Validation set accuracy", linestyle='-', color='r', marker='x', markersize='10.0', linewidth='2.0')

    t_line, = plt.plot(C, test_accuracy, label="Test set accuracy", linestyle='-', color='black', marker='o')

    plt.legend(handles=[c_line, t_line])

    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title("C versus Accuracy")
    plt.show()
