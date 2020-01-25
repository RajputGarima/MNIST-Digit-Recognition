import numpy as np
import pandas as pd
import cvxopt
import timeit
import filter_7_and_8
from svmutil import *
import sys

start = timeit.default_timer()
train_file = sys.argv[1]
test_file = sys.argv[2]
mode = sys.argv[3]

filter_7_and_8.filter_train(train_file)
filter_7_and_8.filter_test(test_file)

df_train = pd.read_csv('7_and_8_train.csv', header=None)
df_test = pd.read_csv('7_and_8_test.csv', header=None)
C = 1.0
sigma = 0.05

test = np.asarray(df_test)
train = np.asarray(df_train)
m = len(train)
n = 784
p = len(test)
shape_train = (m, n)
shape_test = (p, n)
test_data = np.zeros(shape_test)
test_data_output = np.zeros((p, 1))
train_data = np.zeros(shape_train)
train_data_output = np.zeros((m, 1))
kshape = (m, m)
K = np.zeros(kshape)
full_train = np.zeros((m, n + 1))
full_test = np.zeros((p, n + 1))


def pre_process():
    for i in range(p):
        for j in range(n):
            if not test[i][j] == 0:
                test_data[i][j] = test[i][j] / 255.0
                full_test[i][j] = test_data[i][j]
    for i in range(p):
        test_data_output[i] = 1 if test[i][784] == 7 else -1
        full_test[i][784] = test[i][784]

    for i in range(m):
        for j in range(n):
            if not train[i][j] == 0:
                train_data[i][j] = train[i][j] / 255.0
                full_train[i][j] = train_data[i][j]
    for i in range(m):
        train_data_output[i] = 1 if train[i][784] == 7 else -1
        full_train[i][784] = train[i][784]


pre_process()


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def gauss_kernel(x1, x2):
    temp1 = np.linalg.norm((x1 - x2)) ** 2
    return np.exp(-temp1 * sigma)


def k_matrix_linear():
    for i in range(m):
        for j in range(m):
            K[i][j] = linear_kernel(train_data[i], train_data[j])


def k_matrix_gauss():
    for i in range(m):
        for j in range(m):
            K[i][j] = gauss_kernel(train_data[i], train_data[j])


if mode == 'a' or mode == 'b':
    y_matrix = np.outer(train_data_output, train_data_output)
    if mode == 'a':
        k_matrix_linear()
    if mode == 'b':
        k_matrix_gauss()
    P = cvxopt.matrix(np.multiply(y_matrix, K))
    q = cvxopt.matrix(np.ones((m, 1)) * -1)
    A = cvxopt.matrix(train_data_output, (1, m))
    b = cvxopt.matrix(np.zeros(1))
    h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    x = np.diag(np.ones(m) * -1)
    y = np.diag(np.ones(m))
    G = cvxopt.matrix(np.vstack((x, y)))
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.array(solution['x'])
    sv = []
    for i in alpha:
        if i > 1e-5:
            sv.append(i)
    S = (alpha > 1e-5).flatten()
    print("Number of support vectors:", len(sv))
    w = np.dot((np.transpose(alpha * train_data_output)), train_data)


def compute_b_lin():
    print("w: ", w)
    minimum = None
    maximum = None

    for i in range(m):
        if train_data_output[i] == -1:
            negative = np.dot(w, train_data[i])
            if maximum is None or negative > maximum:
                maximum = negative
        else:
            positive = np.dot(w, train_data[i])
            if minimum is None or positive < minimum:
                minimum = positive

    b = -(maximum + minimum) / 2
    return b


def predict_linear():
    b = compute_b_lin()
    print("b: ", b)
    stop = timeit.default_timer()
    print("Training time:", stop - start)
    predicted = []
    x1 = -1.0
    x2 = 1.0
    for i in range(p):
        result = np.dot(w, test_data[i]) + b
        if result < 0:
            predicted.append(x1)
        else:
            predicted.append(x2)

    correct = 0
    for i in range(p):
        if test_data_output[i][0] == predicted[i]:
            correct = correct + 1
    print("Accuracy with linear kernel: ", (correct/p)*100)


# prediction in gaussian kernel
if mode == 'a' or mode == 'b':
    sv_alpha = alpha[S]
    sv_x = train_data[S]
    sv_y = train_data_output[S]


def calculate_w(xj):
    answer = 0
    for alpha, x, y in zip(sv_alpha, sv_x, sv_y):
        answer = answer + alpha * y * gauss_kernel(xj, x)
    return answer


def compute_b():
    ind = np.arange(len(alpha))[S]
    b = 0
    for n in range(len(sv_alpha)):
        b = b + sv_y[n]
        b = b - np.sum(sv_alpha * sv_y * K[ind[n], S])
    b = b / len(alpha)
    return b


def predict_gauss():
    predicted = np.zeros(len(test_data_output))
    x1 = -1.0
    x2 = 1.0
    b = compute_b()
    print("b(gauss): ", b)
    stop = timeit.default_timer()
    print("Training time:", stop - start)
    for i in range(p):
        w = calculate_w(test_data[i])
        if w + b < 0:
            predicted[i] = x1
        else:
            predicted[i] = x2
    correct = 0
    for i in range(p):
        if test_data_output[i][0] == predicted[i]:
            correct = correct + 1
    # print(correct)
    print("Accuracy with gaussian kernel", (correct/p)*100)


if mode == 'a':
    predict_linear()

if mode == 'b':
    predict_gauss()


# -- libsvm

if mode == 'c':
    def toFloatList(a):
        b = []
        for i in a:
            b.append(i[0])
        return b
    y = np.asarray(train_data_output).tolist()
    y = toFloatList(y)
    tp = np.asarray(train_data)
    x = tp.tolist()

    parameter = svm_parameter('-t 0')
    problem = svm_problem(y, x)
    model = svm_train(problem, parameter)

    y1 = np.asarray(test_data_output).tolist()
    y1 = toFloatList(y1)
    tp = np.asarray(test_data)
    x1 = tp.tolist()
    stoplib = timeit.default_timer()
    print("Training time:", stoplib - start)
    _, p_acc, _ = svm_predict(y1, x1, model)
    # print("Accuracy with linear kernel ", p_acc[0])

    parameter = svm_parameter('-t 2 -g 0.05')
    problem = svm_problem(y, x)
    model = svm_train(problem, parameter)
    stoplib2 = timeit.default_timer()
    print("Training time:", stoplib2 - start)
    _, p_acc, _ = svm_predict(y1, x1, model)
    # print("Accuracy with linear kernel ", p_acc[0])
