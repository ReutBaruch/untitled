import csv

import numpy as np


def eta(r, y, y_hat):
    if r == y:
        return 1
    elif r == y_hat:
        return -1
    return 0


def eta2(y, x, wights):
    calculate = np.dot(wights, np.dot(y, x))
    #print(calculate)

    if np.all(calculate > 0):
        #print("heyy")
        return 0
    else:
       #print("nn")
        return -np.dot(y, x)


def _init_weights():
    rows = 3
    columns = 8
    weights = [[0 for column in range(columns)] for row in range(rows)]
    weights = np.asarray(weights)
    weights = weights.astype(np.float)
    return weights


def preceptron(pathX, pathY):
    failCount = 0
    count = 0

    #open files
    train_x = open(pathX, "r")
    train_y = open(pathY, "r")

    reader = csv.reader(train_x, delimiter=',')
    data = list(reader)

    #init wights to 0
    wights = _init_weights()


    for line_x, line_y in zip(data, train_y):

        y = float(line_y)
       # x = line_x.split(",")
        x = np.asarray(line_x)

        if x[0] == 'M':
            x[0] = 0.0
        elif x[0] == 'F':
            x[0] = 1.0
        elif x[0] == 'I':
            x[0] = 2.0

        x = x.astype(np.float)
        #normalizing
        x = x / np.linalg.norm(x)

        y_hat = np.argmax(np.dot(wights, x))
        count += 1

        #print(y)
        print(y_hat)
        if y_hat != y:
            failCount += 1
            y = int(y)
            y_hat = int(y_hat)
            wights[y, :] = wights[y, :] + np.dot(eta(y,y,y_hat), x)
            wights[y_hat, :] = wights[y_hat, :] + np.dot(eta(y_hat, y, y_hat), x)

       # print("in for: ", (failCount/count)*100)

    print((failCount / count) * 100)

    #print(wights)





preceptron("/home/reut/PycharmProjects/untitled/train_x.txt", "/home/reut/PycharmProjects/untitled/train_y.txt")


