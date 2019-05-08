import numpy as np


def eta(y, x, wights):
    calculate = np.dot(wights, np.dot(y, x))
    #print(calculate)

    if np.all(calculate > 0):
#        print("heyy")
        return 0
    else:
 #       print("nn")
        return -np.dot(y, x)


def _init_wights():
    rows = 3
    columns = 8
    wights = [[0 for column in range(columns)] for row in range(rows)]
    wights = np.asarray(wights)
    wights = wights.astype(np.float)
    return wights


def preceptron(pathX, pathY):
    #open files
    train_x = open(pathX, "r")
    train_y = open(pathY, "r")

    #init wights to 0
    wights = _init_wights()


    for line_x, line_y in zip(train_x, train_y):

        y = float(line_y)
        x = line_x.split(",")
        x = np.asarray(x)

        if x[0] == 'M':
            x[0] = 1.0
        elif x[0] == 'F':
            x[0] = 2.0
        elif x[0] == 'I':
            x[0] = 3.0

        x = x.astype(np.float)

        y_hat = np.argmax(np.dot(wights, x))

        if y_hat != y:
            y = int(y)
            y_hat = int(y_hat)
            wights[y,:] = wights[y,:] + np.dot(eta(y, x, wights), x)
            wights[y_hat, :] = wights[y_hat, :] + np.dot(eta(y, x, wights), x)

    print("end")
    #print(wights)





preceptron("/home/reut/PycharmProjects/untitled/train_x.txt", "/home/reut/PycharmProjects/untitled/train_y.txt")


