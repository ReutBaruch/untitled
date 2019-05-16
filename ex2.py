import csv
import random
import numpy as np
import sys

#functions to canculate eta - didn't use it
def eta(r, y, y_hat):
    if r == y:
        return 1
    elif r == y_hat:
        return -1
    return 0

#functions to canculate eta - didn't use it
def eta2(y, x, wights):
    calculate = np.dot(wights, np.dot(y, x))

    if np.all(calculate > 0):
        return 0
    else:
        return -np.dot(y, x)


def PA_loss(x, y, y_hat, w):
    return max(0, 1 - np.dot(w[y, :], x) + np.dot(w[y_hat, :], x))


def _init_weights():
    rows = 3
    columns = 8
    #fill the matrix with 0
    weights = [[0 for column in range(columns)] for row in range(rows)]
    weights = np.asarray(weights)
    weights = weights.astype(np.float)

    return weights


def algorithm(pathX, pathY, pathTest):
    #open files
    train_x = open(pathX, "r")
    train_y = open(pathY, "r")

    reader = csv.reader(train_x, delimiter=',')
    train_x = list(reader)

    train_y = list(train_y)

    #init wights to 0
    preceptron_wights = _init_weights()
    SVM_wights = _init_weights()
    PA_wights = _init_weights()

    #set the eta value
    hiperEta = 2

    #get the lines from files in a random order
    both_Train = list(zip(train_x, train_y))
    random.shuffle(both_Train)
    train_x, train_y = zip(*both_Train)

    #start training
    for iteration in range(20):

        failCountPreceptron = 0
        failCountSVM = 0
        failCountPA = 0
        count = 0

        #reducing the eta in every iteration
        hiperEta = hiperEta / 10

        for line_x, line_y in zip(train_x, train_y):

            y = float(line_y)
            x = np.asarray(line_x)

            if x[0] == 'M':
                x[0] = 0.2
            elif x[0] == 'F':
                x[0] = 0.4
            elif x[0] == 'I':
                x[0] = 0.6

            x = x.astype(np.float)
            #normalizing
            x = x / np.linalg.norm(x)

            #canculating y_hat for every algorithm
            preceptron_y_hat = np.argmax(np.dot(preceptron_wights, x))
            SVM_y_hat = np.argmax(np.dot(SVM_wights, x))
            PA_y_hat = np.argmax(np.dot(PA_wights, x))

            y = int(y)

            count += 1

            #update perceptron
            if preceptron_y_hat != y:
                failCountPreceptron += 1
                preceptron_y_hat = int(preceptron_y_hat)
                preceptron_wights[y, :] = preceptron_wights[y, :] + np.dot(hiperEta, x)
                preceptron_wights[preceptron_y_hat, :] = preceptron_wights[preceptron_y_hat, :] - np.dot(hiperEta, x)

            #update PA
            if PA_y_hat != y:
                failCountPA += 1
                PA_wights[y, :] = PA_wights[y, :] + (PA_loss(x, y, PA_y_hat, PA_wights) / (2 * pow((np.linalg.norm(x)), 2))) * x
                PA_wights[PA_y_hat, :] = PA_wights[PA_y_hat, :] - (PA_loss(x, y, PA_y_hat, PA_wights) / np.linalg.norm(x)) * x

            #update SVM
            if SVM_y_hat != y:
                failCountSVM += 1
                third_y = 0
                y = int(y)
                SVM_y_hat = int(SVM_y_hat)

                if (y == 0) & (SVM_y_hat == 1):
                    third_y = 2
                elif (y == 1) & (SVM_y_hat == 0):
                    third_y = 2
                elif (y == 1) & (SVM_y_hat == 2):
                    third_y = 0
                elif (y == 2) & (SVM_y_hat == 1):
                    third_y = 0
                elif (y == 0) & (SVM_y_hat == 2):
                    third_y = 1
                elif (y == 2) & (SVM_y_hat == 0):
                    third_y = 1

                SVM_wights[y, :] = (1 - 0.075 * hiperEta) * SVM_wights[y, :] + hiperEta * x
                SVM_wights[SVM_y_hat, :] = (1 - 0.075 * hiperEta) * SVM_wights[SVM_y_hat, :] - hiperEta * x
                SVM_wights[third_y, :] = (1 - 0.075 * hiperEta) * SVM_wights[third_y, :]

       # print("Preceptron: ", (failCountPreceptron / count) * 100)
      #  print("SVM: ", (failCountSVM / count) * 100)
     #   print("PA: ", (failCountPA / count) * 100)

    # now workk on the test data after we train
    test = open(pathTest, "r")

    reader = csv.reader(test, delimiter=',')
    test = list(reader)

    for line in test:
        line = np.asarray(line)

        if line[0] == 'M':
            line[0] = 0.2
        elif line[0] == 'F':
            line[0] = 0.4
        elif line[0] == 'I':
            line[0] = 0.6

        line = line.astype(np.float)
        # normalizing
        line = line / np.linalg.norm(line)

        preceptron_y_hat = np.argmax(np.dot(preceptron_wights, line))
        SVM_y_hat = np.argmax(np.dot(SVM_wights, line))
        PA_y_hat = np.argmax(np.dot(PA_wights, line))

        print("perceptron: ", preceptron_y_hat, ", ", "svm: ", SVM_y_hat,", ", "pa: ", PA_y_hat, sep='')


algorithm(sys.argv[1], sys.argv[2], sys.argv[3])

