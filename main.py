import numpy
import csv

import numpy as np


def read_csv():
    matrix = np.ndarray(shape=(5, 4), dtype=float)
    with open('data.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in csv_reader:
            matrix[i] = np.array([float(x) for x in row])
            i += 1
        csvfile.close()
    return [matrix[:-1][:], matrix[-1:][:]]


def criteria_vald(payment_matrix):
    index_dict = {}
    for i in range(0, payment_matrix.shape[0]):
        index_dict[i] = np.min(payment_matrix[i])

    strategy = [key for key, value in index_dict.items() if value == max(index_dict.values())]

    return strategy[0]


def criteria_optimism(payment_matrix):
    index_dict = {}
    for i in range(0, payment_matrix.shape[0]):
        index_dict[i] = np.max(payment_matrix[i])

    strategy = [key for key, value in index_dict.items() if value == max(index_dict.values())]

    return strategy[0]


def criteria_pess(payment_matrix):
    index_dict = {}
    for i in range(0, payment_matrix.shape[0]):
        index_dict[i] = np.min(payment_matrix[i])

    strategy = [key for key, value in index_dict.items() if value == min(index_dict.values())]

    return strategy[0]


def gurvits_criteria(payment_matrix, lmbd):
    index_dict = {}
    for i in range(0, payment_matrix.shape[0]):
        index_dict[i] = np.max(payment_matrix[i]) * lmbd + np.min(payment_matrix[i]) * (1.0 - lmbd)

    strategy = [key for key, value in index_dict.items() if value == max(index_dict.values())]

    return strategy[0]


def servig_criteria(risk_matr):
    index_dict = {}
    for i in range(0, risk_matr.shape[0]):
        index_dict[i] = np.max(risk_matr[i])

    strategy = [key for key, value in index_dict.items() if value == min(index_dict.values())]

    return strategy[0]


def risk_matrix(payment_matrix):
    max_values = list()
    for i in range(0, payment_matrix.shape[0]):
        max_values.append(np.max(payment_matrix[:, i]))

    risk_matr = np.ndarray(shape=(4, 4), dtype=float)
    for i in range(0, payment_matrix.shape[0]):
        for j in range(0, payment_matrix.shape[1]):
            risk_matr[i, j] = abs(payment_matrix[i, j] - max_values[j])

    return risk_matr


########################################

# with risk

#########################################


def bayes(payment_matrix, probabilities):
    index_dict = {}
    for i in range(0, payment_matrix.shape[0]):
        index_dict[i] = np.sum(payment_matrix[i] * probabilities)

    strategy = [key for key, value in index_dict.items() if value == max(index_dict.values())]
    return [strategy[0], index_dict[strategy[0]]]


def bayes_risk(risk_matrix, probabilities):
    index_dict = {}
    for i in range(0, risk_matrix.shape[0]):
        index_dict[i] = np.sum(risk_matrix[i] * probabilities)

    strategy = [key for key, value in index_dict.items() if value == min(index_dict.values())]
    return [strategy[0], index_dict[strategy[0]]]


def hodzha_lemana(payment_matrix, probabilities, mu):
    index_dict = {}
    for i in range(0, payment_matrix.shape[0]):
        index_dict[i] = mu * np.sum(payment_matrix[i] * probabilities) + (1.0 - mu) * np.min(payment_matrix[i])

    strategy = [key for key, value in index_dict.items() if value == max(index_dict.values())]
    return [strategy[0], index_dict[strategy[0]]]


def hodzha_lemana_risk(risk_matr, probabilities, mu):
    index_dict = {}
    for i in range(0, risk_matr.shape[0]):
        index_dict[i] = mu * np.sum(risk_matr[i] * probabilities) + (1.0 - mu) * np.max(risk_matr[i])

    strategy = [key for key, value in index_dict.items() if value == min(index_dict.values())]
    return [strategy[0], index_dict[strategy[0]]]


def geimer(payment_matrix, probabilities):
    index_dict = {}
    for i in range(0, payment_matrix.shape[0]):
        index_dict[i] = np.min(payment_matrix[i] * probabilities)

    strategy = [key for key, value in index_dict.items() if value == max(index_dict.values())]
    return [strategy[0]]


def main():
    lmbd = mu = 0.6
    strategy = {0: 0, 1: 0, 2: 0, 3: 0}
    payment_matrix = read_csv()[0]
    probabilities = read_csv()[1][0]
    risk_matrix(payment_matrix)
    strategy[criteria_vald(payment_matrix)] += 1
    strategy[criteria_optimism(payment_matrix)] += 1
    strategy[criteria_pess(payment_matrix)] += 1
    strategy[gurvits_criteria(payment_matrix, lmbd)] += 1
    strategy[servig_criteria(risk_matrix(payment_matrix))] += 1
    print("With uncertainty: ")
    print(strategy)
    ############################
    strategy = {0: 0, 1: 0, 2: 0, 3: 0}

    strategy[bayes(payment_matrix, probabilities)[0]] += 1
    strategy[bayes_risk(risk_matrix(payment_matrix), probabilities)[0]] += 1
    strategy[hodzha_lemana(payment_matrix, probabilities, mu)[0]] += 1
    strategy[hodzha_lemana_risk(risk_matrix(payment_matrix), probabilities, mu)[0]] += 1
    strategy[geimer(payment_matrix, probabilities)[0]] += 1
    print("With risk: ")
    print(strategy)
main()
