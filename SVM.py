import pandas as pd
import timeit
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# importar dados formato csv/data
def format_file(filename):
    data = pd.read_csv(filename, sep=",", low_memory=False)
    clm = list(data.columns[1:-1])
    dataset = list()
    for sample in clm:
        dataset.append(list(data[sample]))
    return dataset


def split_label(dataset, c):
    label = list()
    for i in range(len(dataset)):
        label.append(dataset[i][c])
        dataset[i] = dataset[i][:c] + dataset[i][c + 1:]
    return dataset, label


# transformar colunas de string para Float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])


# transformar colunas de string para Int
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


if __name__ == '__main__':
    start = timeit.default_timer()

    dataset = format_file("./data/obesity_abundance.txt")

    for i in range(1, len(dataset[0])):
        str_column_to_float(dataset, i)

    str_column_to_int(dataset, 0)

    dataset, label = split_label(dataset, 0)
    # dividindo dataset em treino e teste

    accuracy_scores = list()
    regularization = [pow(2, -i) for i in range(15)]
    for i in range(3):
        regularization.append(pow(2, i))
    for i in regularization:
        data_train, data_test, label_train, label_test = train_test_split(dataset, label, test_size=0.3)

        # criando um classificador svm
        clf = svm.SVC(C=i, kernel='rbf')

        # treinando o modelo usando os sets de treino
        clf.fit(data_train, label_train)

        # predizendo para o dataset
        y_pred = clf.predict(data_test)

        accuracy_scores.append((metrics.accuracy_score(label_test, y_pred), i))
        # print("Confusion Matrix:\n", confusion_matrix(label_test, y_pred))

    accuracy_scores.sort()
    print("\nMean accuracy score: ", mean([i[0] for i in accuracy_scores]))
    print("Highest accuracy score: ", accuracy_scores[-1][0], "with C=", accuracy_scores[-1][1])
    stop = timeit.default_timer()
    print('\nTime: ', stop - start)
