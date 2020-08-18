from csv import reader
import timeit
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# importar dados formato csv/data
def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = dataset[i][:-1]
    file.close()
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
        row[column] = float(row[column].strip())


# transformar colunas de string para Int
def strColumn_toInt(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def reformat_dataset(dataset, row):
    label = dict()
    # get index of each labeled sample
    for i in range(1, len(dataset[row])):
        if dataset[row][i] not in label:
            label[dataset[row][i]] = [i]
        else:
            label[dataset[row][i]].append(i)
    new_dataset = list()
    # append each sample in your respective label into the dict
    for key in label.keys():
        for i in label[key]:
            sample = []
            for j in range(1, len(dataset)):
                sample.append(dataset[j][i])
            sample.append(key)
            new_dataset.append(sample)
    return new_dataset


if __name__ == '__main__':
    start = timeit.default_timer()

    dataset = load_csv("./data/obesity_abundance.txt")
    dataset = reformat_dataset(dataset[2:], 0)

    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)

    strColumn_toInt(dataset, -1)

    dataset, label = split_label(dataset, -1)

    # dividindo dataset em treino e teste
    data_train, data_test, label_train, label_test = train_test_split(dataset, label, test_size=0.3)

    # criando um classificador svm
    clf = svm.SVC(kernel='linear')

    # treinando o modelo usando os sets de treino
    clf.fit(data_train, label_train)

    # predizendo para o dataset
    y_pred = clf.predict(data_test)

    print("Accuracy: ", metrics.accuracy_score(label_test, y_pred))
    print("Precision (average=macro): ", metrics.precision_score(label_test, y_pred, average='macro'))
    print("Recall (average=macro): ", metrics.recall_score(label_test, y_pred, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(label_test, y_pred))

    stop = timeit.default_timer()
    print('\nTime: ', stop - start)
