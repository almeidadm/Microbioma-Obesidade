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
    return dataset

# transformar colunas de string para Float
def strColumn_toFloat(dataset, column):
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


def split_label(dataset, r):
    label = dict()
    for i in range(len(dataset[r])):
        if dataset[r][i] not in label:
            label[dataset[r][i]] = [i]
        else:
            label[dataset[r][i]].append(i)
    data_dict = dict()
    for key in label.keys():
        for i in label[key]:
            line 


if __name__ == '__main__':
    start = timeit.default_timer()

    dataset = load_csv("./data/obesity_abundance.txt")
    dataset, label = split_label(dataset[1:], 1)
    print(label)


def asd():
    dataset = []
    print(dataset[0][-1])
    for i in range(len(dataset[0]) - 1):
        strColumn_toFloat(dataset, i)

    # strColumn_toInt(dataset, len(dataset[0])-1)

    dataset, label = split_label(dataset, len(dataset[0]) - 1)

    # dividindo dataset em treino e teste
    data_train, data_test, label_train, label_test = train_test_split(dataset, label, test_size=0.3)

    # criando um classificador svm
    clf = svm.SVC(kernel='linear')

    # treinando o modelo usando os sets de treino
    clf.fit(data_train, label_train)

    # predizendo para o dataset

    y_pred = clf.predict(data_test)

    # print(label_pred)

    print("Accuracy: ", metrics.accuracy_score(label_test, y_pred))
    # Para predições binárias
    # print("Precision: ", metrics.precision_score(label_test, y_pred))
    # print("Recall: ", metrics.recall_score(label_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(label_test, y_pred))

    stop = timeit.default_timer()
    print('\nTime: ', stop - start)

