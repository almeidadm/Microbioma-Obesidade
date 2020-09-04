import pandas as pd
import timeit
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt


# importar dados formato csv/data
def format_file(filename):
    data = pd.read_csv(filename, sep=",", low_memory=False)
    clm = list(data.columns[1:-1])
    dataset = list()
    for sample in clm:
        dataset.append(list(data[sample]))
    return dataset

#separar classe da amostra
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

# predict calibrated probabilities
def calibrated(trainX, testX, trainy):
	# define model
	model = SVC()
	# define and fit calibration model
	calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=2)
	calibrated.fit(trainX, trainy)
	# predict probabilities
	return calibrated.predict_proba(testX)


if __name__ == '__main__':
    start = timeit.default_timer()

    # Processando arquivo
    dataset = format_file("./data/obesity_abundance.txt")
    for i in range(1, len(dataset[0])):
        str_column_to_float(dataset, i)
    str_column_to_int(dataset, 0)

    dataset, label = split_label(dataset, 0)

    #transformando os arrays em np.arrays para facilitar manipulações
    dataset = np.array(dataset)
    label = np.array(label)

    print('\nDATA')
    print('samples:', len(dataset))
    print('sample distribuition:', np.unique(label, return_counts=True))
    print('features:', len(dataset[0]))


    # Parâmetros a serem testados para otimização dos resultados
    C_parameter = [2**i for i in range(-5, 16, 2)]
    Gamma_parameter = [2**i for i in range(3, -15, -2)]

    # Instanciando um gerador de subconjuntos para cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # Lista para selecionar resultados de acurácia das 5 execuções
    accuracy = list()
    # contador para separar os plots de cada fold
    cnt = 0

    colors = cycle(['green', 'red', 'blue'])

    # Iremos realizar a classificação para cada fold
    for train_ix, test_ix in skf.split(dataset, label):
        cnt = cnt+1
        train_data, test_data = dataset[train_ix], dataset[test_ix]
        train_label, test_label = label[train_ix], label[test_ix]

        uncalibrated_fpr = dict() # false positive
        uncalibrated_tpr = dict() # true positive
        uncalibrated_roc_auc = dict()
        calibrated_fpr = dict()
        calibrated_tpr = dict()
        calibrated_roc_auc = dict()
        uncalibrated_fop = dict()  # prob real
        uncalibrated_mpv = dict()  # prob predita
        calibrated_fop = dict()
        calibrated_mpv = dict()



        # criando um classificador svm oneVSone com parâmetros C e gamma temporariamente fixados
        clf = SVC(C=1, kernel='rbf', decision_function_shape='ovo', probability=True, gamma=Gamma_parameter[5])
        # treinando o modelo usando os sets de treino
        clf.fit(train_data, train_label)
        # classificando
        label_predicted = clf.predict(test_data)
        # Armazenando acurácia da predição
        accuracy.append(metrics.accuracy_score(test_label, label_predicted))
        # Calculando as probabilidades não calibradas de cada classe por amostra
        label_probs = clf.decision_function(test_data)

        # tornando os rótulos de test binários para sua utilização nos próximos métodos
        label_bin = label_binarize(test_label, classes=[0, 1, 2])

        calibrated_label_probs = calibrated(train_data, test_data, train_label)

        # Iniciando o cálculo da roc_curve/auc/roc_auc_score para cada classe
        for i in range(3):
            uncalibrated_fpr[i], uncalibrated_tpr[i], _ = roc_curve(label_bin[:, i], label_probs[:, i])
            calibrated_fpr[i], calibrated_tpr[i], _ = roc_curve(label_bin[:, i], calibrated_label_probs[:, i])
            uncalibrated_roc_auc[i] = auc(uncalibrated_fpr[i], uncalibrated_tpr[i])
            calibrated_roc_auc[i] = auc(calibrated_fpr[i], calibrated_tpr[i])

            uncalibrated_fop[i], uncalibrated_mpv[i] = calibration_curve(label_bin[:, i], label_probs[:, i], n_bins=5, normalize=True)
            calibrated_fop[i], calibrated_mpv[i] = calibration_curve(label_bin[:, i], calibrated_label_probs[:, i], n_bins=5, normalize=True)

        # plotando calibramento perfeito
        plt.plot([0, 1], [0, 1], linestyle='--')

        for i, color in zip(range(3), colors):
            plt.plot(uncalibrated_mpv[i], uncalibrated_fop[i], marker='.', color=color, label='class {0}' ''.format(i))
            plt.legend(loc="lower right")
        fname = './data/images/fold_'+str(cnt)+'_Uncalibrated_SVM_Reliability_Diagram.png'
        plt.savefig(fname)
        plt.clf()

        plt.plot([0, 1], [0, 1], linestyle='--')

        for i, color in zip(range(3), colors):
            plt.plot(calibrated_mpv[i], calibrated_fop[i], marker='.', color=color, label='class {0}' ''.format(i))
            plt.legend(loc="lower right")
        fname = './data/images/fold_'+str(cnt)+'_Calibrated_SVM_Reliability_Diagram.png'
        plt.savefig(fname)
        plt.clf()

        # plot para cada classe não calibrada
        for i, color in zip(range(3), colors):
            plt.plot(uncalibrated_fpr[i], uncalibrated_tpr[i], lw=1.5, color=color, marker=".", label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, uncalibrated_roc_auc[i]))
        # Linha divisória de referência
        plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multi-class data')
        plt.legend(loc="lower right")
        fname = './data/images/fold_'+str(cnt)+'_Uncalibrated_ROC.png'
        plt.savefig(fname)
        plt.clf()


        # plot para cada classe calibrada
        for i, color in zip(range(3), colors):
            plt.plot(calibrated_fpr[i], calibrated_tpr[i], lw=1.5, color=color, marker=".", label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, calibrated_roc_auc[i]))
        # Linha divisória de referência
        plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multi-class data')
        plt.legend(loc="lower right")
        fname = './data/images/fold_'+str(cnt)+'_Calibrated_ROC.png'
        plt.savefig(fname)
        plt.clf()

    accuracy = np.array(accuracy)
    i = accuracy.argmax(axis=0)
    j = accuracy.argmin(axis=0)
    print('Max: %.4f' %accuracy[i])
    print('Min: %.4f' %accuracy[j])

    end = timeit.default_timer()

    print("Time Execution: %.0fs" %(end-start))

