from random import randrange
from csv import reader
import timeit

#importar dados formato csv/data
def load_csv(filename):
	file = open(filename, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset[:-1]

#transformar colunas de string para Float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

#transformar colunas de string para Int
def strColumn_toInt(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

#normalizar os dados
def normalizeDataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0])/(minmax[i][1] - minmax[i][1])

#avalia a corretude do algoritmo
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

#Divide o dataset em k partes
def crossValidationSplit(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset)/n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

#Avalia o algoritmo usando a cross validation split
def evaluate_algorithm(dataset, algorithm, n_fold, *args):
	folds = crossValidationSplit(dataset, n_fold)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

#-----------------------------------------------------------------------

#Calcula o indice Gini para uma  divisao de dataset
def gini_index(groups, classes):
	#conta todas as amostras como um ponto de divisao
	n_instances = float(sum([len(group) for group in groups]))
	#soma os pesos do indice de Gini para cada grupo
	gini = 0.0
	for group in groups:
		size = float(len(group))
		#evitando dividir por 0
		if size == 0:
			continue
		score = 0.0
		#pontuacao do grupo baseado na pontuacao de cada classe
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		gini += (1.0 - score) * (size / n_instances)
	return gini


#Divide os dados baseado em um atributo e seu valor
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

#seleciona o melhor ponto de divisao para o dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999,999,999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini<b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index': b_index, 'value': b_value, 'groups': b_groups}

#cria um valor de no terminal
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

#cria divisoes filhas para um no ou terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	#confere para um nao divisao
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left+right)
		return
	#confere a profundidade maxima
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	#processa o filho esquerdo
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	#processa o filho direito
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

#constroi uma arvore de decisao
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

#Faz uma predicao com a arvore de decisao
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

#Classificacao e algoritmo de arvore de regressao
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return predictions

if __name__ == '__main__':
	start = timeit.default_timer()
	filename = "./data/obesity.csv"
	
	dataset = load_csv(filename)
	for i in range(len(dataset[0])-1):
		str_column_to_float(dataset, i)

	str_column_to_float(dataset, len(dataset[0])-1)

	n_folds = 10
	max_depth = 110
	min_size = 10
	scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)

	print('Scores: %s' % scores)
	print('Mean Accuracy: %5.3f%%' % (sum(scores)/float(len(scores))))
	
	stop = timeit.default_timer()
	print('\nTime: ', stop-start)	

	
#	tree = build_tree(dataset, max_depth, min_size)
#	row = [5.7,2.9,4.2,1.3]
#	label = predict(tree, row)
	
#	print('\nData=%s, Predicted: %s' % (row, label))
