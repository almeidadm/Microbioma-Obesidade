import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, accuracy_score, make_scorer, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import time, datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import sys, getopt, traceback, os, re

if(not os.path.exists("./Resultados/")):
    os.mkdir("./Resultados/")
if(not os.path.exists("./Resultados/Imagens")):
    os.mkdir("./Resultados/Imagens/")
if(not os.path.exists("./Resultados/Performances/")):
    os.mkdir("./Resultados/Performances/")
if(not os.path.exists("./Resultados/Performances/Treinados")):
    os.mkdir("./Resultados/Performances/Treinados/")
if(not os.path.exists("./Resultados/Performances/NTreinados")):
    os.mkdir("./Resultados/Performances/NTreinados/")
if(not os.path.exists("./Resultados/Parametros_GridSearchCV")):
    os.mkdir("./Resultados/Parametros_GridSearchCV/")
if(not os.path.exists("./Resultados/Features_Selecionadas")):
    os.mkdir("./Resultados/Features_Selecionadas/")

class Files:

    def __init__(self):
        self.inputfile = ''
    def set_name(self):
        self.name = re.search(r'_[\w.-]+o', self.inputfile).group()[1:-2]

class Metrics:

    def __init__(self):
        self.iteration = 1
        self.folds = 5
        self.accuracy = True
        self.precision = True
        self.recall = True
        self.auc = True
        self.f1 = True

def get_arg(argv, metrics, files):

    try:
        opts, args = getopt.getopt(argv, "hi:m:p:s:it:accuracy:precision:recall:auc:f1:split", ["ifile=", "metricsfile=", "parametersfile=","sep=", "iteration=", "accuracymetric=", "precisionmetric=", "recallmetric=", "aucmetric=", "f1metric=", "splitmetric="])
    except getopt.GetoptError:
        print("Unexpected error:", sys.exc_info()[0])
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("\t-i <input table>\n\t-m <output metrics file>\n\t-p <output parameters file>\n\t-s <separator id>\n\t--accuracymetric | -accuracy < True or False>\n\t--precisionmetric | -precision < True or False>\n\t--recallmetric | -recall < True or False>\n\t--aucmetric | -auc < True or False>\n\t--f1metric | -f1 < True or False>\n\t--splitmetric | -split <integer to K-fold>")
        elif opt in ("-i", "--ifile"):
            files.inputfile = arg
            files.set_name()
        elif opt in("-accuracy", "--accuracymetric"):
            if arg == "True":
                metrics.accuracy = True
            elif arg != "False":
                print("Incompatible input to ACCURACY metric")
                sys.exit(2)
        elif opt in("-precision", "--precisionmetric"):
            if arg == "False":
                metrics.precision = False
            elif arg != "True":
                print("Incompatible input to PRECISION metric")
                sys.exit(2)
        elif opt in("-recall", "--recallmetric"):
            if arg == "False":
                metrics.recall = False
            elif arg != "True":
                print("Incompatible input to RECALL metric")
                sys.exit(2)
        elif opt in("-auc", "--aucmetric"):
            if arg == "False":
                metrics.auc = False
            elif arg != "True":
                print("Incompatible input to AUC metric")
                sys.exit(2)
        elif opt in("-f1", "--f1metric"):
            if arg == "False":
                metrics.f1 = False
            elif arg != "True":
                print("Incompatible input to F1 metric")
                sys.exit(2)
        elif opt in("-split", "--splitmetric"):
            if int(arg):
                metrics.folds = int(arg)
            elif arg < 2:
                print("Incompatible input split")
                sys.exit(2)

def pre_processing(df):
    y = LabelEncoder().fit_transform(df[df.columns[-1]])
    X = df.drop(columns=df.columns[-1], axis=1)
    return X, y

def get_metrics(X, y, model, cv, iterations=1, title="model",accuracy=False, precision=False, recall=False, f1=False, auc=False):

    metrics = {
        'accuracy': [accuracy, [], accuracy_score],
        'precision': [precision, [], precision_score],
        'recall': [recall, [], recall_score],
        'f1': [f1, [], f1_score],
        'auc':[auc, [], roc_auc_score]
    }

    importance_count = list()

    for _ in range(iterations):

        for train_idx, test_idx in cv.folds(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            for key in metrics.keys():
                if metrics[key][0]:
                    try:
                        metrics[key][1].append(metrics[key][2](y_test, y_pred))
                    except Exception:
                        traceback.print_exc()
                        metrics[key][1].append("")
                        continue

    desired_metrics = dict()
    for key in metrics.keys():
        if metrics[key][0]:
            desired_metrics[key] = np.mean(metrics[key][1])

    return desired_metrics

def evaluate(X, y, algorithms, file, cv, accuracy=True, precision=True, recall=True, f1=True, it=1, auc=True):
    file.write('n_fatures: ' + str(X.shape[1])+'\n')
    file.write('iterations: '+ str(it)+'\n')
    for key in algorithms.keys():
        start_time = time.time()
        print(key, file=sys.stdout)
        file.write(">"+key+"\n")
        metrics = get_metrics(X=X, y=y, model=algorithms[key][0], cv=cv, iterations=it, title=key,accuracy=accuracy, precision=precision, recall=recall, f1=f1, auc=auc)
        algorithms[key].append(metrics)
        for m_key in metrics.keys():
            print("\t%s :"%m_key, metrics[m_key])
            file.write("\t"+m_key+": " + str(metrics[m_key])+"\n")
            try:
                algorithms[key][0].fit(X,y)
                file.write("\tbest parameters: "+str(algorithms[key][0].best_params_)+"\n")
                file.write("\tbest accuracy score: "+str(algorithms[key][0].best_score_)+"\n")
            except Exception:
                traceback.print_exc()
            file.write("\tTime execution: "+str(datetime.timedelta(seconds=int(time.time()-start_time)))+"\n")

    return

if __name__ == "__main__":

    metrics = Metrics()
    files = Files()

    get_arg(sys.argv[1:], metrics, files)

    if files.inputfile == "":
        sys.exit(2)

    df = pd.read_csv(files.inputfile, sep="\t", engine='python')

    df.dropna(axis=1, how='any', inplace=True)

    feature_name = df['Unnamed: 0']

    df = df.transpose()
    df[df.columns[-1]].head()
    df = df.drop('Unnamed: 0', axis=0)

    X, y = pre_processing(df.astype(str))
    X = X.astype(float)

    skf = StratifiedKFold(n_splits=metrics.folds, shuffle=True)
    scorer = make_scorer(accuracy_score)

#modelos a serem treinados
    nmodels = {
        'gauss': [GaussianProcessClassifier(n_jobs=2),
                {'kernel': [1*kernels.RBF(), 1*kernels.DotProduct(), 1*kernels.Matern(), 1*kernels.RationalQuadratic(), 1*kernels.WhiteKernel()]}],

        'nb': [GaussianNB()],

        'rf': [RandomForestClassifier(),
                {'n_estimators': [10, 50, 100, 200, 500],
                'criterion': ["gini", "entropy"]}],

        'dt': [DecisionTreeClassifier(),
                {"criterion": ["gini", "entropy"],
                "splitter": ["best", "random"]}],

        'svm': [SVC(probability=True, random_state=1),
                {'C': [2**i for i in range(-5, 16, 2)],
                'kernel': ['sigmoid', 'poly', 'rgb', 'linear'],
                'gamma': [2**i for i in range(3, -15, -2)]}],

        'knn': [KNeighborsClassifier(),
                {'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': [1,2]}],

        'perceptron': [Perceptron(random_state=1),
                {'penalty': ['l2','l1','elasticnet'],
                'alpha': [2 ** i for i in range(-5, 16, 2)]}],

        'adaboost': [AdaBoostClassifier(),
                {'n_estimators': [10, 50, 100, 200, 500],
                'learning_rate': [0.25,0.5, 1, 1.5, 2]}],

        'xgboost': [XGBClassifier(n_estimators=500,gamma=0),
                {'max_depth': [4, 6, 8, 10, 100, 1000],
                'learning_rate': [0.001, 0.01],
                'subsample': [0.5, 0.75, 1],
                'colsample_bytree': [0.4, 0.6, 0.8, 1.0]}],

        'mlp': [MLPClassifier(max_iter=1000),
                {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                'solver': ['sgd', 'adam', 'lbfgs']}]
    }

    X_ev, X_tune, y_ev, y_tune = train_test_split(X, y, test_size=0.3, stratify=y)

#otimização de parâmetros
    for model in nmodels.keys():
        if model == "nb":
            continue
        try:
            print(">",model)
            params = GridSearchCV(nmodels[model][0], nmodels[model][1], scoring=make_scorer(roc_auc_score), cv=2, n_jobs=2).fit(X_tune,y_tune).best_params_
            nmodels[model].append(params)
        except Exception:
            print("erro", file=sys.stderr)
            traceback.print_exc()
            continue

#modelos treinados
    models = {
        'gauss': [GaussianProcessClassifier(kernel=nmodels['gauss'][2]['kernel'], n_jobs=2)],

        'nb': [GaussianNB()],

        'rf': [RandomForestClassifier(n_estimators=nmodels['rf'][2]['n_estimators'], criterion=nmodels['rf'][2]['criterion'])],

        'dt': [DecisionTreeClassifier(criterion=nmodels['dt'][2]['criterion'], splitter=nmodels['dt'][2]['splitter'])],

        'svm': [SVC(C=nmodels['svm'][2]['C'],kernel=nmodels['svm'][2]['kernel'], gamma=nmodels['svm'][2]['gamma'],probability=True, random_state=1)],

        'knn': [KNeighborsClassifier(n_neighbors=nmodels['knn'][2]['n_neighbors'], weights=nmodels['knn'][2]['weights'], algorithm=nmodels['knn'][2]['algorithm'], p=nmodels['knn'][2]['p'])],

        'perceptron': [Perceptron(penalty=nmodels['perceptron'][2]['penalty'], alpha=nmodels['perceptron'][2]['alpha'], random_state=1)],

        'adaboost': [AdaBoostClassifier(n_estimators=nmodels['adaboost'][2]['n_estimators'], learning_rate=nmodels['adaboost'][2]['learning_rate'])],

        'xgboost': [XGBClassifier(n_estimators=500, gamma=0, max_depth=nmodels['xgboost'][2]['max_depth'], learning_rate=nmodels['xgboost'][2]['learning_rate'], subsample=nmodels['xgboost'][2]['subsample'], colsample_bytree=nmodels['xgboost'][2]['colsample_bytree'])],

        'mlp': [MLPClassifier(hidden_layer_sizes=nmodels['mlp'][2]['hidden_layer_sizes'], solver=nmodels['mlp'][2]['solver'],max_iter=1000)]
    }

#salvando parametros obtidos
    try:
        param = open("./Resultados/Parametros_GridSearchCV/parametros_"+files.name+".txt", "w+")
        for mod in models.keys():
          param.write(str(models[mod][0]))
          param.write("\n\n")
        param.close()
    except Exception:
        traceback.print_exc()

    output = open("./Resultados/Performances/NTreinados/performance_modelos_ntreinados-"+files.name+".txt", "w+")
    evaluate(X=X_ev, y=y_ev, algorithms=nmodels, file=output, cv=skf, accuracy=metrics.accuracy, precision=metrics.precision, recall=metrics.recall, f1=metrics.f1, it=metrics.iteration, auc=metrics.auc)
    output.close()

    output = open("./Resultados/Performances/Treinados/performance_modelos_treinados-"+files.name+".txt", "w+")
    evaluate(X=X_ev, y=y_ev, algorithms=models, file=output, cv=skf, accuracy=metrics.accuracy, precision=metrics.precision, recall=metrics.recall, f1=metrics.f1, it=metrics.iteration, auc=metrics.auc)
    output.close()

#modelos de feature importances não treinados
    FI_nmodels = {
            'rf': [RandomForestClassifier(),
                    {'n_estimators': [10, 50, 100, 200, 500],
                    'criterion': ["gini", "entropy"]}],

            'dt': [DecisionTreeClassifier(),
                    {"criterion": ["gini", "entropy"],
                    "splitter": ["best", "random"]}],

            'adaboost': [AdaBoostClassifier(),
                    {'n_estimators': [10, 50, 100, 200, 500],
                    'learning_rate': [0.25,0.5, 1, 1.5, 2]}],

            'xgboost': [XGBClassifier(n_estimators=500,gamma=0),
                    {'max_depth': [4, 6, 8, 10, 100, 1000],
                    'learning_rate': [0.001, 0.01],
                    'subsample': [0.5, 0.75, 1],
                    'colsample_bytree': [0.4, 0.6, 0.8, 1.0]}]
        }


#otimizando modelos
    for model in FI_nmodels.keys():
      try:
        print(">",model,file=sys.stdout)
        params = GridSearchCV(FI_nmodels[model][0], FI_nmodels[model][1], scoring=make_scorer(roc_auc_score),  cv=StratifiedKFold(n_splits=5),n_jobs=1).fit(X,y).best_params_
        FI_nmodels[model].append(params)
      except ValueError:
        print("\terro",file=sys.stdout)
        pass

#modelos de feature importances treinados
    FI_models = {
            'rf': [RandomForestClassifier(n_estimators=FI_nmodels['rf'][2]['n_estimators'], criterion=FI_nmodels['rf'][2]['criterion'])],

            'dt': [DecisionTreeClassifier(criterion=FI_nmodels['dt'][2]['criterion'], splitter=FI_nmodels['dt'][2]['splitter'])],

            'adaboost': [AdaBoostClassifier(n_estimators=FI_nmodels['adaboost'][2]['n_estimators'], learning_rate=FI_nmodels['adaboost'][2]['learning_rate'])],

            'xgboost': [XGBClassifier(n_estimators=500, gamma=0, max_depth=FI_nmodels['xgboost'][2]['max_depth'], learning_rate=FI_nmodels['xgboost'][2]['learning_rate'], subsample=FI_nmodels['xgboost'][2]['subsample'], colsample_bytree=FI_nmodels['xgboost'][2]['colsample_bytree'])],
         }

    file = open("./Resultados/Parametros_GridSearchCV/FI_parametros_"+files.name+".txt", "w+")
    for mod in FI_models.keys():
      file.write(str(FI_models[mod][0]))
      file.write("\n\n")
    file.close()

# Tomando importancia de todas as features
    for model in FI_models.keys():
      print("> "+model,file=sys.stdout)
      FI_models[model][0].fit(X, y)
      FI_models[model].append(np.array(FI_models[model][0].feature_importances_))

# Selecionando importancias > 0
    indx = dict()
    for key in FI_models.keys():
      indx[key] = []
      for i in range(len(FI_models[key][1])):
        if FI_models[key][1][i] > 0:
          indx[key].append(i)

    # intersecção de features selecionadas por RF, ADABOOST e XGBOOST
    new_features = np.intersect1d(np.intersect1d(indx['rf'], indx["adaboost"]).tolist(), indx["xgboost"]).tolist()
    file = open("./Resultados/Features_Selecionadas/features_selecionada_"+files.name+".txt", "w+")
    for i in new_features:
        file.write(feature[i]+"\n")
    file.close()


    new_X = X[new_features]

    # performance_modelos_treinados
    file = open("./Resultados/Performances/performance_modelos_ntreinados_com_selecao_de_atributos_"+files.name+".txt", "w+")
    evaluate(X=new_X, y=y, algorithms=modelos_ntreinados, file=file, cv=skf, accuracy=metrics.accuracy, precision=metrics.precision, recall=metrics.recall, f1=metrics.f1, it=metrics.iteration, auc=metrics.auc)
    file.close()

    # performance_modelos_treinados
    file = open("./Resultados/Performances/performance_modelos_treinados_com_selecao_de_atributos_"+files.name+".txt", "w+")
    evaluate(X=new_X, y=y, algorithms=modelos_treinados, file=file, cv=skf,  accuracy=metrics.accuracy, precision=metrics.precision, recall=metrics.recall, f1=metrics.f1, it=metrics.iteration, auc=metrics.auc)
    file.close()
