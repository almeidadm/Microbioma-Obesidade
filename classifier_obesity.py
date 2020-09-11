#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from itertools import cycle
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


# In[2]:


file = pd.read_csv("/home/diego/PycharmProjects/Microbioma-Obesidade/data/obesity_abundance.txt", sep=",", low_memory=False)


# In[3]:


file


# In[4]:


file.columns[1:-1]


# In[5]:


samples = file.columns[1:-1]


# In[6]:


dataset = list()
label = list()
for sample in samples:
    if file[sample][0]!= 'n':
        label.append(file[sample][0])
        dataset.append(np.asarray(file[sample][1:], dtype=np.float64))


# In[7]:


dataset = np.array(dataset)
dataset


# In[8]:


label


# In[9]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(label)
label = le.transform(label)


# In[10]:


label


# In[11]:


C_parameter = [2**i for i in range(-5, 16, 2)]
Gamma_parameter = [2**i for i in range(3, -15, -2)]

tuned_parameters = [{'kernel': ['rbf'], 'gamma': Gamma_parameter, 'C': C_parameter}]

scores = ['accuracy', 'recall']

colors = cycle(['green', 'red', 'blue'])


# In[12]:


skf = StratifiedKFold(n_splits=5, shuffle=False)


# In[13]:


svc = SVC(probability=True)


# In[14]:


import timeit
start = timeit.default_timer()
cnt = 0
for train_idx, test_idx in skf.split(dataset, label):
    cnt += 1
    train_data, test_data = dataset[train_idx], dataset[test_idx]
    train_label, test_label = label[train_idx], label[test_idx]
    print("Fold %d" %cnt)
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svc, tuned_parameters, scoring= score )
        clf.fit(train_data, train_label)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        label_pred = clf.predict(test_data)
        print(metrics.classification_report(test_label, label_pred))
end =  timeit.default_timer()
print("Tempo execução: %0.f", end-start)


# In[15]:


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

skf = StratifiedKFold(n_splits=5, shuffle=False)

fig, ax = plt.subplots()
i = 0

for train_idx, test_idx in skf.split(dataset, label):
    train_data, test_data = dataset[train_idx], dataset[test_idx]
    train_label, test_label = label[train_idx], label[test_idx]

    svc = SVC(C=0.03125, kernel='rbf',gamma=8 ,probability=True)

    svc.fit(train_data, train_label)
     
    score = svc.decision_function(test_data)
    predicted = svc.predict(test_data)
      
    fpr, tpr, thresholds = metrics.roc_curve(test_label, score)
    
    roc_auc = metrics.auc(fpr, tpr)
    
   
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc)
    ax.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i+=1
    
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:




