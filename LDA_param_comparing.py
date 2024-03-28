#este cógigo analisa quais são os melhores parâmetros desse classsificador

import os, sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #divide os dados em teste e treino
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV #ira fazer cross validation



#o seguinte bloco de código é para pegar o path até a base, e depois extrair do dados no descritor escolhido
path_pkl = "C:/Users/oluiz/OneDrive/Documentos/IC/Pickles/kthtips2b/"
ChosenPKL = [file for file in os.listdir(path_pkl) if (not file.endswith('CLASS.pkl') and file.endswith('LPQ_kthtips2bFEAS.pkl')) ]
classepkl = [file for file in os.listdir(path_pkl) if file.endswith('CLASS.pkl')]
classepkl = classepkl[0]


with open(path_pkl+classepkl, "rb") as fc:
    classes = pickle.load(fc)
classes = classes["class"]
classes = np.array(classes)


path = path_pkl+ChosenPKL[0]
with open(path, "rb") as f:
    data = pickle.load(f)
    descriptor = data["descriptor"]
    feas_matrix = data["features"]
    feas_matrix = np.array(feas_matrix)
    feas_matrix =feas_matrix.reshape(feas_matrix.shape[0], -1)
    
    
    


param_grid = [{'solver':['svd','lsqr','eigen'],
               'shrinkage': ['auto', 0.1 , 0.2 , 0.5, 1]}]

xtrain,xtest,ytrain,ytest = train_test_split(feas_matrix, classes,train_size=0.2,random_state=0)
op = GridSearchCV(LinearDiscriminantAnalysis(),param_grid,cv=5,scoring='accuracy',verbose=2)
op.fit(xtrain,ytrain)
bestParams = op.best_params_
lda = LinearDiscriminantAnalysis(solver=bestParams['solver'], shrinkage=bestParams['shrinkage'], store_covariance=True)


acc=[]
acc= np.array(acc)
for k in range(0,100):
    xtrain, xtest, ytrain, ytest = train_test_split(feas_matrix, classes, train_size=0.2, random_state=k)
    lda.fit(xtrain,ytrain)
    score = lda.score(xtest,ytest)
    acc = np.append(acc,score)

