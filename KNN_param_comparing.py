#este cógigo analisa quais são os melhores parâmetros desse classsificador




import os, sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #divide os dados em teste e treino
from sklearn import neighbors
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
    
    

param_grid = [{'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
               'weights': ['uniform','distance'],
               'algorithm': ['brute','kd_tree', 'ball_tree']}]

xtrain,xtest,ytrain,ytest = train_test_split(feas_matrix, classes,train_size=0.2,random_state=0)
op = GridSearchCV(neighbors.KNeighborsClassifier(),param_grid,cv=5,scoring='accuracy',verbose=2)
op.fit(xtrain,ytrain)
bestParams = op.best_params_
knn = neighbors.KNeighborsClassifier(n_neighbors=bestParams['n_neighbors'], weights=bestParams['weights'],algorithm=bestParams['algorithm'])


acc=[]
acc= np.array(acc)
for k in range(0,100):
    xtrain, xtest, ytrain, ytest = train_test_split(feas_matrix, classes, train_size=0.2, random_state=k)
    knn.fit(xtrain,ytrain)
    score = knn.score(xtest,ytest)
    acc = np.append(acc,score)

