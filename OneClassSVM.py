import os, sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #divide os dados em teste e treino
from sklearn.svm import SVC #isso criar a svm para classificação
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV #ira fazer cross validation





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
    
    
    
ocsvm = OneClassSVM(kernel = 'rbf', gamma='scale',degree=6,nu=0.03)
print(ocsvm)

ocsvm.fit(feas_matrix,classes)
x = ocsvm.predict(feas_matrix)


matrixOCSVM=[0,0,0,0,0,0,0,0,0,0,0]
listSameClass = []
auxCLASS=0
lastCLASS = classes[0]
for i in range(0,4752):
    if classes[i] != lastCLASS:
        lastCLASS = classes[i]
        matrixOCSVM[auxCLASS] = listSameClass
        listSameClass=[]
        auxCLASS+=1
    listSameClass.append(x[i])   
