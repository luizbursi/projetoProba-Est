#este cógigo analisa quais são os melhores parâmetros desse classsificador

import os, sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #divide os dados em teste e treino
from sklearn.svm import SVC #isso criar a svm para classificação
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



accpoly=[]
accrbf=[]
acclinear=[]
acclinear = np.array(acclinear)
accpoly = np.array(accpoly)
accrbf= np.array(accrbf)
linear_svm = SVC(kernel='linear',random_state=0)
rbf_svm= SVC(kernel='rbf',C=100,gamma='scale',random_state=0)
poly_svm = SVC(kernel='poly', degree=6,C=100, random_state=0)

for k in range(0,100):
    xtrain, xtest, ytrain, ytest = train_test_split(feas_matrix, classes, train_size=0.2, random_state=k)
    linear_svm.fit(xtrain,ytrain)
    rbf_svm.fit(xtrain,ytrain)
    poly_svm.fit(xtrain,ytrain)
    accpoly = np.append(accpoly,poly_svm.score(xtest,ytest))
    accrbf = np.append(accrbf,rbf_svm.score(xtest,ytest))
    acclinear = np.append(acclinear,linear_svm.score(xtest,ytest))

finalArray = []
finalArray = np.array(finalArray)
finalArray = [acclinear,accrbf,accpoly]