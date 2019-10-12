from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import numpy as np


class Training:
    def __init__(self, training, target):
        self.training = training
        self.target = target
        self.model = None

    def knnTraining(self):
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(self.training, self.target)
        self.model = knn
        # Cridar la funcion de fit amb els valors desitjats, buscar l'hiper paràmetre adequat

    def decisionTreeTraining(self):
        params = {
                #'criterion': ('gini', 'entropy'),
                #'splitter': ('best', 'random'),
                #'max_depth': ('None', [250, 1000]),
                #'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                #'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                #'min_weight_fraction_leaf': [0, 0.5],
                #'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                #'min_impurity_decrease': [0, 1, 2, 3, 4, 5],
                #'presort': (True, False)
            }

        params = {'max_depth': [750, 1250]}

        dtc = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=-1, verbose=1)
        dtc.fit(self.training, self.target)
        print(dtc.best_estimator_)
        self.model = dtc

        #clf = tree.DecisionTreeClassifier()
        #clf.fit(self.training, self.target)
        #self.model = clf
        # Cridar la funcion de fit amb els valors desitjats

    def supportVectorMachinesTraining(self):
        svm = SVC()
        svm.fit(self.training, self.target)
        self.model = svm
        # Cridar la funcion de fit amb els valors desitjats

    def logisticRegressionTraining(self):
        lrt = LogisticRegression(C=1, penalty='l2', tol=0.00001)
        lrt.fit(self.training, self.target)
        self.model = lrt
        # Tocar paràmetres i cridar la funció fit

