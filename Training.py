from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


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
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.training, self.target)
        self.model = clf
        # Cridar la funcion de fit amb els valors desitjats

    def decisionForestTraining(self):
        params = {'n_estimators': [10, 50,150],
                  'max_depth': [None, 15, 100]}
        gs = GridSearchCV(RandomForestClassifier(), params, n_jobs=-1, verbose=1)
        gs.fit(self.training.drop('accident_id', axis=1), self.target)
        print(gs.best_estimator_)
        self.model = gs.best_estimator_

    def supportVectorMachinesTraining(self):
        svm = SVC()
        svm.fit(self.training, self.target)
        self.model = svm
        # Cridar la funcion de fit amb els valors desitjats

    def logisticRegressionTraining(self):
        params = {'penalty':['l1', 'l2'],
                  'tol':[0.00001, 0.000001],
                  'C':[0.25,0.5,0.75,1],
                  'solver':['saga']}
        gs = GridSearchCV(LogisticRegression(), params, n_jobs=-1, verbose=1)
        gs.fit(self.training, self.target)
        print(gs.best_estimator_)
        self.model = gs.best_estimator_
        # Tocar paràmetres i cridar la funció fit

    def MLPTraining(self):
        #params = {'solver':['sgd','adam'],'activation':['logistic','tanh','relu'], 'learning_rate':['invscaling','adaptive']}
        #gs = GridSearchCV(MLPClassifier(), params, n_jobs=-1, verbose=1)
        #gs.fit(self.training, self.target)
        #print(gs.best_estimator_)
        gs = MLPClassifier(max_iter=100, learning_rate='invscaling', hidden_layer_sizes=(170,110,50))
        gs.fit(self.training.drop('accident_id', axis=1), self.target)
        self.model = gs
