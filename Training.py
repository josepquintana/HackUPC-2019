from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

    def sgdClassifierTraining(self):
        params = {
            'loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive')
        }
        sgd = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=-1, verbose=1)
        sgd.fit(self.training, self.target)
        self.model = sgd.best_estimator_

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
        self.model = dtc.best_estimator_

    def decisionForestTraining(self):
        params = {'n_estimators': [10, 50,150],
                  'max_depth': [None, 15, 100]}
        gs = GridSearchCV(RandomForestClassifier(), params, n_jobs=-1, verbose=1)
        gs.fit(self.training.drop('accident_id', axis=1), self.target)
        self.model = gs.best_estimator_

    def supportVectorMachinesTraining(self):
        svm = SVC()
        svm.fit(self.training, self.target)
        self.model = svm

    def logisticRegressionTraining(self):
        params = {'penalty':['l1', 'l2'],
                  'tol':[0.00001, 0.000001],
                  'C':[0.25,0.5,0.75,1],
                  'solver':['saga']}
        gs = GridSearchCV(LogisticRegression(), params, n_jobs=-1, verbose=1)
        gs.fit(self.training, self.target)
        self.model = gs.best_estimator_
        # Tocar paràmetres i cridar la funció fit

    def mlpTraining(self):
        #params = {'solver':['sgd','adam'],'activation':['logistic','tanh','relu'], 'learning_rate':['invscaling','adaptive']}
        gs = MLPClassifier(max_iter=100, learning_rate='invscaling', hidden_layer_sizes=(170,110,50))
        gs.fit(self.training.drop('accident_id', axis=1), self.target)
        self.model = gs
