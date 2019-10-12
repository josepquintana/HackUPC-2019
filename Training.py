from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.kernel_approximation import RBFSampler


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
        print(sgd.best_estimator_)
        self.model = sgd.best_estimator_

        #sgd = SGDClassifier(max_iter=1000, tol=1e-3)
        #sgd.fit(self.training, self.target)
        #self.model = sgd

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
        self.model = dtc.best_estimator_

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

