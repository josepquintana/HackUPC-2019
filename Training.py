from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class Training:
    def __init__(self, training, target):
        self.training = training
        self.target = target
        self.model = 0

    def knnTraining(self):
        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(self.training, self.target)
        # Cridar la funcion de fit amb els valors desitjats, buscar l'hiper paràmetre adequat

    def decisionTreeTraining(self):
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.training, self.target)
        self.model = clf
        # Cridar la funcion de fit amb els valors desitjats

    def supportVectorMachinesTraining(self):
        svm = SVC()
        svm.fit(self.training, self.target)
        # Cridar la funcion de fit amb els valors desitjats

    def logisticRegressionTraining(self):
        lrt = LogisticRegression(C=1, penalty='l2', tol=0.00001)
        lrt.fit(self.training, self.target)
        # Tocar paràmetres i cridar la funció fit

