from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as lr





class Training:
    def __init__(self, accidents, vehicles, target):
        self.accidents = accidents
        self.vehicles = vehicles
        self.target = target

    def knnTraining():
        knn = KNeighborsClassifier(n_neighbors=15)
        ### Cridar la funcion de fit amb els valors desitjats, buscar l'hiper paràmetre adequat ###

    def decisionTreeTraining(self):
        clf = tree.DecisionTreeClassifier()
        ### Cridar la funcion de fit amb els valors desitjats ###


    def supportVectorMachinesTraining(self):
        svm = SVC.SVC()
        ### Cridar la funcion de fit amb els valors desitjats ###

    def logisticRegressionTraining(self):
        lrt = lr.LogisticRegression(C=1, penalty='l2', tol=0.00001)
        ### Tocar paràmetres i cridar la funció fit ###
