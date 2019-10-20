import os
import LoadData as ld
import Training as tr
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def main():
    #os.chdir('../') # Set working directory

    print("\nStarting program.\n")

    print("Loading data...\n")
    accidents_data = ld.AccidentsData()
    vehicles_data = ld.VehiclesData()
    merged_data = ld.MergedData(accidents_data, vehicles_data)
    X_test = merged_data.get_merged_test()
    y_test = merged_data.get_target_test()
    X_train = merged_data.get_merged_train()
    y_train = merged_data.get_target_train()

    print("Available Models:\n")
    print("1. K-nearest Neighbors")
    print("2. Stochastic Gradient Descent Classifier")
    print("3. Decision Tree Classifier")
    print("4. Random Forest Classifier")
    print("5. C-Support Vector Classification")
    print("6. Logistic Regression")
    print("7. Multi-Layer Perceptron Classifier")
    print("\n")

    mode = input("Choose Training Model: ")

    print('\nTraining model...\n')
    training = tr.Training(X_train, y_train)

    if mode == "1":
        training.knnTraining()
    elif mode == "2":
        training.sgdClassifierTraining()
    elif mode == "3":
        training.decisionTreeTraining()
    elif mode == "4":
        training.supportVectorMachinesTraining()
    elif mode == "5":
        training.supportVectorMachinesTraining()
    elif mode == "6":
        training.logisticRegressionTraining()
    elif mode == "7":
        training.mlpTraining()
    else:
        print("Bye!")
        quit()

    print('Calculating prediction...')
    y_pred = training.model.predict(X_test.drop('accident_id', axis=1))
    print('F1 score = ', f1_score(y_test,y_pred))

main()