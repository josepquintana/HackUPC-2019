import LoadData as ld
import Training as tr
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def main():
    print("Carregant dades...")
    accidents_data = ld.AccidentsData()
    vehicles_data = ld.VehiclesData()
    merged_data = ld.MergedData(accidents_data, vehicles_data)
    X_test = merged_data.get_merged_test()
    y_test = merged_data.get_target_test()
    X_train = merged_data.get_merged_train()
    y_train = merged_data.get_target_train()

    print('Entrenant models...')
    training = tr.Training(X_train, y_train)
    training.decisionForestTraining()

    print('Calculant predicció...')
    y_pred = training.model.predict(X_test.drop('accident_id', axis=1))
    print('f1 = ', f1_score(y_test,y_pred))

    #target.to_csv(r'output/target.csv')
    #print(merged.loc[merged['accident_id'] == 583864]['number_of_vehicles'])

main()