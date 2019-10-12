import pandas as pd
import LoadData as ld
import Training as tr
from sklearn.metrics import f1_score

def main():
    print("Hello World!")
    accidents_data = ld.AccidentsData()
    vehicles_data = ld.VehiclesData()
    merged_data = ld.MergedData(accidents_data, vehicles_data, 0.8)

    X_test = merged_data.get_merged_test()
    y_test = merged_data.get_target_test()
    X_train = merged_data.get_merged_train()
    y_train = merged_data.get_target_train()

    training = tr.Training(X_train, y_train)
    training.knnTraining()

    y_pred = training.model.predict(X_test)


    print(f1_score(y_test,y_pred))

    #target.to_csv(r'output/target.csv')
    #print(merged.loc[merged['accident_id'] == 583864]['number_of_vehicles'])

main()