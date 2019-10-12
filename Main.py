import pandas as pd
import LoadData as ld
import Training as tr

def main():
    print("Hello World!")
    accidents_data = ld.AccidentsData()
    vehicles_data = ld.VehiclesData()
    merged_data = ld.MergedData(accidents_data, vehicles_data)

    merged = merged_data.get_merged()
    target = merged_data.get_target()

    merged.to_csv('out/merged.csv', sep=',')
    #target.to_csv('out/target.csv', sep=',')

    training = tr.Training(merged, target)
    #training.knnTraining()

    #target.to_csv(r'output/target.csv')
    #print(merged.loc[merged['accident_id'] == 583864]['number_of_vehicles'])

main()