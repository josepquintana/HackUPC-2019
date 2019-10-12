import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class VehiclesData():
    def __init__(self):
        vehicles = pd.read_csv('data/vehicles.csv')
        vehicles = vehicles.drop(columns=['Vehicle_IMD_Decile'])
        vehicles = pd.get_dummies(vehicles, columns=['Vehicle_Type', 'Towing_and_Articulation', 'Vehicle_Manoeuvre',
                                                     'Vehicle_Location-Restricted_Lane', 'Junction_Location',
                                                     'Skidding_and_Overturning', 'Hit_Object_in_Carriageway',
                                                     'Vehicle_Leaving_Carriageway', 'Hit_Object_off_Carriageway',
                                                     '1st_Point_of_Impact',
                                                     'Journey_Purpose_of_Driver', 'Propulsion_Code',
                                                     'Driver_IMD_Decile', 'Driver_Home_Area_Type'])

        cols_acabenmenysu = []
        for colname in vehicles.columns:
            if colname[-3:] == '_-1' or colname[-5:] == '_-1.0':
                cols_acabenmenysu.append(colname)
        vehicles = vehicles.drop(columns=cols_acabenmenysu)

        vehicles = vehicles.drop(vehicles[vehicles.Age_of_Driver < 15].index)

        vehicles['Engine_Capacity_(CC)'].replace(-1, np.nan, inplace=True)
        vehicles['Engine_Capacity_(CC)'].fillna(vehicles['Engine_Capacity_(CC)'].mean(), inplace=True)

        vehicles['Age_of_Driver'].replace(-1, np.nan, inplace=True)
        vehicles['Age_of_Driver'].fillna(vehicles['Age_of_Driver'].mean(), inplace=True)

        vehicles['Age_of_Vehicle'].replace(-1, np.nan, inplace=True)
        vehicles['Age_of_Vehicle'].fillna(vehicles['Age_of_Vehicle'].mean(), inplace=True)



        vehicles['Was_Vehicle_Left_Hand_Drive?'].replace(-1, np.nan, inplace=True)
        vehicles['Was_Vehicle_Left_Hand_Drive?'].replace('-1', np.nan, inplace=True)
        vehicles['Sex_of_Driver'].replace(-1, np.nan, inplace=True)
        vehicles['Sex_of_Driver'].replace('-1', np.nan, inplace=True)
        vehicles['Sex_of_Driver'].replace('Not known', np.nan, inplace=True)

        dicvehicles = {'Sex_of_Driver': {'Male': 1.0, 'Female': 0.0},
                       'Was_Vehicle_Left_Hand_Drive?': {'Yes': 1.0, 'No': 0.0}
                       }
        vehicles.replace(dicvehicles, inplace=True)
        vehicles['Was_Vehicle_Left_Hand_Drive?'].fillna(vehicles['Was_Vehicle_Left_Hand_Drive?'].mean(), inplace=True)
        vehicles['Sex_of_Driver'].fillna(vehicles['Sex_of_Driver'].mean(), inplace=True)

        tobenorm = ['Age_of_Driver', 'Engine_Capacity_(CC)', 'Age_of_Vehicle']

        scaler = MinMaxScaler
        norm = MinMaxScaler()
        vehicles[tobenorm] = pd.DataFrame(norm.fit_transform(vehicles[tobenorm]))

        self.valors = vehicles


v = VehiclesData()









