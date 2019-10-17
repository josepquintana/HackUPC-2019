import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class AccidentsData:
    def __init__(self):
        accidents = pd.read_csv('data/accidents.csv')

        # Eliminar columnes que preeliminarment es consideren irrellevants
        accidents = accidents.drop(columns=['police_force', 'local_authority_district', 'local_authority_highway',
                                            'lsoa_of_accident_location', 'location_easting_osgr',
                                            'location_northing_osgr'])

        # One hot encoding
        accidents = pd.get_dummies(accidents, columns=['1st_road_class', 'junction_detail', 'junction_control',
                                                       '2nd_road_class', 'pedestrian_crossing-human_control',
                                                       'pedestrian_crossing-physical_facilities', 'light_conditions',
                                                       'road_surface_conditions',
                                                       'special_conditions_at_site', 'carriageway_hazards'])

        # Eliminar columnes associades a condició de les one hot que són desconegudes
        cols_acaben_menysu = []
        for colname in accidents.columns:
            if colname[-3:] == '_-1':
                cols_acaben_menysu.append(colname)
        accidents = accidents.drop(columns=cols_acaben_menysu)

        numeritza = {'urban_or_rural_area': {'Urban': 1,
                                             'Rural': 0}
                     }
        accidents.replace(numeritza, inplace=True)

        # Si no hi ha condició excepcional, irrellevant
        accidents = accidents.drop(columns=['special'
                                            '_conditions_at_site_None', 'carriageway_hazards_None',
                                            '1st_road_class_Unclassified',
                                            '2nd_road_class_Unclassified'])

        # Convertir hh:mm:00 a minuts desde mitjanit
        accidents['time'] = accidents['time'].apply(lambda s: int(s[:-4]) * 60 + int(s[-2:]))

        # Convertir aaaa:mm:dd a minuts desde mitjanit
        accidents['date'] = accidents['date'].apply(lambda s: int(s[7:9]) + int(s[-2:-1]) * 30.44)

        # Substituïr -10s per avg de la columna
        accidents['2nd_road_number'].replace(-1, np.nan, inplace=True)
        accidents['2nd_road_number'].fillna(accidents['2nd_road_number'].mean(), inplace=True)

        # Normalitzat de les columnes que els cal
        tobenorm = ['longitude', 'latitude', 'number_of_vehicles', 'number_of_casualties', 'date', 'time',
                    '1st_road_number',
                    'road_type', 'speed_limit', '2nd_road_number', 'weather_conditions']
        norm = MinMaxScaler()
        accidents[tobenorm] = norm.fit_transform(accidents[tobenorm])

        #self.features = accidents.drop('target', axis=1)
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(accidents.drop('target', axis=1),
                                                                            accidents['target'], train_size=.7)

    def get_Xtrain(self):
        return self.Xtrain

    def get_Xtest(self):
        return self.Xtest

    def get_ytrain(self):
        return self.ytrain

    def get_ytest(self):
        return self.ytest


class VehiclesData:
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
        vehicles['Engine_Capacity_(CC)'].replace('-1', np.nan, inplace=True)
        vehicles['Engine_Capacity_(CC)'].fillna(vehicles['Engine_Capacity_(CC)'].mean(), inplace=True)

        vehicles['Age_of_Driver'].replace(-1, np.nan, inplace=True)
        vehicles['Age_of_Driver'].replace('-1', np.nan, inplace=True)
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

        norm = MinMaxScaler()
        vehicles[tobenorm] = norm.fit_transform(vehicles[tobenorm])
        self.valors = vehicles

    def get_valors(self):
        return self.valors


class MergedData:
    def __init__(self, accidents, vehicles):
        acctarg_train = pd.concat([accidents.get_Xtrain(), accidents.get_ytrain()], axis=1)
        acctarg_test = pd.concat([accidents.get_Xtest(), accidents.get_ytest()], axis=1)

        merged_train = pd.merge(acctarg_train, vehicles.get_valors(), on='accident_id')
        merged_test = pd.merge(acctarg_test, vehicles.get_valors(), on='accident_id')

        self.target_train = merged_train['target']
        self.target_test = merged_test['target']

        self.merged_train = merged_train.drop('target', axis=1)
        self.merged_test = merged_test.drop('target', axis=1)


    def get_merged_train(self):
        return self.merged_train

    def get_target_train(self):
        return self.target_train

    def get_merged_test(self):
        return self.merged_test

    def get_target_test(self):
        return self.target_test
