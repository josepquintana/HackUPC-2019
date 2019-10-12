import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

class AccidentsData():
    def __init__(self):
        accidents = pd.read_csv('data/accidents.csv')

        #Eliminar columnes que preeliminarment es consideren irrellevants
        accidents = accidents.drop(columns=['police_force', 'local_authority_district','local_authority_highway',
                                            'lsoa_of_accident_location', 'location_easting_osgr',
                                            'location_northing_osgr'])


        #One hot encoding
        accidents = pd.get_dummies(accidents, columns=['1st_road_class','junction_detail', 'junction_control',
                                                       '2nd_road_class','pedestrian_crossing-human_control',
                                                       'pedestrian_crossing-physical_facilities','light_conditions',
                                                       'road_surface_conditions',
                                                       'special_conditions_at_site','carriageway_hazards'])

        #Eliminar columnes associades a condició de les one hot que són desconegudes
        cols_acaben_menysu = []
        for colname in accidents.columns:
            if colname[-3:] == '_-1':
                cols_acaben_menysu.append(colname)
        accidents = accidents.drop(columns=cols_acaben_menysu)


        numeritza = {'urban_or_rural_area':{'Urban':1,
                                            'Rural':0}
                     }
        accidents.replace(numeritza, inplace=True)

        #Si no hi ha condició excepcional, irrellevant
        accidents = accidents.drop(columns=['special'
                                '_conditions_at_site_None','carriageway_hazards_None', '1st_road_class_Unclassified',
                                '2nd_road_class_Unclassified'])

        #Convertir hh:mm:00 a minuts desde mitjanit
        accidents['time'] = accidents['time'].apply(lambda s: int(s[:-4])*60+int(s[-2:]))

        # Convertir aaaa:mm:dd a minuts desde mitjanit
        accidents['date'] = accidents['date'].apply(lambda s: int(s[7:9]) + int(s[-2:-1])*30.44)

        #Substituïr -10s per avg de la columna
        accidents['2nd_road_number'].replace(-1, np.nan, inplace=True)
        accidents['2nd_road_number'].fillna(accidents['2nd_road_number'].mean(), inplace=True)

        #Normalitzat de les columnes que els cal
        tobenorm = ['longitude','latitude','number_of_vehicles','number_of_casualties','date','time','1st_road_number',
                    'road_type', 'speed_limit','2nd_road_number', 'weather_conditions']
        norm = MinMaxScaler()
        accidents[tobenorm] = pd.DataFrame(norm.fit_transform(accidents[tobenorm]))

        self.target = accidents['target']
        self.features = accidents.drop('target', axis=1)



a = AccidentsData()

vehicles = pd.read_csv('data/vehicles.csv')
#print(vehicles['Age_of_Driver'].value_counts())
# test = pd.read_csv('data/test.csv')

#print(accidents['accident_id'].head())