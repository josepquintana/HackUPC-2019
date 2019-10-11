import pandas as pandas

accidents = pandas.read_csv('data/accidents.csv')
vehicles = pandas.read_csv('data/vehicles.csv')
test = pandas.read_csv('data/test.csv')

print('Vehicles')
print(vehicles.head())