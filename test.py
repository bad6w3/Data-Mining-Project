#!/usr/bin/python
from sklearn.ensemble import RandomForestRegressor

def test(testing_data_X, forest):
   forest_y = forest.predict(testing_data_X)
   return forest_y
