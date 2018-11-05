#!/usr/bin/python
from sklearn.ensemble import RandomForestRegressor

def train(num_trees, training_data_X, training_data_Y):
    forest = RandomForestRegressor(n_estimators=num_trees)
    forest.fit(training_data_X, training_data_Y)
    return forest