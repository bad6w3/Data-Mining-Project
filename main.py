#!/usr/bin/python
import train
import test
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from pandas import read_csv
from pandas import concat
from pandas import Series

#Read in CSV's Change these to the path to downloaded csv's
train_df = read_csv("train.csv")
test_df = read_csv("test.csv")
#Pull out Y value for training data
y = train_df.iloc[0:1461,-1]
#Pull out X values for training data
x = train_df.iloc[0:1461,1:80]
#Pull out X values for testing data
x_test = test_df.iloc[0:1460,1:80]
x_test_id = test_df.iloc[0:1460, 0]
#Data Preprocessing
#Declare Numeric values for Categorical Attributes
numeric_vals = {"MSZoning": {"A": 1, "C": 2, "FV": 3, "I": 4, "RH": 5, "RL": 6, "RP": 7, "RM": 8, "C (all)": 9},
                "Street": {"Grvl": 1, "Pave": 2},
                "Alley": {"Grvl": 1, "Pave": 2},
                "LotShape": {"Reg": 1, "IR1": 2, "IR2": 3, "IR3": 4},
                "LandContour": {"Lvl": 1, "Bnk": 2, "HLS": 3, "Low": 4},
                "Utilities": {"AllPub": 1, "NoSewr": 2, "NoSeWa": 3, "ELO": 4},
                "LotConfig": {"Inside": 1, "Corner": 2, "CulDSac": 3, "FR2": 4, "FR3": 5},
                "LandSlope": {"Gtl": 1, "Mod": 2, "Sev": 3},
                "Neighborhood": {"Blmngtn": 1, "Blueste": 2, "BrDale": 3, "BrkSide": 25,"ClearCr": 4, "CollgCr": 5, "Crawfor": 6, "Edwards": 7, "Gilbert": 8, "IDOTRR": 9,
                                 "MeadowV": 10, "Mitchel": 11, "Names": 12, "NAmes": 12, "NoRidge": 13, "NPkVill": 14, "NridgHt": 15, "NWAmes": 16, "OldTown": 17,
                                 "SWISU": 18, "Sawyer": 19, "SawyerW": 20, "Somerst": 21, "StoneBr": 22, "Timber": 23, "Veenker": 24},
                "Condition1": {"Artery": 1, "Feedr": 2, "Norm": 3, "RRNn": 4, "RRAn": 5, "PosN": 6, "PosA": 7, "RRNe": 8, "RRAe": 9},
                "Condition2": {"Artery": 1, "Feedr": 2, "Norm": 3, "RRNn": 4, "RRAn": 5, "PosN": 6, "PosA": 7, "RRNe": 8, "RRAe": 9},
                "BldgType": {"1Fam": 1, "2FmCon": 2, "2fmCon": 2, "Duplx": 3, "Duplex": 3, "TwnhsE": 4, "TwnhsI": 5, "Twnhs": 6},
                "HouseStyle": {"1Story": 1, "1.5Fin": 2, "1.5Unf": 3, "2Story": 4, "2.5Fin": 5, "2.5Unf": 6, "SFoyer": 7, "SLvl": 8},
                "RoofStyle": {"Flat": 1, "Gable": 2, "Gambrel": 3, "Hip": 4, "Mansard": 5, "Shed": 6},
                "RoofMatl": {"ClyTile": 1, "CompShg": 2, "Membran": 3, "Metal": 4, "Roll": 5, "Tar&Grv": 6, "WdShake": 7, "WdShngl": 8},
                "Exterior1st": {"AsbShng": 1, "AsphShn": 2, "BrkComm": 3, "Brk Cmn": 3, "BrkFace": 4, "CBlock": 5, "CemntBd": 6, "CmentBd": 6, "HdBoard": 7, "ImStucc": 8,
                                "MetalSd": 9, "Other": 10, "Plywood": 11, "PreCast": 12, "Stone": 13, "Stucco": 14, "VinylSd": 15,
                                "Wd Sdng": 16, "WdShing": 17, "Wd Shng": 17},
                "Exterior2nd": {"AsbShng": 1, "AsphShn": 2, "BrkComm": 3, "Brk Cmn": 3, "BrkFace": 4, "CBlock": 5, "CemntBd": 6, "CmentBd": 6, "HdBoard": 7, "ImStucc": 8,
                                "MetalSd": 9, "Other": 10, "Plywood": 11, "PreCast": 12, "Stone": 13, "Stucco": 14, "VinylSd": 15,
                                "Wd Sdng": 16, "WdShing": 17, "Wd Shng": 17},
                "MasVnrType": {"BrkCmn": 1, "Brk Cmn": 1, "BrkFace": 2, "CBlock": 3, "None": 4, "Stone": 5},
                "ExterQual": {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5},
                "ExterCond": {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5},
                "Foundation": {"BrkTil": 1, "CBlock": 2, "PConc": 3, "Slab": 4, "Stone": 5, "Wood": 6},
                "BsmtQual": {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5},
                "BsmtCond": {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5},
                "BsmtExposure": {"Gd": 1, "Av": 2, "Mn": 3, "No": 4},
                "BsmtFinType1": {"GLQ": 1, "ALQ": 2, "BLQ": 3, "Rec": 4, "LwQ": 5, "Unf": 6},
                "BsmtFinType2": {"GLQ": 1, "ALQ": 2, "BLQ": 3, "Rec": 4, "LwQ": 5, "Unf": 6},
                "Heating": {"Floor": 1, "GasA": 2, "GasW": 3, "Grav": 4, "OthW": 5, "Wall": 6},
                "HeatingQC": {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5},
                "CentralAir": {"N": 1, "Y": 2},
                "Electrical": {"SBrkr": 1, "FuseA": 2, "FuseF": 3, "FuseP": 4, "Mix": 5},
                "KitchenQual": {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5},
                "Functional": {"Typ": 1, "Min1": 2, "Min2": 3, "Mod": 4, "Maj1": 5, "Maj2": 6, "Sev": 7, "Sal": 8},
                "FireplaceQu": {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5},
                "GarageType": {"2Types": 1, "Attchd": 2, "Basment": 3, "BuiltIn": 4, "CarPort": 5, "Detchd": 6},
                "GarageFinish": {"Fin": 1, "RFn": 2, "Unf": 3},
                "GarageQual": {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5},
                "GarageCond": {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5},
                "PavedDrive": {"Y": 1, "P": 2, "N": 3},
                "PoolQC": {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4},
                "Fence": {"GdPrv": 1, "MnPrv": 2, "GdWo": 3, "MnWw": 4},
                "MiscFeature": {"Elev": 1, "Gar2": 2, "Othr": 3, "Shed": 4, "TenC": 5},
                "SaleType": {"WD": 1, "CWD": 2, "VWD": 3, "New": 4, "COD": 5, "Con": 6, "ConLw": 7, "ConLI": 8, "ConLD": 9, "Oth": 10},
                "SaleCondition": {"Normal": 1, "Abnorml": 2, "AdjLand": 3, "Alloca": 4, "Family": 5, "Partial": 6}}
#Apply these numeric values to both the training and testing X values
x.replace(numeric_vals, inplace=True)
x.fillna(0, inplace=True)
x_test.replace(numeric_vals, inplace=True)
x_test.fillna(0, inplace=True)

#Declare number of trees in the forest
num_trees = 50
#Generate Trained Random Forest
forest = train.train(num_trees, x.values, y.values)
#Generate Prediction
forest_y = test.test(x_test.values, forest)
#Format for Kaggle Submission
final_group = concat([x_test_id, Series(forest_y)], axis=1)
final_group.columns = ['Id', 'SalePrice']
final_group.to_csv("Results.csv", sep=',', index=False)
