
print("Hello World!")

import numpy as np
import pandas as pd
import time
import os as os
import numerox as nx
from numpy import loadtxt
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model, model_selection
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv('C:/Users/Heidi/Downloads/data/modelling/submission/xgboosting/numerai_dataset/numerai_training_data.csv', header=0)
tournament = pd.read_csv('C:/Users/Heidi/Downloads/data/modelling/submission/xgboosting/numerai_dataset/numerai_tournament_data.csv', header=0)
validation = tournament[tournament['data_type']=='validation']
x_train = np.array(train_data.iloc[:,3:-5])
y_bernie_train = np.array(train_data.loc[:,['target_bernie']])
x_test = np.array(validation.iloc[:,3:-5])
y_bernie_test = np.array(validation.loc[:,['target_bernie']])

#model = xgb.XGBRegressor(learning_rate=0.1, subsample=0.4, max_depth=2, n_estimators=5, seed=0, nthread=-1)
#model.fit(dfit.x, dfit.y)
 #       yhat = model.predict_proba(dpre.x)[:, 1]
#dmatrix=xgb.DMatrix(data=x_train, label=y_bernie_train)
learning_rate = [0.1]
subsample = [0.4]
max_depth = [2]
n_estimators = [5,10]
seed = [2]
parameters = {'learning_rate': learning_rate,
             'subsample': subsample,
             'max_depth': max_depth,
             'n_estimators': n_estimators,
             'seed': seed}

params = {"n_estimators":"10", "learning_rate":"0.1", "subsample":"0.4", "max_depth":"2", "seed":"123"}
max_depths = [2, 5, 10, 20]
best_rmse = []

for curr_val in max_depths:

    params["max_depth"] = curr_val
    
    # Perform cross-validation
    clf = model_selection.GridSearchCV(xgb.XGBClassifier(),params, scoring="neg_log_loss", cv=3)
    clf.fit(x_train,y_bernie_train)
    best_params = clf.best_params_
    print (">> best params: ", best_params)

y_pred = clf.predict_proba(x_test) 
sub["probability"]=y_pred[:,1] 
sub.to_csv("C:/Users/Heidi/Downloads/data/modelling/submission/xgboosting/SimplePrediction.csv", index=False)

#xg_cl = xgb.XGBClassifierer(n_estimators=10, learning_rate=0.1, subsample=0.4, max_depth=2, seed=123) 
#xg_cl.fit(x_train,y_bernie_train) 
#y_pred = xg_cl.predict_proba(x_test) 
##sub["probability"]=y_pred[:,1] 
#sub.to_csv("C:/Users/Heidi/Downloads/data/modelling/submission/xgboosting/SimplePrediction.csv", index=False)