print("test")

import numpy as np
import pandas as pd
import time
import os as os
import numerox as nx
from numpy import loadtxt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model, model_selection
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import GridSearchCV

os.chdir(os.path.join(os.getcwd()))
data = nx.download("numerai_dataset.zip")

#xgboost setup
class xgboost(nx.Model):

    def __init__(self, params):
        self.p = params
     #   if not HAS_XGBOOST:
      #      raise ImportError("You must install xgboost to use this model")

    def fit_predict(self, dfit, dpre):
        model = xgb.XGBRegressor(learning_rate=self.p['learning_rate'],
                            subsample=self.p['subsample'],
                            max_depth=self.p['max_depth'],
                            n_estimators=self.p['n_estimators'],
                            seed=self.p['seed'],
                            nthread=-1)
        model.fit(dfit.x, dfit.y)
        yhat = model.predict_proba(dpre.x)[:, 1]
        return dpre.ids, yhat

#parameters 
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

#input data
train_data = pd.read_csv('C:/Users/Heidi/Desktop/MMAI Course/FINANCE/Numerai_Dec 29/numerai_training_data.csv', header=0)
tournament = pd.read_csv('C:/Users/Heidi/Desktop/MMAI Course/FINANCE/Numerai_Dec 29/numerai_tournament_data.csv', header=0)
validation = tournament[tournament['data_type']=='validation']
tournaments = ["bernie"]
X = np.array(train_data.loc[:, :"feature50"])
#name = ["bernie", "elizabeth", "jordan", "ken", "charles", "frank", "hillary"]

#output
os.chdir(os.path.join(os.getcwd()))
MODEL_NAME = "xgboosting"
FOLDER_NAME = "submission"
os.chdir(os.path.join(os.getcwd()))

# define kfold cross validation split
kfold_split = 3

# loop through each tournament and print the input for train and validation
for index in range(0, len(tournaments)):
    # get the tournament name
    tournament = tournaments[index]
    
    print ("*********** TOURNAMENT " + tournament + " ***********")
    
    # set the target name for the tournament
    target = "target_" + tournament 
    
    # set the y train with the target variable
    y = train_data.iloc[:, train_data.columns == target].values.reshape(-1,)
    
    # use GroupKFold for splitting the era
    group_kfold = model_selection.GroupKFold(n_splits=kfold_split)
    
    counter = 1
    
    print (">> group eras using kfold split\n")
    
    for train_index, test_index in group_kfold.split(X, y, groups=train_data['era']):
        # X_train takes the 50 features only for training and leave the other columns
        X_train = X[train_index][:,3:]
        # y_train remains the same
        y_train = y[train_index]
        
        print (">> running split #", counter)
        
        print (">> finding best params")
        xgreg = model_selection.GridSearchCV(xgb.XGBRegressor(), parameters, scoring="neg_mean_squared_error", cv=kfold_split)
        xgreg.fit(X_train, y_train)
        best_params = xgreg.best_params_
        print (">> best params: ", best_params)

        # create a new logistic regression model for the tournament
        model = xgboost(best_params)

        print (">> training info:")
        train = nx.backtest(model, data,verbosity=2)

       # print (">> validation info:")
        #validation = nx.production(model, data)

        print (">> saving validation info: ")
        validation.to_csv(MODEL_NAME + "-" + tournament + "-" + str(counter) + ".csv")
        print (">> done saving validation info")

        print ("\n")
        
        counter=counter+1