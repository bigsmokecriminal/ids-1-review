
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import warnings
# return 0 as default 
# we want our model to be very cautious to save us money!
class default:
    def predict(X,y):
        return [0]

def train_air_model(train):
    
    X = train.loc[:, train.columns != "AIRLINE_DELAY"].values
    y = train.loc[:, "AIRLINE_DELAY"].values

    if len(X) == 0:
        X = [[0, 0]]
        y = [0]
    if len(X) <= 3: 
        X = np.append(X,[X[0]], axis=0)
        X = np.append(X,[X[0]], axis=0)
        y = np.append(y,[y[0]])
        y = np.append(y,[y[0]])

    MLP_param = {
        'max_iter' : [30],
        'hidden_layer_sizes': [(24,12,8)],
        'activation': ['relu'],
        
    }

    mlp = MLPRegressor()
    # to use scorer, run eval from c) code below
    mlp_model =  GridSearchCV(estimator=mlp, param_grid=MLP_param, refit="neg_mean_squared_error",
                                 verbose=0, cv=3, scoring="neg_mean_squared_error")
    mlp_model.fit(X,y)
    estimator = mlp_model.best_estimator_
    #train_f_air = estimator.predict(X)
    #predict_air = estimator.predict(test.loc[:, test.columns != "AIRLINE_DELAY"].values)
    #truth = test["AIRLINE_DELAY"].values
    #print(mean_squared_error(truth, predict_air))
    return estimator


def train_ori_ap_model(train):
    
    X = train.loc[:, train.columns != "DEPARTURE_DELAY"].values
    y = train.loc[:, "DEPARTURE_DELAY"].values
    
    if len(X) == 0:
        X = [[0, 0]]
        y = [0]
    if len(X) <= 3: 
        X = np.append(X,[X[0]], axis=0)
        X = np.append(X,[X[0]], axis=0)
        y = np.append(y,[y[0]])
        y = np.append(y,[y[0]])

    MLP_param = {
        'max_iter' : [30],
        'hidden_layer_sizes': [(24,12,8)],
        'activation': ['relu'],
        
    }
    
    mlp = MLPRegressor()
    
    # to use scorer, run eval from c) code below
    mlp_model =  GridSearchCV(estimator=mlp, param_grid=MLP_param, refit="neg_mean_squared_error",
                                 verbose=0, n_jobs=4, cv=3, scoring="neg_mean_squared_error")
    mlp_model.fit(X,y)
    estimator = mlp_model.best_estimator_
    #train_f_dept = estimator.predict(X)
    #predict_dept = estimator.predict(test.loc[:, test.columns != "DEPARTURE_DELAY"].values)
    #truth = test["DEPARTURE_DELAY"].values
    #print(mean_squared_error(truth, predict_dept))
    return estimator


def estimator_maker(train_set, test_set):
    
    # frist we group out data just as we did with the manual approach
    print("\tGrouping data sets...")
    train_groups_airline = train_set.groupby("AIRLINE")
    #test_groups_airline = test_set.groupby("AIRLINE")
    train_groups_ori_ap = train_set.groupby("ORIGIN_AIRPORT")
    #test_groups_ori_ap = test_set.groupby("ORIGIN_AIRPORT")

    # than we learn the AIRLINE_DELAY per group (per AIRLINE)
    print("\tLearning AIRLINE_DELAY features....")
    dict_airline = {}
    airlines = list(test_set["AIRLINE"].unique())
    for al in airlines:
        #print(name)
        try:
            train = train_groups_airline.get_group(al)[["DAY_YEARLY", "SD_MIN", "AIRLINE_DELAY"]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator = train_air_model(train)
                dict_airline[al] = estimator
        except KeyError as e:
            dict_airline[al] = default()
    print("\tDone")

    # next we learn the DEPARTURE_DELAY per group (per ORIGIN_AIRPORT)
    print("\tLearning DEPARTURE_DELAY features....", end='')
    dict_origin = {}
    airport = list(test_set["ORIGIN_AIRPORT"].unique())
    for ap in airport:
        #print(name)
        try:
            train = train_groups_ori_ap.get_group(ap)[["DAY_YEARLY", "SD_MIN", "DEPARTURE_DELAY"]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator = train_ori_ap_model(train)
                dict_origin[ap] = estimator
        except KeyError as e:
            dict_origin[ap] = default()
    print("Done")

    # now we can apply our models on the test_set
    # here we can build a quick wrapper

    def _feature_maker_wraper(al, ori_ap, day, sd_min):
   
        result = [
            *dict_airline[al].predict([[day, sd_min]]),
            *dict_origin[ori_ap].predict([[day, sd_min]]),
        ]
        
        return pd.Series(result)

    # note that we can do this! We are not learning from test_set.
    #print("\tTaking AIRLINE_DELAY, DEPARTURE_DELAY from train_set (AIRD, DEPD)...", end='')
    #train_set["AIRD"] = train_set["AIRLINE_DELAY"]
    #train_set["DEPD"] = train_set["DEPARTURE_DELAY"]
    print("\tApplying Model on train_set...")
    train_set[["AIRD", "DEPD"]] = train_set.parallel_apply(lambda i: _feature_maker_wraper(
                                        i["AIRLINE"], i["ORIGIN_AIRPORT"],
                                        i["DAY_YEARLY"], i["SD_MIN"]), axis=1)
    print("Done")
    # we have to apply the model on test_set to have these features for predicting!
    print("\tApplying Model on test_set (AIRD, DEPD)...", end='')
    test_set[["AIRD", "DEPD"]] = test_set.parallel_apply(lambda i: _feature_maker_wraper(
                                        i["AIRLINE"], i["ORIGIN_AIRPORT"],
                                        i["DAY_YEARLY"], i["SD_MIN"]), axis=1)
    print("Done")
    return train_set, test_set
