# Basic Dataframe Imports
from ntpath import join
from tabnanny import verbose
from black import Line
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import itertools

# Shallow Learning Imports
import sklearn
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.metrics
import sklearn.decomposition
import sklearn.experimental
import sklearn.impute
import sklearn.ensemble
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer  
from sklearn.impute import SimpleImputer 
from sklearn import datasets, model_selection, preprocessing, svm, metrics, decomposition
from sklearn.model_selection import GridSearchCV, learning_curve

# ANN Imports
import tensorflow as tf
from tensorflow import keras as ks

import datetime as dt
import matplotlib.dates as mdates

# ============================ Data Visualization ================================

def load_data(filename):
    return pd.read_csv(filename)

def visualize(data):
    plt.plot(data['UNRATE.1'], label = "Next Period Unemployment Rate")
    plt.plot(data['CPI'], label = "Inflation")
    plt.plot(data['ICSA'], label = "Initial Claims")
    plt.plot(data['CCSA'], label = "Continuing Claims")
    x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in data['Date']]
    y = range(len(x)) # many thanks to Kyss Tao for setting me straight here
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(x,y)
    plt.gcf().autofmt_xdate()
    plt.savefig("plot1.jpg")

# ============================ Data Preprocessing ================================

def preprocess_data(X_train, X_test):

    # X_train = X_train.drop("ICSA", axis=1).drop("CCSA", axis=1)
    # X_test = X_test.drop("ICSA", axis=1).drop("CCSA", axis=1)

    # Impute data
    imp = SimpleImputer(missing_values=np.nan, strategy='constant')
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    # Apply a scaler
    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

# Choose label column to predict
def get_focus_data(data, focus):
    y = 0
    
    if focus == 0:
        y = data["UNRATE"]
        X = data.drop("UNRATE", axis=1).drop("UNRATE+1", axis=1).drop("UNRATE+3", axis=1).drop("UNRATE+6", axis=1).drop("UNRATE+9", axis=1).drop("UNRATE+12", axis=1).drop("Date", axis=1)
    else:
        if focus == 1:
            y = data["UNRATE+1"]
        elif focus == 3:
            y = data["UNRATE+3"]
        elif focus == 6:
            y = data["UNRATE+6"]
        elif focus == 9:
            y = data["UNRATE+9"]
        elif focus == 12:
            y = data["UNRATE+12"]
        X = data.drop("UNRATE+1", axis=1).drop("UNRATE+3", axis=1).drop("UNRATE+6", axis=1).drop("UNRATE+9", axis=1).drop("UNRATE+12", axis=1).drop("Date", axis=1)
    return sklearn.model_selection.train_test_split(X, y, test_size = 0.2, shuffle = True)
    
# ============================ Shallow Learning Models ================================

def elastic_net(X_train, y_train, X_test):
    param = {
        "alpha": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        "l1_ratio": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        "max_iter":[10000, 15000, 20000], 
        }
        
    # model = optimize(X_train, y_train, sklearn.linear_model.ElasticNet(), param).fit(X_train,y_train)
    model = sklearn.linear_model.ElasticNet(alpha = 0.1, l1_ratio = 0.1, max_iter = 10000)
    model.fit(X_train,y_train)
    return model.predict(X_test)
    
def linear_regression(X_train, y_train, X_test):
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train,y_train)
    return model.predict(X_test)

def random_forest(X_train, y_train, X_test):
    model = sklearn.ensemble.RandomForestRegressor()
    model.fit(X_train,y_train)
    return model.predict(X_test)

def knn(X_train, y_train, X_test):
    model = sklearn.neighbors.KNeighborsRegressor()
    model.fit(X_train,y_train)
    return model.predict(X_test)

# ============================ Deep Learning Models ================================

def fnn(X_train, y_train, X_test):
    
    input_layer = ks.layers.Input(shape=[X_train.shape[1],])

    h11 = ks.layers.Dense(500, activation="sigmoid")(input_layer)
    h12 = ks.layers.Dense(500, activation="sigmoid")(h11)
    h13 = ks.layers.Dense(500, activation="sigmoid")(h12)

    concat_1 = ks.layers.Concatenate()([input_layer, h13])

    h21 = ks.layers.Dense(250, activation="sigmoid")(concat_1)
    h22 = ks.layers.Dense(250, activation="sigmoid")(h21)
    h23 = ks.layers.Dense(250, activation="sigmoid")(h22)

    concat_2 = ks.layers.Concatenate()([concat_1, h23])

    h31 = ks.layers.Dense(125, activation="sigmoid")(concat_2)
    h32 = ks.layers.Dense(125, activation="sigmoid")(h31)
    h33 = ks.layers.Dense(125, activation="sigmoid")(h32)

    concat_3 = ks.layers.Concatenate()([concat_2, h33])

    output_layer = ks.layers.Dense(1, activation="linear")(concat_3)

    # Create a model using the layers from above
    model = ks.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
    early_stopping_callback = ks.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=100, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, validation_split=0.15, callbacks=[early_stopping_callback], verbose=False)
    return model.predict(X_test)

def rnn(X_train, y_train, X_test):
    
    model = ks.Sequential([
        ks.Input(shape=(None,)),                   # Input for variable-length sequences
        ks.layers.Embedding(max_features, 1),      # Embed each integer in a 1-dimensional vector
        ks.layers.LSTM(1, activation="sigmoid")    # Single LSTM node 
    ])

    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
    early_stopping_callback = ks.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=100, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, validation_split=0.15, callbacks=[early_stopping_callback], verbose=False)
    return model.predict(X_test)

# ============================ Model Optimization ================================

def optimize(X_train, y_train, model, param):

    # Initialize GridSearchCV object with decision tree regressor and hyperparameters
    opt_model = GridSearchCV(estimator=model, param_grid = param, cv = 10, return_train_score=True, scoring='neg_mean_absolute_error', refit=True)
   
    # Train and cross-validate, print results
    opt_model.fit(X_train, y_train)
    # opt_model_result = pd.DataFrame(opt_model.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
    # print(opt_model_result[["param_"+x for x in param.keys()]+ ['mean_test_score']].head)
    return opt_model

# ============================ Main Function ================================
def log_results(results):
    with open("logfile.txt",'w') as f:
        f.write("\n")
        f.write("========================================================================\n")
        for period in results.keys():
            f.write(f"------------------ Period +{period} ------------------\n")
            for result in results[period]:
                f.write(result+"\n")
        f.write("========================================================================\n")

def main():
    # Read Data
    data = load_data("FRED_MD.csv")
    focuses = [0, 1 ,3 ,6 ,9 ,12]
    results = {}
    for focus in focuses:  
        results_verbose = []

        print(f"\n ======================== Forecasting Period: +{focus} =================================")
        X_train, X_test, y_train, y_test = get_focus_data(data, focus)

        # Preprocess Data 
        X_train, X_test = preprocess_data(X_train, X_test)

        # ============================ Replication of Previous Papers - Baseline ================================

        # Cook, Thomas R.. Aaron Smalter Hall - Macroeconomic Indicator Forecasting with Deep Neural Networks - 2017
        y_pred = fnn(X_train, y_train, X_test)
        print(f"FNN MAE(%):\t\t\t\t{round(sklearn.metrics.mean_absolute_error(y_test, y_pred),3)}")
        results_verbose += [f"FNN MAE(%):\t\t\t\t{round(sklearn.metrics.mean_absolute_error(y_test, y_pred),3)}"]

        # Smalter Hall, Aaron - Machine Learning Approaches to Macroeconomic Forecasting - 2018
        y_pred = elastic_net(X_train, y_train, X_test)
        print(f"ElasticNet MAE(%):\t\t\t{round(sklearn.metrics.mean_absolute_error(y_test, y_pred),3)}")
        results_verbose += [f"ElasticNet MAE(%):\t\t\t{round(sklearn.metrics.mean_absolute_error(y_test, y_pred),3)}"]
        # ============================ Own Work ================================
        
        #Baseline linear regression
        y_pred = linear_regression(X_train, y_train, X_test)
        print(f"Linear Regression MAE(%):\t\t{round(sklearn.metrics.mean_absolute_error(y_test, y_pred),3)}")
        results_verbose += [f"Linear Regression MAE(%):\t\t{round(sklearn.metrics.mean_absolute_error(y_test, y_pred),3)}"]

        # Baseline KNN
        y_pred = knn(X_train, y_train, X_test)
        print(f"KNN MAE(%):\t\t\t\t{round(sklearn.metrics.mean_absolute_error(y_test, y_pred),3)}")
        results_verbose += [f"KNN MAE(%):\t\t\t\t{round(sklearn.metrics.mean_absolute_error(y_test, y_pred),3)}"]

        # Baseline Random Forest
        y_pred = random_forest(X_train, y_train, X_test)
        print(f"Random Forest MAE(%):\t\t\t{round(sklearn.metrics.mean_absolute_error(y_test, y_pred),3)}")
        results_verbose += [f"Random Forest MAE(%):\t\t\t{round(sklearn.metrics.mean_absolute_error(y_test, y_pred),3)}"]

        results[focus] = results_verbose

    # log_results(results)

main()


def deprecated():
    y = data["W875RX1"]
    X = data.drop("W875RX1", axis=1).drop("UNRATE+1", axis=1).drop("UNRATE+3", axis=1).drop("UNRATE+6", axis=1).drop("UNRATE+9", axis=1).drop("UNRATE+12", axis=1).drop("Date", axis=1)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, shuffle = True)

    # Preprocess Data 
    X_train, X_test = preprocess_data(X_train, X_test)

    #Baseline linear regression
    y_pred = linear_regression(X_train, y_train, X_test)
    print(f"Linear Regression MAE(%): {sklearn.metrics.mean_absolute_error(y_test, y_pred)}")

    y_pred = knn(X_train, y_train, X_test)
    print(f"KNN MAE(%): {sklearn.metrics.mean_absolute_error(y_test, y_pred)}")

    y_pred = random_forest(X_train, y_train, X_test)
    print(f"Random Forest MAE(%): {sklearn.metrics.mean_absolute_error(y_test, y_pred)}")