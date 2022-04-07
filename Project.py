
# Basic Dataframe Imports
from ntpath import join
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

# ANN Imports
import tensorflow as tf
from tensorflow import keras as ks

import datetime as dt
import matplotlib.dates as mdates


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

def load_data(filename):
    return pd.read_csv(filename)
    
def jointPlot(csv):
    # Print out joint plots for each feature linked with next period unemployment in the dataset
    sns.jointplot(x="CPI", y="UNRATE.1", data=csv,
                    kind="reg", truncate=False,
                    xlim=(0, 16), ylim=(2, 12),
                    color="m", height=7)
    plt.savefig("JointPlotCPI.png")
    sns.jointplot(x="ICSA", y="UNRATE.1", data=csv,
                    kind="reg", truncate=False,
                    xlim=(500000, 3500000), ylim=(2, 12),
                    color="m", height=7)
    plt.savefig("JointPlotICSA.png")
    sns.jointplot(x="CCSA", y="UNRATE.1", data=csv,
                    kind="reg", truncate=False,
                    xlim=(750000, 27500000), ylim=(2, 12),
                    color="m", height=7)
    plt.savefig("JointPlotCCSA.png")
    sns.jointplot(x="INDPROD", y="UNRATE.1", data=csv,
                    kind="reg", truncate=False,
                    xlim=(-17, 14), ylim=(2, 12),
                    color="m", height=7)
    plt.savefig("JointPlotIndust.png")
    sns.jointplot(x="CONSUMPTION", y="UNRATE.1", data=csv,
                    kind="reg", truncate=False,
                    xlim=(-4, 15), ylim=(2, 12),
                    color="m", height=7)
    plt.savefig("JointPlotConsum.png")

def analyze_model(model, X_train):
    print(pd.DataFrame(model.coef_, columns=X_train.columns))

def linear_regression(X_train, y_train, X_test):
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train,y_train)
    
    # get importance
    importance = model.coef_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.savefig("JFeatures.png")

    return model.predict(X_test)

def svr(X_train, y_train, X_test):
    model = svm.LinearSVR() 
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

def preprocess_data(X_train, X_test):
    # Apply a scaler
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test

def main():
    
    # Read Data
    data = load_data("econ_data_1.csv")
    
    # Break down Data
    y = data["UNRATE.1"]
    X = data.drop("UNRATE.1", axis=1).drop("Date", axis=1)
    X_train_alpha, X_test_alpha, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, shuffle = True)
    
    # Preprocess Data 
    X_train, X_test = preprocess_data(X_train_alpha, X_test_alpha)

    y_pred = linear_regression(X_train, y_train, X_test)
    print(f"Linear Regression MSE: {sklearn.metrics.mean_squared_error(y_test, y_pred)}")

    # y_pred = knn(X_train, y_train, X_test)
    # print(f"KNN MSE: {sklearn.metrics.mean_squared_error(y_test, y_pred)}")

    # y_pred = svr(X_train, y_train, X_test)
    # print(f"SVR MSE: {sklearn.metrics.mean_squared_error(y_test, y_pred)}")

    # y_pred = random_forest(X_train, y_train, X_test)
    # print(f"Random Forest MSE: {sklearn.metrics.mean_squared_error(y_test, y_pred)}")

main()