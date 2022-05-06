# Basic Dataframe Imports
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Visualization Imports
import seaborn as sns
import datetime as dt
import matplotlib.dates as mdates

# Data Preprocessing Imports
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer  
from sklearn.model_selection import GridSearchCV

# Shallow Learning Imports
import sklearn
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.metrics
import sklearn.decomposition
import sklearn.experimental
import sklearn.impute
import sklearn.ensemble

# ANN Imports
import tensorflow as tf
from tensorflow import keras as ks

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

# Removes columns for unemployment rate from X, adds column for unemployment rate in the desired period to Y, produces train/test split
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
    return X,y

# Split the dataset into a training set and test set
def get_train_test_split(X, y, split_idx, timestep = 0): # Use timestep > 0 to split data for RNNs
    # Drop every row in the training set before the row with index split_idx
    X_train = X.drop(labels=range(split_idx - timestep, X.shape[0]), axis=0)
    y_train = y.drop(labels=range(split_idx - timestep, y.shape[0]), axis=0)
    # Drop every row in the test set except the row with index split_idx
    X_test = X.drop(labels=range(split_idx + 1, X.shape[0]), axis=0)
    X_test = X_test.drop(labels=range(0, split_idx - timestep), axis=0)
    y_test = y.drop(labels=range(split_idx + 1, X.shape[0]), axis=0)
    y_test = y_test.drop(labels=range(0, split_idx - timestep), axis=0)
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):
    # Impute data
    imp = IterativeImputer()
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    # Apply a scaler
    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

# Preprocesses data for RNN by creating time series of a desired length
def preprocess_rnn_data(X_train, X_test, y_train, y_test, timestep):
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    X_train_rnn = []
    y_train_rnn = []
    X_test_rnn = []
    y_test_rnn = []

    for i in range(timestep, X_train.shape[0]):
        X_train_rnn.append(X_train[i - timestep:i, :])
        y_train_rnn.append(y_train[i])

    for i in range(timestep, X_test.shape[0]):    
        X_test_rnn.append(X_test[i - timestep:i, :])
        y_test_rnn.append(y_test[i])
    
    X_train_rnn = np.array(X_train_rnn)
    X_test_rnn = np.array(X_test_rnn)
    y_train_rnn = np.array(y_train_rnn)
    y_test_rnn = np.array(y_test_rnn)
    
    return X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn
    
# ============================ Shallow Learning Models ================================

def elastic_net(X_train, y_train, X_test):
    model = sklearn.linear_model.ElasticNet(alpha = 0.1, l1_ratio = 0.1, max_iter = 10000)
    model.fit(X_train,y_train)
    return model.predict(X_test)
    
def linear_regression(X_train, y_train, X_test):        
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train,y_train)
    return model.predict(X_test)

def svr(X_train, y_train, X_test):        
    model = sklearn.svm.SVR()
    model.fit(X_train,y_train)
    return model.predict(X_test)

def knn(X_train, y_train, X_test):
    model = sklearn.neighbors.KNeighborsRegressor()
    model.fit(X_train,y_train)
    return model.predict(X_test)

def random_forest(X_train, y_train, X_test):
    model = sklearn.ensemble.RandomForestRegressor()
    model.fit(X_train,y_train)
    return model.predict(X_test)

# ============================ Deep Learning Models ================================

def fnn(X_train, y_train, X_test):
    input_layer = ks.layers.Input(shape=[X_train.shape[1],])

    h11 = ks.layers.Dense(100, activation="sigmoid")(input_layer)
    h12 = ks.layers.Dense(100, activation="sigmoid")(h11)
    h13 = ks.layers.Dense(100, activation="sigmoid")(h12)

    concat_1 = ks.layers.Concatenate()([input_layer, h13])

    h21 = ks.layers.Dense(50, activation="sigmoid")(concat_1)
    h22 = ks.layers.Dense(50, activation="sigmoid")(h21)
    h23 = ks.layers.Dense(50, activation="sigmoid")(h22)

    concat_2 = ks.layers.Concatenate()([concat_1, h23])

    h31 = ks.layers.Dense(25, activation="sigmoid")(concat_2)
    h32 = ks.layers.Dense(25, activation="sigmoid")(h31)
    h33 = ks.layers.Dense(25, activation="sigmoid")(h32)

    concat_3 = ks.layers.Concatenate()([concat_2, h33])

    output_layer = ks.layers.Dense(1, activation="linear")(concat_3)

    model = ks.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
    early_stopping_callback = ks.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=100, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, validation_split=0.15, callbacks=[early_stopping_callback], verbose=False)
    return model.predict(X_test)

def rnn(X_train, y_train, X_test, timestep):
    input_layer = ks.layers.Input(shape=(timestep, X_train.shape[2]))

    lstm = ks.layers.LSTM(100, return_sequences=False, input_shape=(timestep, X_train.shape[2]))(input_layer) 

    h11 = ks.layers.Dense(100, activation="sigmoid")(lstm)
    h12 = ks.layers.Dense(100, activation="sigmoid")(h11)
    h13 = ks.layers.Dense(100, activation="sigmoid")(h12)

    concat_1 = ks.layers.Concatenate()([lstm, h13])

    h21 = ks.layers.Dense(50, activation="sigmoid")(concat_1)
    h22 = ks.layers.Dense(50, activation="sigmoid")(h21)
    h23 = ks.layers.Dense(50, activation="sigmoid")(h22)

    concat_2 = ks.layers.Concatenate()([concat_1, h23])

    h31 = ks.layers.Dense(25, activation="sigmoid")(concat_2)
    h32 = ks.layers.Dense(25, activation="sigmoid")(h31)
    h33 = ks.layers.Dense(25, activation="sigmoid")(h32)

    concat_3 = ks.layers.Concatenate()([concat_2, h33])

    output_layer = ks.layers.Dense(1, activation="linear")(concat_3)

    model = ks.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
    early_stopping_callback = ks.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=100, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, validation_split=0.15, callbacks=[early_stopping_callback], verbose=False)
    return model.predict(X_test)

def encoder_decoder(X_train, y_train, X_test, timestep):
    input_layer = ks.layers.Input(shape=(timestep, X_train.shape[2]))

    encoder = ks.layers.LSTM(100, return_sequences=False, input_shape=(timestep, X_train.shape[2]))(input_layer) 
    repeat = ks.layers.RepeatVector(timestep)(encoder)
    decoder = ks.layers.LSTM(100, return_sequences=False, input_shape=(timestep, X_train.shape[2]))(repeat)

    output_layer = ks.layers.Dense(1, activation="linear")(decoder)

    model = ks.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
    early_stopping_callback = ks.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=100, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, validation_split=0.15, callbacks=[early_stopping_callback], verbose=False)
    return model.predict(X_test)

# ============================ Model Optimization ================================

# Trains models using the given parameters and returns model with highest validation score
def optimize(X_train, y_train, model, param): 
    opt_model = GridSearchCV(estimator=model, param_grid = param, cv = 10, return_train_score=True, scoring='neg_mean_absolute_error', refit=True, verbose=True)
    # Train and cross-validate, print results
    opt_model.fit(X_train, y_train)
    opt_model_result = pd.DataFrame(opt_model.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
    print(opt_model_result[["param_"+x for x in param.keys()]+ ['mean_test_score']].head)
    return opt_model

# ============================ Main Function ================================

# Logs prediction errors onto a file
def log_results(results): 
    with open("log_file.txt",'a') as f:
        f.write("\n")
        f.write("========================================================================\n")
        for period in results.keys():
            f.write(f"------------------ Period +{period} ------------------\n")
            for result in results[period]:
                f.write(result+"\n")
        f.write("========================================================================\n")

def print_scores(models, errors, focus):
    for i in range(len(models)):
        print(models[i] + " error metrics:")
        print("\tMean:", np.mean(errors[i]))
        print("\tMedian:", np.median(errors[i]))
        print("\tStandard Deviation:", np.std(errors[i]))
        print("\tVariance:", np.var(errors[i]))

def main():
    # Read Data
    data = load_data("FRED_MD.csv")
    # Row that test set should start on
    test_set_starting_row = 530 
    # Contains the number of months we would like to predict unemployment ahead for 
    focuses = [3, 6, 9, 12]
    # Names of models that will be trained and tested; denotes which indices in 'predictions' and 'errors' correspond to which models
    models = ["FNN", "RNN", "Encoder - Decoder", "Elastic Net", "Linear Regression", "SVR", "KNN", "Random Forest"] 
    # 3D array, each outer array stores predictions for all models in a focus period, each inner array stores predictions for a model
    predictions = [[[] for model in range(len(models))] for focus in range(len(focuses))] 
    rnn_timestep = 2

    # Each iteration represents one focus period
    for focus in focuses:  
        # 2D array, each inner array stores error scores for a model
        errors = [[] for model in range(len(models))] 
        focus_num = 0
        print(f"\n======================== Forecasting Period: +{focus} =================================")
        for test_row in range(test_set_starting_row, data.shape[0]):
            model_num = 0
            X, y = get_focus_data(data, focus)
            X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_row - 2)
            X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = get_train_test_split(X, y, test_row - 2, rnn_timestep)
            # Preprocess Data 
            X_train, X_test = preprocess_data(X_train, X_test)
            X_train_rnn, X_test_rnn = preprocess_data(X_train_rnn, X_test_rnn)
            X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = preprocess_rnn_data(X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn, rnn_timestep)

            # ============================ Replication of Previous Papers - Baseline ================================

            # Cook, Thomas R.. Aaron Smalter Hall - Macroeconomic Indicator Forecasting with Deep Neural Networks - 2017
            # FNN
            y_pred = fnn(X_train, y_train, X_test)
            predictions[focus_num][model_num].append(y_pred[0])
            errors[model_num].append(sklearn.metrics.mean_absolute_error(y_test, y_pred))
            model_num += 1

            # RNN
            y_pred = rnn(X_train_rnn, y_train_rnn, X_test_rnn, rnn_timestep)
            predictions[focus_num][model_num].append(y_pred[0])
            errors[model_num].append(sklearn.metrics.mean_absolute_error(y_test, y_pred))
            model_num += 1

            # Encoder-Decoder
            y_pred = encoder_decoder(X_train_rnn, y_train_rnn, X_test_rnn, rnn_timestep)
            predictions[focus_num][model_num].append(y_pred[0])
            errors[model_num].append(sklearn.metrics.mean_absolute_error(y_test, y_pred))
            model_num += 1

            # Smalter Hall, Aaron - Machine Learning Approaches to Macroeconomic Forecasting - 2018
            y_pred = elastic_net(X_train, y_train, X_test)
            predictions[focus_num][model_num].append(y_pred[0])
            errors[model_num].append(sklearn.metrics.mean_absolute_error(y_test, y_pred))
            model_num += 1

            # ============================ Own Work ================================

            # Linear regression
            y_pred = linear_regression(X_train, y_train, X_test)
            predictions[focus_num][model_num].append(y_pred[0])
            errors[model_num].append(sklearn.metrics.mean_absolute_error(y_test, y_pred))
            model_num += 1

            # Baseline SVR
            y_pred = svr(X_train, y_train, X_test)
            predictions[focus_num][model_num].append(y_pred[0])
            errors[model_num].append(sklearn.metrics.mean_absolute_error(y_test, y_pred))
            model_num += 1

            # Baseline KNN
            y_pred = knn(X_train, y_train, X_test)
            predictions[focus_num][model_num].append(y_pred[0])
            errors[model_num].append(sklearn.metrics.mean_absolute_error(y_test, y_pred))
            model_num += 1

            # Baseline Random Forest
            y_pred = random_forest(X_train, y_train, X_test)
            predictions[focus_num][model_num].append(y_pred[0])
            errors[model_num].append(sklearn.metrics.mean_absolute_error(y_test, y_pred))
            model_num += 1
            
        print_scores(models, errors, focus)
        focus_num += 1
    
main()