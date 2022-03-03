
# Basic Dataframe Imports
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

def visualize(data):
    print(data.head())
    print(data.shape)
    data.hist(figsize=(14,14))
    plt.savefig("Histo.pdf")
    sns.pairplot(data=data, hue="Potability", kind='kde')
    plt.savefig("PairPlotKDE.pdf")
    
def preprocess_data(X_train, X_test):

    # Impute Missing Values
    # imp = IterativeImputer(max_iter=10, random_state=0)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)
    
    # Apply a scaler
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test
    
def read_data():
    return pd.read_csv("water_potability.csv")
        
def analyze_model(model, X_train):
    print(pd.DataFrame(model.coef_, columns=X_train.columns))

def logistic_regression(X_train, y_train, X_test):
    model = sklearn.linear_model.LogisticRegression(n_jobs=-1)
    model.fit(X_train,y_train)
    return model.predict(X_test)
    
    
def random_forest(X_train,y_train):
    # Cross-validation folds
    k = 10

    # Hyperparameters to tune:
    params = {'n_estimators': range(50,500, 50)}

    # Initialize GridSearchCV object with decision tree classifier and hyperparameters
    grid_forest = sklearn.model_selection.GridSearchCV(estimator=sklearn.ensemble.RandomForestRegressor(verbose =3, n_jobs=-1, bootstrap=True, criterion="squared_error"),
                                                    param_grid=params, 
                                                    cv=k,
                                                    return_train_score=True,
                                                    scoring='neg_mean_absolute_error',
                                                    refit=True) 

    # Train and cross-validate, print results
    grid_forest.fit(X_train, y_train)
    grid_forest_result = pd.DataFrame(grid_forest.cv_results_).sort_values(by=['mean_test_score'], ascending=False)

    print(grid_forest_result[['param_n_estimators', 'mean_test_score']])
    
    return grid_forest

def gradient_boosting(X_train, y_train):
    n_rounds = range(1, 100, 3)
    val_scores = []
    train_scores = []

    # Loop over number of boosting rounds
    for rounds in n_rounds:
        
        # Train classifier
        gb = sklearn.ensemble.GradientBoostingClassifier(n_estimators=rounds)
        
        # Cross-validation scores
        scores = sklearn.model_selection.cross_validate(gb, X_train, y_train, scoring='f1', cv=10, verbose=1, return_train_score=True, n_jobs= -1)
        val_scores.append(scores["test_score"].mean()) 
        train_scores.append(scores["train_score"].mean())

    # Print best score
    print(f'Max Validation Score: {max(val_scores)}')
    # Plot F1 score versus number of boosting rounds
    plt.close()
    sns.lineplot(x=n_rounds, y=val_scores, label="F1 Val Score")
    sns.lineplot(x=n_rounds, y=train_scores, label="F1 Train Score")
    plt.ylim((0, 1.05))
    plt.ylabel("Cross Validation Accuracy")
    plt.xlabel("Number of Boosting Rounds")
    plt.title("Gradient Boosting")
    plt.grid()
    plt.show()

def FNN(X_train, y_train, X_test, y_test):
    model = ks.Sequential()
    model.add(ks.layers.BatchNormalization())
    for n in [256, 128, 64, 32]:
        model.add(ks.layers.Dense(n, activation="relu"))
    model.add(ks.layers.Dense(10, activation="softmax"))
    model.compile(loss= "sparse_categorical_crossentropy",optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=100, validation_split=0.15)
    print(model.summary())
    test_predictions = np.argmax(model.predict(X_test), axis =1)
    test_accuracy = sklearn.metrics.accuracy_score(y_test, test_predictions)
    print(test_accuracy)

def SVC_model(X_train, y_train):
    # Keyword arguments for options above

    kernel = ["rbf"]# if kernel is "polynomial", we'll use the default degree of 3 (cubic)
    cs = [1]

    # Create an empty Pandas dataframe to store the results
    results = pd.DataFrame()

    # Loop over all combinations of the SVC options using the itertools.product() function
    for k, c in itertools.product(kernel, cs):
        
        # create model
        svc = svm.SVC(C=c, kernel=k) 
        
        # Train model and compute average 5-fold cross-validation accuracy
        accuracies = model_selection.cross_val_score(svc, X_train, y_train, n_jobs=-1, verbose=2, cv= 50) 
        avg_accuracy = accuracies.mean()
        
        # Concatenate accuracy results to storage dataframe
        results = pd.concat([results, pd.DataFrame([[k, c, avg_accuracy]], columns=["kernel", "c", "Accuracy"])], ignore_index=True)

    # Sort results by accuracy and display
    print(results.sort_values(by="Accuracy", ascending=False))

def main():
    
    # Read Data
    water_data = read_data()
    
    water_data.hist(figsize=(14,14))
    plt.savefig("histo.pdf")
    # Visualize Data
    #visualize(pd.DataFrame(water_data))
    
    # Break down Data
    y = water_data["Potability"]
    X = water_data.drop("Potability", axis=1)
    X_train_alpha, X_test_alpha, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2, shuffle = True)
    
    # Preprocess Data 
    X_train, X_test = preprocess_data(X_train_alpha, X_test_alpha)
    
    # Supervised Shallow Learning Models
    #gradient_boosting(X_train, y_train)
    #random_forest(X_train,y_train)
    
    #print(f"Accuracy: {sklearn.metrics.accuracy_score(y_test, y_pred).round(2) * 100}%")
    #analyze_model(X_train=X_train_alpha, model=log_regression)

    # Artificial Neural Networks
    svc = svm.SVC(C=1, kernel="rbf") 
    svc.fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    print(f"SVC with RBF Accuracy: {sklearn.metrics.accuracy_score(y_test, y_pred).round(2) * 100}%")

    forest = sklearn.ensemble.RandomForestClassifier(n_jobs=-1)
    forest.fit(X_train,y_train)
    y_pred = forest.predict(X_test)
    print(f"RandomForest Accuracy: {sklearn.metrics.accuracy_score(y_test, y_pred).round(2) * 100}%")

    # FNN(X_train, y_train, X_test, y_test)

    
main()